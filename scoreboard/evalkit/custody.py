"""evalkit sealed custody for an independent tester (eval milestone M6, item D7; rule book R23 tooling).

G1, G5, G6. No em dashes.

What it does:

  provider / entrant                     organiser (holds the files, not the key)          tester (holds the key)
  ------------------                     ----------------------------------------          ----------------------
  seal(labels or entry, tester cert) --> sealed .cms files + custody.json + custody log
                                         open_sealed(..., organiser key)  -> refused, logged
                                                                                           open_sealed(..., tester key)
                                                                                           after the freeze -> bytes, logged

- Labels and entries are encrypted to the tester's public certificate. The organiser keeps the sealed files but has
  no private key that opens them. The tester role is one key away: the same call with the tester's private key opens
  them.
- Entries are sealed until the freeze: `seal` refuses a new entry at or after the freeze, and `open_sealed` refuses
  every object before the freeze. The time check is policy, not cryptography (the tester could open early with the
  openssl command line); the log is what makes an early open visible.
- Every seal and every open attempt (opened, refused before the freeze, or failed to decrypt) writes one line to a
  hash-chained custody log before any plaintext is returned. A broken chain refuses every new line. The lines carry
  no key, so the chain catches naive edits only: someone with write access who rewrites a line and recomputes every
  later hash, or who cuts the tail, gets a log that still verifies. The log is append-only in effect only once its
  head is pinned outside the custody folder (committed or stamped); `verify_log(path, expected_head=...)` then
  checks that the pinned line is still in the chain.
- Each sealed file's SHA-256 is logged at seal time, and an open checks it. A file swapped after sealing is refused,
  even though anyone with the public certificate can make a new valid ciphertext.
- The custody record pins the tester certificate's SHA-256, and every seal line names the recipient certificate's
  SHA-256. Sealing to any other certificate is refused. custody.json is checked against the record hash in the
  custody_start log line on every seal, open and verify, so an edited record (a swapped certificate, an earlier
  freeze) is refused. This holds while the log itself is honest; see the chain limit above.

Limits:

- Each seal line carries the plaintext SHA-256 and length, and anyone who reads the log sees them. For a very small
  label set with a known site list and a known file format, the organiser could confirm a guessed file against that
  hash. The ciphertext length already shows the plaintext length.

Crypto: OpenSSL CMS (RFC 5652 / RFC 5083 AuthEnvelopedData), RSA-OAEP with SHA-256 key transport and AES-256-GCM,
through the `openssl` command (OpenSSL 3 or later; LibreSSL is refused). No crypto is written here; this module only
calls a vetted library and keeps the record. Python `cryptography`, `pyrage` and `age` are not installed in this
environment, so the OpenSSL CLI is the vetted library already available. `EVALKIT_OPENSSL` overrides the binary.

What custody does not add (own-work-list check page, D7): it protects label sets that arrive from outside and that
nobody here has read. It adds nothing for label sets we built and already hold or read; sealing those would protect
nothing and must not be described as blind.

Keys: the tester makes the key pair on the tester's machine (`keygen`) and sends only the certificate. Private keys
are never committed. Tests make keys in a temp dir.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

LOG_VERSION = 1
GENESIS = "0" * 64
KINDS = ("labels", "entry")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class CustodyError(Exception):
    """A custody step was refused. The message says why."""


# ---------------------------------------------------------------------------------------------------------------
# OpenSSL
# ---------------------------------------------------------------------------------------------------------------

def openssl_bin() -> str:
    cand = os.environ.get("EVALKIT_OPENSSL") or shutil.which("openssl") or ""
    for c in (cand, "/opt/homebrew/bin/openssl", "/usr/local/bin/openssl"):
        if c and Path(c).exists():
            out = subprocess.run([c, "version"], capture_output=True, text=True).stdout
            m = re.match(r"OpenSSL (\d+)\.", out)
            if m and int(m.group(1)) >= 3:
                return c
    raise CustodyError("no OpenSSL 3 or later found (LibreSSL lacks CMS AES-GCM); set EVALKIT_OPENSSL")


def _run(args: list[str], stdin: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([openssl_bin(), *args], input=stdin, capture_output=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path | str) -> str:
    return sha256_bytes(Path(p).read_bytes())


def cert_sha256(cert: Path | str) -> str:
    """SHA-256 of the certificate in DER form (the fingerprint the tester commits with the acceptance)."""
    r = _run(["x509", "-in", str(cert), "-outform", "DER"])
    if r.returncode != 0:
        raise CustodyError(f"not a certificate: {cert}")
    return sha256_bytes(r.stdout)


def keygen(out_dir: Path | str, name: str = "tester") -> tuple[Path, Path]:
    """Make a key pair: NAME.key.pem (private, mode 0600, never committed) and NAME.cert.pem (public, send this)."""
    if not _NAME.match(name):
        raise CustodyError(f"bad key name {name!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key, cert = out / f"{name}.key.pem", out / f"{name}.cert.pem"
    if key.exists() or cert.exists():
        raise CustodyError(f"refused: {key} or {cert} exists; never overwrite a key")
    old = os.umask(0o077)
    try:
        r = _run(["req", "-x509", "-newkey", "rsa:3072", "-nodes", "-keyout", str(key), "-out", str(cert),
                  "-days", "3650", "-subj", f"/CN=evalkit custody {name}"])
    finally:
        os.umask(old)
    if r.returncode != 0:
        raise CustodyError("openssl keygen failed: " + r.stderr.decode(errors="replace")[-300:])
    os.chmod(key, 0o600)
    return key, cert


def _encrypt(plaintext: bytes, cert: Path) -> bytes:
    r = _run(["cms", "-encrypt", "-binary", "-aes-256-gcm", "-recip", str(cert),
              "-keyopt", "rsa_padding_mode:oaep", "-keyopt", "rsa_oaep_md:sha256", "-outform", "DER"], plaintext)
    if r.returncode != 0 or not r.stdout:
        raise CustodyError("openssl encrypt failed: " + r.stderr.decode(errors="replace")[-300:])
    return r.stdout


def _decrypt(ciphertext: bytes, key: Path) -> bytes | None:
    """Plaintext, or None when this key cannot open it. Plaintext goes to a pipe, never to disk."""
    r = _run(["cms", "-decrypt", "-binary", "-inform", "DER", "-inkey", str(key)], ciphertext)
    return r.stdout if r.returncode == 0 else None


# ---------------------------------------------------------------------------------------------------------------
# Hash-chained custody log
# ---------------------------------------------------------------------------------------------------------------

def _canon(d: dict) -> bytes:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _line_hash(rec: dict) -> str:
    return sha256_bytes(_canon({k: v for k, v in rec.items() if k != "hash"}))


def read_log(path: Path | str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def verify_log(path: Path | str, expected_head: str | None = None) -> str:
    """Check the whole chain. Returns the head hash (GENESIS for an empty log); raises CustodyError on a break.

    The lines are unkeyed, so this catches naive edits only. A rewrite that recomputes every later hash, or a cut
    tail, still verifies. Pass `expected_head`, a head hash pinned outside the custody folder (committed or
    stamped), to require that the pinned line is still in the chain: a rewrite at or before the pinned line, or a
    cut through it, is then refused. Lines after the pin are covered only by a newer pin.
    """
    prev = GENESIS
    seen = expected_head is None or expected_head == GENESIS
    for i, rec in enumerate(read_log(path)):
        if rec.get("seq") != i:
            raise CustodyError(f"custody log broken at line {i + 1}: seq {rec.get('seq')} != {i}")
        if rec.get("prev") != prev:
            raise CustodyError(f"custody log broken at line {i + 1}: prev hash does not match line {i}")
        if rec.get("hash") != _line_hash(rec):
            raise CustodyError(f"custody log broken at line {i + 1}: line hash does not match its content")
        prev = rec["hash"]
        seen = seen or prev == expected_head
    if not seen:
        raise CustodyError("custody log does not contain the pinned head hash; it was rewritten or cut")
    return prev


def append_log(path: Path | str, actor: str, action: str, now: dt.datetime, **fields) -> dict:
    """Append one line under an exclusive lock, after verifying the chain. Never rewrites earlier lines."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        prev = verify_log(p)
        seq = len(read_log(p))
        rec = {"v": LOG_VERSION, "seq": seq, "time_utc": _iso(now), "actor": actor, "action": action, **fields,
               "prev": prev}
        rec["hash"] = _line_hash(rec)
        os.write(fd, _canon(rec) + b"\n")
        os.fsync(fd)
        return rec
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ---------------------------------------------------------------------------------------------------------------
# Custody record, seal, open
# ---------------------------------------------------------------------------------------------------------------

def _iso(t: dt.datetime) -> str:
    if t.tzinfo is None:
        raise CustodyError("times must carry a time zone")
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


@dataclass(frozen=True)
class Custody:
    """A custody folder: custody.json (freeze time, pinned tester cert), sealed/, and custody_log.jsonl."""
    root: Path

    @property
    def record_path(self) -> Path:
        return self.root / "custody.json"

    @property
    def log_path(self) -> Path:
        return self.root / "custody_log.jsonl"

    @property
    def sealed_dir(self) -> Path:
        return self.root / "sealed"

    def record(self) -> dict:
        """custody.json, after checking the log chain and the record hash pinned in the custody_start line."""
        if not self.record_path.exists():
            raise CustodyError(f"no custody record at {self.record_path}")
        verify_log(self.log_path)
        log = read_log(self.log_path)
        if not log or log[0].get("action") != "custody_start":
            raise CustodyError("refused: the custody log does not start with a custody_start line")
        if sha256_file(self.record_path) != log[0].get("record_sha256"):
            raise CustodyError("refused: custody.json differs from the record hash in the custody_start log line")
        return json.loads(self.record_path.read_text())

    def freeze(self) -> dt.datetime:
        return _parse(self.record()["freeze_utc"])


def init_custody(root: Path | str, tester_cert: Path | str, freeze: dt.datetime, actor: str,
                 now: dt.datetime | None = None) -> Custody:
    """Write custody.json once: the freeze time and the tester certificate's SHA-256. Never overwritten."""
    c = Custody(Path(root))
    if c.record_path.exists():
        raise CustodyError(f"refused: {c.record_path} exists; a custody record is written once")
    now = now or _now()
    if freeze <= now:
        raise CustodyError("refused: the freeze must be in the future when custody starts")
    c.root.mkdir(parents=True, exist_ok=True)
    c.sealed_dir.mkdir(exist_ok=True)
    rec = {"v": LOG_VERSION, "freeze_utc": _iso(freeze), "tester_cert_sha256": cert_sha256(tester_cert),
           "cipher": "CMS AuthEnvelopedData, RSA-OAEP-SHA256 key transport, AES-256-GCM (OpenSSL)",
           "note": "Custody protects label sets that arrive from outside unread. It adds nothing for label sets "
                   "the organiser built or has read."}
    c.record_path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    append_log(c.log_path, actor, "custody_start", now, freeze_utc=rec["freeze_utc"],
               tester_cert_sha256=rec["tester_cert_sha256"], record_sha256=sha256_file(c.record_path))
    return c


def _sealed_path(c: Custody, kind: str, name: str) -> Path:
    if kind not in KINDS:
        raise CustodyError(f"kind must be one of {KINDS}")
    if not _NAME.match(name):
        raise CustodyError(f"bad object name {name!r} (letters, digits, . _ - only)")
    return c.sealed_dir / f"{kind}__{name}.cms"


def seal(c: Custody, kind: str, name: str, plaintext: bytes, tester_cert: Path | str, actor: str,
         now: dt.datetime | None = None) -> Path:
    """Encrypt plaintext to the pinned tester certificate and log it. The plaintext is never written to disk here."""
    now = now or _now()
    rec = c.record()  # checks the log chain and the custody.json hash before anything is encrypted
    recip = cert_sha256(tester_cert)
    if recip != rec["tester_cert_sha256"]:
        raise CustodyError("refused: this certificate is not the tester certificate pinned in custody.json")
    if kind == "entry" and now >= _parse(rec["freeze_utc"]):
        raise CustodyError("refused: entries close at the freeze")
    out = _sealed_path(c, kind, name)
    if out.exists():
        raise CustodyError(f"refused: {out.name} is already sealed; a sealed object is never replaced")
    ct = _encrypt(plaintext, Path(tester_cert))
    tmp = out.with_name(f"{out.name}.tmp.{os.getpid()}")
    tmp.write_bytes(ct)
    try:
        os.link(tmp, out)  # fails if out exists, so a second seal can never overwrite the first
    except FileExistsError:
        raise CustodyError(f"refused: {out.name} is already sealed; a sealed object is never replaced") from None
    finally:
        tmp.unlink()
    try:
        append_log(c.log_path, actor, "seal", now, kind=kind, object=name, sealed_sha256=sha256_bytes(ct),
                   plain_sha256=sha256_bytes(plaintext), plain_bytes=len(plaintext), recipient_cert_sha256=recip)
    except BaseException:
        out.unlink()  # no seal line, so no sealed file either
        raise
    return out


def _sealed_record(c: Custody, kind: str, name: str) -> dict:
    hits = [r for r in read_log(c.log_path)
            if r.get("action") == "seal" and r.get("kind") == kind and r.get("object") == name]
    if len(hits) != 1:
        raise CustodyError(f"no single seal line for {kind} {name} in the custody log")
    return hits[0]


def open_sealed(c: Custody, kind: str, name: str, key: Path | str, actor: str,
                now: dt.datetime | None = None) -> bytes:
    """Decrypt one sealed object with a private key. Every attempt writes a log line before anything is returned.

    Refused before the freeze (logged "open_refused_before_freeze"). Refused when the sealed file's hash differs
    from its seal line (logged "open_refused_hash_mismatch"). A key that cannot open it is logged "open_failed".
    """
    now = now or _now()
    rec = c.record()  # checks the log chain and the custody.json hash
    freeze = _parse(rec["freeze_utc"])
    path = _sealed_path(c, kind, name)
    seal_line = _sealed_record(c, kind, name)
    ct = path.read_bytes()
    base = {"kind": kind, "object": name, "sealed_sha256": sha256_bytes(ct), "key_id": _key_id(key)}
    if now < freeze:
        append_log(c.log_path, actor, "open_refused_before_freeze", now, **base)
        raise CustodyError(f"refused: {kind} {name} stays sealed until the freeze {rec['freeze_utc']}")
    if base["sealed_sha256"] != seal_line["sealed_sha256"]:
        append_log(c.log_path, actor, "open_refused_hash_mismatch", now, **base)
        raise CustodyError(f"refused: {path.name} differs from the file logged at seal time")
    pt = _decrypt(ct, Path(key))
    if pt is None or sha256_bytes(pt) != seal_line["plain_sha256"]:
        append_log(c.log_path, actor, "open_failed", now, **base)
        raise CustodyError(f"this key cannot open {kind} {name}")
    append_log(c.log_path, actor, "open", now, plain_sha256=sha256_bytes(pt), **base)
    return pt


def _key_id(key: Path | str) -> str:
    """SHA-256 of the public half of a private key (DER), so the log names the key without holding it."""
    r = _run(["pkey", "-in", str(key), "-pubout", "-outform", "DER"])
    return sha256_bytes(r.stdout) if r.returncode == 0 and r.stdout else "unreadable-key"


def log_head(c: Custody, expected_head: str | None = None) -> str:
    """The chain head hash, after checking the chain and custody.json. A caller may commit or stamp it
    (OpenTimestamps) to pin it; this module never stamps. `expected_head` checks an earlier pin."""
    c.record()
    return verify_log(c.log_path, expected_head)


# ---------------------------------------------------------------------------------------------------------------
# CLI: python -m scoreboard.evalkit.custody keygen|init|seal|open|verify
# ---------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scoreboard.evalkit.custody", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keygen", help="tester: make a key pair; send only the .cert.pem")
    k.add_argument("--out", required=True)
    k.add_argument("--name", default="tester")
    i = sub.add_parser("init", help="start custody: freeze time and pinned tester certificate")
    i.add_argument("--root", required=True)
    i.add_argument("--cert", required=True)
    i.add_argument("--freeze", required=True, help="UTC, YYYY-MM-DDTHH:MM:SSZ")
    i.add_argument("--actor", required=True)
    s = sub.add_parser("seal", help="encrypt a labels or entry file to the tester certificate")
    s.add_argument("--root", required=True)
    s.add_argument("--kind", choices=KINDS, required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--cert", required=True)
    s.add_argument("--actor", required=True)
    o = sub.add_parser("open", help="tester: decrypt after the freeze (writes a log line)")
    o.add_argument("--root", required=True)
    o.add_argument("--kind", choices=KINDS, required=True)
    o.add_argument("--name", required=True)
    o.add_argument("--key", required=True)
    o.add_argument("--actor", required=True)
    o.add_argument("--out", required=True)
    v = sub.add_parser("verify", help="check the custody log chain and print its head hash")
    v.add_argument("--root", required=True)
    v.add_argument("--expected-head", default=None, help="a head hash pinned earlier; it must still be in the chain")
    for p in (i, s, o):
        p.add_argument("--now", default=None, help=argparse.SUPPRESS)  # tests only: a fixed UTC time
    a = ap.parse_args(argv)
    now = _parse(a.now) if getattr(a, "now", None) else None
    try:
        if a.cmd == "keygen":
            key, cert = keygen(a.out, a.name)
            print(f"private key (keep, never commit): {key}\ncertificate (send): {cert}\n"
                  f"certificate sha256: {cert_sha256(cert)}")
        elif a.cmd == "init":
            c = init_custody(a.root, a.cert, _parse(a.freeze), a.actor, now=now)
            print(f"custody started: {c.record_path}")
        elif a.cmd == "seal":
            print(seal(Custody(Path(a.root)), a.kind, a.name, Path(a.inp).read_bytes(), a.cert, a.actor, now=now))
        elif a.cmd == "open":
            if Path(a.out).exists():  # checked before the open, so no open is logged without a written file
                raise CustodyError(f"refused: {a.out} exists")
            pt = open_sealed(Custody(Path(a.root)), a.kind, a.name, a.key, a.actor, now=now)
            fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(pt)
            print(a.out)
        else:
            print(log_head(Custody(Path(a.root)), a.expected_head))
    except CustodyError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
