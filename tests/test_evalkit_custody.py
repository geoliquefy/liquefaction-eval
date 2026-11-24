"""evalkit sealed custody (eval M6, item D7; R23 tooling). Synthetic labels and entries only; every key is made in a
temp dir and removed with it, never committed. No em dashes."""
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import custody as C  # noqa: E402

try:
    C.openssl_bin()
    HAVE_OPENSSL = True
except C.CustodyError:
    HAVE_OPENSSL = False
pytestmark = pytest.mark.skipif(not HAVE_OPENSSL, reason="needs OpenSSL 3 or later")

UTC = dt.timezone.utc
T0 = dt.datetime(2027, 1, 1, 0, 0, tzinfo=UTC)
FREEZE = dt.datetime(2027, 2, 1, 0, 0, tzinfo=UTC)
BEFORE = FREEZE - dt.timedelta(days=1)
AFTER = FREEZE + dt.timedelta(days=1)
LABELS = b"site_id,label,x_m,y_m\ns001,1,0,0\ns002,0,10,0\ns003,,20,0\n"
ENTRY = b"site_id,p\ns001,0.8\ns002,0.1\ns003,0.4\n"


@pytest.fixture()
def world(tmp_path):
    """Tester and organiser each hold a key pair in their own dir. The organiser has the tester CERT only."""
    tester_key, tester_cert = C.keygen(tmp_path / "tester_home", "tester")
    org_key, org_cert = C.keygen(tmp_path / "organiser_home", "organiser")
    c = C.init_custody(tmp_path / "custody", tester_cert, FREEZE, actor="organiser", now=T0)
    C.seal(c, "labels", "synthetic-quake-a", LABELS, tester_cert, actor="label-provider", now=T0)
    C.seal(c, "entry", "entrant-x", ENTRY, tester_cert, actor="entrant-x", now=BEFORE)
    return {"c": c, "tester_key": tester_key, "tester_cert": tester_cert, "org_key": org_key, "org_cert": org_cert}


def actions(c):
    return [r["action"] for r in C.read_log(c.log_path)]


def test_private_keys_are_owner_only(world):
    assert world["tester_key"].stat().st_mode & 0o077 == 0
    assert world["org_key"].stat().st_mode & 0o077 == 0


def test_sealed_files_hold_no_plaintext(world):
    c = world["c"]
    for p in c.sealed_dir.iterdir():
        blob = p.read_bytes()
        assert b"site_id" not in blob and b"s001" not in blob
    assert not list(c.root.rglob("*.csv"))


def test_organiser_without_tester_key_cannot_read_sealed_labels(world):
    """The organiser account has its own key but not the tester's: refused before the freeze and after it."""
    c = world["c"]
    with pytest.raises(C.CustodyError, match="until the freeze"):
        C.open_sealed(c, "labels", "synthetic-quake-a", world["org_key"], actor="organiser", now=BEFORE)
    with pytest.raises(C.CustodyError, match="cannot open"):
        C.open_sealed(c, "labels", "synthetic-quake-a", world["org_key"], actor="organiser", now=AFTER)
    assert actions(c)[-2:] == ["open_refused_before_freeze", "open_failed"]


def test_organiser_cannot_read_by_going_around_the_tool(world):
    """The barrier is cryptographic, not the freeze check: openssl itself fails with the organiser key."""
    c = world["c"]
    ct = (c.sealed_dir / "labels__synthetic-quake-a.cms").read_bytes()
    assert C._decrypt(ct, world["org_key"]) is None
    r = subprocess.run([C.openssl_bin(), "cms", "-decrypt", "-binary", "-inform", "DER", "-inkey",
                        str(world["org_key"]), "-recip", str(world["org_cert"])], input=ct, capture_output=True)
    assert r.returncode != 0 and LABELS not in r.stdout


def test_tester_role_is_one_key_away(world):
    """The same call the organiser made opens the labels when the only change is the tester's private key."""
    c = world["c"]
    for key, ok in ((world["org_key"], False), (world["tester_key"], True)):
        try:
            got = C.open_sealed(c, "labels", "synthetic-quake-a", key, actor="someone", now=AFTER)
        except C.CustodyError:
            got = None
        assert (got == LABELS) is ok


def test_every_decrypt_attempt_writes_one_log_line(world):
    c = world["c"]
    n0 = len(C.read_log(c.log_path))
    attempts = [(world["tester_key"], BEFORE), (world["org_key"], AFTER), (world["tester_key"], AFTER),
                (world["tester_key"], AFTER)]
    for key, t in attempts:
        try:
            C.open_sealed(c, "entry", "entrant-x", key, actor="tester", now=t)
        except C.CustodyError:
            pass
    log = C.read_log(c.log_path)
    assert len(log) == n0 + len(attempts)
    assert [r["action"] for r in log[n0:]] == ["open_refused_before_freeze", "open_failed", "open", "open"]
    tester_id = C._key_id(world["tester_key"])
    for r in log[n0:]:
        assert r["object"] == "entrant-x" and r["kind"] == "entry" and r["time_utc"] and r["actor"] == "tester"
        assert len(r["key_id"]) == 64
    assert log[-1]["key_id"] == tester_id and log[-1]["plain_sha256"] == C.sha256_bytes(ENTRY)


def test_log_names_keys_and_hashes_but_holds_no_content(world):
    C.open_sealed(world["c"], "labels", "synthetic-quake-a", world["tester_key"], actor="tester", now=AFTER)
    text = world["c"].log_path.read_text()
    assert "s001" not in text and "PRIVATE" not in text and "0.8" not in text


def test_entries_close_at_the_freeze(world):
    with pytest.raises(C.CustodyError, match="entries close"):
        C.seal(world["c"], "entry", "late", ENTRY, world["tester_cert"], actor="late-entrant", now=FREEZE)


def test_seal_refuses_a_certificate_that_is_not_the_pinned_tester(world):
    with pytest.raises(C.CustodyError, match="not the tester certificate"):
        C.seal(world["c"], "labels", "quake-b", LABELS, world["org_cert"], actor="organiser", now=T0)


def test_a_sealed_object_is_never_replaced(world):
    with pytest.raises(C.CustodyError, match="already sealed"):
        C.seal(world["c"], "labels", "synthetic-quake-a", b"other", world["tester_cert"], actor="x", now=T0)


def test_a_swapped_sealed_file_is_refused_and_logged(world):
    """Anyone with the public cert can make a valid ciphertext; the seal-time hash catches the swap."""
    c = world["c"]
    fake = C._encrypt(b"site_id,label,x_m,y_m\ns001,0,0,0\n", world["tester_cert"])
    (c.sealed_dir / "labels__synthetic-quake-a.cms").write_bytes(fake)
    with pytest.raises(C.CustodyError, match="differs"):
        C.open_sealed(c, "labels", "synthetic-quake-a", world["tester_key"], actor="tester", now=AFTER)
    assert actions(c)[-1] == "open_refused_hash_mismatch"


def test_log_is_hash_chained_and_a_naive_rewrite_breaks_it(world):
    """A naive edit (hashes not recomputed) breaks the chain. A rewrite that recomputes every later hash would still
    verify; only a head pinned outside the custody folder catches that (tests below)."""
    c = world["c"]
    head = C.log_head(c)
    assert head == C.read_log(c.log_path)[-1]["hash"]
    lines = c.log_path.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["actor"] = "someone-else"
    lines[1] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    c.log_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(C.CustodyError, match="broken at line 2"):
        C.verify_log(c.log_path)
    with pytest.raises(C.CustodyError, match="broken"):
        C.open_sealed(c, "labels", "synthetic-quake-a", world["tester_key"], actor="tester", now=AFTER)


def test_a_naively_deleted_log_line_breaks_the_chain(world):
    """A middle line deleted without recomputing hashes breaks the chain. A cut tail would still verify."""
    c = world["c"]
    lines = c.log_path.read_text().splitlines()
    c.log_path.write_text("\n".join(lines[:1] + lines[2:]) + "\n")
    with pytest.raises(C.CustodyError, match="broken"):
        C.verify_log(c.log_path)


def _rechain(lines):
    """What an editor with write access can do: recompute every hash after an edit."""
    out, prev = [], C.GENESIS
    for i, x in enumerate(lines):
        rec = json.loads(x)
        rec["seq"], rec["prev"] = i, prev
        rec["hash"] = C._line_hash(rec)
        prev = rec["hash"]
        out.append(C._canon(rec).decode())
    return out


def test_a_recomputed_rewrite_verifies_until_a_pinned_head_is_checked(world):
    """The honest limit: an unkeyed chain cannot catch a full rewrite. A pinned head can."""
    c = world["c"]
    pinned = C.log_head(c)
    lines = c.log_path.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["actor"] = "someone-else"
    lines[1] = json.dumps(rec)
    c.log_path.write_text("\n".join(_rechain(lines)) + "\n")
    C.verify_log(c.log_path)  # passes: the rewrite is invisible without a pin
    with pytest.raises(C.CustodyError, match="pinned head"):
        C.verify_log(c.log_path, expected_head=pinned)


def test_a_cut_tail_verifies_until_a_pinned_head_is_checked(world):
    c = world["c"]
    pinned = C.log_head(c)
    lines = c.log_path.read_text().splitlines()
    c.log_path.write_text("\n".join(lines[:-1]) + "\n")
    C.verify_log(c.log_path)
    with pytest.raises(C.CustodyError, match="pinned head"):
        C.log_head(c, expected_head=pinned)


def test_a_pinned_head_allows_later_lines(world):
    c = world["c"]
    pinned = C.log_head(c)
    C.open_sealed(c, "labels", "synthetic-quake-a", world["tester_key"], actor="tester", now=AFTER)
    assert C.log_head(c, expected_head=pinned) != pinned
    assert C.main(["verify", "--root", str(c.root), "--expected-head", pinned]) == 0
    assert C.main(["verify", "--root", str(c.root), "--expected-head", "f" * 64]) == 2


def test_an_edited_custody_record_is_refused(world):
    """Swap the pinned certificate to the organiser one and move the freeze earlier: seal, open, verify refuse."""
    c = world["c"]
    rec = json.loads(c.record_path.read_text())
    rec["tester_cert_sha256"] = C.cert_sha256(world["org_cert"])
    rec["freeze_utc"] = "2027-01-02T00:00:00Z"
    c.record_path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    n0 = len(C.read_log(c.log_path))
    with pytest.raises(C.CustodyError, match="custody.json differs"):
        C.seal(c, "labels", "quake-b", LABELS, world["org_cert"], actor="organiser", now=T0)
    with pytest.raises(C.CustodyError, match="custody.json differs"):
        C.open_sealed(c, "labels", "synthetic-quake-a", world["tester_key"], actor="tester", now=BEFORE)
    with pytest.raises(C.CustodyError, match="custody.json differs"):
        C.log_head(c)
    assert C.main(["verify", "--root", str(c.root)]) == 2
    assert len(C.read_log(c.log_path)) == n0
    assert not (c.sealed_dir / "labels__quake-b.cms").exists()


def test_seal_line_names_the_recipient_certificate(world):
    seals = [r for r in C.read_log(world["c"].log_path) if r["action"] == "seal"]
    assert seals and all(r["recipient_cert_sha256"] == C.cert_sha256(world["tester_cert"]) for r in seals)


def test_seal_on_a_broken_chain_writes_no_file(world):
    c = world["c"]
    lines = c.log_path.read_text().splitlines()
    c.log_path.write_text("\n".join(lines[:1] + lines[2:]) + "\n")
    with pytest.raises(C.CustodyError, match="broken"):
        C.seal(c, "labels", "quake-b", LABELS, world["tester_cert"], actor="provider", now=T0)
    assert sorted(p.name for p in c.sealed_dir.iterdir()) == ["entry__entrant-x.cms", "labels__synthetic-quake-a.cms"]


def test_custody_record_is_written_once_and_freeze_must_be_future(world, tmp_path):
    with pytest.raises(C.CustodyError, match="written once"):
        C.init_custody(world["c"].root, world["tester_cert"], FREEZE, actor="organiser", now=T0)
    with pytest.raises(C.CustodyError, match="future"):
        C.init_custody(tmp_path / "late", world["tester_cert"], T0, actor="organiser", now=FREEZE)


def test_keygen_never_overwrites(world):
    with pytest.raises(C.CustodyError, match="never overwrite"):
        C.keygen(world["tester_key"].parent, "tester")


def test_cli_round_trip(tmp_path):
    """keygen, init, seal, open (refused for the organiser key, then the tester key), verify; through main()."""
    th, oh, root = tmp_path / "t", tmp_path / "o", tmp_path / "custody"
    assert C.main(["keygen", "--out", str(th)]) == 0
    assert C.main(["keygen", "--out", str(oh), "--name", "organiser"]) == 0
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    t0, freeze, after = (x.strftime(fmt) for x in (T0, FREEZE, AFTER))
    assert C.main(["init", "--root", str(root), "--cert", str(th / "tester.cert.pem"), "--freeze", freeze,
                   "--actor", "organiser", "--now", t0]) == 0
    src = tmp_path / "labels.csv"
    src.write_bytes(LABELS)
    assert C.main(["seal", "--root", str(root), "--kind", "labels", "--name", "q", "--in", str(src),
                   "--cert", str(th / "tester.cert.pem"), "--actor", "provider", "--now", t0]) == 0
    out = tmp_path / "opened.csv"
    common = ["open", "--root", str(root), "--kind", "labels", "--name", "q", "--out", str(out)]
    assert C.main(common + ["--key", str(oh / "organiser.key.pem"), "--actor", "organiser", "--now", t0]) == 2
    assert not out.exists()
    c = C.Custody(root)
    assert C.main(common + ["--key", str(oh / "organiser.key.pem"), "--actor", "organiser", "--now", after]) == 2
    assert C.main(common + ["--key", str(th / "tester.key.pem"), "--actor", "tester", "--now", after]) == 0
    assert out.read_bytes() == LABELS
    # --out exists now: refused before the open, so no open line is logged
    assert C.main(common + ["--key", str(th / "tester.key.pem"), "--actor", "tester", "--now", after]) == 2
    assert C.main(["verify", "--root", str(root)]) == 0
    assert actions(c) == ["custody_start", "seal", "open_refused_before_freeze", "open_failed", "open"]
