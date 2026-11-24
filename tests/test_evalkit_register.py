"""evalkit register and receipt (R20; D3, eval M6): submit, receipt, verify, tamper found. Synthetic only.

OpenTimestamps is never run here: every stamp test passes a fake runner.
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoreboard.evalkit import register as reg  # noqa: E402
from scoreboard.evalkit.spec import Entry, EntryManifest, SpecError, entry_sha256, save_entry  # noqa: E402

T0 = datetime(2026, 11, 23, 12, 0, 0, tzinfo=timezone.utc)


def clock(start=T0, step_s=1.0):
    """A fake clock that moves forward by step_s on each call."""
    state = {"t": start}

    def now():
        t = state["t"]
        state["t"] = t + timedelta(seconds=step_s)
        return t
    return now


def write_entry(d: Path, name: str, seed: int = 0) -> tuple[Path, Entry]:
    rng = np.random.default_rng(seed)
    m = EntryManifest(model=f"synthetic {name}", output_kind="target_probability", statement="synthetic test entry",
                      exposure={"synthetic-eq": "unseen"}, house=False)
    e = Entry(m, tuple(f"s{k}" for k in range(30)), rng.uniform(0.05, 0.95, 30))
    save_entry(e, d / f"{name}.json", d / f"{name}.csv")
    return d / f"{name}.json", e


def lines_of(path: Path) -> list[str]:
    return path.read_text().splitlines()


def test_end_to_end_submit_receipt_verify(tmp_path):
    r = tmp_path / "register.jsonl"
    now = clock()
    receipts = []
    for k in range(3):
        mp, e = write_entry(tmp_path, f"e{k}", seed=k)
        rec = reg.submit_entry(r, mp, entrant="team-a", now=now)
        assert rec["entry_sha256"] == entry_sha256(e)            # the receipt carries the kit's R4 hash
        assert rec["seq"] == k + 1 and rec["item_type"] == "entry" and rec["entrant"] == "team-a"
        receipts.append(rec)
    assert receipts[0]["prev_sha256"] == reg.GENESIS
    for a, b in zip(receipts, receipts[1:]):
        assert b["prev_sha256"] == a["line_sha256"]               # each line holds the previous line's hash
    chk = reg.check_register(r)
    assert chk.status == reg.VERIFIED and chk.n_lines == 3 and chk.head_sha256 == receipts[-1]["line_sha256"]
    for k, rec in enumerate(receipts):
        res = reg.verify_receipt(r, rec, entry_sha=reg.entry_file_sha256(tmp_path / f"e{k}.json"))
        assert res["status"] == reg.VERIFIED, res["reasons"]
    for text in lines_of(r):                                     # one canonical JSON object per line
        assert reg.line_text(json.loads(text)) == text


def test_a_changed_line_is_found(tmp_path):
    r = tmp_path / "register.jsonl"
    now = clock()
    recs = [reg.submit_entry(r, write_entry(tmp_path, f"e{k}", k)[0], now=now) for k in range(3)]
    ls = lines_of(r)
    d = json.loads(ls[1])
    d["item_sha256"] = "f" * 64                                  # swap in another entry's hash, kept canonical
    ls[1] = reg.line_text(d)
    r.write_text("\n".join(ls) + "\n")
    chk = reg.check_register(r)
    assert chk.status == reg.REGISTER_FAULT
    assert any(f["line"] == 3 and "prev_sha256" in f["fault"] for f in chk.faults)
    res = reg.verify_receipt(r, recs[1])
    assert res["status"] == reg.REGISTER_FAULT
    assert any("line 2 changed" in x for x in res["reasons"])
    # a receipt for a later line also fails, since the chain under it broke
    assert reg.verify_receipt(r, recs[2])["status"] == reg.REGISTER_FAULT


def test_a_rehashed_chain_still_fails_the_receipt(tmp_path):
    """An editor who rewrites a line and recomputes every later link still breaks the receipt of that line."""
    r = tmp_path / "register.jsonl"
    now = clock()
    recs = [reg.submit_entry(r, write_entry(tmp_path, f"e{k}", k)[0], now=now) for k in range(3)]
    ls = [json.loads(x) for x in lines_of(r)]
    ls[0]["item_sha256"] = "a" * 64
    prev = reg.GENESIS
    out = []
    for d in ls:
        d["prev_sha256"] = prev
        t = reg.line_text(d)
        out.append(t)
        prev = reg.line_sha256(t)
    r.write_text("\n".join(out) + "\n")
    assert reg.check_register(r).status == reg.VERIFIED          # the chain alone reads clean
    for rec in recs:                                             # every receipt at or after line 1 does not
        assert reg.verify_receipt(r, rec)["status"] == reg.REGISTER_FAULT


def test_a_removed_line_is_found(tmp_path):
    r = tmp_path / "register.jsonl"
    now = clock()
    recs = [reg.submit_entry(r, write_entry(tmp_path, f"e{k}", k)[0], now=now) for k in range(4)]
    ls = lines_of(r)
    r.write_text("\n".join(ls[:1] + ls[2:]) + "\n")              # drop line 2
    chk = reg.check_register(r)
    assert chk.status == reg.REGISTER_FAULT
    kinds = [f["fault"] for f in chk.faults]
    assert sum("seq" in k for k in kinds) == 1 and sum("prev_sha256" in k for k in kinds) == 1
    res = reg.verify_receipt(r, recs[1])
    assert res["status"] == reg.REGISTER_FAULT and any("missing" in x for x in res["reasons"])
    # the last line removed: its receipt finds it missing
    r.write_text("\n".join(ls[:3]) + "\n")
    assert reg.check_register(r).status == reg.VERIFIED
    res = reg.verify_receipt(r, recs[3])
    assert res["status"] == reg.REGISTER_FAULT and any("missing" in x for x in res["reasons"])


def test_reordered_lines_and_decreasing_time_are_faults(tmp_path):
    r = tmp_path / "register.jsonl"
    now = clock()
    for k in range(3):
        reg.submit_entry(r, write_entry(tmp_path, f"e{k}", k)[0], now=now)
    ls = lines_of(r)
    r.write_text("\n".join([ls[0], ls[2], ls[1]]) + "\n")
    assert reg.check_register(r).status == reg.REGISTER_FAULT
    # a chain that links but whose intake time goes back is a REGISTER FAULT (R20)
    r2 = tmp_path / "r2.jsonl"
    a = {"v": 1, "seq": 1, "intake_utc": "2026-11-23T12:00:05.000000Z", "item_type": "entry",
         "item_sha256": "1" * 64, "prev_sha256": reg.GENESIS, "entrant": None}
    b = dict(a, seq=2, intake_utc="2026-11-23T12:00:01.000000Z", prev_sha256=reg.line_sha256(reg.line_text(a)))
    r2.write_text(reg.line_text(a) + "\n" + reg.line_text(b) + "\n")
    chk = reg.check_register(r2)
    assert [f["fault"] for f in chk.faults] == ["intake time decreases (R20)"]


def test_non_canonical_and_cut_lines_are_faults(tmp_path):
    r = tmp_path / "register.jsonl"
    reg.submit_entry(r, write_entry(tmp_path, "e0")[0], now=clock())
    text = lines_of(r)[0]
    r.write_text(json.dumps(json.loads(text), indent=1).replace("\n", "") + "\n")   # same content, other bytes
    assert any("canonical" in f["fault"] for f in reg.check_register(r).faults)
    r.write_text(text)                                                                # no trailing newline
    assert any("newline" in f["fault"] for f in reg.check_register(r).faults)


def test_intake_refuses_a_broken_register_and_a_backward_clock(tmp_path):
    r = tmp_path / "register.jsonl"
    mp, _ = write_entry(tmp_path, "e0")
    reg.submit_entry(r, mp, now=clock(T0))
    before = r.read_bytes()
    with pytest.raises(reg.RegisterError, match="earlier than the head"):
        reg.submit_entry(r, mp, now=clock(T0 - timedelta(seconds=5)))
    assert r.read_bytes() == before                              # nothing appended
    r.write_bytes(before.replace(b'"seq":1', b'"seq":7'))
    broken = r.read_bytes()
    with pytest.raises(reg.RegisterError, match="REGISTER FAULT"):
        reg.submit_entry(r, mp, now=clock(T0 + timedelta(hours=1)))
    assert r.read_bytes() == broken


def test_intake_checks_the_entry_schema(tmp_path):
    r = tmp_path / "register.jsonl"
    mp, _ = write_entry(tmp_path, "e0")
    m = json.loads(mp.read_text())
    m["output_kind"] = "guess"
    (tmp_path / "bad.json").write_text(json.dumps(m))
    (tmp_path / "bad.csv").write_text((tmp_path / "e0.csv").read_text())
    with pytest.raises(SpecError, match="output_kind"):
        reg.submit_entry(r, tmp_path / "bad.json", now=clock())
    rows = (tmp_path / "e0.csv").read_text().splitlines()
    (tmp_path / "dup.json").write_text(mp.read_text())
    (tmp_path / "dup.csv").write_text("\n".join(rows + [rows[1]]) + "\n")
    with pytest.raises(SpecError, match="duplicate"):
        reg.submit_entry(r, tmp_path / "dup.json", now=clock())
    (tmp_path / "lone.json").write_text(mp.read_text())
    with pytest.raises(SpecError, match="no matching"):
        reg.submit_entry(r, tmp_path / "lone.json", now=clock())
    assert not r.exists() or r.read_text() == ""                 # a refused entry writes no line (open R20 gap)


def test_receipt_finds_other_entry_files_and_bad_receipts(tmp_path):
    r = tmp_path / "register.jsonl"
    mp, _ = write_entry(tmp_path, "e0", seed=0)
    rec = reg.submit_entry(r, mp, now=clock())
    other, _ = write_entry(tmp_path, "e1", seed=1)
    res = reg.verify_receipt(r, rec, entry_sha=reg.entry_file_sha256(other))
    assert res["status"] == reg.REGISTER_FAULT and any("entry files in hand" in x for x in res["reasons"])
    forged = dict(rec, entry_sha256="b" * 64)
    assert reg.verify_receipt(r, forged)["status"] == reg.REGISTER_FAULT
    with pytest.raises(reg.RegisterError):
        reg.verify_receipt(r, {k: v for k, v in rec.items() if k != "seq"})


def test_stamp_is_wired_behind_a_mocked_runner(tmp_path):
    r = tmp_path / "register.jsonl"
    rec = reg.submit_entry(r, write_entry(tmp_path, "e0")[0], now=clock())
    calls = []

    def fake_runner(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    out = reg.stamp_head(r, runner=fake_runner)
    assert out["stamped"] and out["head_sha256"] == rec["line_sha256"] and out["head_seq"] == 1
    assert calls == [["ots", "stamp", out["head_file"]]]
    assert rec["line_sha256"] in Path(out["head_file"]).read_text()
    assert out["proof"] == out["head_file"] + ".ots"

    def failing_runner(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, "", "calendar unreachable")
    assert reg.stamp_head(r, runner=failing_runner)["stamped"] is False

    def missing_runner(cmd, **kw):
        raise FileNotFoundError(cmd[0])
    with pytest.raises(reg.RegisterError, match="not found"):
        reg.stamp_head(r, runner=missing_runner)
    with pytest.raises(reg.RegisterError, match="empty"):
        reg.stamp_head(tmp_path / "none.jsonl", runner=fake_runner)


def test_cli_submit_verify_tamper_and_stamp_is_off_by_default(tmp_path, capsys, monkeypatch):
    stamped = []
    monkeypatch.setattr(reg, "stamp_head", lambda *a, **k: stamped.append(a) or {"stamped": True})
    r, rp = tmp_path / "register.jsonl", tmp_path / "receipt.json"
    mp, _ = write_entry(tmp_path, "e0")
    assert reg.main(["submit", "--register", str(r), str(mp), "--entrant", "team-a", "--receipt-out", str(rp)]) == 0
    assert stamped == []                                         # no --stamp, no stamp
    out = json.loads(capsys.readouterr().out)
    assert out["receipt"] == json.loads(rp.read_text()) and "stamp" not in out
    assert reg.main(["verify", "--register", str(r), "--receipt", str(rp), "--entry", str(mp)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == reg.VERIFIED
    assert reg.main(["check", "--register", str(r)]) == 0
    capsys.readouterr()
    assert reg.main(["submit", "--register", str(r), str(mp), "--stamp"]) == 0
    assert len(stamped) == 1                                     # --stamp calls the (mocked) stamp step
    capsys.readouterr()
    r.write_text(r.read_text().replace('"team-a"', '"team-b"'))
    assert reg.main(["verify", "--register", str(r), "--receipt", str(rp)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == reg.REGISTER_FAULT
    assert reg.main(["submit", "--register", str(tmp_path / "x.jsonl"), str(tmp_path / "nope.json")]) == 2


def test_python_dash_m_entry_point(tmp_path):
    mp, _ = write_entry(tmp_path, "e0")
    r = tmp_path / "register.jsonl"
    p = subprocess.run([sys.executable, "-m", "scoreboard.evalkit.register", "submit", "--register", str(r), str(mp)],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["receipt"]["seq"] == 1


# ---- review 1 fixes ----------------------------------------------------------------------------------------------

GOLDEN_LINE = ('{"entrant":"team-a","intake_utc":"2026-11-23T12:00:00.000000Z","item_sha256":"' + "1" * 64
               + '","item_type":"entry","prev_sha256":"' + "0" * 64 + '","seq":1,"v":1}')
GOLDEN_SHA = "896984c45650a6b9f95c73d1d25dbf837b5bae9b78bb59cb973ee51bd0cf54f1"   # checked with shasum -a 256


def test_golden_line_vector_for_an_outside_verifier(tmp_path):
    """The line hash is sha256 of the exact line bytes, with no newline. An outside verifier needs only hashlib."""
    d = {"v": 1, "seq": 1, "intake_utc": "2026-11-23T12:00:00.000000Z", "item_type": "entry",
         "item_sha256": "1" * 64, "prev_sha256": reg.GENESIS, "entrant": "team-a"}
    assert reg.line_text(d) == GOLDEN_LINE
    assert reg.line_sha256(GOLDEN_LINE) == GOLDEN_SHA
    r = tmp_path / "register.jsonl"
    r.write_bytes(GOLDEN_LINE.encode("ascii") + b"\n")
    chk = reg.check_register(r)
    assert chk.status == reg.VERIFIED and chk.head_sha256 == GOLDEN_SHA
    # a real intake: the receipt's line hash is sha256 of the raw first line of the file
    r2 = tmp_path / "r2.jsonl"
    rec = reg.submit_entry(r2, write_entry(tmp_path, "e0")[0], now=clock())
    assert rec["line_sha256"] == hashlib.sha256(r2.read_bytes().split(b"\n")[0]).hexdigest()


PARALLEL_CODE = """
import sys
from scoreboard.evalkit import register as reg
p, n = int(sys.argv[2]), int(sys.argv[3])
for k in range(n):
    reg.append_item(sys.argv[1], "entry", format(p * 1000 + k, "064x"))
"""


def test_parallel_intakes_take_distinct_seqs(tmp_path):
    """Several processes append at once; the file lock keeps one seq per line and an unbroken chain."""
    r = tmp_path / "register.jsonl"
    n_proc, n_each = 6, 15
    procs = [subprocess.Popen([sys.executable, "-c", PARALLEL_CODE, str(r), str(p), str(n_each)], cwd=ROOT,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for p in range(n_proc)]
    for p in procs:
        _, err = p.communicate(timeout=120)
        assert p.returncode == 0, err
    chk = reg.check_register(r)
    assert chk.status == reg.VERIFIED, chk.faults[:3]
    assert chk.n_lines == n_proc * n_each
    assert len({d["item_sha256"] for d in chk.lines}) == n_proc * n_each


def test_cli_stamp_failure_still_prints_the_receipt(tmp_path, capsys, monkeypatch):
    """The line is in the register before the stamp. A failed stamp must not hide the receipt or say REFUSED."""
    def missing_runner(cmd, **kw):
        raise FileNotFoundError(cmd[0])
    real = reg.stamp_head
    monkeypatch.setattr(reg, "stamp_head", lambda register, **k: real(register, runner=missing_runner))
    r = tmp_path / "register.jsonl"
    mp, _ = write_entry(tmp_path, "e0")
    assert reg.main(["submit", "--register", str(r), str(mp), "--stamp"]) == 1
    cap = capsys.readouterr()
    out = json.loads(cap.out)
    assert "REFUSED" not in cap.err
    assert out["receipt"]["seq"] == 1 and out["receipt"]["line_sha256"] == reg.check_register(r).head_sha256
    assert out["stamp"]["stamped"] is False and "not found" in out["stamp"]["error"]
    assert len(lines_of(r)) == 1
    # a stamp client that runs but fails: also exit 1, with the receipt
    monkeypatch.setattr(reg, "stamp_head", lambda register, **k: real(
        register, runner=lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "calendar unreachable")))
    assert reg.main(["submit", "--register", str(r), str(mp), "--stamp"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["receipt"]["seq"] == 2 and out["stamp"]["stamped"] is False


def test_cli_verify_refuses_csv_without_entry(tmp_path, capsys):
    r, rp = tmp_path / "register.jsonl", tmp_path / "receipt.json"
    mp, _ = write_entry(tmp_path, "e0")
    assert reg.main(["submit", "--register", str(r), str(mp), "--receipt-out", str(rp)]) == 0
    capsys.readouterr()
    assert reg.main(["verify", "--register", str(r), "--receipt", str(rp), "--csv", str(tmp_path / "e0.csv")]) == 2
    assert "--csv needs --entry" in capsys.readouterr().err
