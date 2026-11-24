"""Closed division package table (R15; D5 follow-ups, wave 10; eval M5). Synthetic files only. No em dashes."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoreboard.evalkit import package as P  # noqa: E402
from scoreboard.evalkit.spec import SpecError  # noqa: E402


@pytest.fixture
def pkg(tmp_path):
    d = tmp_path / "package"
    (d / "shakemap").mkdir(parents=True)
    (d / "sites.csv").write_text("site_id,lon,lat\nex:0001,172.6,-43.5\n", encoding="utf-8")
    (d / "shakemap" / "grid.xml").write_bytes(b"<grid/>\n")
    return d


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_the_table_is_the_kit_form_and_its_hash_is_the_sha256_of_its_bytes(tmp_path, pkg):
    res = P.write_table(pkg, tmp_path / "t.csv")
    raw = (tmp_path / "t.csv").read_bytes()
    want = ("file_id,sha256\n"
            f"shakemap/grid.xml,{_sha(b'<grid/>' + bytes([10]))}\n"
            f"sites.csv,{_sha((pkg / 'sites.csv').read_bytes())}\n").encode("utf-8")
    assert raw == want                                           # sorted ids, "/" separators, LF, last LF
    assert res == {"package_sha256": _sha(raw), "n_files": 2, "path": str(tmp_path / "t.csv")}
    assert P.table_sha256(tmp_path / "t.csv") == _sha(raw)
    rows, h = P.read_table(tmp_path / "t.csv")
    assert h == _sha(raw) and P.table_sha256_of_rows(rows) == h
    with pytest.raises(SpecError, match="never replaced"):
        P.write_table(pkg, tmp_path / "t.csv")
    assert (tmp_path / "t.csv").read_bytes() == raw


@pytest.mark.parametrize("edit", [
    lambda t: t.replace("\n", "\r\n"),                                   # CRLF
    lambda t: t[:-1],                                                    # no last LF
    lambda t: "\n".join([t.split("\n")[0]] + t.split("\n")[1:3][::-1]) + "\n",   # rows not sorted
    lambda t: t.replace("file_id,sha256", "file,sha256"),               # header
    lambda t: t + t.split("\n")[1] + "\n",                               # repeated id
    lambda t: t.replace("\nsites.csv,", "\nsites.csv, "),               # a space in a row
    lambda t: t.upper().replace("FILE_ID,SHA256", "file_id,sha256"),     # uppercase hex
    lambda t: "file_id,sha256\n",                                        # no file
    lambda t: t + "../x," + "0" * 64 + "\n",                             # id leaves the folder
])
def test_a_table_not_in_the_kit_form_is_refused(tmp_path, pkg, edit):
    P.write_table(pkg, tmp_path / "t.csv")
    text = (tmp_path / "t.csv").read_text(encoding="utf-8")
    bad = edit(text)
    if bad == text:
        pytest.skip("edit is a no-op on this table")
    (tmp_path / "b.csv").write_bytes(bad.encode("utf-8"))
    with pytest.raises(SpecError, match="package table"):
        P.read_table(tmp_path / "b.csv")


def test_check_names_each_file_that_differs_or_is_missing(tmp_path, pkg, capsys):
    P.write_table(pkg, tmp_path / "t.csv")
    ok = P.check_files(tmp_path / "t.csv", pkg)
    assert ok["status"] == "PASS" and set(ok["files"].values()) == {"PASS"} and ok["not_in_table"] == []
    assert P.main(["check", str(tmp_path / "t.csv"), str(pkg)]) == 0
    (pkg / "sites.csv").write_text("site_id,lon,lat\nex:0001,172.6,-43.6\n", encoding="utf-8")
    (pkg / "shakemap" / "grid.xml").unlink()
    (pkg / "extra.txt").write_text("x\n", encoding="utf-8")
    bad = P.check_files(tmp_path / "t.csv", pkg)
    assert bad["status"] == "FAIL"
    assert bad["files"] == {"shakemap/grid.xml": "MISSING", "sites.csv": "FAIL"}
    assert bad["not_in_table"] == ["extra.txt"]
    capsys.readouterr()
    assert P.main(["check", str(tmp_path / "t.csv"), str(pkg)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"


def test_a_file_that_is_only_missing_is_fail(tmp_path, pkg, capsys):
    """D5 follow-ups review 1 (Q8): MISSING alone, with every other file PASS, is FAIL and exit 1."""
    P.write_table(pkg, tmp_path / "t.csv")
    (pkg / "shakemap" / "grid.xml").unlink()
    bad = P.check_files(tmp_path / "t.csv", pkg)
    assert bad["files"] == {"shakemap/grid.xml": "MISSING", "sites.csv": "PASS"}
    assert bad["status"] == "FAIL" and bad["not_in_table"] == []
    capsys.readouterr()
    assert P.main(["check", str(tmp_path / "t.csv"), str(pkg)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"


def test_write_leaves_out_hidden_files(tmp_path, pkg):
    """D5 follow-ups review 1: a .DS_Store is never a row, so a fetcher without it does not read MISSING."""
    (pkg / ".DS_Store").write_bytes(b"\0finder")
    (pkg / "shakemap" / ".hidden").write_bytes(b"x")
    (pkg / ".git").mkdir()
    (pkg / ".git" / "HEAD").write_bytes(b"ref\n")
    res = P.write_table(pkg, tmp_path / "t.csv")
    rows, _ = P.read_table(tmp_path / "t.csv")
    assert sorted(rows) == ["shakemap/grid.xml", "sites.csv"] and res["n_files"] == 2
    fetched = tmp_path / "fetched"
    (fetched / "shakemap").mkdir(parents=True)
    for fid in rows:
        (fetched / fid).write_bytes((pkg / fid).read_bytes())
    assert P.check_files(tmp_path / "t.csv", fetched)["status"] == "PASS"


def test_command_line_write_hash_and_refusal(tmp_path, pkg, capsys):
    assert P.main(["write", str(pkg), "--out", str(tmp_path / "t.csv")]) == 0
    w = json.loads(capsys.readouterr().out)
    assert w["status"] == "WRITTEN" and w["n_files"] == 2
    assert P.main(["hash", str(tmp_path / "t.csv")]) == 0
    assert json.loads(capsys.readouterr().out)["package_sha256"] == w["package_sha256"]
    (tmp_path / "b.csv").write_text("nope\n", encoding="utf-8")
    assert P.main(["hash", str(tmp_path / "b.csv")]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "REFUSED"
    assert P.main(["write", str(pkg), "--out", str(tmp_path / "t.csv")]) == 2       # never replaces
    assert "never replaced" in json.loads(capsys.readouterr().out)["reason"]
