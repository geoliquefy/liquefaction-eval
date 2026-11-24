"""B4 package layout (ruling 58c, wave 12, eval M5): the forward runner writes the rule book B4 /inputs/ layout,
FILE_ID is the path inside /inputs/, ids are printable ASCII with no space or quote, and a symbolic link is hashed
as its target in write, rebuild and check. Planted packages in both layouts. Synthetic files only. No em dashes.
"""
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoreboard.evalkit import forward as F  # noqa: E402
from scoreboard.evalkit import package as P  # noqa: E402
from scoreboard.evalkit.spec import SpecError, canonical_json  # noqa: E402
from tests.test_evalkit_forward import FLAG, entries, write_cpts, write_grid  # noqa: E402

B4 = ["event.json", "package/cpt_list.csv", "shakemap/grid.xml", "sites.csv"]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def src(tmp_path):
    s = tmp_path / "src"
    s.mkdir()
    reg = s / "register.jsonl"
    F.write_register(reg, entries())
    return {"run": tmp_path / "run", "reg": reg, "grid": write_grid(s / "g.xml"),
            "gz": write_grid(s / "g.xml.gz", gz=True), "cpt": write_cpts(s / "cpts.csv")}


def _freeze(src, t="A", grid="grid"):
    return F.freeze_inputs(src["run"], FLAG, src[grid], src["cpt"], input_type=t, register_path=src["reg"])


# ---------------------------------------------------------------------------------------------------------------
# The forward runner writes the B4 layout
# ---------------------------------------------------------------------------------------------------------------

def test_freeze_writes_exactly_the_b4_files_and_freezes_their_table(src):
    m = _freeze(src)["manifest"]
    d = src["run"] / "inputs" / "A"
    files = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
    assert files == B4                                                   # nothing else inside /inputs/
    assert m["schema"] == F.SCHEMA_INPUTS == "evalkit-forward-inputs/2"
    assert list(m["inputs"]["files"]) == B4 and m["inputs"]["root"] == "/inputs/"
    assert m["inputs"]["files"] == {f: _sha((d / f).read_bytes()) for f in B4}
    assert not (d / "manifest.json").exists() and (src["run"] / "manifests" / "A.json").exists()
    # The table `package write` makes of the folder is the frozen one: FILE_ID is the path inside /inputs/.
    w = P.write_table(d, src["run"] / "table-A.csv")
    assert w["package_sha256"] == m["inputs"]["package_sha256"] == P.rebuild_sha256(d)[0]
    assert P.read_table(src["run"] / "table-A.csv")[0] == m["inputs"]["files"]


def test_event_json_holds_the_seven_b4_fields_from_the_manifest_event(src):
    m = _freeze(src)["manifest"]
    raw = (src["run"] / "inputs" / "A" / "event.json").read_bytes()
    ev = m["event"]
    want = {"comcat_id": ev["id"], "origin_utc": ev["time_utc"], "lon": ev["lon"], "lat": ev["lat"],
            "depth_km": ev["depth_km"], "mw": ev["mag"], "mw_type": ev["mag_type"]}
    assert raw == (canonical_json(want) + "\n").encode("utf-8") == F.event_json_bytes(ev)
    assert json.loads(raw) == {"comcat_id": "synth-fwd-1", "origin_utc": "2030-01-01T00:00:00Z", "lon": 175.0,
                               "lat": -40.0, "depth_km": 8.0, "mw": 6.4, "mw_type": "mww"}


def test_sites_csv_is_site_id_lon_lat_and_matches_the_kit_cpt_list(src):
    m = _freeze(src)["manifest"]
    d = src["run"] / "inputs" / "A"
    lines = (d / "sites.csv").read_bytes().decode("utf-8").split("\n")
    assert lines[0] == "site_id,lon,lat" and lines[-1] == "" and "\r" not in "".join(lines)
    cpt = F.read_cpt_list(d / "package" / "cpt_list.csv")
    assert F.read_sites(d / "sites.csv") == cpt["site_ids"] and len(cpt["site_ids"]) == m["sites"]["n_sites"] == 395
    first = lines[1].split(",")
    assert float(first[1]) == cpt["lon"][0] and float(first[2]) == cpt["lat"][0]       # lon before lat (B4)


def test_a_gzip_grid_is_stored_unzipped_as_grid_xml(src):
    m = _freeze(src, grid="gz")["manifest"]
    g = src["run"] / "inputs" / "A" / "shakemap" / "grid.xml"
    assert g.read_bytes() == gzip.decompress(src["gz"].read_bytes())
    assert m["shakemap"]["source_gzip"] is True and m["shakemap"]["source_sha256"] == _sha(src["gz"].read_bytes())
    assert F.frozen_inputs(src["run"], "A").pga_g.size == 395


def test_an_added_file_in_inputs_is_caught_a_hidden_one_too(src):
    _freeze(src)
    d = src["run"] / "inputs" / "A"
    (d / ".DS_Store").write_bytes(b"finder")             # B4-LAYOUT review 1 N4: the runner alone writes /inputs/
    with pytest.raises(F.RunRefused, match=r"hidden files the freeze did not write \(\.DS_Store\)"):
        F.load_frozen(src["run"], "A")
    (d / ".DS_Store").unlink()
    F.load_frozen(src["run"], "A")
    os.symlink(src["run"] / "nowhere", d / "package" / ".#cpt_list.csv")   # a hidden broken link: refused, named
    with pytest.raises(F.RunRefused, match=r"package/\.#cpt_list\.csv"):
        F.load_frozen(src["run"], "A")
    (d / "package" / ".#cpt_list.csv").unlink()
    (d / "package" / "extra.csv").write_text("x\n")
    with pytest.raises(F.RunRefused, match=r"changed after the freeze \(package/extra.csv"):
        F.load_frozen(src["run"], "A")


@pytest.mark.parametrize("key", ["depth_km", "mag_type"])
@pytest.mark.parametrize("value", ["absent", None, ""])
def test_a_record_with_no_depth_or_no_magnitude_type_is_refused_before_any_write(src, key, value):
    rec = {k: v for k, v in FLAG.items() if k != key}            # B4-LAYOUT review 1 F2: no made-up depth or mw_type
    if value != "absent":
        rec[key] = value
    with pytest.raises(F.RunRefused, match=f"the watcher record has no '{key}'.*nothing was written"):
        F.freeze_inputs(src["run"], rec, src["grid"], src["cpt"], input_type="A", register_path=src["reg"])
    assert not src["run"].exists()
    assert F.event_of(rec)["depth_km" if key == "depth_km" else "mag_type"] in (0.0, None, "")   # event_of unchanged


def test_a_true_depth_of_zero_is_frozen_as_zero(src):
    F.freeze_inputs(src["run"], {**FLAG, "depth_km": 0.0}, src["grid"], src["cpt"], input_type="A",
                    register_path=src["reg"])
    assert json.loads((src["run"] / "inputs" / "A" / "event.json").read_bytes())["depth_km"] == 0.0


def test_a_stray_manifest_json_in_a_b4_folder_is_an_added_file_not_the_old_layout(src):
    _freeze(src)
    d = src["run"] / "inputs" / "A"
    (d / "manifest.json").write_text(json.dumps({"schema": F.SCHEMA_INPUTS}))     # B4-LAYOUT review 1 N3
    with pytest.raises(F.RunRefused, match=r"changed after the freeze \(manifest.json") as e:
        F.load_frozen(src["run"], "A")
    assert "old kit layout" not in str(e.value)
    (d / "manifest.json").write_text(json.dumps({"schema": F.SCHEMA_INPUTS_OLD}))  # the old schema: still old layout
    with pytest.raises(F.RunRefused, match="old kit layout"):
        F.load_frozen(src["run"], "A")


def test_a_stopped_freeze_folder_is_not_overwritten(src):
    d = src["run"] / "inputs" / "A"
    d.mkdir(parents=True)
    (d / "sites.csv").write_text("left over\n")
    with pytest.raises(F.RunRefused, match="not empty but no freeze is logged"):
        _freeze(src)
    assert (d / "sites.csv").read_text() == "left over\n"


# ---------------------------------------------------------------------------------------------------------------
# The old kit layout: planted run folder refused, its table still readable
# ---------------------------------------------------------------------------------------------------------------

@pytest.fixture
def old_run(src):
    """A planted run folder in the kit layout before ruling 58c (schema evalkit-forward-inputs/1)."""
    run = src["run"]
    d = run / "inputs" / "A"
    d.mkdir(parents=True)
    (d / "shakemap_grid.xml").write_bytes(src["grid"].read_bytes())
    (d / "cpt_list.csv").write_bytes(src["cpt"].read_bytes())
    m = {"schema": F.SCHEMA_INPUTS_OLD, "input_type": "A", "event": F.event_of(FLAG),
         "shakemap": {"file": "shakemap_grid.xml", "sha256": _sha((d / "shakemap_grid.xml").read_bytes())},
         "cpt_list": {"file": "cpt_list.csv", "sha256": _sha((d / "cpt_list.csv").read_bytes())}}
    (d / "manifest.json").write_text(json.dumps(m))
    F._append(run, "inputs_frozen", "A", F._sha_json(m))
    rows = {m[k]["file"]: m[k]["sha256"] for k in ("shakemap", "cpt_list")}   # the old score() table rows
    (run / "scores" / "A").mkdir(parents=True)
    (run / "scores" / "A" / "package_table.csv").write_text(P.table_text(rows), encoding="utf-8", newline="")
    return run


def test_an_old_layout_run_is_refused_with_a_clear_message(src, old_run):
    for call in (lambda: F.load_frozen(old_run, "A"), lambda: F.frozen_inputs(old_run, "A"),
                 lambda: _freeze(src, t="B")):                             # no mixed layouts in one run folder
        with pytest.raises(F.RunRefused, match=r"old kit layout \(evalkit-forward-inputs/1.*freeze the inputs "
                                               r"again into a new run folder"):
            call()


def test_an_old_layout_table_stays_readable(old_run):
    table = old_run / "scores" / "A" / "package_table.csv"
    rows, h = P.read_table(table)
    assert sorted(rows) == ["cpt_list.csv", "shakemap_grid.xml"] and h == _sha(table.read_bytes())
    res = P.check_files(table, old_run / "inputs" / "A")
    assert res["status"] == "PASS" and res["not_in_table"] == ["manifest.json"]
    assert P.main(["hash", str(table)]) == 0


# ---------------------------------------------------------------------------------------------------------------
# FILE_ID: printable ASCII, no space, no quote (ruling 58c; amendment 2 review F1, F5)
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["a b.csv", 'q"x.csv', "q'x.csv", " lead.csv", "caf\u00e9.csv", "cafe\u0301.csv",
                                  "\u00e9t\u00e9/x.csv"])
def test_a_bad_file_name_is_refused_by_write_rebuild_check_and_a_table_holding_it(tmp_path, name):
    d = tmp_path / "pkg"
    (d / Path(name).parent).mkdir(parents=True, exist_ok=True)
    (d / "ok.csv").write_text("a\n")
    (d / name).write_text("b\n")
    for call in (lambda: P.write_table(d, tmp_path / "t.csv"), lambda: P.rebuild_sha256(d)):
        with pytest.raises(SpecError, match="printable ASCII, or holds a space or a quote"):
            call()
    assert not (tmp_path / "t.csv").exists()
    # A table that names the id (built by hand in the old rules) is refused by every reader.
    rows = {name: "0" * 64, "ok.csv": "1" * 64}
    text = "file_id,sha256\n" + "".join(f"{fid},{rows[fid]}\n" for fid in sorted(rows))
    (tmp_path / "hand.csv").write_bytes(text.encode("utf-8"))
    with pytest.raises(SpecError, match="bad file_id"):
        P.read_table(tmp_path / "hand.csv")
    assert P.main(["check", str(tmp_path / "hand.csv"), str(d)]) == 2


def test_every_printable_ascii_but_the_banned_ones_is_allowed():
    ok = "".join(chr(c) for c in range(33, 127) if chr(c) not in ",\\\"'/")
    assert P._check_id(ok) == ok and P._check_id("a/" + ok) == "a/" + ok


# ---------------------------------------------------------------------------------------------------------------
# Symbolic links: hashed as the target, the same in write, rebuild and check (review F6)
# ---------------------------------------------------------------------------------------------------------------

@pytest.fixture
def linked(tmp_path):
    outside = tmp_path / "store"
    outside.mkdir()
    (outside / "grid.xml").write_bytes(b"<grid v='1'/>\n")
    d = tmp_path / "inputs"
    (d / "shakemap").mkdir(parents=True)
    (d / "sites.csv").write_text("site_id,lon,lat\nex:1,172.6,-43.5\n")
    os.symlink(outside / "grid.xml", d / "shakemap" / "grid.xml")
    return {"dir": d, "target": outside / "grid.xml", "tmp": tmp_path}


def test_a_link_to_a_file_is_hashed_as_its_target_everywhere(linked):
    d, tgt, tmp = linked["dir"], linked["target"], linked["tmp"]
    w = P.write_table(d, tmp / "t.csv")
    rows, h = P.read_table(tmp / "t.csv")
    assert rows["shakemap/grid.xml"] == _sha(tgt.read_bytes())             # the target's bytes, the link's path
    assert P.rebuild_sha256(d) == (h, 2) and w["package_sha256"] == h == P.table_sha256(tmp / "t.csv")
    assert P.folder_rows(d) == rows
    copy = tmp / "copy"                                                    # a fetcher with a plain file agrees
    (copy / "shakemap").mkdir(parents=True)
    (copy / "shakemap" / "grid.xml").write_bytes(tgt.read_bytes())
    (copy / "sites.csv").write_bytes((d / "sites.csv").read_bytes())
    assert P.check_files(tmp / "t.csv", copy)["status"] == "PASS" == P.check_files(tmp / "t.csv", d)["status"]
    assert P.rebuild_sha256(copy)[0] == h
    tgt.write_bytes(b"<grid v='2'/>\n")                                    # the target moves: the link reads FAIL
    assert P.check_files(tmp / "t.csv", d)["files"]["shakemap/grid.xml"] == "FAIL"
    link_table = tmp / "link-table.csv"                                    # `hash` reads a linked table as its target
    os.symlink(tmp / "t.csv", link_table)
    assert P.table_sha256(link_table) == h


@pytest.mark.parametrize("kind", ["folder", "broken"])
def test_a_link_to_a_folder_or_to_nothing_is_refused_everywhere(linked, kind):
    d, tmp = linked["dir"], linked["tmp"]
    P.write_table(d, tmp / "t.csv")
    if kind == "folder":
        os.symlink(linked["target"].parent, d / "more")
    else:
        os.symlink(tmp / "nowhere.csv", d / "gone.csv")
    for call in (lambda: P.write_table(d, tmp / "t2.csv"), lambda: P.rebuild_sha256(d),
                 lambda: P.check_files(tmp / "t.csv", d)):
        with pytest.raises(SpecError, match="symbolic link to a folder or to nothing"):
            call()
    assert P.main(["check", str(tmp / "t.csv"), str(d)]) == 2


def test_a_linked_file_inside_frozen_inputs_is_read_as_its_target(src):
    _freeze(src)
    d = src["run"] / "inputs" / "A"
    real = src["run"] / "grid-copy.xml"
    real.write_bytes((d / "shakemap" / "grid.xml").read_bytes())
    (d / "shakemap" / "grid.xml").unlink()
    os.symlink(real, d / "shakemap" / "grid.xml")
    F.load_frozen(src["run"], "A")                                         # same target bytes: still the frozen table
    real.write_bytes(real.read_bytes() + b"\n")
    with pytest.raises(F.RunRefused, match="shakemap/grid.xml"):
        F.load_frozen(src["run"], "A")


# ---------------------------------------------------------------------------------------------------------------
# The amendment 2 byte form did not move
# ---------------------------------------------------------------------------------------------------------------

def test_amendment_2_worked_example_still_hashes_the_same(tmp_path):
    d = tmp_path / "pkg"
    d.mkdir()
    (d / "cpt_list.csv").write_bytes(b"site_id,lon,lat\nex:0001,172.60,-43.50\n")
    (d / "shakemap_grid.xml").write_bytes(b'<shakemap_grid example="1"/>\n')
    w = P.write_table(d, tmp_path / "TABLE.csv")
    assert (tmp_path / "TABLE.csv").stat().st_size == 176
    assert w["package_sha256"] == "1c858d5ec6fa1c733a7d4b6fda6530f69ef286831ebd77f57885f1fa87efeb10"


# ---------------------------------------------------------------------------------------------------------------
# Hidden paths: one answer from write, rebuild, check and both intake paths (B4-LAYOUT review 1, F1 and N2)
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("plant", ["hidden_broken_link", "hidden_folder_with_space"])
def test_a_hidden_odd_path_never_refuses_write_rebuild_check_or_intake(tmp_path, plant):
    from scoreboard.evalkit import intake
    d = tmp_path / "inputs"
    (d / "package").mkdir(parents=True)
    (d / "event.json").write_text('{"comcat_id":"x"}\n')
    (d / "package" / "vs30.csv").write_text("site_id,vs30\nex:1,200\n")
    table = tmp_path / "t.csv"
    w = P.write_table(d, table)
    if plant == "hidden_broken_link":
        os.symlink(tmp_path / "nowhere", d / "package" / ".#vs30.csv")    # an editor lock file
        odd = "package/.#vs30.csv"
    else:
        (d / ".Trash 1").mkdir()
        (d / ".Trash 1" / "x").write_text("old\n")
        odd = ".Trash 1/x"
    assert P.rebuild_sha256(d) == (w["package_sha256"], 2)
    assert P.folder_rows(d) == P.read_table(table)[0]
    res = P.check_files(table, d)
    assert res["status"] == "PASS" and res["not_in_table"] == [odd]
    assert P.main(["check", str(table), str(d)]) == 0
    assert P.write_table(d, tmp_path / "t2.csv")["package_sha256"] == w["package_sha256"]
    want = w["package_sha256"]
    assert intake.files_against(want, d) == "PASS"                         # the rebuild path
    assert intake.files_against(want, d, table) == "PASS"                  # the table path
    (d / "package" / "vs30.csv").write_text("site_id,vs30\nex:1,201\n")   # a real change still reads FAIL
    assert intake.files_against(want, d).startswith("FAIL")
    assert "package/vs30.csv FAIL" in intake.files_against(want, d, table)


def test_a_hidden_table_row_is_still_tested_by_check(tmp_path):
    d = tmp_path / "pkg"
    d.mkdir()
    (d / "a.csv").write_text("a\n")
    (d / ".params").write_text("k=1\n")
    rows = {"a.csv": _sha(b"a\n"), ".params": _sha(b"k=1\n")}               # a hidden input row, by hand
    table = tmp_path / "t.csv"
    table.write_text(P.table_text(rows), encoding="utf-8", newline="")
    assert P.check_files(table, d)["status"] == "PASS"
    (d / ".params").unlink()
    os.symlink(tmp_path / "nowhere", d / ".params")                       # the row is a broken link: refused
    with pytest.raises(SpecError, match="symbolic link to a folder or to nothing"):
        P.check_files(table, d)


@pytest.mark.parametrize("name", ["caf\u00e9.csv", "a b.csv"])
def test_a_non_hidden_badly_named_extra_makes_check_refused_as_write_and_rebuild(tmp_path, name):
    d = tmp_path / "pkg"
    d.mkdir()
    (d / "a.csv").write_text("a\n")
    P.write_table(d, tmp_path / "t.csv")
    (d / name).write_text("b\n")                                          # N2: REFUSED, as the docstring now says
    for call in (lambda: P.check_files(tmp_path / "t.csv", d), lambda: P.rebuild_sha256(d)):
        with pytest.raises(SpecError, match="printable ASCII, or holds a space or a quote"):
            call()
    assert P.main(["check", str(tmp_path / "t.csv"), str(d)]) == 2


def test_a_space_is_refused_by_the_range_check_alone():
    assert " " not in P._ID_BANNED                                         # N1: the range check 33..126 does it
    with pytest.raises(SpecError, match="printable ASCII, or holds a space"):
        P._check_id("a b.csv")
