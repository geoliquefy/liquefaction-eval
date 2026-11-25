"""B4 sandbox and the scorer-side pin check (founder ruling 59a, wave 13; eval M5). Synthetic and planted only.

Planted entries (tests/b4_sandbox_entries.py): one honest, one that reads a label file, one that writes outside
/outputs/, one that opens a socket, one that starts a process, one that lists a folder outside /inputs/, and one that
reads a kit file. Review 1 added: a folder handle read through dir_fd, a relative '..' read, a sqlite3 database written
outside /outputs/, an honest standard library import, and a packaged entry whose __init__.py must never run. Each bad
one hides its own error; the sandbox refuses it anyway. A task with an off-pin seed (and
tile_m, clip) is refused before the label file is opened. No Canterbury data is read. No em dashes.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoreboard.evalkit import forward as F  # noqa: E402
from scoreboard.evalkit import sandbox as SB  # noqa: E402
from scoreboard.evalkit.cli import main as cli_main  # noqa: E402
from scoreboard.evalkit.spec import SpecError, save_task  # noqa: E402
from scoreboard.evalkit.task_builder import check_pins, load_pins  # noqa: E402
from tests.test_evalkit_forward import FLAG, TASK, planted_labels, write_cpts, write_grid  # noqa: E402

MOD = "tests.b4_sandbox_entries"


def entry(name, fn, **params):
    spec = f"{MOD}:{fn}"
    return {"name": name, "runner": {"kind": "python", "callable": spec, "code_sha256": F.callable_sha256(spec),
                                     "params": params},
            "manifest": {"model": name, "output_kind": "score", "statement": "a planted sandbox entry",
                         "exposure": {"*": "unseen"}, "house": False}}


@pytest.fixture
def frozen(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    labels = src / "PLANTED-labels.csv"
    labels.write_text("site_id,label\ns0000,1\n")            # a stand-in label file the bad entries reach for
    leak = tmp_path / "leak.txt"
    dbleak = tmp_path / "leak.sqlite"
    reg = src / "register.jsonl"
    entries = [
        {"name": "shaking score", "runner": {"kind": "builtin", "name": "shaking_score"},
         "manifest": {"model": "shaking score", "output_kind": "score", "statement": "PGA / MSF(M)",
                      "exposure": {"*": "unseen"}, "house": False}},
        entry("honest", "honest"),
        entry("reads label", "reads_label", path=str(labels)),
        entry("writes outside", "writes_outside", path=str(leak)),
        entry("opens socket", "opens_socket"),
        entry("starts process", "starts_process", path=str(labels)),
        entry("lists outside", "lists_outside", path=str(src)),
        entry("reads kit", "reads_label", path=str(ROOT / "scoreboard" / "evalkit" / "task_pins.json")),
        entry("imports stdlib", "imports_stdlib"),
        entry("dir fd read", "dir_fd_read", path=str(labels)),
        entry("dotdot read", "dotdot_read", path=str(labels)),
        entry("sqlite outside", "sqlite_outside", path=str(dbleak)),
    ]
    F.write_register(reg, entries)
    run = tmp_path / "run"
    F.freeze_inputs(run, FLAG, write_grid(src / "gA.xml"), write_cpts(src / "cpts.csv"), input_type="A",
                    register_path=reg)
    return {"run": run, "reg": reg, "labels": labels, "leak": leak, "dbleak": dbleak, "src": src}


REFUSED = ["dir fd read", "dotdot read", "lists outside", "opens socket", "reads kit", "reads label", "sqlite outside",
           "starts process", "writes outside"]
RUN = ["honest", "imports stdlib", "shaking score"]


def test_planted_entries_are_refused_and_the_honest_one_runs(frozen):
    out = F.predict(frozen["run"], frozen["reg"], "A")
    rec = out["record"]
    assert sorted(rec["entries"]) == RUN
    why = {d["name"]: d for d in rec["sandbox_refused"]}
    assert sorted(why) == REFUSED
    assert why["reads label"]["reason"] == "SANDBOX REFUSED: read outside /inputs/"
    assert why["reads label"]["path"] == str(frozen["labels"])
    assert why["reads kit"]["reason"] == "SANDBOX REFUSED: read outside /inputs/"
    assert why["writes outside"]["reason"] == "SANDBOX REFUSED: write outside /outputs/"
    assert why["opens socket"]["reason"] == "SANDBOX REFUSED: network or socket"
    assert why["starts process"]["reason"] == "SANDBOX REFUSED: new process or signal"
    assert why["lists outside"]["reason"] == "SANDBOX REFUSED: read outside /inputs/"
    assert why["dir fd read"]["reason"] == "SANDBOX REFUSED: folder handle (dir_fd paths are not checked)"
    assert why["dotdot read"]["reason"] == "SANDBOX REFUSED: relative path with '..'"
    assert why["sqlite outside"]["reason"] == "SANDBOX REFUSED: sqlite3 file outside /outputs/"
    assert not frozen["leak"].exists() and not frozen["dbleak"].exists()
    assert rec["sandbox"]["mode"] == "process" and rec["run_failed"] == [] and "audit_only" not in rec["sandbox"]
    log = F.read_log(frozen["run"])
    assert log[-1]["step"] == "predictions_written" and log[-1]["n_sandbox_refused"] == len(REFUSED)
    # the honest entry's values equal the same function run on the frozen inputs in this process
    d = F._pred_dir(frozen["run"], "A")
    got = np.loadtxt(d / "honest.csv", delimiter=",", skiprows=1, usecols=1)
    inp = F.frozen_inputs(frozen["run"], "A")
    assert np.allclose(got, -np.hypot(inp.x_m, inp.y_m), rtol=0, atol=0)


def test_the_audit_layer_alone_refuses_the_same_entries(frozen, monkeypatch):
    """With the kernel layer off and audit-only allowed, the audit hook still refuses every planted entry, and the
    honest standard library import runs with the container's read roots (the child adds its own install)."""
    monkeypatch.setitem(SB._SBX_OK, "ok", False)
    rec = F.predict(frozen["run"], frozen["reg"], "A", sandbox_opts={"allow_audit_only": True})["record"]
    assert rec["sandbox"]["kernel_layer"].startswith("none") and "allow_audit_only=True" in rec["sandbox"]["audit_only"]
    assert sorted(rec["entries"]) == RUN and sorted(d["name"] for d in rec["sandbox_refused"]) == REFUSED
    assert not frozen["leak"].exists() and not frozen["dbleak"].exists()


def test_no_kernel_layer_fails_closed(frozen, monkeypatch):
    """Review 1, S1: with no kernel layer, process mode is refused before anything is written, unless allowed."""
    monkeypatch.setitem(SB._SBX_OK, "ok", False)
    with pytest.raises(SpecError, match="no kernel layer here"):
        F.predict(frozen["run"], frozen["reg"], "A")
    assert not F._pred_dir(frozen["run"], "A").exists()
    assert not any(d["step"] == "predictions_written" for d in F.read_log(frozen["run"]))
    inp = F.frozen_inputs(frozen["run"], "A")
    manifest, _ = F.load_frozen(frozen["run"], "A")
    with pytest.raises(SpecError, match="no kernel layer here"):
        F._run_entry(entry("honest", "honest"), inp, frozen["run"], manifest, sandbox="process", sandbox_opts={})


def test_the_child_reads_its_own_stdlib_with_container_roots(frozen, monkeypatch):
    """Review 1, S2: the parent names no Python install in the read roots (as in the container); the child adds its
    own, so an honest import of a module not yet loaded runs. Kernel layer off, so only the child's roots count."""
    monkeypatch.setitem(SB._SBX_OK, "ok", False)
    inp = F.frozen_inputs(frozen["run"], "A")
    manifest, _ = F.load_frozen(frozen["run"], "A")
    p = F._run_entry(entry("imports stdlib", "imports_stdlib"), inp, frozen["run"], manifest, sandbox="process",
                     sandbox_opts={"allow_audit_only": True})
    assert np.array_equal(p, -np.hypot(inp.x_m, inp.y_m))


def test_a_packaged_entry_never_runs_its_parent_init(tmp_path, monkeypatch):
    """Review 1, B1: pkg/__init__.py of a 'pkg.entry:fn' callable must not run in the kit process (or anywhere)."""
    pkg = tmp_path / "pkgroot" / "b4_planted_pkg"
    pkg.mkdir(parents=True)
    flag = tmp_path / "INIT-RAN"
    (pkg / "__init__.py").write_text(f"open({str(flag)!r}, 'w').write(str(__import__('os').getpid()))\n")
    (pkg / "entry.py").write_text("import numpy as np\n\n\ndef honest(inp):\n    return -np.hypot(inp.x_m, inp.y_m)\n")
    monkeypatch.syspath_prepend(str(tmp_path / "pkgroot"))
    spec = "b4_planted_pkg.entry:honest"
    h = F.callable_sha256(spec)
    assert not flag.exists() and "b4_planted_pkg" not in sys.modules
    assert SB.module_file("b4_planted_pkg.entry") == (pkg / "entry.py").resolve() or \
        SB.module_file("b4_planted_pkg.entry") == pkg / "entry.py"
    assert SB.module_file("b4_planted_pkg.missing") is None and SB.module_file("os.path.x") is None
    src = tmp_path / "src"
    src.mkdir()
    reg = src / "register.jsonl"
    F.write_register(reg, [{"name": "packaged", "runner": {"kind": "python", "callable": spec, "code_sha256": h},
                            "manifest": {"model": "packaged", "output_kind": "score", "statement": "planted",
                                         "exposure": {"*": "unseen"}, "house": False}}])
    run = tmp_path / "run"
    F.freeze_inputs(run, FLAG, write_grid(src / "gA.xml"), write_cpts(src / "cpts.csv"), input_type="A",
                    register_path=reg)
    rec = F.predict(run, reg, "A")["record"]
    assert sorted(rec["entries"]) == ["packaged"] and rec["sandbox_refused"] == []
    assert not flag.exists() and "b4_planted_pkg" not in sys.modules


def test_the_refusal_names_the_entry_and_the_attempt(frozen):
    inp = F.frozen_inputs(frozen["run"], "A")
    manifest, _ = F.load_frozen(frozen["run"], "A")
    e = entry("reads label", "reads_label", path=str(frozen["labels"]))
    with pytest.raises(SB.SandboxRefused, match=r"SANDBOX REFUSED: entry 'reads label': read outside /inputs/"):
        F._run_entry(e, inp, frozen["run"], manifest, sandbox="process", sandbox_opts={})


def test_the_entry_sees_only_the_inputs_folder(frozen):
    """The sandboxed inputs equal the frozen ones; the event comes from event.json alone."""
    a = F.frozen_inputs(frozen["run"], "A")
    b = F.inputs_from_folder(F._inputs_dir(frozen["run"], "A"), "A")
    assert a.site_ids == b.site_ids and np.array_equal(a.pga_g, b.pga_g) and np.array_equal(a.x_m, b.x_m)
    assert b.event == {k: a.event[k] for k in ("id", "time_utc", "lon", "lat", "depth_km", "mag", "mag_type")}


def test_a_changed_inputs_copy_is_refused_before_the_run(frozen, tmp_path):
    manifest, _ = F.load_frozen(frozen["run"], "A")
    rows = dict(manifest["inputs"]["files"], **{"sites.csv": "0" * 64})
    with pytest.raises(SpecError, match="differs from the frozen table"):
        SB.run_entry(entry("honest", "honest"), F._inputs_dir(frozen["run"], "A"), rows, input_type="A")


def test_kernel_layer_blocks_a_home_read_without_python(tmp_path):
    """macOS only: the sandbox-exec profile alone stops /bin/cat reading a file in the kit folder."""
    if SB.layers().kernel_layer != "macos sandbox-exec":
        pytest.skip("sandbox-exec is not available here")
    run, out = tmp_path / "run", tmp_path / "run" / "outputs"
    out.mkdir(parents=True)
    py = SB._python_roots()
    py = (py + [py[-1]] * 4)[:4]
    d = {"OUT": str(out.resolve()), "HOME": SB._real(Path.home()), "KIT": SB._real(SB.KIT_ROOT),
         "RUN": str(run.resolve()), "CODE": str(run.resolve()), "KITCODE": SB._real(SB.KIT_ROOT / "scoreboard"),
         **{f"PY{k}": v for k, v in enumerate(py)}}
    argv = ["/usr/bin/sandbox-exec", "-p", SB.sbpl_profile()] + [x for k, v in d.items() for x in ("-D", f"{k}={v}")]
    secret = SB.KIT_ROOT / "README.md"
    assert subprocess.run(argv + ["/bin/cat", str(secret)], capture_output=True).returncode != 0
    assert subprocess.run(argv + ["/usr/bin/touch", str(tmp_path / "x")], capture_output=True).returncode != 0
    assert subprocess.run(argv + ["/usr/bin/touch", str(out / "x")], capture_output=True).returncode == 0


def test_container_argv_mounts_inputs_read_only_and_no_network(tmp_path):
    argv = SB.container_argv("evalkit:test", tmp_path / "run", tmp_path / "code")
    s = " ".join(argv)
    assert argv[:3] == ["docker", "run", "--rm"] and "--network none" in s and "--read-only" in s
    assert f"{tmp_path / 'run' / 'inputs'}:/inputs:ro" in argv and f"{tmp_path / 'run' / 'outputs'}:/outputs:rw" in argv
    assert "--cap-drop ALL" in s and "no-new-privileges" in s and "--cpus 8" in s and "--memory 32g" in s
    assert all(not a.endswith(":rw") or a.endswith("/outputs:rw") for a in argv if a.count(":") >= 1 and "/" in a)


def test_unknown_sandbox_mode_is_refused():
    with pytest.raises(SpecError, match="sandbox mode"):
        SB.layers("none")


# ---------------------------------------------------------------------------------------------------------------
# Scorer-side pin check (amended R2; rule book v1 amendment 1 section 4)
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("field,value", [("seed", 7), ("tile_m", 1000.0), ("clip", 1e-3)])
def test_score_refuses_an_off_pin_task_before_the_labels_open(frozen, field, value):
    F.predict(frozen["run"], frozen["reg"], "A")
    labels = planted_labels(frozen["run"], frozen["src"] / "labels-planted.csv")
    with pytest.raises(F.RunRefused, match=rf"off its pins for card version 'later': {field} "):
        F.score(frozen["run"], labels, "A", {**TASK, field: value})
    assert not any(d["step"] == "labels_opened" for d in F.read_log(frozen["run"]))
    card = F.score(frozen["run"], labels, "A", TASK)                 # the pinned task scores
    assert card["task"]["seed"] == 20261106 and card["forward"]["sandbox"]["mode"] == "process"
    assert card["forward"]["pins_checked"] == {"card_version": "later", "pins": load_pins("later")}
    assert {d["name"] for d in card["forward"]["sandbox_refused"]} >= {"reads label", "opens socket"}


def test_check_pins_reads_every_card_version():
    for v in ("v2", "v3", "v3.1", "later"):
        pins = load_pins(v)
        check_pins(dict(pins), v)
        with pytest.raises(SpecError, match="seed"):
            check_pins(dict(pins, seed=pins["seed"] + 1), v)
        with pytest.raises(SpecError, match=r"tile_m '\(missing\)'"):
            check_pins({k: x for k, x in pins.items() if k != "tile_m"}, v)
    with pytest.raises(SpecError, match="no pinned values"):
        check_pins(load_pins("later"), "v9")


def test_cli_score_with_card_version_refuses_an_off_pin_task(tmp_path, capsys):
    from scoreboard.evalkit import synth
    task, entries, labels, shaking = synth.demo_inputs(n=300, n_boot=50)     # demo task: tile_m 2000, its own seed
    save_task(task, tmp_path / "task.json")
    (tmp_path / "entries").mkdir()
    rc = cli_main(["score", "--task", str(tmp_path / "task.json"), "--entries", str(tmp_path / "entries"),
                   "--labels", str(tmp_path / "no-labels.csv"), "--out", str(tmp_path / "card.json"),
                   "--card-version", "v3.1"])
    err = capsys.readouterr().err
    assert rc == 2 and "off its pins for card version 'v3.1'" in err and "tile_m" in err
    assert not (tmp_path / "card.json").exists()
    with pytest.raises(FileNotFoundError):             # no label file: the warning comes before the labels
        cli_main(["score", "--task", str(tmp_path / "task.json"), "--entries", str(tmp_path / "entries"),
                  "--labels", str(tmp_path / "no-labels.csv"), "--out", str(tmp_path / "card.json")])
    err = capsys.readouterr().err                     # review 1, N2: without the flag the gap is printed
    assert "WARNING: no --card-version" in err and "NOT checked" in err
