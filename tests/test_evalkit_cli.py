"""evalkit command line: gates, score from files, demo, and refusals. Synthetic only."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoreboard.evalkit import synth  # noqa: E402
from scoreboard.evalkit.cli import main  # noqa: E402
from scoreboard.evalkit.spec import save_entry, save_labels, save_task  # noqa: E402


def write_inputs(tmp_path, n_boot=100):
    task, entries, labels, shaking = synth.demo_inputs(n=700, n_boot=n_boot)
    save_task(task, tmp_path / "task.json")
    save_labels(labels, tmp_path / "labels.csv")
    d = tmp_path / "entries"
    d.mkdir()
    for k, e in enumerate(entries):
        save_entry(e, d / f"e{k}.json", d / f"e{k}.csv")
    rows = ["site_id,p"] + [f"{s},{float(v)!r}" for s, v in zip(shaking.site_ids, shaking.p)]
    (tmp_path / "shaking.csv").write_text("\n".join(rows) + "\n")
    return task


def test_score_from_files(tmp_path, capsys):
    task = write_inputs(tmp_path)
    out = tmp_path / "card.json"
    rc = main(["score", "--task", str(tmp_path / "task.json"), "--entries", str(tmp_path / "entries"),
               "--labels", str(tmp_path / "labels.csv"), "--out", str(out), "--shaking-only",
               str(tmp_path / "shaking.csv"), "--registered-task-sha256", task.sha256()])
    assert rc == 0
    card = json.loads(out.read_text())
    assert card["status"] == "SCORED" and card["task_sha256"] == task.sha256()
    assert "baseline: shaking only" in card["metrics"]
    assert json.loads(capsys.readouterr().out)["status"] == "SCORED"


def test_score_refuses_with_exit_2(tmp_path, capsys):
    write_inputs(tmp_path)
    lines = (tmp_path / "labels.csv").read_text().splitlines()
    (tmp_path / "labels.csv").write_text("\n".join(lines + [lines[1]]) + "\n")      # a duplicate id
    rc = main(["score", "--task", str(tmp_path / "task.json"), "--entries", str(tmp_path / "entries"),
               "--labels", str(tmp_path / "labels.csv"), "--out", str(tmp_path / "card.json")])
    assert rc == 2 and "duplicate" in capsys.readouterr().err
    assert not (tmp_path / "card.json").exists()
    (tmp_path / "entries" / "orphan.json").write_text("{}")
    rc = main(["score", "--task", str(tmp_path / "task.json"), "--entries", str(tmp_path / "entries"),
               "--labels", str(tmp_path / "labels.csv"), "--out", str(tmp_path / "card.json")])
    assert rc == 2


def test_gates_and_demo_through_python_dash_m(tmp_path):
    py = sys.executable
    g = subprocess.run([py, "-m", "scoreboard.evalkit", "gates"], cwd=ROOT, capture_output=True, text=True)
    assert g.returncode == 0 and json.loads(g.stdout)["pass"] is True
    out = tmp_path / "demo.json"
    d = subprocess.run([py, "-m", "scoreboard.evalkit", "demo", "--n-boot", "100", "--out", str(out)], cwd=ROOT,
                       capture_output=True, text=True)
    assert d.returncode == 0, d.stderr
    card = json.loads(out.read_text())
    assert card["status"] == "SCORED" and card["duplicates"]["pairs"][0]["b"] == "House model, steeper copy"
    assert card["metrics"]["Published index"]["probability_metrics"].startswith("NOT APPLICABLE")
    assert any("NOT FOR RELEASE" in n for n in card["notes"])
    shares = card["metrics"]["House manifestation model"]["clip_sensitivity"]
    assert np.isclose(shares["share_exact_0"], 0.0934, atol=0.01)
