"""evalkit spec: strict validation, JSON round trips, canonical hashes (rule book R2, R3, R4, R6, R7)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.g5_amend import canonical_sha256  # noqa: E402
from scoreboard.evalkit.spec import (Entry, EntryManifest, Labels, SpecError, Task, entry_sha256,  # noqa: E402
                                     labels_sha256, load_entry, load_labels, load_task, predictions_text,
                                     probability_eligible, save_entry, save_labels, save_task, site_list_sha256)

H = "ab" * 32


def task_dict(**kw):
    d = {"event_id": "us7000test", "target": "ejecta seen at the site", "label_source": "survey v1",
         "label_rule_sha256": H, "site_list_sha256": H, "track": "forward", "division": "closed", "tile_m": 2000.0,
         "clip": 1e-6, "seed": 1, "label_caveats": "aerial grading"}
    d.update(kw)
    return d


def manifest(**kw):
    d = {"model": "M", "output_kind": "target_probability", "statement": "p of ejecta",
         "exposure": {"us7000test": "unseen"}, "house": False}
    d.update(kw)
    return EntryManifest.from_dict(d)


def test_task_round_trip_and_hash(tmp_path):
    t = Task.from_dict(task_dict(label_error={"sensitivity": 0.9, "specificity": 0.8}))
    save_task(t, tmp_path / "t.json")
    t2 = load_task(tmp_path / "t.json")
    assert t2 == t and t2.sha256() == t.sha256()
    assert Task.from_dict(task_dict(tile_m=2000)).sha256() == Task.from_dict(task_dict()).sha256()
    assert Task.from_dict(task_dict(seed=2)).sha256() != t.sha256()


@pytest.mark.parametrize("change, field", [
    ({"colour": "red"}, "unknown field"), ({"track": "backward"}, "task.track"), ({"clip": 1e-5}, "task.clip"),
    ({"label_rule_sha256": "xyz"}, "label_rule_sha256"), ({"target": "  "}, "task.target"),
    ({"seed": True}, "task.seed"), ({"n_boot": 10}, "task.n_boot"),
    ({"label_error": {"sensitivity": 0.9}}, "label_error"), ({"earlier_base_rate": 1.0}, "earlier_base_rate")])
def test_task_refuses_bad_fields(change, field):
    with pytest.raises(SpecError, match=field):
        Task.from_dict(task_dict(**change))


def test_task_refuses_missing_field():
    d = task_dict()
    del d["label_caveats"]
    with pytest.raises(SpecError, match="missing"):
        Task.from_dict(d)


def test_task_constructor_validates_too():
    with pytest.raises(SpecError):
        Task(**task_dict(division="secret"))


@pytest.mark.parametrize("change, field", [
    ({"output_kind": "probability"}, "output_kind"), ({"exposure": {}}, "exposure"),
    ({"exposure": {"us7000test": "maybe"}}, "exposure"), ({"house": "yes"}, "house"),
    ({"conversion_id": H}, "conversion_id"), ({"division": "middle"}, "division"), ({"author": "x"}, "unknown")])
def test_manifest_refuses_bad_fields(change, field):
    with pytest.raises(SpecError, match=field):
        manifest(**change)


def test_probability_eligible_is_r3():
    assert probability_eligible(manifest())
    assert not probability_eligible(manifest(output_kind="other_probability"))
    assert not probability_eligible(manifest(output_kind="score"))
    assert probability_eligible(manifest(output_kind="other_probability", conversion_id=H))
    assert probability_eligible(manifest(output_kind="score", conversion_id=H))


def test_entry_validation():
    ids = ("a", "b", "c")
    with pytest.raises(SpecError, match="\\[0, 1\\]"):
        Entry(manifest(), ids, [0.1, 1.2, 0.3])
    Entry(manifest(output_kind="score"), ids, [0.1, 7.0, -3.0])
    with pytest.raises(SpecError, match="finite"):
        Entry(manifest(output_kind="score"), ids, [0.1, np.nan, 0.3])
    with pytest.raises(SpecError, match="duplicate"):
        Entry(manifest(), ("a", "a", "c"), [0.1, 0.2, 0.3])
    with pytest.raises(SpecError, match="comma"):
        Entry(manifest(), ("a,b", "c", "d"), [0.1, 0.2, 0.3])
    with pytest.raises(SpecError, match="one value per site"):
        Entry(manifest(), ids, [0.1, 0.2])


def test_entry_hash_matches_g5_canonical_text_and_binds_manifest(tmp_path):
    rng = np.random.default_rng(0)
    ids = tuple(f"s{k}" for k in range(50))
    e = Entry(manifest(), ids, np.round(rng.random(50), 6) * 0.99)   # far from a 1e-9 rounding edge
    save_entry(e, tmp_path / "e.json", tmp_path / "e.csv")
    assert canonical_sha256(tmp_path / "e.csv", str_cols=frozenset({"site_id"})) == \
        __import__("hashlib").sha256(predictions_text(ids, e.p).encode()).hexdigest()
    e2 = load_entry(tmp_path / "e.json", tmp_path / "e.csv")
    assert entry_sha256(e2) == entry_sha256(e)
    tiny = Entry(manifest(), ids, e.p + 1e-12)
    assert entry_sha256(tiny) == entry_sha256(e)                       # values hashed at 1e-9
    moved = Entry(manifest(), ids, np.where(np.arange(50) == 3, e.p + 1e-6, e.p))
    assert entry_sha256(moved) != entry_sha256(e)
    exposed = Entry(manifest(exposure={"us7000test": "fitted_on"}), ids, e.p)
    assert entry_sha256(exposed) != entry_sha256(e)                    # a changed manifest is a new entry (R4)


def test_labels_round_trip_and_validation(tmp_path):
    lab = Labels(("a", "b", "c"), [1, 0, np.nan], [0.0, 1.0, 2.0], [5.0, 6.0, 7.0])
    save_labels(lab, tmp_path / "l.csv")
    assert (tmp_path / "l.csv").read_text().splitlines()[3].startswith("c,,")
    lab2 = load_labels(tmp_path / "l.csv")
    assert labels_sha256(lab2) == labels_sha256(lab)
    assert lab2.labelled.tolist() == [True, True, False]
    with pytest.raises(SpecError, match="label must be"):
        Labels(("a", "b"), [1, 2], [0, 0], [0, 0])
    (tmp_path / "dup.csv").write_text("site_id,label,x_m,y_m\na,1,0,0\na,0,1,1\n")
    with pytest.raises(SpecError, match="duplicate"):
        load_labels(tmp_path / "dup.csv")
    (tmp_path / "head.csv").write_text("id,label,x_m,y_m\na,1,0,0\n")
    with pytest.raises(SpecError, match="header"):
        load_labels(tmp_path / "head.csv")


def test_site_list_hash_ignores_order():
    assert site_list_sha256(["b", "a", "c"]) == site_list_sha256(("a", "c", "b"))
    assert site_list_sha256(["a", "b"]) != site_list_sha256(["a", "b", "c"])


def test_manifest_json_is_strict(tmp_path):
    (tmp_path / "m.json").write_text(json.dumps({"model": "M"}))
    (tmp_path / "m.csv").write_text("site_id,p\na,0.1\n")
    with pytest.raises(SpecError, match="missing"):
        load_entry(tmp_path / "m.json", tmp_path / "m.csv")
