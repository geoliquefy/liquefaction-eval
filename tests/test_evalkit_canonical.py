"""evalkit canonical card hash and the effort-slot allowlist record (eval plan step 2b; review 2 nit N3)."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import effort as E  # noqa: E402
from scoreboard.evalkit import synth  # noqa: E402
from scoreboard.evalkit.card import CardRefused, build_card, write_card  # noqa: E402
from scoreboard.evalkit.cli import main  # noqa: E402
from scoreboard.evalkit.spec import (CARD_HASH_SIG, Entry, EntryManifest, SpecError, canonical_card_json,  # noqa: E402
                                     card_sha256, entry_sha256, load_card_sha256)


@pytest.fixture(scope="module")
def demo():
    task, entries, labels, shaking = synth.demo_inputs(n=600, n_boot=50)
    return task, entries, labels, shaking, build_card(task, entries, labels, shaking_only=shaking)


def floats(obj, path=()):
    """Every float in a card, with its path."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from floats(v, path + (k,))
    elif isinstance(obj, list):
        for k, v in enumerate(obj):
            yield from floats(v, path + (k,))
    elif isinstance(obj, float):
        yield path, obj


def set_at(obj, path, value):
    for k in path[:-1]:
        obj = obj[k]
    obj[path[-1]] = value


def calibration_paths(card):
    return [p for p, v in floats(card) if "calibration" in p and v != 0.0][:30]


# Canonical hash ------------------------------------------------------------------------------------------------

def test_cards_that_differ_below_the_precision_hash_the_same(demo):
    """30 calibration values moved by up to 1.7e-16 (the container vs Mac gap): same card hash."""
    card = demo[4]
    other = copy.deepcopy(card)
    paths = calibration_paths(card)
    assert len(paths) == 30
    for p in paths:
        v = dict(floats(card))[p]
        set_at(other, p, float(np.nextafter(v, np.inf)))   # one ulp: 1.1e-16 at 0.5 to 1, 2.2e-16 at 1 to 2
    assert other != card
    assert card_sha256(other) == card_sha256(card)


def test_cards_that_differ_above_the_precision_hash_differently(demo):
    card = demo[4]
    p = calibration_paths(card)[0]
    v = dict(floats(card))[p]
    other = copy.deepcopy(card)
    set_at(other, p, v * (1 + 10 ** -(CARD_HASH_SIG - 2)))   # a change at the 11th significant digit
    assert card_sha256(other) != card_sha256(card)
    other2 = copy.deepcopy(card)
    set_at(other2, p, v + 1e-6)
    assert card_sha256(other2) != card_sha256(card)


def test_the_hash_ignores_key_order_and_negative_zero_but_not_types():
    a = {"b": 1.0, "a": [0.0, 2], "c": None, "d": True}
    b = {"d": True, "c": None, "a": [-0.0, 2], "b": 1.0}
    assert card_sha256(a) == card_sha256(b)
    assert card_sha256({"x": 1}) != card_sha256({"x": True})
    assert canonical_card_json({"x": 0.1 + 0.2}) == '{"x":0.3}'
    with pytest.raises(SpecError, match="finite"):
        card_sha256({"x": float("nan")})


def test_near_zero_noise_hashes_as_zero():
    """Review 1 of step 2, MINOR 5: -2.78e-17, 0, -0.0 and 1e-13 all hash as 0.0; 1e-12 and above do not."""
    h0 = card_sha256({"x": 0.0})
    for v in (-2.78e-17, -0.0, 1e-13, -9.99e-13):
        assert card_sha256({"x": v}) == h0
    assert canonical_card_json({"x": -2.78e-17}) == '{"x":0.0}'
    assert card_sha256({"x": 1e-12}) != h0 and card_sha256({"x": -1e-11}) != h0


def test_file_hash_reads_and_never_rewrites(demo, tmp_path, capsys):
    task, _, labels, _, card = demo
    path = tmp_path / "card.json"
    write_card(card, path, labels.site_ids)
    before = path.read_bytes()
    assert load_card_sha256(path) == card_sha256(json.loads(before))
    assert main(["hash", str(path)]) == 0
    assert capsys.readouterr().out.strip() == card_sha256(card)
    assert path.read_bytes() == before


def test_cli_summary_prints_the_canonical_hash(tmp_path, capsys):
    out = tmp_path / "demo.json"
    assert main(["demo", "--out", str(out), "--n-boot", "50"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["card_sha256"] == load_card_sha256(out)


# N3: the effort slot needs the pre-event allowlist recorded in the manifest ------------------------------------

def test_effort_entry_records_its_allowlist():
    ids = ("syn:00000", "syn:00001")
    good = E.entry(ids, np.array([0.2, 0.4]), "q", "s")
    assert good.manifest.feature_allowlist == E.ALLOWLIST_NAME == "pre-event"
    assert good.manifest.to_dict()["feature_allowlist"] == "pre-event"
    proxy = E.entry(ids, np.array([0.2, 0.4]), "q", "s", allowlist=E.PROXY_FEATURES)
    assert proxy.manifest.feature_allowlist == "proxy"
    with pytest.raises(E.EffortRefused, match="allowlist"):
        E.entry(ids, np.array([0.2, 0.4]), "q", "s", allowlist={"pga": "made up"})
    assert entry_sha256(good) != entry_sha256(proxy)


def test_older_manifests_keep_their_hash():
    """The new field is omitted when unset, so every entry built before step 2b keeps its R4 hash."""
    m = EntryManifest(model="m", output_kind="score", statement="s", exposure={"q": "unseen"}, house=False)
    assert "feature_allowlist" not in m.to_dict()
    assert EntryManifest.from_dict(m.to_dict()) == m
    e = Entry(m, ("syn:00000",), np.array([0.5]))
    # sha256 of '{"conversion_id":null,"division":null,"exposure":{"q":"unseen"},"house":false,"model":"m",...'
    assert entry_sha256(e) == entry_sha256(Entry(EntryManifest.from_dict(json.loads(json.dumps(m.to_dict()))),
                                                 ("syn:00000",), np.array([0.5])))
    with pytest.raises(SpecError, match="feature_allowlist"):
        EntryManifest.from_dict({**m.to_dict(), "feature_allowlist": ""})


def test_card_refuses_an_effort_baseline_without_the_pre_event_allowlist(demo):
    task, entries, labels, shaking, _ = demo
    ids = labels.site_ids
    p = np.linspace(0.1, 0.9, len(ids))
    base = dict(model=E.NAME, output_kind="other_probability", statement="effort", exposure={task.event_id: E.EXPOSURE},
                house=False)
    for allow in (None, "proxy", "post-event"):
        m = EntryManifest(**base, feature_allowlist=allow)
        with pytest.raises(CardRefused, match="feature_allowlist 'pre-event'"):
            build_card(task, entries, labels, shaking_only=shaking, effort_only=Entry(m, ids, p))
    proxy = E.entry(ids, p, task.event_id, "proxy predictions under the baseline name", allowlist=E.PROXY_FEATURES)
    with pytest.raises(CardRefused, match="N3"):
        build_card(task, entries, labels, shaking_only=shaking, effort_only=proxy)
    good = Entry(EntryManifest(**base, feature_allowlist="pre-event"), ids, p)
    card = build_card(task, entries, labels, shaking_only=shaking, effort_only=good)
    assert E.NAME in [r["name"] for r in card["entries"]]
