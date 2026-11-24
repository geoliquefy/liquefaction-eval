"""Ruling 45 correction 8 across B1 (losses.py) and B2 (outside.py), wave 1 integration. Synthetic data only.

"LSN + global fragility" and "LSN + CES fragility" carry the SAME mark: both are our run of one published table
(Maurer, engrXiv 121, Table 3), and our chain fails parity with that paper. The card entry note (outside) and the
LOSS line (losses) must name the two rows alike and in the same words. No em dashes.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import losses, outside, synth  # noqa: E402
from scoreboard.evalkit.card import build_card  # noqa: E402
from scoreboard.evalkit.spec import Entry, EntryManifest, Task, site_list_sha256  # noqa: E402

EV = "eq-test"
NAMES = ["LSN + CES fragility [authors-maurer-2026-11-19]", "LSN + global fragility [authors-maurer-2026-11-19]",
         "LSN + CES fragility [default-2026-11]", "LSN + global fragility [default-2026-11]",
         "LSN + CES fragility [authors-new-chain]", "LSN + global fragility", "LSN + CES fragility",
         "Level 0", "M0 (Level 0 + earthquake band)", "USGS Zhu 2017", "GEM TS22", "LSN + local fragility [x]",
         "my LSN + CES fragility", "LSN + CES fragility [x] extra", "baseline: coin"]


@pytest.mark.parametrize("name", NAMES)
def test_both_modules_agree_on_which_rows_are_our_run_of_a_published_model(name):
    assert losses.is_published_ours(name) == outside.is_published_run(name)
    twin = outside.fragility_twin(name)
    if outside.is_published_run(name):
        assert twin is not None and losses.is_published_ours(twin) and outside.is_published_run(twin)


def test_both_modules_use_the_same_words():
    assert outside.PUBLISHED_RUN.startswith(losses.PUBLISHED_OURS_KIND)
    assert losses.PUBLISHED_OURS_MARK.lower() == outside.PUBLISHED_RUN.lower()
    assert "engrXiv 121, Table 3" in outside.PUBLISHED_RUN and "fails parity" in outside.PUBLISHED_RUN


def _task(ev):
    return Task(event_id=EV, target="ejecta seen at the site", label_source="synthetic v1",
                label_rule_sha256="1" * 64, site_list_sha256=site_list_sha256(ev["site_ids"]), track="record",
                division="closed", tile_m=2000.0, clip=1e-3, seed=3, label_caveats="synthetic labels", n_boot=60)


def test_on_one_card_both_lsn_rows_get_the_same_note_and_the_same_loss_mark():
    """Both author-chain LSN rows confirmed and both worse than the coin: same entry note, same LOSS mark and kind."""
    ev = synth.synthetic_event(900, 21, unlabelled_share=0.1)
    bad = np.clip(1.0 - ev["p_target"], 0.02, 0.98)
    names = ("LSN + global fragility [authors-test]", "LSN + CES fragility [authors-test]")
    entries = [Entry(EntryManifest(model=n, output_kind="score", statement="synthetic score", exposure={EV: "unseen"},
                                   house=True, conversion_id="c" * 64), ev["site_ids"], bad + 0.001 * i)
               for i, n in enumerate(names)]
    card = build_card(_task(ev), entries, synth.labels_of(ev), losses_first=True,
                      author_confirmed={names[0]: True, names[1]: True})
    meta = {e["name"]: e for e in card["entries"]}
    assert meta[names[0]]["published_run"] == meta[names[1]]["published_run"] == outside.PUBLISHED_RUN
    by = {ln["house"]: ln for ln in card["losses"]["lines"]}
    assert set(names) <= set(by)
    a, b = by[names[0]], by[names[1]]
    assert a["kind"] == b["kind"] == losses.PUBLISHED_OURS_KIND
    assert a["mark"] == b["mark"] == losses.PUBLISHED_OURS_MARK
    assert a["mark"].lower() == meta[names[0]]["published_run"].lower()
