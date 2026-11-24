"""C4 review 1 (wave 3): a v3 card holds no per-earthquake word anywhere, with the B5 geographic null and the B7
placement block on. Ruling 47: until a method certifies, words only on the pooled line. Synthetic data only.

The fixtures are the B5 and B7 test worlds (imported, not copied), rebuilt at fixed 4 km tiles for v3.
"""
import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_evalkit_b5_baselines as B5  # noqa: E402
import test_evalkit_placement as TP  # noqa: E402
from test_evalkit_b5_baselines import world  # noqa: E402,F401  (pytest fixture)
from scoreboard.evalkit import geo_null as GN  # noqa: E402
from scoreboard.evalkit import interval_policy as IP  # noqa: E402
from scoreboard.evalkit import site_memory as SM  # noqa: E402
from scoreboard.evalkit import synth  # noqa: E402
from scoreboard.evalkit.card import DESCRIPTIVE, build_card, scan_aggregates  # noqa: E402
from scoreboard.evalkit.spec import Entry  # noqa: E402

pytestmark = pytest.mark.skipif(IP.C4_OUTCOME["per_earthquake"] is not None, reason="a per-earthquake method certified")

ANY_WORD = IP.PER_EARTHQUAKE_WORDS


def _tile4(task):
    return dataclasses.replace(task, tile_m=4000.0, tile_rule="fixed")


def test_v3_with_geo_null_withholds_word_vs_null(world):
    ev, lon, lat, grids, eg, ep, labels, task, model = world
    g = GN.build(B5.EV, B5.EV_DATE, SM.sites_of(labels), eg)
    copy = Entry(synth.manifest("copy of the null", "score", B5.EV), ev["site_ids"], 3.0 * g.entry.p + 1.0)
    t4 = _tile4(task)
    v2 = B5.build_card(t4, [model, copy], labels, geo_null=g)
    v3 = B5.build_card(t4, [model, copy], labels, geo_null=g, card_version="v3")
    # the v2 card at the same tile prints the words this test looks for, so the test can fail
    assert {r["word_vs_null"] for r in v2["geographic_null"]["calibrated_auc"].values()} <= ANY_WORD
    assert IP.words_left(v2)
    cal = v3["geographic_null"]["calibrated_auc"]
    assert cal and all(r["word_vs_null"] == DESCRIPTIVE for r in cal.values())
    for name, r in cal.items():                                    # numbers and intervals stay
        assert r["value"] == pytest.approx(v2["geographic_null"]["calibrated_auc"][name]["value"])
        assert r["lo"] == pytest.approx(v2["geographic_null"]["calibrated_auc"][name]["lo"])
    assert v3["words"] == DESCRIPTIVE and IP.words_left(v3) == []
    scan_aggregates(v3, labels.site_ids)


@pytest.mark.parametrize("mode", ["placement_only", "continuous"])
def test_v3_with_placement_withholds_the_proxy_word(mode):
    task, entries, labels, ps = TP._world_card_inputs(4 if mode == "placement_only" else 1, mode)
    t4 = _tile4(task)
    v2 = build_card(t4, entries, labels, placement=ps)
    v3 = build_card(t4, entries, labels, placement=ps, card_version="v3")
    assert v2["placement"]["proxy"]["word"] in ANY_WORD
    px = v3["placement"]["proxy"]
    assert px["word"] == DESCRIPTIVE and "residual_excludes_0.5" not in px and "rule" not in px
    assert "ruling 47" in px["word_note"]
    for k in ("auc_overall", "auc_within_density_strata", "open_bias_size_auc"):     # numbers stay
        assert px[k] == v2["placement"]["proxy"][k]
    assert IP.words_left(v3) == []
    scan_aggregates(v3, labels.site_ids)


def test_v3_refuses_a_card_that_still_holds_a_word():
    with pytest.raises(IP.IntervalPolicyError, match="still holds words"):
        IP._withhold_other_words({"pgv": {"word": "SEPARATES"}}, DESCRIPTIVE)
    assert IP.words_left({"note": "OPEN BIAS: 95% interval excludes 0.5"}) == []    # rule text is not a word
