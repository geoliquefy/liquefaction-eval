"""B8 step 2 (eval M4): the B5 geographic null on a card with B2 on (author_confirmed). An outside row that prints NOT
RUN BY AUTHOR SETTINGS has no score, so calibrated_auc marks it and does not read its metrics. Synthetic data only.
"""
import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_evalkit_b5_baselines as B5  # noqa: E402
from test_evalkit_b5_baselines import world  # noqa: E402,F401  (pytest fixture)
from scoreboard.evalkit import geo_null as GN  # noqa: E402
from scoreboard.evalkit import interval_policy as IP  # noqa: E402
from scoreboard.evalkit import outside  # noqa: E402
from scoreboard.evalkit import site_memory as SM  # noqa: E402
from scoreboard.evalkit import synth  # noqa: E402
from scoreboard.evalkit.card import scan_aggregates  # noqa: E402
from scoreboard.evalkit.spec import Entry  # noqa: E402


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_geo_null_with_an_outside_row_not_run(world, version):
    ev, lon, lat, grids, eg, ep, labels, task, model = world
    g = GN.build(B5.EV, B5.EV_DATE, SM.sites_of(labels), eg)
    house = Entry(synth.manifest("h", "score", B5.EV, house=True), ev["site_ids"], model.p * 0.5 + 1.0)
    t = dataclasses.replace(task, tile_m=4000.0, tile_rule="fixed")
    card = B5.build_card(t, [house, model], labels, geo_null=g, author_confirmed={}, card_version=version)
    assert card["metrics"]["m"] == outside.NOT_RUN                  # model "m" is an outside row (house False)
    cal = card["geographic_null"]["calibrated_auc"]
    assert cal["m"] == outside.NOT_RUN
    assert cal["h"]["value"] == pytest.approx(card["metrics"]["h"]["auc"]["value"]
                                              - card["metrics"][GN.NAME]["auc"]["value"] + 0.5)
    if version == "v3":
        assert IP.words_left(card) == []
    scan_aggregates(card, labels.site_ids)
