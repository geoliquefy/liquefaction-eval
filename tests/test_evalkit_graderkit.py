"""evalkit grader kit (work list E3): blinding, licence check, GeoTIFF reader, tiles, form server, power.
Synthetic only. No em dashes."""
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit.graderkit import blind, demo, geo, power, serve, sources  # noqa: E402


# ---------- blinding ----------

def test_blind_item_keeps_only_allowed_fields():
    u = {"queue_id": "Q1", "x": 1.0, "y": 2.0, "earthquake": "feb2011", "cpt": "C", "score_band": 2,
         "weight": 3.0, "label": 1}
    assert blind.blind_item(u) == {"queue_id": "Q1", "x": 1.0, "y": 2.0, "earthquake": "feb2011"}


def test_blind_queue_mixes_flagged_seeds_and_hides_the_flag():
    units = [{"queue_id": "Q2", "x": 0, "y": 0, "earthquake": "sep2010"}]
    seeds = [{"queue_id": "Q1", "x": 0, "y": 0, "earthquake": "feb2011", "in_first_24": True, "site_code": "S"},
             {"queue_id": "Q3", "x": 0, "y": 0, "earthquake": "feb2011", "in_first_24": False}]
    q = blind.blind_queue(units, seeds)
    assert [i["queue_id"] for i in q] == ["Q1", "Q2"]
    assert all(set(i) == set(blind.ALLOWED) for i in q)


def test_blind_queue_refuses_duplicates_and_unknown_earthquake():
    u = {"queue_id": "Q1", "x": 0, "y": 0, "earthquake": "sep2010"}
    with pytest.raises(blind.BlindError):
        blind.blind_queue([u, dict(u)])
    with pytest.raises(blind.BlindError):
        blind.blind_item({**u, "earthquake": "kaikoura2016"})


def test_check_packet_catches_a_leak(tmp_path):
    (tmp_path / "items").mkdir()
    (tmp_path / "items" / "a.json").write_text(json.dumps({"queue_id": "Q1", "evidence": [{"score_band": 1}]}))
    (tmp_path / "items" / "b.json").write_text(json.dumps({"note": "see the CGD0300 map"}))
    probs = blind.check_packet(tmp_path)
    assert any("score_band" in p for p in probs) and any("CGD0300" in p for p in probs)


# ---------- licence check ----------

def _toml(tmp_path, body):
    p = tmp_path / "s.toml"
    p.write_text(body)
    return p


def test_unknown_licence_is_refused(tmp_path):
    p = _toml(tmp_path, '[[source]]\nid="x"\nkind="imagery_kml"\ndate="2011-02-24"\nlicence="all-rights"\n'
                        'attribution="a"\ntiles=[]\n')
    with pytest.raises(sources.LicenceError):
        sources.load_registry(p)
    reg = sources.load_registry(p, strict=False)
    assert reg.imagery == [] and reg.refused[0][0] == "x"


def test_nzgd_licence_needs_its_terms_file(tmp_path):
    body = ('[[source]]\nid="n"\nkind="imagery_kml"\ndate="2011-02-24"\nlicence="NZGD-ToU-2024"\n'
            'licence_file="terms.txt"\nattribution="a"\nnotice="Important notice text"\ntiles=[]\n')
    with pytest.raises(sources.LicenceError):
        sources.load_registry(_toml(tmp_path, body))
    (tmp_path / "terms.txt").write_text("terms")
    reg = sources.load_registry(_toml(tmp_path, body))
    assert reg.imagery[0].share == "local" and reg.imagery[0].notice == "Important notice text"


def test_nzgd_source_without_the_eqc_notice_is_refused(tmp_path):
    (tmp_path / "terms.txt").write_text("terms")
    body = ('[[source]]\nid="n"\nkind="imagery_kml"\ndate="2011-02-24"\nlicence="NZGD-ToU-2024"\n'
            'licence_file="terms.txt"\nattribution="a"\ntiles=[]\n')
    with pytest.raises(sources.LicenceError, match="notice"):
        sources.load_registry(_toml(tmp_path, body))


def test_pillar_sources_carry_notice_quality_and_dec_flag():
    import tomllib
    cfg = tomllib.loads((Path(sources.__file__).parent / "sources.pillar.toml").read_text())
    for s in cfg["source"]:
        assert s["licence"] != "NZGD-ToU-2024" or "Important notice" in s["notice"]
    lid = [s for s in cfg["source"] if s["kind"] == "lidar_dem"][0]
    q = {k: v["rating"] for k, v in lid["quality"].items()}
    assert q == {"sep2010": "unusable", "feb2011": "good", "jun2011": "caution", "dec2011": "unchecked"}
    # the Dec post grid is flown after the Dec window end; no other post grid is
    for eq, pair in lid["pairs"].items():
        late = [t for t in pair["post"] if lid["grids"][t]["flown_end"] >= blind.EARTHQUAKES[eq]["next"]]
        assert bool(late) == (eq == "dec2011")


def test_cc_by_is_open_and_needs_attribution(tmp_path):
    body = '[[source]]\nid="l"\nkind="imagery_kml"\ndate="2011-02-24"\nlicence="CC-BY-4.0"\ntiles=[]\n'
    with pytest.raises(sources.LicenceError):
        sources.load_registry(_toml(tmp_path, body))
    reg = sources.load_registry(_toml(tmp_path, body + 'attribution="LINZ, CC BY 4.0"\n'))
    assert reg.imagery[0].share == "open"


# ---------- GeoTIFF reader ----------

def test_geotiff_round_trip_and_window(tmp_path):
    arr = np.arange(20 * 30, dtype=np.float32).reshape(20, 30)
    p = tmp_path / "g.tif"
    geo.write_geotiff(p, arr, 1000.0, 5000.0, 5.0)
    g = geo.open_grid(p)
    assert (g.x0, g.y0, g.dx, g.width, g.height) == (1000.0, 5000.0, 5.0, 30, 20)
    w = g.window(1000.0 + 5 * 7, 5000.0 - 5 * 4, 5.0)
    assert w.shape == (3, 3) and w[1, 1] == arr[4, 7] and w[0, 0] == arr[3, 6]
    edge = g.window(1000.0, 5000.0, 5.0)
    assert np.isnan(edge[0, 0]) and edge[1, 1] == arr[0, 0]
    assert g.covers(1000.0, 5000.0) and not g.covers(990.0, 5000.0)


def test_area_tiepoint_is_shifted_to_cell_centre(tmp_path):
    from PIL import Image, TiffImagePlugin
    info = TiffImagePlugin.ImageFileDirectory_v2()
    info[33550] = (5.0, 5.0, 0.0)
    info[33922] = (0.0, 0.0, 0.0, 100.0, 200.0, 0.0)
    info.tagtype[34735] = 3
    info[34735] = (1, 1, 0, 1, 1025, 0, 1, 1)
    p = tmp_path / "a.tif"
    Image.fromarray(np.zeros((4, 4), np.float32), mode="F").save(p, tiffinfo=info)
    g = geo.open_grid(p)
    assert (g.x0, g.y0) == (102.5, 197.5)


# ---------- demo build: tiles, blind packet ----------

@pytest.fixture(scope="module")
def demo_packet(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    return demo.run_demo(out, log=lambda *_: None)


def test_demo_packet_is_blind_and_complete(demo_packet):
    m = json.loads((demo_packet / "manifest.json").read_text())
    assert len(m["queue"]) == 7  # 6 units + 1 flagged seed; the unflagged seed stays out
    assert m["counts"]["with_lidar"] == 7 and m["counts"]["with_imagery_after"] == 7
    # 'before' falls back from a newer source that misses the point to the older one that covers it
    assert m["counts"]["with_imagery_before"] == 7
    assert m["counts"]["lidar_quality"] == {"unchecked": 7}
    assert blind.check_packet(demo_packet) == []
    text = " ".join(p.read_text() for p in demo_packet.rglob("*.json"))
    for leak in ("DEMO_CPT", "demo_truth", "DEMO_SEED", "in_first_24", "weight"):
        assert leak not in text
    assert (demo_packet / "RULE.md").exists()


def test_demo_lidar_change_shows_the_planted_settlement(demo_packet):
    items = [json.loads(p.read_text()) for p in demo_packet.glob("items/*/item.json")]
    sites = {s["queue_id"]: s for s in blind.load_site_list(demo.DEMO_SITES)}
    for it in items:
        if it["queue_id"] not in sites:
            continue
        lid = [e for e in it["evidence"] if e["kind"] == "lidar_change"][0]
        if sites[it["queue_id"]]["demo_truth"]:
            assert lid["disc_minus_ring_m"] < -0.15
        else:
            assert abs(lid["disc_minus_ring_m"]) < 0.05


def test_imagery_tile_has_disc_size_and_resolution(demo_packet):
    from PIL import Image
    from scoreboard.evalkit.graderkit import tiles
    p = next(demo_packet.glob("items/*/imagery-after.png"))
    with Image.open(p) as im:
        assert im.size == (700, 700)
        rgb = im.convert("RGB")
        r = int(tiles.DISC_M / tiles.IMG_RES_M)  # 250 px for a 25 m disc at 0.1 m
        assert r == 250
        yellow = (255, 230, 0)
        assert any(rgb.getpixel((350 + r - k, 350)) == yellow for k in range(3))  # outline on the radius
        assert rgb.getpixel((350 + r - 20, 350)) != yellow and rgb.getpixel((350 + r + 20, 350)) != yellow


def test_lidar_item_carries_quality_and_window_flag(demo_packet):
    it = json.loads(next(demo_packet.glob("items/*/item.json")).read_text())
    lid = [e for e in it["evidence"] if e["kind"] == "lidar_change"][0]
    assert lid["quality"] == "unchecked" and lid["post_flown_after_window_end"] is False
    assert json.loads((demo_packet / "manifest.json").read_text())["notices"] == []  # synthetic needs none


def test_lidar_window_flag_and_quality_from_registry(tmp_path):
    out = tmp_path / "d"
    toml = demo.make_sources(out)
    text = toml.read_text().replace('flown = "synthetic post"}', 'flown = "synthetic post", flown_end = "2012-02-18"}')
    toml.write_text(text + '[source.quality]\ndec2011 = {rating = "unchecked", why = "w"}\n'
                           'feb2011 = {rating = "good", why = "g"}\n')
    reg = sources.load_registry(toml)
    s = blind.load_site_list(demo.DEMO_SITES)[0]
    from scoreboard.evalkit.graderkit import tiles
    dec = tiles.cut_lidar(reg.lidar[0], "dec2011", s["x"], s["y"], tmp_path / "a.png", window_end="2012-01-02")
    feb = tiles.cut_lidar(reg.lidar[0], "feb2011", s["x"], s["y"], tmp_path / "b.png", window_end="2012-03-01")
    assert dec["post_flown_after_window_end"] and dec["quality"] == "unchecked"
    assert not feb["post_flown_after_window_end"] and feb["quality"] == "good"
    bad = text + '[source.quality]\nsep2010 = {rating = "fine", why = "x"}\n'
    toml.write_text(bad)
    with pytest.raises(ValueError):
        sources.load_registry(toml)


def test_build_stops_when_an_attribution_leaks_an_inventory_name(tmp_path):
    from scoreboard.evalkit.graderkit import build
    toml = demo.make_sources(tmp_path / "d")
    reg = sources.load_registry(toml)
    for src in reg.imagery:
        src.attribution = "tiles cut along CGD0200 polygons"
    queue = blind.blind_queue(blind.load_site_list(demo.DEMO_SITES))[:1]
    with pytest.raises(blind.BlindError, match="CGD0200"):
        build.build_packets(queue, reg, tmp_path / "packet", log=lambda *_: None)


# ---------- form server ----------

def _grade(**kw):
    g = {"grader": "grader_a", "queue_id": "Q003", "ejecta": "YES", "probability": 0.75, "severity": "minor",
         "confidence": 2, "recognise": "no", "evidence_used": ["imagery-after.png"], "seconds": 300}
    g.update(kw)
    return g


def test_validate_rules():
    q = {"Q003"}
    assert serve.validate(_grade(), q)["probability"] == 0.75
    for bad in (dict(probability=0.6), dict(ejecta="MAYBE"), dict(severity="huge"), dict(confidence=5),
                dict(recognise=None), dict(queue_id="Q999"), dict(grader="a b"), dict(probability=None)):
        with pytest.raises(serve.GradeError):
            serve.validate(_grade(**bad), q)
    cg = serve.validate(_grade(cannot_grade=True, reason="R2", ejecta=None), q)
    assert cg["cannot_grade"] and "ejecta" not in cg
    with pytest.raises(serve.GradeError):
        serve.validate(_grade(cannot_grade=True, reason="R6", reason_text=""), q)


def test_server_round_trip_lock_and_independence(demo_packet, tmp_path):
    grades = tmp_path / "grades"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(demo_packet, grades))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def post(path, obj, token="", ctype="application/json", origin=None):
        h = {"Content-Type": ctype, "X-Grader-Token": token}
        if origin:
            h["Origin"] = origin
        req = urllib.request.Request(base + path, json.dumps(obj).encode(), h)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def get(path, token=""):
        req = urllib.request.Request(base + path, headers={"X-Grader-Token": token})
        with urllib.request.urlopen(req) as r:
            return r.read()

    def raw_get_status(path):
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1])
        c.request("GET", path)  # sent as written, no path normalising
        s = c.getresponse().status
        c.close()
        return s

    try:
        assert b"Blind audit grading" in get("/")
        assert json.loads(get("/packet/manifest.json"))["queue"]
        # a real file outside the packet (the demo sources sit next to it) is not served
        assert (demo_packet.parent / "sources" / "sources.toml").exists()
        assert raw_get_status("/packet/../sources/sources.toml") == 404
        assert raw_get_status("/packet/manifest.json") == 200
        ta = post("/api/start", {"grader": "grader_a"})[1]["token"]
        tb = post("/api/start", {"grader": "grader_b"})[1]["token"]
        assert ta != tb
        assert post("/api/start", {"grader": "grader_a"})[0] == 403  # name taken: b cannot become a
        assert post("/api/start", {"grader": "grader_a"}, token=ta)[1]["token"] == ta  # a resumes
        assert post("/api/grade", _grade())[0] == 403  # no token
        assert post("/api/grade", _grade(), token=tb)[0] == 403  # b's token cannot write a's grade
        assert post("/api/grade", _grade(), token=ta)[0] == 200
        assert post("/api/grade", _grade(probability=0.95), token=ta)[0] == 200
        assert post("/api/grade", _grade(grader="grader_b", ejecta="NO", probability=0.05), token=tb)[0] == 200
        with pytest.raises(urllib.error.HTTPError) as e:
            get("/api/grades?grader=grader_a", token=tb)  # b cannot read a's grades
        assert e.value.code == 403
        mine = json.loads(get("/api/grades?grader=grader_a", token=ta))
        assert mine["grades"]["Q003"]["probability"] == 0.95  # latest wins
        assert len((grades / "grader_a" / "Q003.jsonl").read_text().splitlines()) == 2  # history kept
        assert all(g["grader"] == "grader_a" for g in mine["grades"].values())
        assert post("/api/grade", _grade(probability=0.3), token=ta)[0] == 400
        # cross-site writes: wrong content type, foreign origin
        assert post("/api/grade", _grade(), token=ta, ctype="text/plain")[0] == 415
        assert post("/api/grade", _grade(), token=ta, origin="http://evil.example")[0] == 403
        assert post("/api/grade", _grade(), token=ta, origin=base)[0] == 200
        assert post("/api/lock?grader=grader_a", {}, token=tb)[0] == 403  # b cannot lock a
        assert not (grades / "grader_a" / "LOCKED").exists()
        assert post("/api/lock?grader=grader_a", {}, token=ta)[0] == 200
        assert post("/api/grade", _grade(), token=ta)[0] == 400
        assert power.minutes_from_grades(grades) == pytest.approx(5.0)
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------- power and hours ----------

def test_registered_hours_match_the_design_table():
    h = power.Effort().hours()
    assert h["total"] == pytest.approx(170.7, abs=0.05)
    assert power.Effort(minutes=20, disagree=0.35).hours()["total"] > 220  # the design's upper end is too low


def test_hanley_mcneil_reproduces_the_design_half_widths():
    # audit design section 2: Sep 2010 set A 0.16, Feb 2011 set A 0.25, Dec 2011 0.16
    assert power.precision("sep2010", 75)["A"]["auc_hw"] == pytest.approx(0.16, abs=0.006)
    assert power.precision("feb2011", 75)["A"]["auc_hw"] == pytest.approx(0.25, abs=0.006)
    assert power.precision("dec2011", 75)["A"]["auc_hw"] == pytest.approx(0.16, abs=0.006)


def test_proposal_keeps_ranking_earthquakes_and_respects_floors():
    p = power.propose()
    assert p["sep2010"]["keep"] == p["jun2011"]["keep"] == p["dec2011"]["keep"] == 75
    assert 24 <= p["feb2011"]["keep"] < 75
    pr = power.precision("feb2011", p["feb2011"]["keep"])
    assert pr["A"]["prev_hw"] <= power.PREV_HW and pr["B"]["prev_hw"] <= power.PREV_HW


def test_subsample_keeps_targets_and_rescales_inclusion():
    rng = np.random.default_rng(1)
    units = []
    k = 0
    for eq in ("e1", "e2"):
        for band in range(3):
            for dens in ("low", "high"):
                N, n = int(rng.integers(50, 500)), 12
                for _ in range(n):
                    units.append({"queue_id": f"Q{k:04d}", "earthquake": eq, "placement": "after", "density": dens,
                                  "score_band": band, "inclusion_probability": n / N, "weight": N / n, "n_h": n})
                    k += 1
    sub = power.subsample(units, {"e1": 40}, seed=7)
    e1 = [u for u in sub if u["earthquake"] == "e1"]
    assert len(e1) == 40 and len([u for u in sub if u["earthquake"] == "e2"]) == 72
    # the weights of each stratum still sum to its frame size N_h
    orig = {}
    for u in units:
        key = (u["earthquake"], u["density"], u["score_band"])
        orig[key] = u["weight"] * u["n_h"]
    tot = {}
    for u in sub:
        key = (u["earthquake"], u["density"], u["score_band"])
        tot[key] = tot.get(key, 0) + u["weight"]
    for key, N in orig.items():
        assert tot[key] == pytest.approx(N)
    assert power.subsample(units, {"e1": 40}, seed=7) == sub  # seeded


def test_note_prints_hours_and_options():
    s = power.note(10.0)
    assert "170.7 h" in s and "measured median of 10.0 minutes" in s and "P1" in s
    assert "not measured" in s and "210 of 300 units" in s


def test_option_hours_are_pinned():
    # 24 seeds: the design's own table and the first-draft option figures
    assert power.Effort(double_fraction=0.3).hours()["total"] == pytest.approx(112.95, abs=0.01)
    assert power.Effort(seeds=36).hours()["total"] == pytest.approx(176.4, abs=0.01)
    # adjudication covers double-graded items only
    assert power.Effort(double_fraction=0.3).hours()["adjudication"] == pytest.approx(114 * 0.3 * 0.25)
    kept = sum(v["keep"] for v in power.propose().values())
    o = {k: round(e.hours()["total"], 2) for k, e in power.options(kept).items()}
    assert o["registered"] == pytest.approx(176.4) and o["P1"] == pytest.approx(118.65)
    assert power.p1_label_cost(0.30) == pytest.approx(0.15)
