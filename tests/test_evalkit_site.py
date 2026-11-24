"""Item D9 (eval M6): the static scorecard site. Builds the site from the sealed v2 card and checks the launch-check
items that a test can hold: the Markdown twin carries the same numbers as each page, every cell of the sealed
SCORECARD.md tables sits on the site under the same row, earthquake and set or population (and nothing else does),
no page runs a script, internal links resolve, icons equal the brand files,
the build is deterministic and refuses a card whose files differ from its seal. No em dashes.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
from pathlib import Path

import numpy as np

import pytest

from scoreboard.evalkit import synth
from scoreboard.evalkit.card import SITE_ID_SHAPE, build_card
from scoreboard.evalkit.site import build as site
from scoreboard.evalkit.spec import Entry, EntryManifest, Task, site_list_sha256

HERE = Path(__file__).resolve()
CARD = next((p / "docs" / "scorecard-v2-2026-11-19" for p in HERE.parents
             if (p / "docs" / "scorecard-v2-2026-11-19" / "scorecard.json").is_file()), None)
needs_card = pytest.mark.skipif(CARD is None, reason="the sealed v2 card folder (pillar docs) is not above this repo")

# md5 of repos/geoliquefy-design/brand/assets/<name> (checked 2026-11-23); the site copies them byte for byte.
BRAND_MD5 = {"favicon.svg": "f60f6f83b1d627fbd0646c79b93fa720"}
INTERVAL = re.compile(r"[+-]?\d\.\d{3} \[[+-]?\d\.\d{3}, [+-]?\d\.\d{3}\]")
DECIMAL = re.compile(r"[+-]?\d+\.\d+")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if CARD is None:
        pytest.skip("no sealed card")
    out = tmp_path_factory.mktemp("site") / "site"
    man = site.build(CARD, out)
    return out, man


def _pages(out: Path):
    return sorted(p for p in out.rglob("index.html") if "card" not in p.relative_to(out).parts)


def _visible(h: str) -> str:
    body = h.split("<body>", 1)[1]
    body = re.sub(r"<svg.*?</svg>", " ", body, flags=re.S)
    body = re.sub(r'<nav class="crumbs".*?</nav>', " ", body, flags=re.S)
    return html.unescape(re.sub(r"<[^>]+>", " ", body))


@needs_card
def test_every_page_has_a_twin_with_the_same_numbers(built):
    out, _ = built
    pages = _pages(out)
    assert len(pages) >= 30
    for p in pages:
        md = (p.parent / "index.md").read_text(encoding="utf-8")
        md_body = md.rsplit("\n---\n", 1)[0]
        h = p.read_text(encoding="utf-8")
        assert '<link rel="alternate" type="text/markdown" href="index.md">' in h
        assert sorted(DECIMAL.findall(_visible(h))) == sorted(DECIMAL.findall(md_body)), p


@needs_card
def test_every_interval_on_the_site_is_in_the_sealed_card_text(built):
    out, _ = built
    sealed = (CARD / "SCORECARD.md").read_text(encoding="utf-8")
    seen = 0
    for p in _pages(out):
        for iv in INTERVAL.findall((p.parent / "index.md").read_text(encoding="utf-8")):
            assert iv in sealed, (p, iv)
            seen += 1
    assert seen > 300


@needs_card
def test_static_facts_in_html_and_no_script_or_inline_style(built):
    out, _ = built
    for p in _pages(out) + [out / "404.html"]:
        h = p.read_text(encoding="utf-8")
        assert re.search(r"<title>[^<]+</title>", h) and '<meta name="description"' in h and "<h1>" in h
        scripts = re.findall(r"<script([^>]*)>", h)
        assert all('type="application/ld+json"' in s for s in scripts), p
        assert "<style" not in h and " style=" not in h and " on" + "click" not in h
        if p.name == "index.html":
            assert h.count('rel="canonical"') == 1 and 'property="og:image"' in h
    assert "script-src 'none'" in (out / "_headers").read_text(encoding="utf-8")
    assert "frame-ancestors 'none'" in (out / "_headers").read_text(encoding="utf-8")
    assert 'name="robots" content="noindex"' in (out / "404.html").read_text(encoding="utf-8")


@needs_card
def test_internal_links_resolve(built):
    out, _ = built
    for p in list(out.rglob("*.html")):
        for href in re.findall(r'(?:href|src)="(/[^"#?]*)', p.read_text(encoding="utf-8")):
            t = out / href.lstrip("/")
            assert (t / "index.html").is_file() if href.endswith("/") else t.is_file(), (p, href)
    for p in out.rglob("index.md"):
        for href in re.findall(r"\]\((/[^)\s]*)\)", p.read_text(encoding="utf-8")):
            assert (out / href.lstrip("/")).is_file(), (p, href)


@needs_card
def test_no_dash_no_site_id_and_agent_files(built):
    out, man = built
    for p in out.rglob("*"):
        if p.is_file() and p.suffix in (".html", ".md", ".txt", ".xml", ".webmanifest", ".json") or p.name == "_headers":
            t = p.read_text(encoding="utf-8")
            assert not re.search("[\u2013\u2014]", t), p
            assert not SITE_ID_SHAPE.search(t), p
    robots = (out / "robots.txt").read_text(encoding="utf-8")
    for agent in ("GPTBot", "ClaudeBot", "Claude-User", "PerplexityBot", "Google-Extended", "CCBot"):
        assert f"User-agent: {agent}" in robots
    llms = (out / "llms.txt").read_text(encoding="utf-8")
    assert llms.startswith("# ") and "\n> " in llms and "agents@geoliquefy.com" in llms
    sm = (out / "sitemap.xml").read_text(encoding="utf-8")
    assert sm.count("<loc>") == len(_pages(out)) and "<lastmod>2026-11-19</lastmod>" in sm
    assert man["card_sha256"] == site.card_notes.V2_CARD_SHA256


@needs_card
def test_icons_equal_brand_files_and_card_files_equal_seal(built):
    out, man = built
    for n, md5 in BRAND_MD5.items():
        assert hashlib.md5((out / n).read_bytes()).hexdigest() == md5
    for n in ("favicon.ico", "apple-touch-icon.png", "icon-512-maskable.png", "logo-mark.svg"):
        assert (out / n).read_bytes() == (site.STATIC / n).read_bytes()
    for n in site.CARD_FILES:
        assert (out / "card" / "v2" / n).read_bytes() == (CARD / n).read_bytes()
    assert man["card_files_sha256"]["scorecard.json"] in (CARD / "SEAL.txt").read_text(encoding="utf-8")


@needs_card
def test_build_is_deterministic_and_refuses_foreign_folders(built, tmp_path):
    out, man = built
    again = site.build(CARD, tmp_path / "again")
    assert again["files"] == man["files"]
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("not a site build")
    with pytest.raises(site.SiteRefused):
        site.build(CARD, foreign)
    assert (foreign / "keep.txt").exists()


@needs_card
def test_refuses_a_card_that_differs_from_its_seal(tmp_path):
    d = tmp_path / "card"
    d.mkdir()
    for n in site.CARD_FILES:
        shutil.copyfile(CARD / n, d / n)
    (d / "SCORECARD.md").write_text((CARD / "SCORECARD.md").read_text(encoding="utf-8") + "\nedited\n")
    with pytest.raises(site.SiteRefused, match="SEAL"):
        site.build(d, tmp_path / "out")


def test_lint_refuses_dashes_and_charlatan_phrases():
    with pytest.raises(site.SiteRefused):
        site.lint("a \u2014 b", "x")
    with pytest.raises(site.SiteRefused):
        site.lint("to be honest, it held", "x")
    assert site.lint("the row passes", "x") == ["x: passes"]


def test_forest_is_deterministic_and_marks_differ_by_shape():
    rows = [("Level 0 (HOUSE)", 0.6, 0.5, 0.7), ("USGS Zhu 2017 (ORGANISER-RUN)", 0.55, None, None),
            ("coin", 0.5, 0.5, 0.5)]
    kw = dict(lo=0.0, hi=1.0, ref=0.5, ref_label="coin 0.5", ticks=(0, 0.5, 1), panel_note="n 3", title="t")
    a, b = site.forest(rows, **kw), site.forest(rows, **kw)
    assert a == b and "<circle" in a and "<rect" in a and "<path" in a and "\u2014" not in a


# ---------------------------------------------------------------------------------------------------------------
# Cell fidelity (review 1, should-fix 1): every sealed table cell, by row, earthquake and set or population
# ---------------------------------------------------------------------------------------------------------------
# Test-local names, written out here so that a label swap in build.py cannot also change what the test expects.
EQ_OF_PATH = {"sep2010": "Sep 2010", "feb2011": "Feb 2011", "jun2011": "Jun 2011", "dec2011": "Dec 2011"}
SET_OF_NAME = {"set B (primary)": "B", "set A (beside)": "A"}
WORLD_EXPOSURE = {"B&I chain (Level 0, M0, LSN, LPI)": "B&I exposure", "USGS Zhu 2017": "Zhu 2017 exposure",
                  "LSN global fragility": "LSN global fragility exposure"}
EQ = r"(\w{3} \d{4})"
SETN = r"(set [AB] \((?:primary|beside)\))"


def _md_tables(text: str):
    """[(h1, h2, h3, caption, header, rows)] for every pipe table in a Markdown text."""
    out, h = [], {"#": "", "##": "", "###": ""}
    cap, block = "", []
    for ln in text.split("\n") + [""]:
        if ln.startswith("|"):
            block.append([c.strip().replace("\\|", "|") for c in ln.strip()[1:-1].split(" | ")])
            continue
        if block:
            out.append((h["#"], h["##"], h["###"], cap, block[0], [r for r in block[2:]]))
            block, cap = [], ""
        m = re.match(r"^(#{1,3}) (.*)$", ln)
        if m:
            h[m.group(1)] = m.group(2)
            if m.group(1) != "###":
                h["###"] = ""
        elif ln.startswith("Table: "):
            cap = ln[len("Table: "):]
    return out


def _sealed_cells(text: str) -> dict:
    cells = {}
    for _, h2, h3, _, head, rows in _md_tables(text):
        mc = re.match(r"Canterbury, set ([AB])", h2)
        mw = re.match(r"World case histories, (P\d)", h2)
        if mc:
            s = mc.group(1)
            if h3.startswith("UNSEEN-only"):
                for r in rows:
                    cells[("unseen", s, r[1], r[0])] = tuple(r[2:])
                continue
            kind = ("all" if h3 == "All labelled sites" else "pre" if h3.startswith("Beside: only CPTs") else
                    "exposure" if h3.startswith("Exposure per cell") else "logloss" if h3.startswith("Probability")
                    else None)
            assert kind, h3
            for r in rows:
                for eq, v in zip(head[1:], r[1:]):
                    cells[(kind, s, eq, r[0])] = v
        elif h2.startswith("Pooled lines"):
            for r in rows:
                cells[("pooled", r[0], r[3])] = tuple(r[1:])
        elif mw and len(head) == 11:
            for r in rows:
                eq = r[0].replace(" (see Kobe note)", "")
                for col, v in zip(head[1:], r[1:]):
                    cells[("world", mw.group(1), eq, col)] = v
        elif mw and len(head) == 2:
            for r in rows:
                cells[("under", mw.group(1), r[0])] = r[1]
        elif h2.startswith("Diagnostic only"):
            for r in rows:
                for eq, v in zip(head[1:], r[1:]):
                    cells[("diag", r[0], eq)] = v
    for m in re.finditer(r"^Menon AUC ceilings for set ([AB]) \(step 1a\): (.*)\.$", text, re.M):
        for part in m.group(2).split("; "):
            pm = re.match(EQ + r" (.*)$", part)
            cells[("menon", m.group(1), pm.group(1))] = pm.group(2)
    return cells


def _site_cells(out: Path) -> dict:
    cells = {}

    def put(key, v):
        assert key not in cells, ("a cell printed twice", key)
        cells[key] = v

    for p in sorted(out.rglob("index.md")):
        if "card" in p.relative_to(out).parts:
            continue
        text = p.read_text(encoding="utf-8").rsplit("\n---\n", 1)[0]
        h1 = text.split("\n", 1)[0]
        for _, _, _, cap, head, rows in _md_tables(text):
            m = re.match(rf"AUC, (all labelled sites|pre-event CPTs only), {EQ}, {SETN}; n \(yes\), 4 km tiles "
                         r"(.+); status (.+)$", cap)
            if m:
                kind, eq, s = ("all" if m.group(1).startswith("all") else "pre"), m.group(2), SET_OF_NAME[m.group(3)]
                put((kind, s, eq, "n (yes), 4 km tiles"), m.group(4))
                put((kind, s, eq, "status"), m.group(5))
                for r in rows:
                    put((kind, s, eq, r[0]), r[1] + (f"; UNSEEN-only {r[2]}" if kind == "all" and r[2] else ""))
                continue
            m = re.match(rf"(Log loss \[95%\]|Exposure counts on labelled sites|UNSEEN-only AUC \[95%\]), {EQ}, "
                         r"(set [AB])", cap)
            if m:
                kind = {"L": "logloss", "E": "exposure", "U": "unseen"}[m.group(1)[0]]
                s = m.group(3)[-1]
                for r in rows:
                    put((kind, s, m.group(2), r[0]), tuple(r[1:]) if kind == "unseen" else r[1])
                continue
            m = re.match(r"AUC \[95%\], (.+), world (P\d); n \(yes\), 1 km tiles (.+)$", cap)
            if m:
                put(("world", m.group(2), m.group(1), "n (yes), tiles"), m.group(3))
                for r in rows:
                    put(("world", m.group(2), m.group(1), r[0]), r[1])
                continue
            m = re.match(r"Exposure counts, (.+), world (P\d)$", cap)
            if m:
                for r in rows:
                    put(("world", m.group(2), m.group(1), WORLD_EXPOSURE[r[0]]), r[1])
                continue
            m = re.match(r"World (P\d): earthquakes under the record gate$", cap)
            if m:
                for r in rows:
                    put(("under", m.group(1), r[0]), r[1])
                continue
            if cap == "Pooled lines (all DESCRIPTIVE)":
                for r in rows:
                    put(("pooled", r[0], r[3]), tuple(r[1:]))
        mw = re.match(r"^# (.+), world (P\d) \(", h1)
        for ln in text.split("\n"):
            m = re.match(r"^Level 0 minus shaking only, paired \[95%\]: (.+)\.$", ln)
            if m and mw:
                put(("world", mw.group(2), mw.group(1), "Level 0 minus shaking [95%]"), m.group(1))
            m = re.match(rf"^Investigation-density proxy AUC, {SETN}, {EQ}: (.+)\. It is never", ln)
            if m:
                put(("diag", SET_OF_NAME[m.group(1)][-1], m.group(2)), m.group(3))
            m = re.match(rf"^Menon AUC ceiling, {EQ}, {SETN} \(step 1a\): (.+)\.$", ln)
            if m:
                put(("menon", SET_OF_NAME[m.group(2)], m.group(1)), m.group(3))
    return cells


def _diff(sealed: dict, built: dict) -> list:
    keys = sorted(set(sealed) | set(built), key=str)
    return [(k, sealed.get(k, "MISSING ON THE CARD"), built.get(k, "MISSING ON THE SITE")) for k in keys
            if sealed.get(k) != built.get(k)]


@needs_card
def test_every_sealed_table_cell_is_on_the_site_under_the_same_row_earthquake_and_set(built):
    """Review 1 should-fix 1. Each cell of the sealed SCORECARD.md tables (Canterbury sets B and A: all sites,
    pre-event, exposure, UNSEEN-only, log loss; world P1 and P2 with the paired line, exposure and the Zhu empty-cell
    notes; under-gate counts; pooled lines; diagnostic proxy; Menon ceilings) equals the cell the site prints for the
    same row, earthquake and set or population, read from the labels the page prints. The site prints no cell that
    the card lacks."""
    out, _ = built
    sealed = _sealed_cells((CARD / "SCORECARD.md").read_text(encoding="utf-8"))
    on_site = _site_cells(out)
    assert len(sealed) > 500
    bad = _diff(sealed, on_site)
    assert not bad, f"{len(bad)} cells differ, first: {bad[:5]}"


@needs_card
def test_page_paths_and_index_links_agree_with_the_labels_they_print(built):
    """A path (from the card key) and the label on the page or the index link must name the same earthquake and set."""
    out, _ = built
    for p in sorted((out / "canterbury").rglob("index.md")):
        s, e = p.parent.parent.name[-1].upper(), p.parent.name
        h1 = p.read_text(encoding="utf-8").split("\n", 1)[0]
        assert h1.startswith(f"# Canterbury {EQ_OF_PATH[e]}, set {s} ("), (p, h1)
    sealed = _sealed_cells((CARD / "SCORECARD.md").read_text(encoding="utf-8"))
    front = (out / "index.md").read_text(encoding="utf-8")
    n = 0
    for _, _, _, cap, _, rows in _md_tables(front):
        for r in rows:
            m = re.match(rf"\[{EQ}, {SETN}\]\(/canterbury/set-([ab])/(\w+)/index\.md\)$", r[0])
            if cap == "Canterbury pages":
                s = SET_OF_NAME[m.group(2)]
                assert (m.group(3).upper(), EQ_OF_PATH[m.group(4)]) == (s, m.group(1)), r
                assert [r[1], r[2]] == [sealed[("all", s, m.group(1), k)] for k in ("n (yes), 4 km tiles", "status")]
                assert r[3] == sealed[("all", s, m.group(1), "Level 0 (HOUSE)")].split("; ")[0]
                n += 1
            elif cap == "World pages":
                m = re.match(r"\[(.+), (P\d) \([a-z, ]+\)\]\(/world/(p\d)/[a-z0-9-]+/index\.md\)$", r[0])
                assert m.group(3) == m.group(2).lower(), r
                assert [r[1], r[2]] == [sealed[("world", m.group(2), m.group(1), k)] for k in ("n (yes), tiles",
                                                                                                  "Level 0")]
                n += 1
    assert n == 8 + 15


# ---------------------------------------------------------------------------------------------------------------
# B1 LOSS block (review 1, should-fix 2): the site reads the kit's schema
# ---------------------------------------------------------------------------------------------------------------

def _loss_card(**task_kw):
    ev = synth.synthetic_event(900, 21, unlabelled_share=0.1)
    d = dict(event_id="eq-test", target="ejecta seen at the site", label_source="synthetic v1",
             label_rule_sha256="1" * 64, site_list_sha256=site_list_sha256(ev["site_ids"]), track="record",
             division="closed", tile_m=2000.0, clip=1e-3, seed=3, label_caveats="synthetic labels", n_boot=60)
    d.update(task_kw)
    bad = np.clip(1.0 - ev["p_target"], 0.02, 0.98)
    m = EntryManifest(model="house bad", output_kind="target_probability", statement="synthetic",
                      exposure={"eq-test": "unseen"}, house=True)
    return build_card(Task(**d), [Entry(m, ev["site_ids"], bad)], synth.labels_of(ev), losses_first=True)


def test_loss_items_print_the_lost_to_texts_and_the_not_scored_reason():
    card = _loss_card()
    block = card["losses"]
    texts = [i["text"] for ln in block["lines"] for i in ln["lost_to"]]
    assert texts and site.loss_items(block) == texts
    under = _loss_card(min_sites=10_000)["losses"]
    assert under["status"] == "NOT SCORED"
    assert site.loss_items(under) == [f"NOT SCORED: {under['why']}."]
    assert site.loss_items({**block, "lines": [], "no_proper_score": ["h2"]}) == [
        "No LOSS line on this card.", "House rows with no proper score (no LOSS check): h2."]
    for t in site.loss_items(block):
        site.lint(t, "loss")


@needs_card
def test_front_page_leads_with_a_loss_block_when_the_card_has_one():
    J = json.loads((CARD / "scorecard.json").read_text(encoding="utf-8"))
    block = _loss_card()["losses"]
    pg = site.front_page({"losses": block, **J}, None, [], [])
    assert pg.blocks[0][0] == "p" and pg.blocks[1][:2] == ("h2", "LOSS block")
    lists = [b[1] for b in pg.blocks if b[0] == "ul"]
    assert lists[0] == site.loss_items(block) and all(lists[0])
    plain = site.front_page(J, None, [], [])
    assert plain.blocks[0][0] == "p" and plain.blocks[1][1].startswith("Read this first")


@needs_card
def test_published_manifest_holds_no_lint_output_and_the_caveat_keeps_its_emphasis(built):
    out, man = built
    pub = json.loads((out / "build-manifest.json").read_text(encoding="utf-8"))
    assert set(pub) == {"card_sha256", "card_files_sha256", "files"} and "verdict_grep_hits" in man
    note = "<strong>not certified; lab coverage 0.80 to 0.96 depending on an unidentified range</strong>"
    assert note in (out / "index.html").read_text(encoding="utf-8")
    assert "**not certified; lab coverage" in (out / "index.md").read_text(encoding="utf-8")
    for ln in (out / "llms.txt").read_text(encoding="utf-8").split("\n"):
        if "/canterbury/" in ln or "/world/" in ln:
            assert re.search(r"index\.md\): \S", ln), ln
    rb = (out / "rebuild" / "index.md").read_text(encoding="utf-8")
    assert "not public yet" in rb and "sha256sum scorecard.json SCORECARD.md" in rb
