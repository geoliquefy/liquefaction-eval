"""Build the public scorecard site (scorecard.geoliquefy.com, ruling 43a) from a sealed card folder.

G1+G5+G6 (eval M6), item D9. Presentation only: every number is read from the card JSON and printed with the
sealed renderer's formats; nothing is scored here. Every page has a Markdown twin built from the same blocks.
The site needs no JavaScript. Aggregates only (NZGD clause 6.2): the card passes card.scan_aggregates before any
page is written. No em dashes.

Usage: python -m scoreboard.evalkit.site --card-dir <docs/scorecard-v2-2026-11-19> --out <dir>
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .. import card as card_mod
from .. import spec
from . import card_notes

SITE = "https://scorecard.geoliquefy.com"
STATIC = Path(__file__).resolve().parent / "static"
FONTS = ("https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;500;700&family=Inter:wght@400;600"
         "&display=swap")
CSP = ("default-src 'self'; script-src 'none'; style-src 'self' https://fonts.googleapis.com; "
       "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; manifest-src 'self'; "
       "object-src 'none'; base-uri 'self'; form-action 'none'; frame-ancestors 'none'; upgrade-insecure-requests")
CARD_FILES = ("scorecard.json", "SCORECARD.md", "SEAL.txt")
STATIC_FILES = ("favicon.ico", "favicon.svg", "apple-touch-icon.png", "icon-512-maskable.png", "logo-mark.svg",
                "styles.css", "og.png")

EVN = {"sep2010": "Sep 2010", "feb2011": "Feb 2011", "jun2011": "Jun 2011", "dec2011": "Dec 2011"}
ROWS = ["Level 0", "M0 (Level 0 + earthquake band)", "LSN + global fragility [authors-maurer-2026-11-19]",
        "LSN + CES fragility [authors-maurer-2026-11-19]", "LSN + global fragility [default-2026-11]",
        "LSN + CES fragility [default-2026-11]", "USGS Zhu 2017", "GEM TS22", "baseline: coin",
        "baseline: earlier-earthquake rate", "baseline: shaking only", "baseline: effort only"]
SHORT = {"Level 0": "Level 0 (HOUSE)", "M0 (Level 0 + earthquake band)": "M0 (HOUSE)",
         "LSN + global fragility [authors-maurer-2026-11-19]": "LSN + global frag., authors id (HOUSE)",
         "LSN + CES fragility [authors-maurer-2026-11-19]": "LSN + CES frag., authors id (HOUSE)",
         "LSN + global fragility [default-2026-11]": "LSN + global frag., default id (beside)",
         "LSN + CES fragility [default-2026-11]": "LSN + CES frag., default id (beside)",
         "USGS Zhu 2017": "USGS Zhu 2017 (ORGANISER-RUN)",
         "GEM TS22": "GEM TS22 (ORGANISER-RUN, STAND-IN INPUTS, NOT MONOTONE)",
         "baseline: coin": "coin", "baseline: earlier-earthquake rate": "earlier-earthquake rate",
         "baseline: shaking only": "shaking only", "baseline: effort only": "pre-event effort"}
WORLD_ROWS = [("Level 0", "Level 0"), ("BI14-LSN (both LSN fragility rows)", "BI14-LSN"), ("BI14-LPI", "BI14-LPI"),
              ("USGS Zhu 2017", "Zhu 2017"), ("baseline: shaking only", "shaking only")]
FIG_LABEL = {"GEM TS22": "GEM TS22 (ORGANISER-RUN, STAND-IN, NOT MONOTONE)"}   # fits the figure width
SET_NAME = {"B": "set B (primary)", "A": "set A (beside)"}
POP_NAME = {"P1": "P1 (primary)", "P2": "P2 (beside, union)"}
AUC_GLOSS = ("AUC is the chance that a site that liquefied gets a higher score than a site that did not. A coin "
             "scores 0.5; a perfect ranking scores 1.")

# memo-comms section 5 greps (docs/COMMS-FRAMEWORK-2026-08-24.md), run over every page as a build lint.
CHARLATAN = re.compile(r"we refuse|we owe|honest (number|ordering|answer)|would not defend|rather than omit|"
                       r"audit.*(forced|surfaced)|initially wrote|asserted, not measured|we never report|"
                       r"to be (candid|frank|honest)")
VERDICT = re.compile(r"closer|better|worse|beats|outperform|badly|about right|matches|agrees with|reproduce|"
                     r"impossible|passes|fails")
DASHES = re.compile("[\u2013\u2014]")       # en and em dash, written as escapes


class SiteRefused(Exception):
    """The card folder or the built pages break a site rule."""


# ---------------------------------------------------------------------------------------------------------------
# Number formats (the sealed renderer's formats)
# ---------------------------------------------------------------------------------------------------------------

def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def iv(d) -> str:
    if not isinstance(d, dict) or d.get("value") is None:
        return "n/a"
    return f"{d['value']:.3f} [{f3(d.get('lo'))}, {f3(d.get('hi'))}]"


def wiv(d, tiles: int) -> str:
    s = iv(d)
    return s + " (uninformative: under 10 tiles)" if tiles < 10 and s != "n/a" else s


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


# ---------------------------------------------------------------------------------------------------------------
# Page model: one list of blocks gives both the HTML page and its Markdown twin
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class Page:
    path: str                     # "" for the root, else "a/b/" (always ends with a slash)
    title: str
    description: str
    h1: str
    blocks: list = field(default_factory=list)
    crumbs: list = field(default_factory=list)   # [(path, label)]

    def h2(self, text, anchor=None):
        self.blocks.append(("h2", text, anchor or slug(text)[:60]))

    def p(self, text):
        self.blocks.append(("p", text))

    def ul(self, items):
        self.blocks.append(("ul", list(items)))

    def table(self, caption, head, rows):
        self.blocks.append(("table", caption, list(head), [list(r) for r in rows]))

    def fig(self, svg, caption):
        self.blocks.append(("fig", svg, caption))

    def pre(self, text):
        self.blocks.append(("pre", text))


LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
CODE = re.compile(r"`([^`]+)`")
STRONG = re.compile(r"\*\*(.+?)\*\*")


def inline_html(text: str) -> str:
    """Escape, then turn `code`, **strong** and [text](href) into HTML. Hrefs are site paths or https URLs."""
    out, last = [], 0
    for m in LINK.finditer(text):
        out.append(_code_html(text[last:m.start()]))
        out.append(f'<a href="{html.escape(m.group(2), quote=True)}">{_code_html(m.group(1))}</a>')
        last = m.end()
    out.append(_code_html(text[last:]))
    return "".join(out)


def _code_html(t: str) -> str:
    parts = CODE.split(t)
    return "".join(f"<code>{html.escape(x)}</code>" if i % 2 else _strong_html(x) for i, x in enumerate(parts))


def _strong_html(t: str) -> str:
    parts = STRONG.split(t)
    return "".join(f"<strong>{html.escape(x)}</strong>" if i % 2 else html.escape(x) for i, x in enumerate(parts))


def md_link(text: str) -> str:
    """Markdown twin links: site paths point at the twin (index.md); other links stay as written."""
    def fix(m):
        href = m.group(2)
        if href.startswith("/") and href.endswith("/"):
            href = href + "index.md"
        return f"[{m.group(1)}]({href})"
    return LINK.sub(fix, text)


def md_cell(text: str) -> str:
    return md_link(text).replace("|", "\\|")


def render_md(pg: Page, card_hash: str) -> str:
    L = [f"# {pg.h1}", "", f"> {pg.description}", ""]
    for b in pg.blocks:
        k = b[0]
        if k == "h2":
            L += [f"## {b[1]}", ""]
        elif k == "p":
            L += [md_link(b[1]), ""]
        elif k == "ul":
            L += [f"- {md_link(x)}" for x in b[1]] + [""]
        elif k == "table":
            L += [f"Table: {b[1]}", "", "| " + " | ".join(md_cell(h) for h in b[2]) + " |",
                  "|" + "---|" * len(b[2])]
            L += ["| " + " | ".join(md_cell(str(c)) for c in r) + " |" for r in b[3]] + [""]
        elif k == "fig":
            L += [f"Figure: {md_link(b[2])}", ""]
        elif k == "pre":
            L += ["```", b[1], "```", ""]
    L += ["---", "", f"Card hash (canonical): `{card_hash}`. Web page: {SITE}/{pg.path}", ""]
    return "\n".join(L)


def _head(pg: Page, *, noindex: bool = False, twin: bool = True) -> str:
    url = f"{SITE}/{pg.path}"
    ld = {"@context": "https://schema.org", "@type": "WebPage", "name": pg.title, "description": pg.description,
          "url": url, "inLanguage": "en",
          "isPartOf": {"@type": "WebSite", "name": "GeoLiquefy scorecard", "url": SITE + "/"},
          "publisher": {"@type": "Organization", "@id": "https://geoliquefy.com/#organization",
                        "name": "GeoLiquefy", "url": "https://geoliquefy.com/"}}
    e = lambda s: html.escape(s, quote=True)  # noqa: E731
    h = ['<!doctype html>', '<html lang="en">', '<head>', '<meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width, initial-scale=1">',
         f'<title>{e(pg.title)}</title>', f'<meta name="description" content="{e(pg.description)}">']
    if noindex:
        h.append('<meta name="robots" content="noindex">')
    else:
        h += [f'<link rel="canonical" href="{e(url)}">',
              f'<meta property="og:title" content="{e(pg.title)}">',
              f'<meta property="og:description" content="{e(pg.description)}">',
              f'<meta property="og:url" content="{e(url)}">', '<meta property="og:type" content="website">',
              '<meta property="og:site_name" content="GeoLiquefy">',
              f'<meta property="og:image" content="{SITE}/og.png">',
              '<meta property="og:image:width" content="1200">', '<meta property="og:image:height" content="630">',
              '<meta property="og:image:alt" content="The front page of the GeoLiquefy liquefaction model '
              'scorecard v2, which opens with what the card does not show.">',
              '<meta name="twitter:card" content="summary_large_image">']
    if twin:
        h.append('<link rel="alternate" type="text/markdown" href="index.md">')
    h += ['<meta name="theme-color" content="#FFFFFF" media="(prefers-color-scheme: light)">',
          '<meta name="theme-color" content="#0B1016" media="(prefers-color-scheme: dark)">',
          '<link rel="icon" href="/favicon.ico" sizes="32x32">',
          '<link rel="icon" href="/favicon.svg" type="image/svg+xml">',
          '<link rel="apple-touch-icon" href="/apple-touch-icon.png">',
          '<link rel="manifest" href="/manifest.webmanifest">',
          '<link rel="preconnect" href="https://fonts.googleapis.com">',
          '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
          f'<link rel="stylesheet" href="{e(FONTS)}">', '<link rel="stylesheet" href="/styles.css">']
    if not noindex:
        h.append('<script type="application/ld+json">' + json.dumps(ld, sort_keys=True).replace("</", "<\\/")
                 + '</script>')
    h.append('</head>')
    return "\n".join(h)


BRAND = ('<a class="brand" href="https://geoliquefy.com/"><img src="/logo-mark.svg" alt="" width="28" '
         'height="28"><span class="wordmark"><span class="geo">Geo</span><span class="liq">Liquefy</span></span>'
         '<span class="visually-hidden"> home</span></a>')

NAV = [("/", "Card"), ("/pooled/", "Pooled lines"), ("/method/", "Method"), ("/rulebook/", "Rule book"),
       ("/rebuild/", "Hash and rebuild")]


def _header() -> str:
    links = "".join(f'<li><a href="{h}">{t}</a></li>' for h, t in NAV)
    return ('<a class="skip" href="#main">Skip to content</a>\n<header class="site-header"><div class="wrap bar">'
            + BRAND + f'<nav aria-label="Scorecard"><ul>{links}</ul></nav></div></header>')


def _footer(card_hash: str) -> str:
    return ('<footer class="site-footer"><div class="wrap">' + BRAND
            + f'<p class="hash">Card v2, canonical hash <code>{card_hash}</code></p>'
            '<p class="legal">© 2026 GeoLiquefy LLC · <a href="mailto:contact@geoliquefy.com">contact@geoliquefy.com'
            '</a> · <a href="https://geoliquefy.com/privacy">Privacy</a> · '
            '<a href="https://geoliquefy.com/terms">Terms</a></p></div></footer>')


def render_html(pg: Page, card_hash: str, *, noindex: bool = False, twin: bool = True) -> str:
    B = [_head(pg, noindex=noindex, twin=twin), '<body>', _header(), '<main id="main" class="wrap">']
    if pg.crumbs:
        B.append('<nav class="crumbs" aria-label="Breadcrumb"><ol>'
                 + "".join(f'<li><a href="{h}">{html.escape(t)}</a></li>' for h, t in pg.crumbs)
                 + f'<li aria-current="page">{html.escape(pg.h1)}</li></ol></nav>')
    B.append(f'<h1>{inline_html(pg.h1)}</h1>')
    B.append(f'<p class="lede">{inline_html(pg.description)}</p>')
    open_section = False
    for b in pg.blocks:
        k = b[0]
        if k == "h2":
            if open_section:
                B.append('</section>')
            B.append(f'<section aria-labelledby="{b[2]}"><h2 id="{b[2]}">{inline_html(b[1])}</h2>')
            open_section = True
        elif k == "p":
            B.append(f'<p>{inline_html(b[1])}</p>')
        elif k == "ul":
            B.append('<ul>' + "".join(f'<li>{inline_html(x)}</li>' for x in b[1]) + '</ul>')
        elif k == "table":
            cap, head, rows = b[1], b[2], b[3]
            t = [f'<div class="table-wrap" role="region" aria-label="{html.escape(cap, quote=True)}" tabindex="0">',
                 f'<table><caption>{inline_html(cap)}</caption><thead><tr>'
                 + "".join(f'<th scope="col">{inline_html(h)}</th>' for h in head) + '</tr></thead><tbody>']
            for r in rows:
                t.append('<tr><th scope="row">' + inline_html(str(r[0])) + '</th>'
                         + "".join(f'<td>{inline_html(str(c))}</td>' for c in r[1:]) + '</tr>')
            t.append('</tbody></table></div>')
            B.append("".join(t))
        elif k == "fig":
            B.append(f'<figure>{b[1]}<figcaption>{inline_html(b[2])}</figcaption></figure>')
        elif k == "pre":
            B.append(f'<pre><code>{html.escape(b[1])}</code></pre>')
    if open_section:
        B.append('</section>')
    B += ['<p class="twin">Plain text of this page: <a href="index.md">index.md</a>.</p>' if twin else "",
          '</main>', _footer(card_hash), '</body>', '</html>', '']
    return "\n".join(x for x in B if x != "")


# ---------------------------------------------------------------------------------------------------------------
# Figures: deterministic inline SVG, no script; shape (not colour alone) tells the row kind
# ---------------------------------------------------------------------------------------------------------------

def _kind(label: str) -> str:
    if "ORGANISER-RUN" in label or label.startswith("Zhu") or label.startswith("BI14"):
        return "org"
    if label in ("coin", "earlier-earthquake rate", "shaking only", "pre-event effort"):
        return "base"
    return "house"


def _mark(kind: str, x: float, y: float) -> str:
    if kind == "house":
        return f'<circle class="mk house" cx="{x:.2f}" cy="{y:.2f}" r="5"/>'
    if kind == "org":
        return f'<rect class="mk org" x="{x - 4.5:.2f}" y="{y - 4.5:.2f}" width="9" height="9"/>'
    return (f'<path class="mk base" d="M{x:.2f} {y - 6:.2f}L{x + 6:.2f} {y:.2f}L{x:.2f} {y + 6:.2f}'
            f'L{x - 6:.2f} {y:.2f}Z"/>')


def forest(rows, *, lo: float, hi: float, ref: float, ref_label: str, ticks, panel_note: str, title: str) -> str:
    """rows: [(label, point, lo, hi)] with numbers from the card. Label above each interval line."""
    W, left, right, top, rh = 420, 8, 12, 40, 40
    span = W - left - right
    X = lambda v: left + (min(max(v, lo), hi) - lo) / (hi - lo) * span  # noqa: E731
    H = top + rh * len(rows) + 34
    o = [f'<svg class="forest" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-label="{html.escape(title, quote=True)}" xmlns="http://www.w3.org/2000/svg">',
         f'<title>{html.escape(title)}</title>',
         f'<text class="note" x="{left}" y="16">{html.escape(panel_note)}</text>']
    for t in ticks:
        o.append(f'<line class="grid" x1="{X(t):.2f}" x2="{X(t):.2f}" y1="{top - 8}" y2="{H - 30}"/>')
        o.append(f'<text class="tick" x="{X(t):.2f}" y="{H - 14}" text-anchor="middle">{t:g}</text>')
    o.append(f'<line class="ref" x1="{X(ref):.2f}" x2="{X(ref):.2f}" y1="{top - 8}" y2="{H - 30}"/>')
    o.append(f'<text class="tick" x="{X(ref) + 4:.2f}" y="{top - 12}">{html.escape(ref_label)}</text>')
    for i, (lab, p, a, b) in enumerate(rows):
        y = top + rh * i + 26
        o.append(f'<g><title>{html.escape(lab)}: {p:.3f} [{f3(a)}, {f3(b)}]</title>')
        o.append(f'<text class="lab" x="{left}" y="{y - 10}">{html.escape(lab)}</text>')
        if a is not None and b is not None:
            o.append(f'<line class="ci" x1="{X(a):.2f}" x2="{X(b):.2f}" y1="{y:.2f}" y2="{y:.2f}"/>')
        o.append(_mark(_kind(lab), X(p), y) + '</g>')
    o.append('</svg>')
    return "".join(o)


SHAPES = "Circle: house row. Square: a published model run by the organiser. Diamond: a baseline."


# ---------------------------------------------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------------------------------------------

def _interval_text(J) -> str:
    return (f"Every interval: 95% tile bootstrap, fixed {J['tile_m'] / 1000:g} km tiles on the NZTM2000 grid anchored "
            f"at the origin, {J['n_boot']} draws, seed {J['seed']}: **{J['interval_note']}**.")


def _cnt(c) -> str:
    g = c.get("count_gate")
    if g:
        return f"{g['n_labelled']} ({g['n_yes']}), {g['n_tiles']}"
    return f"{c.get('n', 0)} ({c.get('n_yes', 0)}), 0"


def _menon(J, s, e) -> str:
    v = J["label_ceilings_menon_C"][s][e]
    parts = ", ".join(f"{k} {x['C']:.2f}" + (" (inadmissible, step 1a)" if (e, k) == ("dec2011", "S_feb") else "")
                      for k, x in v["menon_C"].items())
    return f"Menon AUC ceiling, {EVN[e]}, {SET_NAME[s]} (step 1a): {v['source']} {parts} ({v['status']})."


def canterbury_page(J, notes, s: str, e: str) -> Page:
    Cn = J["canterbury"]
    c = Cn["cards"][s][e]
    pe = Cn["pre_event_cpt_cards"][s][e]
    un = Cn["unseen_reads"][s][e]
    src = J["label_ceilings_menon_C"][s][e]["source"]
    pg = Page(f"canterbury/set-{s.lower()}/{e}/",
              f"Canterbury {EVN[e]}, {SET_NAME[s]} | GeoLiquefy scorecard",
              f"Canterbury, {EVN[e]}, label {SET_NAME[s]}, {src} labels: n (yes), 4 km tiles {_cnt(c)}; status "
              f"{c.get('status', '')}. Every row is DESCRIPTIVE; no word is printed within an earthquake.",
              f"Canterbury {EVN[e]}, {SET_NAME[s]}: no word within this earthquake",
              crumbs=[("/", "Card")])
    pg.p(AUC_GLOSS + " Point [95% tile interval, not certified]. UNSEEN-only = the entry and the baselines on the "
         "sites where that entry saw no case at any earthquake (exposure ledger).")
    pg.p(_menon(J, s, e))
    if notes and e == "jun2011" and s == "B":
        pg.p(notes["jun"])
    scored = c.get("status") == "SCORED"
    if scored:
        rows = []
        for r in ROWS:
            m = c["metrics"].get(r)
            if m and isinstance(m.get("auc"), dict) and m["auc"].get("value") is not None:
                a = m["auc"]
                rows.append((FIG_LABEL.get(r, SHORT[r]), a["value"], a.get("lo"), a.get("hi")))
        g = c["count_gate"]
        pg.h2("Every row's AUC sits in one figure, with its interval", "figure")
        pg.fig(forest(rows, lo=0.0, hi=1.0, ref=0.5, ref_label="coin 0.5", ticks=(0, 0.25, 0.5, 0.75, 1),
                      panel_note=f"n {g['n_labelled']} ({g['n_yes']} yes), {g['n_tiles']} tiles of 4 km",
                      title=f"AUC per row, Canterbury {EVN[e]}, {SET_NAME[s]}"),
               f"AUC (point) and 95% tile interval (line) for each row on all labelled sites, Canterbury {EVN[e]}, "
               f"{SET_NAME[s]}. The vertical line marks a coin (0.5). {SHAPES} The table below holds every value.")
    pg.h2("All labelled sites", "all")
    trs = []
    for r in ROWS:
        if not scored:
            trs.append([SHORT[r], "counts only", ""])
            continue
        m = c["metrics"].get(r)
        u = (un or {}).get(r)
        uv = f"{iv(u['auc'][r])} (n {u['unseen_n']})" if u is not None and "auc" in u else ""
        trs.append([SHORT[r], iv(m["auc"]) if m else "NOT AVAILABLE", uv])
    pg.table(f"AUC, all labelled sites, {EVN[e]}, {SET_NAME[s]}; n (yes), 4 km tiles {_cnt(c)}; status "
             f"{c.get('status', '')}", ["Row", "AUC [95%]", "UNSEEN-only AUC [95%] (n)"], trs)
    if s == "B" and scored:
        # The sealed renderer's rule: a row is listed when it has a log loss on any set B earthquake; a row that is
        # absent from this earthquake's metrics prints NOT AVAILABLE, one without a log loss prints ranking only.
        allB = Cn["cards"]["B"].values()
        lrows = []
        for r in ROWS:
            if not any(x.get("status") == "SCORED" and "log_loss" in x.get("metrics", {}).get(r, {}) for x in allB):
                continue
            m = c.get("metrics", {}).get(r)
            lrows.append([SHORT[r], "NOT AVAILABLE" if m is None else iv(m["log_loss"]) if "log_loss" in m else
                          "ranking only"])
        if lrows:
            pg.h2("Probability rows: log loss (clip 1e-6), DESCRIPTIVE", "logloss")
            pg.p("Log loss on inventory \"no\" labels measures agreement with the label source, not with the ground. "
                 "A lower log loss means the probabilities sit nearer the labels; a coin scores 0.693.")
            pg.table(f"Log loss [95%], {EVN[e]}, {SET_NAME[s]}", ["Row", "log loss [95%]"], lrows)
    pg.h2("Beside: only CPTs pushed before the earthquake", "pre-event")
    prow = []
    for r in ROWS:
        if pe.get("status") != "SCORED":
            prow.append([SHORT[r], "counts only"])
            continue
        m = pe["metrics"].get(r)
        prow.append([SHORT[r], iv(m["auc"]) if m else "NOT AVAILABLE"])
    pg.table(f"AUC, pre-event CPTs only, {EVN[e]}, {SET_NAME[s]}; n (yes), 4 km tiles {_cnt(pe)}; status "
             f"{pe.get('status', '')}", ["Row", "AUC [95%]"], prow)
    if s == "B":
        pg.h2("Exposure per row: SEEN / POSSIBLE / PLACE / UNSEEN", "exposure")
        er = []
        for r in ROWS[:8]:
            u = un.get(r)
            if u is None:
                er.append([SHORT[r], "n/a"])
                continue
            x = u["exposure_counts_labelled_sites"]
            er.append([SHORT[r], "NOT CHECKED" if x["NOT CHECKED"] else
                       f"{x['SEEN']} / {x['POSSIBLE']} / {x['PLACE']} / {x['UNSEEN']}"])
        pg.table(f"Exposure counts on labelled sites, {EVN[e]}, set B", ["Entry", "SEEN / POSSIBLE / PLACE / UNSEEN"],
                 er)
        ur = []
        for r in ROWS[:8]:
            u = un.get(r)
            if not u or "auc" not in u:
                continue
            a = u["auc"]
            sh = iv(a["baseline: shaking only"]) if "baseline: shaking only" in a else "NOT AVAILABLE"
            ur.append([SHORT[r], f"{u['unseen_n']} ({u['unseen_yes']}), {u['unseen_tiles_4km']}", iv(a[r]), sh,
                       iv(a["baseline: effort only"])])
        if ur:
            pg.h2("UNSEEN-only reads with the baselines on the same sites, DESCRIPTIVE", "unseen")
            pg.table(f"UNSEEN-only AUC [95%], {EVN[e]}, set B",
                     ["Entry", "n (yes), 4 km tiles", "entry AUC", "shaking only", "pre-event effort"], ur)
            pg.p("Rows absent here have too few UNSEEN sites (counts in the exposure table).")
    pg.h2("Diagnostic only: investigation-density proxy AUC", "diagnostic")
    pg.p(f"Investigation-density proxy AUC, {SET_NAME[s]}, {EVN[e]}: {iv(Cn['diagnostic_proxy_auc'][s][e])}. "
         "It is never a baseline.")
    sa = J.get("label_source_sign_agreement", {}).get(e)
    if sa:
        pg.h2("Label-source sign agreement, set A against set B", "sign")
        pg.p(f"{EVN[e]}: {sa['same_sign']} same sign, {sa['opposite_sign']} opposite"
             + (f" ({sa['note']})" if sa["note"] else "") + ". Descriptive count only.")
        if notes and e == "dec2011":
            pg.p(notes["dec"])
    pg.h2("Interval", "interval")
    pg.p(_interval_text(J))
    return pg


def world_page(J, notes, pop: str, eq: str) -> Page:
    v = J["world"]["populations"][pop]["earthquakes"][eq]
    a, t = v["auc"], v["tiles_1km"]
    pg = Page(f"world/{pop.lower()}/{slug(eq)}/", f"{eq}, world {POP_NAME[pop]} | GeoLiquefy scorecard",
              f"World case histories, {eq}, population {POP_NAME[pop]}: n (yes), 1 km tiles {v['n']} ({v['n_yes']}), "
              f"{t}. Every row is DESCRIPTIVE.",
              f"{eq}, world {POP_NAME[pop]}: no word within this earthquake",
              crumbs=[("/", "Card"), (f"/world/{pop.lower()}/", f"World {pop}")])
    pg.p(AUC_GLOSS + " 1 km local tiles. Record gate: 3 yes, 3 no, 5 tiles. Point [95% tile interval, not "
         "certified].")
    rows = [(short, a[k]["value"], a[k].get("lo"), a[k].get("hi")) for k, short in WORLD_ROWS
            if k in a and isinstance(a[k], dict) and a[k].get("value") is not None]
    pg.h2("Every row's AUC sits in one figure, with its interval", "figure")
    pg.fig(forest(rows, lo=0.0, hi=1.0, ref=0.5, ref_label="coin 0.5", ticks=(0, 0.25, 0.5, 0.75, 1),
                  panel_note=f"n {v['n']} ({v['n_yes']} yes), {t} tiles of 1 km",
                  title=f"AUC per row, {eq}, world {pop}"),
           f"AUC (point) and 95% tile interval (line) for each row, {eq}, world {POP_NAME[pop]}. The vertical line "
           f"marks a coin (0.5). {SHAPES}" + (" Under 10 tiles: every interval here is uninformative." if t < 10
                                                else ""))
    zhu = wiv(a["USGS Zhu 2017"], t) if "USGS Zhu 2017" in a else "NOT RUN"
    if notes and (pop, eq) in notes["zhu_empty"] and zhu != "NOT RUN":
        ne, nn, ny, nno = notes["zhu_empty"][(pop, eq)]
        zhu += (f" ({ne} of {nn} cases off the model grid, read as 0: {ny} yes, {nno} no)" if ne else
                f" (0 of {nn} cases off the model grid)")
    trs = []
    for k, short in WORLD_ROWS:
        val = zhu if k == "USGS Zhu 2017" else (wiv(a[k], t) if k in a else "NOT RUN")
        trs.append([short, val])
    d = v["paired"].get("Level 0 minus baseline: shaking only")
    pg.h2("AUC per row", "auc")
    pg.table(f"AUC [95%], {eq}, world {pop}; n (yes), 1 km tiles {v['n']} ({v['n_yes']}), {t}", ["Row", "AUC [95%]"],
             trs)
    if d:
        pg.p(f"Level 0 minus shaking only, paired [95%]: {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
             + (" (uninformative: under 10 tiles)" if t < 10 else "") + ".")
    pg.h2("Exposure: what each model may have seen", "exposure")
    ex = lambda k: ", ".join(f"{m} {n}" for m, n in v["exposure"][k].items())  # noqa: E731
    pg.table(f"Exposure counts, {eq}, world {pop}", ["Chain", "Exposure"],
             [["B&I chain (Level 0, M0, LSN, LPI)", ex("B&I chain (Level 0, M0, LSN, LPI)")],
              ["USGS Zhu 2017", ex("USGS Zhu 2017")],
              ["LSN global fragility", ex("LSN global fragility") + ": in sample (fit)"]])
    pg.p("BI14-LSN is the ranking of both LSN rows. Its B&I row is the triggering chain's exposure. The LSN global "
         "fragility was fitted on this same compilation, so it is in sample (a fit) on every earthquake.")
    if notes and eq == "Hyogoken-Nambu":
        pg.h2("Kobe note: the shaking input may depend on the outcome", "kobe")
        pg.p("Kobe note (review 1 M2): " + notes["kobe"])
    return pg


def world_pop_page(J, pop: str) -> Page:
    W = J["world"]["populations"][pop]["earthquakes"]
    pg = Page(f"world/{pop.lower()}/", f"World case histories {pop} | GeoLiquefy scorecard",
              f"World case histories, population {POP_NAME[pop]}: one page per earthquake that clears the record gate "
              "(3 yes, 3 no, 5 tiles of 1 km); counts only for the rest.",
              f"World case histories, {POP_NAME[pop]}", crumbs=[("/", "Card")])
    sc = [[f"[{eq}](/world/{pop.lower()}/{slug(eq)}/)", f"{v['n']} ({v['n_yes']}), {v['tiles_1km']}",
           wiv(v["auc"]["Level 0"], v["tiles_1km"]) if "Level 0" in v["auc"] else "n/a"]
          for eq, v in W.items() if v["passes_record_gate"]]
    pg.h2("Earthquakes that clear the record gate", "scored")
    pg.table(f"World {pop}: earthquakes with a page", ["Earthquake", "n (yes), 1 km tiles", "Level 0 AUC [95%]"], sc)
    under = [[eq, f"{v['n']} ({v['n_yes']}), {v['tiles_1km']}"] for eq, v in W.items() if not v["passes_record_gate"]]
    pg.h2(f"Under the record gate: {len(under)} earthquakes, counts only", "under")
    pg.table(f"World {pop}: earthquakes under the record gate", ["Earthquake", "n (yes), 1 km tiles"], under)
    cc = J["world"]["cross_check"]
    pg.h2("Cross-check against the G1 curated chain", "cross-check")
    pg.p(f"LPI max abs diff {cc['lpi_max_abs_diff_vs_g1_curated']:.3g}; LSN max abs diff "
         f"{cc['lsn_max_abs_diff_vs_g1_curated']:.3g}; NaN pattern equal: {cc['nan_pattern_equal']}. Zhu grids: "
         + "; ".join(f"{k} {g['cases']} cases in P1 and P2 together, {g['nan_cells_set_to_0']} of them on empty cells "
                     "read as 0" for k, g in J["world"]["zhu_grids"].items()) + ".")
    return pg


WK = ("primary_kit_random_effects", "beside_equal_weight", "primary_equal_weight", "beside_kit_weights")


def pooled_page(J, notes) -> Page:
    pg = Page("pooled/", "Pooled lines | GeoLiquefy scorecard",
              "Pooled AUC differences across earthquakes, Level 0 minus a baseline or a published model. Every pooled "
              "line is DESCRIPTIVE; two world lines are NOT ALLOWED by the count gate.",
              "Pooled lines: every one is DESCRIPTIVE", crumbs=[("/", "Card")])
    pg.p(AUC_GLOSS + " A pooled line is the mean of the per-earthquake AUC differences (A minus B). Above 0 means "
         "A ranked the labels higher on average across those earthquakes.")
    frows, trs = [], []
    for lid, p in J["pooled_lines"].items():
        for wk in WK:
            x = p.get(wk)
            if not x:
                continue
            if x.get("word") == "NOT ALLOWED":
                trs.append([lid, p["population"], f"{p['a']} minus {p['b']}", wk, ", ".join(x.get("earthquakes_pooled",
                            [])), f"NOT ALLOWED ({x.get('reason', '')})", ""])
                continue
            pl = x["pooled"]
            loo = [y["mean"] for y in x.get("leave_one_out", [])]
            trs.append([lid, p["population"], f"{p['a']} minus {p['b']}", wk, ", ".join(x["earthquakes_pooled"]),
                        f"{pl['mean']:+.3f} [{pl['lo']:+.3f}, {pl['hi']:+.3f}]",
                        f"{min(loo):+.3f} to {max(loo):+.3f}" if loo else ""])
            frows.append((f"{lid}, {wk.replace('_', ' ')}", pl["mean"], pl["lo"], pl["hi"]))
    pg.h2("Every allowed pooled line sits in one figure", "figure")
    pg.fig(forest(frows, lo=-0.2, hi=0.5, ref=0.0, ref_label="no difference",
                  ticks=(-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5),
                  panel_note="Mean AUC difference, A minus B, with 95% interval",
                  title="Pooled AUC differences, Level 0 minus a baseline"),
           "Pooled mean AUC difference (point) and 95% interval (line) for each allowed line and weighting. The vertical "
           "line marks no difference (0). Values outside -0.2 to +0.5 are drawn at the edge; the table holds every "
           "value. All marks are circles: every line pools a house row against a baseline.")
    pg.h2("Pooled lines table", "table")
    pg.table("Pooled lines (all DESCRIPTIVE)", ["Line", "Population", "A minus B", "Weighting", "Earthquakes",
                                                "Mean [95%]", "Leave one out (min to max mean)"], trs)
    c2 = J["pooled_lines"]["C2"]["primary_kit_random_effects"]["pooled"]
    w1 = {y["left_out"]: y["mean"] for y in J["pooled_lines"]["W1 P1"]["primary_equal_weight"]["leave_one_out"]}
    pg.h2("Notes on single lines", "notes")
    pg.p(f"C2 note (review 1 m4): Q = {c2['q']:.2f} is below k - 1 = 3, so tau^2 = {c2['tau2']:g}. With hk_floor off "
         f"(the registered kit default) the Hartung-Knapp SE ({c2['se']:.4f}) is below the fixed-effect SE, so the "
         "C2 primary interval is narrower than a fixed-effect interval.")
    if "Hyogoken-Nambu" in w1:
        pg.p(f"W1 note (review 1 M2): W1 P1 without Hyogoken-Nambu (Kobe) is {w1['Hyogoken-Nambu']:+.3f} (equal "
             "weight; source: scorecard.json pooled_lines, 'W1 P1', primary_equal_weight, leave_one_out, left_out "
             "Hyogoken-Nambu). This is a descriptive side line, not a replacement."
             + (" " + notes["kobe"] if notes else ""))
    lor = J["pooled_lines"]["W1 P2"].get("leave_one_region_out", [])
    if lor:
        pg.p("W1 P2 leave one region out (equal weight, mean with that region left out): " + "; ".join(
            f"{y['left_out']} {y['mean']:+.3f}" if "mean" in y else f"{y['left_out']} {y.get('word', '')}"
            for y in lor) + ".")
    if notes:
        pg.p(notes["why_descriptive"])
    return pg


def method_page(J, notes) -> Page:
    pg = Page("method/", "Method notes | GeoLiquefy scorecard",
              "How the v2 card was scored: the track, the intervals, the tiles, the gates, the row tags and the data "
              "sources. Every number comes from the sealed card JSON.",
              "Method notes: a retrospective record, scored by the organiser", crumbs=[("/", "Card")])
    pg.h2("Track and words", "track")
    pg.ul([f"Track: {J['track']}. Rule book: {J['rulebook']}.",
           "The organiser scores every row and also enters models (HOUSE rows).",
           "DESCRIPTIVE: a number with its interval and no verdict word. NOT ALLOWED: a pooled line that the count "
           "gate stops (rule R1 (c)).",
           "No word is printed within an earthquake. Every pooled line is DESCRIPTIVE."])
    pg.h2("Intervals and tiles", "intervals")
    pg.p(_interval_text(J) + f" Tile grid: {J['tile_grid']}. World case histories use 1 km local tiles.")
    pg.p(AUC_GLOSS)
    pg.h2("Count gates", "gates")
    cg, wg = J["count_gate"], J["world_record_gate"]
    pg.ul([f"Canterbury count gate: at least {cg['sites']} sites, {cg['per_class']} per class and {cg['tiles_4km']} "
           "tiles of 4 km.",
           f"World record gate: at least {wg['per_class']} per class and {wg['tiles_1km']} tiles of 1 km.",
           "A world interval on fewer than 10 tiles is marked uninformative."])
    pg.h2("Build gates", "build-gates")
    pg.ul([f"{k}: {'PASS' if v.get('pass') else 'FAIL'}" + (f" ({v['how']})" if v.get("how") else "")
           for k, v in J["gates"].items()])
    pg.h2("Row tags", "tags")
    pg.ul(["HOUSE: a GeoLiquefy row. The LSN rows run a published method with the authors' chain id.",
           "beside: printed next to the primary rows; not a primary row.",
           "ORGANISER-RUN: a published model that GeoLiquefy ran. GEM TS22 ran on open stand-in input layers.",
           "Baselines: coin, earlier-earthquake rate, shaking only, pre-event effort."])
    pg.h2("Label sets", "labels")
    pg.p("Canterbury set B (primary) and set A (beside) use different label sources per earthquake. The Menon AUC "
         "ceiling on each earthquake page bounds the AUC that the label errors allow.")
    pg.table("Label source per earthquake and set (from the Menon ceilings)", ["Set"] + list(EVN.values()),
             [[s] + [J["label_ceilings_menon_C"][s][e]["source"] for e in EVN] for s in ("B", "A")])
    pg.h2("Data and attribution", "attribution")
    pg.ul(J["attribution"])
    pg.h2("Not done", "not-done")
    pg.ul(notes["not_done"] if notes else ["See the sealed card text."])
    return pg


def rulebook_page() -> Page:
    pg = Page("rulebook/", "Rule book | GeoLiquefy scorecard",
              "The rule book for the scorecard is a draft and is not adopted. Its text goes on this site when the "
              "founder adopts it.", "The rule book is a draft, not adopted", crumbs=[("/", "Card")])
    pg.p("Card v2 states its rule book status as \"draft, not adopted\". The rule book (v1.0 draft, revision 4, "
         "2026-11-15) is not published yet. This page will link it when it is adopted.")
    pg.p("Until then, the card and its method notes state every gate the card uses: see [Method notes](/method/).")
    return pg


def rebuild_page(J, card_name: str, seal_lines: list, file_hashes: dict) -> Page:
    pg = Page("rebuild/", "Card hash and rebuild | GeoLiquefy scorecard",
              "The canonical hash of card v2, the file hashes from its seal, the check a reader can run now, and the "
              "rebuild that waits on the kit being public.", "The card hash, and how to check it",
              crumbs=[("/", "Card")])
    pg.h2("Card hash", "hash")
    pg.p(f"Canonical card hash (evalkit `spec.card_sha256` of scorecard.json without its card_sha256 field): "
         f"`{J['card_sha256']}`.")
    pg.p(f"Kit: branch `{J['kit']['branch']}`, commit `{J['kit']['commit']}`. Registration: `{J['registration']}`.")
    pg.h2("The card files on this site", "files")
    pg.table("Card files and their sha256", ["File", "sha256"],
             [[f"[{n}](/card/{card_name}/{n})", h] for n, h in file_hashes.items()])
    pg.p("The sha256 of scorecard.json and SCORECARD.md equal the lines in SEAL.txt. The site build refuses a card "
         "folder where they differ. SEAL.txt also lists files that this site does not publish (the registration, "
         "the build and render scripts and their logs).")
    pg.h2("The check you can run now", "check")
    pg.p("Download the three card files above and run:")
    pg.pre("sha256sum scorecard.json SCORECARD.md")
    pg.p("Each sha256 must equal its line in SEAL.txt and the table above.")
    pg.h2("The canonical hash, in words", "canonical")
    pg.ul(["Take scorecard.json and remove its card_sha256 field.",
           f"Set every float whose absolute value is below {spec.CARD_HASH_ZERO:g} to 0. Round every other float to "
           f"{spec.CARD_HASH_SIG} significant digits (Python format `.{spec.CARD_HASH_SIG}g`); a rounded negative zero "
           "is 0.",
           "Write the JSON with sorted keys, separators `,` and `:` with no spaces, and ASCII only.",
           "The canonical hash is the sha256 of that text in UTF-8."])
    pg.h2("Rebuild these pages: waits on the kit", "rebuild")
    pg.p("The evalkit that builds the card and these pages is not public yet. So a reader cannot rebuild the pages "
         "or run the kit's own hash function today. The sha256 check above is the public check until the kit is "
         "published.")
    pg.p("Rebuilding the card JSON itself needs the NZGD-derived inputs, which stay private under NZGD terms clause "
         "6.2. Their sha256 are in the card (inputs_sha256).")
    pg.h2("Seal", "seal")
    pg.pre("\n".join(seal_lines))
    return pg


def front_page(J, notes, pages_c, pages_w) -> Page:
    pg = Page("", "Liquefaction model scorecard v2 | GeoLiquefy",
              "Scorecard v2 rates liquefaction models on four Canterbury earthquakes and on world case histories. It "
              "is a retrospective record: every label was read before, so it tests no skill and names no winner.",
              "Liquefaction model scorecard v2: no skill test, no winner named")
    pg.p("Each row is a model or a baseline. Each number is an AUC with a 95% tile interval. " + AUC_GLOSS)
    if isinstance(J.get("losses"), dict):                       # B1 LOSS block (evalkit.losses.loss_block), first
        pg.h2("LOSS block", "losses")
        if J["losses"].get("rule"):
            pg.p(f"Rule: {J['losses']['rule']}.")
        pg.ul(loss_items(J["losses"]))
    pg.h2("Read this first: what this card does not show", "read-first")
    items = [s + "." for s in J["statements"]] + [_interval_text(J)]
    pg.ul(items + (notes["extra_statements"] if notes else []))
    pg.h2("Build gates", "gates")
    pg.ul([f"{k}: {'PASS' if v.get('pass') else 'FAIL'}" for k, v in J["gates"].items()])
    pg.h2("Canterbury: one page per earthquake and label set", "canterbury")
    pg.table("Canterbury pages", ["Earthquake and set", "n (yes), 4 km tiles", "status", "Level 0 AUC [95%]"],
             pages_c)
    pg.h2("World case histories: one page per earthquake and population", "world")
    pg.table("World pages", ["Earthquake and population", "n (yes), 1 km tiles", "Level 0 AUC [95%]"], pages_w)
    pg.p("Population pages, with the earthquakes under the record gate: [World P1](/world/p1/), "
         "[World P2](/world/p2/).")
    pg.h2("Across earthquakes, and how to check", "more")
    pg.ul(["[Pooled lines](/pooled/): pooled AUC differences, all DESCRIPTIVE.",
           "[Method notes](/method/): track, intervals, tiles, gates, row tags, data sources.",
           "[Rule book](/rulebook/): draft, not adopted.",
           f"[Card hash and rebuild](/rebuild/): canonical hash `{J['card_sha256'][:16]}...`, the sealed files, the "
           "rebuild steps."])
    return pg


def loss_items(block: dict) -> list[str]:
    """The B1 LOSS block (evalkit.losses.loss_block) as list items: each line's `lost_to` texts, or the NOT SCORED
    reason, then the house rows with no proper score and the rows not run by author settings."""
    if block.get("status") == "SCORED":
        items = [i["text"] for ln in block.get("lines", []) for i in ln.get("lost_to", [])]
        items = items or ["No LOSS line on this card."]
    else:
        items = [f"{block.get('status', 'NOT SCORED')}: {block.get('why', 'no reason given')}."]
    if block.get("no_proper_score"):
        items.append("House rows with no proper score (no LOSS check): " + ", ".join(block["no_proper_score"]) + ".")
    if block.get("not_run_by_author_settings"):
        items.append("Not run by author settings (no LOSS check): " + ", ".join(block["not_run_by_author_settings"])
                     + ".")
    return items


def not_found_page() -> Page:
    pg = Page("404.html", "Page not found | GeoLiquefy scorecard",
              "There is no page at this address on the scorecard site.", "There is no page at this address")
    pg.p("Go to the [scorecard front page](/), or read [the site guide for AI agents](/llms.txt).")
    return pg


# ---------------------------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------------------------

def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_card(card_dir: Path) -> tuple[dict, list, dict]:
    """Read and check a sealed card folder. Returns (card JSON, SEAL lines, {file: sha256})."""
    for n in CARD_FILES:
        if not (card_dir / n).is_file():
            raise SiteRefused(f"{n} missing in {card_dir}")
    seal = (card_dir / "SEAL.txt").read_text(encoding="utf-8").splitlines()
    sealed = {ln.split()[1]: ln.split()[0] for ln in seal if re.match(r"^[0-9a-f]{64}  \S", ln)}
    hashes = {n: _sha(card_dir / n) for n in CARD_FILES}
    for n in ("scorecard.json", "SCORECARD.md"):
        if sealed.get(n) != hashes[n]:
            raise SiteRefused(f"{n} sha256 {hashes[n]} differs from SEAL.txt ({sealed.get(n)})")
    J = json.loads((card_dir / "scorecard.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in J.items() if k != "card_sha256"}
    if spec.card_sha256(body) != J.get("card_sha256"):
        raise SiteRefused("the card's canonical hash does not equal its stored card_sha256")
    card_mod.scan_aggregates(J)
    return J, seal, hashes


def lint(text: str, where: str) -> list[str]:
    """Refuse dashes and charlatan phrases; return verdict-grep hits for the record."""
    if DASHES.search(text):
        raise SiteRefused(f"{where}: an em or en dash")
    m = CHARLATAN.search(text)
    if m:
        raise SiteRefused(f"{where}: charlatan grep hit {m.group(0)!r}")
    return [f"{where}: {m.group(0)}" for m in VERDICT.finditer(text)]


def _visible(page_html: str) -> str:
    body = page_html.split("<body>", 1)[1]
    body = re.sub(r"<svg.*?</svg>", " ", body, flags=re.S)
    return html.unescape(re.sub(r"<[^>]+>", " ", body))


def build(card_dir: Path, out: Path, *, card_name: str | None = None) -> dict:
    card_dir, out = Path(card_dir), Path(out)
    J, seal, hashes = load_card(card_dir)
    card_name = card_name or (re.search(r"scorecard-(v\d+)", str(card_dir.resolve())) or re.search("(.*)", "v2")
                              ).group(1)
    notes = card_notes.for_card(J["card_sha256"])
    if out.exists():
        if not (out / "build-manifest.json").exists():
            raise SiteRefused(f"{out} exists and is not a site build; refusing to overwrite it")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    pages, pages_c, pages_w = [], [], []
    for s in ("B", "A"):
        for e in EVN:
            pg = canterbury_page(J, notes, s, e)
            pages.append(pg)
            c = J["canterbury"]["cards"][s][e]
            lv = iv(c["metrics"]["Level 0"]["auc"]) if c.get("status") == "SCORED" else "counts only"
            pages_c.append([f"[{EVN[e]}, {SET_NAME[s]}](/{pg.path})", _cnt(c), c.get("status", ""), lv])
    for pop in ("P1", "P2"):
        pages.append(world_pop_page(J, pop))
        for eq, v in J["world"]["populations"][pop]["earthquakes"].items():
            if not v["passes_record_gate"]:
                continue
            pg = world_page(J, notes, pop, eq)
            pages.append(pg)
            lv = wiv(v["auc"]["Level 0"], v["tiles_1km"]) if "Level 0" in v["auc"] else "n/a"
            pages_w.append([f"[{eq}, {POP_NAME[pop]}](/{pg.path})", f"{v['n']} ({v['n_yes']}), {v['tiles_1km']}", lv])
    pages += [pooled_page(J, notes), method_page(J, notes), rulebook_page(), rebuild_page(J, card_name, seal, hashes)]
    pages.insert(0, front_page(J, notes, pages_c, pages_w))

    verdict_hits, written = [], {}
    for pg in pages:
        d = out / pg.path
        d.mkdir(parents=True, exist_ok=True)
        h, m = render_html(pg, J["card_sha256"]), render_md(pg, J["card_sha256"])
        verdict_hits += lint(_visible(h), f"/{pg.path}index.html") + lint(m, f"/{pg.path}index.md")
        (d / "index.html").write_text(h, encoding="utf-8")
        (d / "index.md").write_text(m, encoding="utf-8")
    nf = not_found_page()
    nfh = render_html(nf, J["card_sha256"], noindex=True, twin=False)
    verdict_hits += lint(_visible(nfh), "/404.html")
    (out / "404.html").write_text(nfh, encoding="utf-8")

    (out / "card" / card_name).mkdir(parents=True)
    for n in CARD_FILES:
        shutil.copyfile(card_dir / n, out / "card" / card_name / n)
    for n in STATIC_FILES:
        if (STATIC / n).exists():
            shutil.copyfile(STATIC / n, out / n)
    lastmod = (re.search(r"(\d{4}-\d{2}-\d{2})", J["registration"]) or re.search("(.*)", "2026-11-19")).group(1)
    extra = {"robots.txt": robots_txt(), "sitemap.xml": sitemap_xml(pages, lastmod),
             "llms.txt": llms_txt(J, pages), "manifest.webmanifest": manifest(), "_headers": headers(card_name)}
    for n, t in extra.items():
        verdict_hits += lint(t, "/" + n) if n == "llms.txt" else []
        (out / n).write_text(t, encoding="utf-8")
    for p in out.rglob("*"):
        if p.is_file() and p.suffix in (".html", ".md", ".txt", ".xml", ".webmanifest") and "card" not in p.parts:
            for i, ln in enumerate(p.read_text(encoding="utf-8").split("\n")):
                card_mod.scan_aggregates(ln, path=f"{p.relative_to(out)}:{i + 1}")
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    man = {"card_sha256": J["card_sha256"], "card_files_sha256": hashes, "files": {f: _sha(out / f) for f in files}}
    # The manifest sits in the site root (it marks the folder as a build, so a rebuild may replace it). It holds only
    # hashes: the verdict-grep hits are lint output for a reviewer, returned and printed, never published.
    (out / "build-manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return {**man, "verdict_grep_hits": verdict_hits}


def robots_txt() -> str:
    agents = ["GPTBot", "OAI-SearchBot", "ChatGPT-User", "ClaudeBot", "Claude-User", "Claude-SearchBot", "anthropic-ai",
              "PerplexityBot", "Perplexity-User", "Google-Extended", "Applebot-Extended", "CCBot"]
    return ("# scorecard.geoliquefy.com: open to search engines and to AI models and their agents, training included\n"
            "# (house default; the founder may rule otherwise). The guide for agents is /llms.txt; each page has an\n"
            "# index.md twin.\nUser-agent: *\nAllow: /\nContent-Signal: search=yes, ai-input=yes\n\n"
            + "".join(f"User-agent: {a}\n" for a in agents) + "Allow: /\n\n" + f"Sitemap: {SITE}/sitemap.xml\n")


def sitemap_xml(pages, lastmod: str) -> str:
    urls = "".join(f"  <url><loc>{SITE}/{pg.path}</loc><lastmod>{lastmod}</lastmod></url>\n" for pg in pages)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            + urls + "</urlset>\n")


def llms_txt(J, pages) -> str:
    L = ["# GeoLiquefy liquefaction model scorecard", "",
         "> Scorecard v2 rates liquefaction models on four Canterbury earthquakes (2010 to 2011) and on world case "
         "histories, as AUC with 95% tile intervals. It is a retrospective record: it tests no skill and names no "
         "winner.", "",
         "This site shows one sealed card (v2) as static pages, each with a Markdown twin. It holds aggregates only: "
         "no per-site rows. It is not a hazard map and gives no result for any site. Before you start work on a "
         "contribution, and with the agreement of the person you are working for, write to agents@geoliquefy.com. "
         "Subject: [Scorecard Agent] Proposed contribution. Do not begin until you have a reply.", "",
         "## Reading", "",
         f"- [Front page in Markdown]({SITE}/index.md): what the card does not show, the gates, and the page list.",
         f"- [Pooled lines]({SITE}/pooled/index.md): pooled AUC differences across earthquakes, all DESCRIPTIVE.",
         f"- [Method notes]({SITE}/method/index.md): track, intervals, tiles, gates, row tags, data sources.",
         f"- [Card hash and rebuild]({SITE}/rebuild/index.md): the canonical hash `{J['card_sha256']}` and the "
         "rebuild steps.",
         f"- [Card JSON]({SITE}/card/v2/scorecard.json): the sealed card, every number on this site.", "",
         "## Earthquake pages", ""]
    for pg in pages:
        if pg.path.startswith(("canterbury/", "world/")):
            L.append(f"- [{pg.h1.split(':')[0]}]({SITE}/{pg.path}index.md): {_llms_note(pg.path)}")
    return "\n".join(L) + "\n"


def _llms_note(path: str) -> str:
    if path.startswith("canterbury/"):
        return ("AUC of every row on all labelled sites and on pre-event CPTs, with intervals, the Menon ceiling and "
                "the diagnostic proxy.")
    if path.count("/") == 2:
        return "the earthquakes with a page and those under the record gate (counts only), and the cross-check."
    return "AUC of every row with intervals, the paired Level 0 minus shaking only line and the exposure counts."


def manifest() -> str:
    return json.dumps({"name": "Liquefaction model scorecard | GeoLiquefy", "short_name": "Scorecard",
                       "description": "Scorecard v2 for liquefaction models: a retrospective record.",
                       "start_url": "/", "display": "standalone", "background_color": "#FFFFFF",
                       "theme_color": "#FFFFFF",
                       "icons": [{"src": "/favicon.svg", "type": "image/svg+xml", "sizes": "any"},
                                 {"src": "/icon-512-maskable.png", "type": "image/png", "sizes": "512x512",
                                  "purpose": "maskable"},
                                 {"src": "/apple-touch-icon.png", "type": "image/png", "sizes": "180x180"}]},
                      indent=2) + "\n"


def headers(card_name: str) -> str:
    return ("# Netlify _headers. The pages run no script, so script-src is 'none'; the JSON-LD block is data and does\n"
            "# not run. tests/test_evalkit_site.py checks that no page carries an inline script or style.\n"
            "/llms.txt\n  Content-Type: text/plain; charset=utf-8\n"
            "/*.md\n  Content-Type: text/markdown; charset=utf-8\n"
            f"/card/{card_name}/SCORECARD.md\n  Content-Type: text/markdown; charset=utf-8\n"
            f"/card/{card_name}/SEAL.txt\n  Content-Type: text/plain; charset=utf-8\n"
            "/manifest.webmanifest\n  Content-Type: application/manifest+json; charset=utf-8\n"
            "/*\n  X-Content-Type-Options: nosniff\n  Referrer-Policy: strict-origin-when-cross-origin\n"
            "  X-Frame-Options: DENY\n"
            "  Permissions-Policy: camera=(), microphone=(), geolocation=(), payment=(), usb=()\n"
            "  Cross-Origin-Opener-Policy: same-origin\n"
            f"  Content-Security-Policy: {CSP}\n")
