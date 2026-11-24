# Scorecard site generator (item D9, eval M6)

G1+G5+G6 (eval M6). Builds the public scorecard site (scorecard.geoliquefy.com, ruling 43a) as static pages from a
sealed card folder. Nothing is deployed from here. No em dashes.

```
card folder (scorecard.json, SCORECARD.md, SEAL.txt)
   |  load_card: sha256 = SEAL.txt, canonical hash = card_sha256, card.scan_aggregates
   v
build.py  -->  one Page (blocks) per earthquake and label set  -->  index.html + index.md twin
          -->  front page (a B1 LOSS block leads when the card has one, else Read this first), pooled/, method/, rulebook/, rebuild/, 404.html
          -->  robots.txt, sitemap.xml, llms.txt, manifest, _headers, icons, og.png, card/v2/ (sealed files)
          -->  build-manifest.json (sha256 of every file; no lint output)
```

## Build

```
python -m scoreboard.evalkit.site --card-dir <pillar>/docs/scorecard-v2-2026-11-19 --out <site folder>
```

The output folder is new or an earlier build (it holds `build-manifest.json`); any other folder is refused.
Keep builds outside git (`scorecard-site-build/` is ignored).

## Rules the build holds

- Presentation only. Every number is read from the card JSON with the sealed renderer's formats (`render_v2.py`).
  Notes that the sealed text prints but the JSON lacks sit in `card_notes.py`, keyed by the card hash.
- Aggregates only (NZGD clause 6.2): the card and every built text line pass `card.scan_aggregates`.
- No JavaScript. CSP `script-src 'none'`; the only `<script>` is the JSON-LD data block.
- Lint: an em or en dash, or a memo-comms charlatan-grep phrase, stops the build; verdict-grep hits are printed by
  the CLI for a reviewer and are not published.

## Local launch check

```
python -m scoreboard.evalkit.site.serve <site folder> 8795      # applies _headers, 404 with status 404
node scoreboard/evalkit/site/tools/check.mjs http://127.0.0.1:8795 check.json   # axe, console, no-JS, links
node scoreboard/evalkit/site/tools/og.mjs http://127.0.0.1:8795 og-2x.png       # then sips to 1200x630
npx lighthouse http://127.0.0.1:8795/ [--preset=desktop]
```

The node tools need `playwright@1.49.1`, `axe-core@4.10.2` and `lighthouse@12.2.1` in the folder you run from.
Tests: `tests/test_evalkit_site.py`.
