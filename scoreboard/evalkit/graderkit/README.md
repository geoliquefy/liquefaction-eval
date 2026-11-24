# Grader kit: start in 10 minutes

The grader kit serves a blind web form for the 300-site label audit (audit design
`docs/evals/AUDIT-DESIGN-2026-11-19.md` in the pillar). This page starts you on the synthetic demo set.
The demo points lie offshore. They are not real sites and carry no real label.

```
 site list (local) -> blind queue -> tiles per item -> packet folder -> form on 127.0.0.1 -> grades/<you>/*.jsonl
   queue id, x, y,     every other    imagery 70 m,    item.json +     you grade one          one line per save,
   earthquake only     field dropped  LiDAR change     PNG tiles       item at a time         latest wins
```

## Start (about 10 minutes)

You need Python 3.11 or later with `numpy`, `pillow`, `pyproj` and `matplotlib`. The kit repo venv has them.

1. Open a terminal in the kit repo root (the folder that holds `scoreboard/` and `.venv/`).
2. Run the demo with the kit venv Python (a system Python may lack `pyproj` or `matplotlib`):

   ```
   .venv/bin/python -m scoreboard.evalkit.graderkit demo --out graderkit-demo
   ```

   It builds 7 demo items in about 5 seconds, then prints `grader form at http://127.0.0.1:8765/`.
3. Open http://127.0.0.1:8765/ in a browser.
4. Press "Read the rule". Read the one page (also in `RULE.md` here). It takes about 5 minutes.
5. Type a grader name, for example `grader_a`, and press "Start". The first start of a name binds it to a token.
   The browser keeps the token. Nobody else can then use that name.
6. For each item, look at the tiles, fill in the form, and press "Save and next".
7. When every item has a grade, press "Lock all". You cannot change grades after this.
8. Stop the server with Ctrl+C. Your grades are in `graderkit-demo/grades/grader_a/`.

To come back later, run step 2 again with the same `--out` folder and type the same name in the same browser.
The form resumes. In another browser, paste the token when the form asks; it is in `grades/<name>/TOKEN`.

## What the form shows and hides

- It shows a queue id, the earthquake, the date window, the point, and the tiles with their attribution.
- It hides every model output, score band, stratum, weight, CPT name and label. The site list may hold them; the
  blind step keeps only `queue_id`, `x`, `y` and `earthquake`. The build then scans the packet and stops if a
  forbidden field name or an inventory source name appears.
- Known-answer seed items look exactly like the others.
- The server binds to 127.0.0.1 only. A grader name is bound to a token at its first start. Reading, saving
  and locking grades all need that token, so one grader cannot read or lock another grader's grades by typing
  the other name. Every save must be JSON from the form's own origin, so another web page cannot write grades.
- Each LiDAR caption gives a quality rating for that earthquake (good, caution, unusable, unchecked). It also
  flags a post grid flown after the window end (Dec 2011).

## For the audit custodian (real packets, local only)

Site lists, tiles and grades derived from NZGD data stay on this machine (NZGD terms clause 6.2). Never commit them.

1. Build the packet from the frozen draw:

   ```
   .venv/bin/python -m scoreboard.evalkit.graderkit build \
     --sites ~/.geoliquefy-runs/audit/audit-sample-2026-11-19-r2.json \
     --seeds ~/.geoliquefy-runs/audit/audit-seeds-2026-11-19-r2.json \
     --sources scoreboard/evalkit/graderkit/sources.pillar.toml \
     --base "<path to your data root>" \
     --out ~/.geoliquefy-runs/audit/packets-r2
   ```

   Run it through `ops/heavy_run.py` (it reads the large DEM grids). Add `--only 5` for a trial.
2. Check the packet: `.venv/bin/python -m scoreboard.evalkit.graderkit check --packet ~/.geoliquefy-runs/audit/packets-r2`.
3. Serve it: `.venv/bin/python -m scoreboard.evalkit.graderkit serve --packet ~/.geoliquefy-runs/audit/packets-r2 --grades ~/.geoliquefy-runs/audit/grades`.
4. Recompute the hours with the timer from a pilot: `.venv/bin/python -m scoreboard.evalkit.graderkit power --grades <grades folder>`.

`manifest.json` in the packet says `share_rule: local` when any tile comes from an NZGD source. A packet marked
`local` goes to an outside grader only after the founder rules on it (see the pillar note
`docs/evals/GRADER-KIT-2026-11-23.md`). LINZ CC BY 4.0 sources are `open`: they may go out with their attribution.

## Files

| File | Job |
|---|---|
| `RULE.md` | the grading rule in one page |
| `blind.py` | blind queue and the packet leak check |
| `sources.py`, `sources.pillar.toml` | source registry and licence check; the sources on disk |
| `geo.py`, `tiles.py` | GeoTIFF reader (PIL only), imagery and LiDAR change tiles |
| `build.py` | packet folder |
| `serve.py`, `form/index.html` | local JSON writer and the static form |
| `power.py` | grader hours, power per earthquake, smaller-sample proposal, second-phase subsample |
| `demo.py`, `demo/` | synthetic demo set |

Tests: `tests/test_evalkit_graderkit.py` (synthetic only).
