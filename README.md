# Liquefaction eval: rule book draft, scoring kit and the sealed v2 scorecard

This repository holds three things, read only. The first is the draft rule book for scoring liquefaction models
against observed ground failure. The second is the Python kit that applies those rules (`scoreboard/evalkit`). The
third is the sealed v2 scorecard, which the kit built on 2026-11-19. You can check the card without trusting us: the
command in "Check the card hash" recomputes its canonical hash from the card file alone.

There is no licence yet. All rights are reserved until GeoLiquefy names one. You may read the files and run the
tests. The repository takes no issues or pull requests.

## The tree has four parts

| Path | What it is | Source |
|---|---|---|
| `rulebook/RULEBOOK-V1-DRAFT-2026-11-15.md` | the rule book draft | unchanged |
| `rulebook/RULEBOOK-V1-AMENDMENT-1-2026-11-24.md` | amendment 1: per-version values of `clip`, `tile_m` and `seed` (rule R2) | unchanged |
| `scoreboard/` | the kit package `scoreboard.evalkit` and the 11 `scoreboard` modules it or its tests import | kit commit f77667a, tag `evalkit-v0.9.0` |
| `tests/` | the kit's `test_evalkit_*.py` files and their fixtures | kit commit f77667a |
| `docs/scorecard-v2-2026-11-19/` | the sealed v2 card: `SCORECARD.md`, `scorecard.json`, `REGISTRATION.md`, review, build and render scripts, `SEAL.txt` | unchanged |

## The mirror differs from the kit commit in eight named places

The kit repository is private. The mirror is refreshed at each tagged kit release. The first release held tag
`evalkit-v0.6.0`. This one holds `evalkit-v0.9.0`: a subset of kit commit f77667a. These are the only differences:

1. Two files of the sealed v2 folder are held back: `build_v2-attempt1.log` and `build_v2-attempt2.log`. They are
   build logs. They contain machine paths from the build computer. `SEAL.txt` still lists their sha256 values. The
   canonical card hash does not depend on them.
2. `requirements.txt` drops one line: an editable install of a private plotting-style package. The kit does not
   import it.
3. `container/` holds only the two lock files. The Dockerfile and its context scripts name private paths and are held
   back.
4. `scoreboard/evalkit/graderkit/sources.pillar.toml` is held back. It lists private data paths.
5. `scoreboard/evalkit/graderkit/README.md` line 60 reads `--base "<path to your data root>"` in place of a machine path.
6. `tests/test_evalkit_parity.py` and `tests/test_evalkit_rules_from_code.py` are held back. Both read private files.
7. `conftest.py` at the root is new. It skips 25 named tests that read private scripts or data lists. Each skip
   prints its reason. The list is in that file.
8. `README.md` and `NOTICE` are new.

## Run the tests

Python 3.14 is the tested version.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
```

Run from the repository root, without `-I`. On 2026-11-24 a clean environment with an empty home folder gave
833 passed, 28 skipped, 0 failed. The 28 skips are the 25 named in `conftest.py` and 3 tests that need data outside
this repository.

## Check the card hash

```
.venv/bin/python -c "import json; from scoreboard.evalkit import spec; c = json.load(open('docs/scorecard-v2-2026-11-19/scorecard.json')); c.pop('card_sha256'); print(spec.card_sha256(c))"
```

The printed value is `cc9f7b744d6bd6b3b4d5ce0956f04e3a17539ef9014de0277223033c08c8847d`. The same value is in
`scorecard.json` and `SEAL.txt`. The file hashes in `SEAL.txt` check with `shasum -a 256 -c` for every file present.

## What the repository does not hold

It holds no per-site data from the New Zealand Geotechnical Database (NZGD) or the Next Generation Liquefaction
database (NGL). NZGD terms (clause 6.2) allow aggregates only, so the card prints counts and scores, never site rows.
The coordinates in the test fixtures are public earthquake epicentres or synthetic grid nodes. The site ids in the
tests are planted bad values that the card scan must refuse.

Third-party data in the fixtures is credited in `NOTICE`.
