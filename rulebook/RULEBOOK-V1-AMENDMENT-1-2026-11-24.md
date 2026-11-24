G1+G5+G6 (eval M5): rule book v1 amendment 1, ruling 52c. Pins clip, tile_m and seed per card version in R2.

# Liquefaction model eval: rule book v1, amendment 1 (R2: clip, tile_m, seed)

This file amends rule R2 of `docs/evals/RULEBOOK-V1-DRAFT-2026-11-15.md` (revision 4). That file is not edited. It is
the first rule book amendment. It is committed alone, as one file. It is frozen once committed. The kit does not
change: it stays at main `a2637b2` (tag `evalkit-v0.6.0`). No card, card hash or task hash moves. No em dashes.

```
 RULE BOOK R2 (draft, 2026-11-15)          AMENDMENT 1 (this file)
 "It names no R27 parameter"         -->   a task names clip, tile_m and seed;
                                           each value must equal the pinned value of its card version
 R9 check: "a task naming a clip     -->   reads: a task naming a clip other than the pinned clip
 is refused (R2)"                          is refused (R2)
 kit spec.Task at a2637b2                  unchanged: requires clip, tile_m, seed
 sealed cards v2, v3, v3.1                 unchanged: every task holds clip 1e-6, tile_m 4000.0, seed 20261106
```

## 1. The ruling

Founder ruling 52 (c), `GOALS.md`, 2026-11-24, quoted:

> (c) Rule book: a dated amendment pins clip, tile_m and seed per version in R2 at the values the kit uses now.
> It is committed alone. The kit does not change.

## 2. Why

1. The D5 second outsider dry run (`docs/evals/D5-OUTSIDER-DRY-RUN-2-2026-11-23.md`, point 8) found a conflict. R2
   says a task names no R27 parameter. R9 says "A task file naming a clip is refused (R2)". The kit refuses a task
   file without `clip`, `tile_m` and `seed`.
2. D5B review 1 (`docs/evals/D5B-REVIEW-1-2026-11-24.md`, point 8 and founder question Q1) confirmed it. Its
   recommended answer: change the rule book, not the kit. Removing the fields from `spec.Task` would change every
   task hash, registered ones included.
3. The wave 5 record (`docs/EVAL-WAVE5-2026-11-24.md`, section 8, item 5) carried the question to the founder.
   Ruling 52 (c) answers it.

## 3. Amended R2

### 3.1 Text replaced

R2, **Rule**, first bullet, last sentence. The draft reads:

> It names no R27 parameter; the scorer reads those from the named version.

### 3.2 Text that applies from this amendment

> It also names `clip`, `tile_m` and `seed`, by those names (the kit's `spec.Task` fields). Each value must equal the
> value pinned for the task's card version (rule book v1 amendment 1, section 4). A task cannot choose them. It names
> no other R27 parameter; the scorer reads those from the named version.

R2, **Check**, gets one more sentence:

> The scorer refuses a task whose `clip`, `tile_m` or `seed` differs from the pinned value of its card version.

### 3.3 Effect on the R9 check sentence

The R9 text is not edited. Its check sentence, "A task file naming a clip is refused (R2)", points to R2. Under
amended R2 it refuses a task file whose clip differs from the pinned clip. A task file without a clip is refused as
well (the kit requires the field). R9's aim stays: a task cannot choose its clip.

## 4. Pinned values per card version

`clip` is the main clip. It gives the words and the calibration fits (R9). Every card also prints the other clips of
`metrics.CLIPS` (1e-6, 1e-4, 1e-3, 1e-2) beside. `tile_m` is in metres. `seed` is the integer given to the tile block
bootstrap.

| Card version | clip | tile_m | seed | tile_rule | Card hash (unchanged) |
|---|---|---|---|---|---|
| v2 (`docs/scorecard-v2-2026-11-19/`, sealed) | 1e-6 | 4000.0 | 20261106 | `fixed` | cc9f7b744d6bd6b3b4d5ce0956f04e3a17539ef9014de0277223033c08c8847d |
| v3 (`docs/scorecard-v3-2026-11-23/`, sealed) | 1e-6 | 4000.0 | 20261106 | `fixed` | c810ef8b4ebba74a407c78ace8b45fdb37a174a3f45eadde14170ca84563342c |
| v3.1 (`docs/scorecard-v3.1-2026-11-23/`, sealed) | 1e-6 | 4000.0 | 20261106 | `fixed` | 2ca66984e8262e9ab79e9c32fc7f1b604053f515bf3af3ad310815f7007b96f3 |
| every later card version, until a new rule book amendment changes this table | 1e-6 | 4000.0 | 20261106 | `fixed` | (none yet) |

Evidence (read 2026-11-24 from the sealed `scorecard.json` files; counts only): every task block holds exactly
these four values and `n_boot` 2000. v2 has 14 task blocks, v3 has 12 and v3.1 has 12. No task block holds another
value.

## 5. Where the code sets and takes these values

### 5.1 Where each card version sets them (pillar, sealed builders)

| Card version | clip and seed | tile_m | Task built |
|---|---|---|---|
| v2 | `docs/scorecard-v1-2026-11-15/build_v1.py` line 39: `SEED, N_BOOT, CLIP = 20261106, 2000, 1e-6`; read through `docs/evals/effort_baseline.py` (module `B`, line 54) by `docs/scorecard-v2-2026-11-19/build_v2.py` line 76 | `build_v2.py` line 43: `TILE_M = 4000.0` | `build_v2.py` lines 196 to 200, `tile_rule="fixed"` |
| v3 | same `build_v1.py` line 39, read by `docs/scorecard-v3-2026-11-23/build_v3.py` line 383 | `build_v3.py` line 60: `TILE_M = 4000.0` | `build_v3.py` lines 493 to 498; `card_version="v3"` at line 519 |
| v3.1 | same `build_v1.py` line 39, read by `docs/scorecard-v3.1-2026-11-23/build_v31.py` line 418 | `build_v31.py` line 67: `TILE_M = 4000.0` | `build_v31.py` lines 556 to 561; `card_version="v3"` at line 583 |

The kit holds no default for these three fields. Each card builder passes them into `spec.Task`.

### 5.2 Where the kit takes and checks them (kit `repos/liquefaction-model` at `a2637b2`)

| What | File and line at a2637b2 |
|---|---|
| The three required fields `tile_m: float`, `clip: float`, `seed: int` | `scoreboard/evalkit/spec.py` lines 220 to 222 (class `Task`, line 199) |
| `tile_m` range 100 to 100000 m | `scoreboard/evalkit/spec.py` line 248 |
| `clip` must be one of `CLIPS` | `scoreboard/evalkit/spec.py` lines 249 to 250 |
| `seed` must be an integer, 0 or more | `scoreboard/evalkit/spec.py` line 251 |
| `CLIPS = (1e-6, 1e-4, 1e-3, 1e-2)` | `scoreboard/evalkit/metrics.py` line 25 |
| v3 card version: refuses any task that is not `tile_rule` `fixed` with `tile_m` 4000 (`V3_TILE`) | `scoreboard/evalkit/interval_policy.py` line 65 (`V3_TILE`), lines 68 to 76 (`check_task`); called from `scoreboard/evalkit/card.py` line 159 |
| `seed` starts the tile block bootstrap: `np.random.default_rng(seed)` | `scoreboard/evalkit/resample.py` line 33 (function `bootstrap`, line 26) |
| Card versions the kit knows: `("v2", "v3")`; v3.1 builds as `"v3"` | `scoreboard/evalkit/interval_policy.py` line 32 |

### 5.3 What the kit checks today, and what it does not

1. The kit checks that each field is present and in range.
2. For a v3 card version, the kit refuses a `tile_m` other than 4000.0 and a `tile_rule` other than `fixed`.
3. The kit does not check that `clip` is 1e-6 or that `seed` is 20261106. A v2 card version has no tile check.
4. Until a kit change adds that check, the pinned values are checked by reading the `task` block that every card
   prints. A kit check needs a separate go, because ruling 52 (c) says the kit does not change.

## 6. Beside: the world blocks (not R2 tasks)

The world blocks of v2, v3 and v3.1 are built by the v2 world path (`build_v2.py` `world_population`; v3 and v3.1
call it as `V2.world_population`, v3 registration amendment 1, item A4). They are not R2 tasks and build no
`spec.Task`. They print AUC only, so they use no clip. They use local 1 km tiles (`build_v2.py` line 44:
`W_TILE_M = 1000.0`, used at line 448) and the same seed 20261106 (`build_v2.py` line 473). This amendment records
those values. It does not make them task fields.

## 7. What this amendment does not change

1. The draft file `docs/evals/RULEBOOK-V1-DRAFT-2026-11-15.md` is not edited. No other rule text is edited.
2. The kit does not change. Its README note (`scoreboard/evalkit/README.md` line 121) still says the conflict waits
   for the founder. Updating that note is a kit change, so it is left.
3. No card is rebuilt. No card hash and no task hash moves.
4. Two gaps stay open for the founder. This amendment states them and does not resolve them:
   - R27 (Bootstrap row) sets the seed from a Bitcoin block hash. The pinned seed is the fixed integer 20261106.
   - R11 c sets the main tile size from a residual semivariogram. The pinned tile is a fixed 4 km tile. For v3 and
     later this follows ruling 47 and C4 (the interval C4 measured is the plain fixed 4 km tile bootstrap).
5. Versions (R14). The rule book is a draft and is not adopted. So no rule book version number moves now. The
   amended R2 is part of the text that an adoption fixes.
