# evalkit: the eval rule book in code

G1, G5, G6 (ruling 42 goal B). `scoreboard/evalkit/` enforces the rules of
`docs/evals/RULEBOOK-V1-DRAFT-2026-11-15.md` (pillar root; DRAFT, not adopted) so that a card that breaks one is
refused or marked, not printed quietly. It is a library: no side effects, it reads only the files a caller names
and writes only the card a caller asks for. Pure numpy and scipy, plus imports from `scoreboard/scorer.py`,
`g5_amend.py` and `label_ceiling.py` (unchanged). Synthetic tests only. No em dashes.

```
 Task (R2) ----+                        +--> gates.run_gates (R12 a-c) must pass, or no card
 Entries (R3,R4,R6) --> spec: validate --+
 Labels (R7) --+        join by id       +--> card.build_card --> scan_aggregates (R17) --> card.json
                                              |  metrics (R9)  resample (R11)  baselines (R10)
                                              |  words + Holm + MDD, 1/2/4 km and clip words beside
                                              '  best entry: forward track only (R5), UNSEEN only (R6), not house (R16)
 cards of several earthquakes --> pool.card_item --> pool.pooled_line (R1): HOLDS / FRAGILE / CANNOT TELL / NOT ALLOWED
```

## Use

```python
from scoreboard.evalkit import build_card, load_task, load_labels, load_entry, write_card
card = build_card(load_task("task.json"), [load_entry("m.json", "m.csv")], load_labels("labels.csv"))
```

```
python -m scoreboard.evalkit gates
python -m scoreboard.evalkit score --task task.json --entries DIR --labels labels.csv --out card.json \
    [--shaking-only shaking.csv] [--registered-task-sha256 HEX]
python -m scoreboard.evalkit demo [--out card.json] [--n-boot N]
```

Files: an entry is `NAME.json` (manifest) plus `NAME.csv` (`site_id,p`). Labels are `site_id,label,x_m,y_m`
with label 1, 0 or blank (not labelled) and projected metres. Exit codes: 0 done, 1 gates failed, 2 refused.

## Entering and checking an entry (D5, eval M5)

This section answers the stuck points of the two outsider dry runs (pillar `docs/evals/D5-OUTSIDER-DRY-RUN-2026-11-23.md`
and `docs/evals/D5-OUTSIDER-DRY-RUN-2-2026-11-23.md`, reviewed in `docs/evals/D5B-REVIEW-1-2026-11-24.md`).

**Where the kit lives.** Repo `repos/liquefaction-model` (pillar root; GitHub `geoliquefy/liquefaction-model`, private
today), package `scoreboard/evalkit/`. Run every command from the repo root. Do not pass `-I`: isolated mode drops
the current directory from the import path, so `import scoreboard` fails. A fresh clone or worktree has no `.venv`.
The kit needs numpy and scipy only. Make an environment for it with the pins of `requirements.txt`:

```
python3 -m venv .venv
.venv/bin/pip install numpy==2.5.4 scipy==1.18.1
```

Every kit module imports in that environment, and `gates` runs. Do not use `pip install -r requirements.txt` for the
kit: that file also pins `geoliquefy-figstyle` from a private GitHub repo, so the full install fails for an outsider.
The kit does not need figstyle.

**Which kit a card used.** The kit has no version string of its own. A card names the kit by a git ref and a commit.
The sealed v2 card says "evalkit-2026-11-15 at 6638d75". `evalkit-2026-11-15` is a BRANCH on origin, not a tag (its
REGISTRATION.md section 2 says "branch"). Its head is 6638d75 today (`git ls-remote origin
refs/heads/evalkit-2026-11-15`), but a branch can move, so trust the commit, not the name. Commit 6638d75 is also
reachable from `origin/main`: `git checkout 6638d75` gives the v2 kit. Later kit releases carry tags
(`evalkit-v0.3.0` and on). `repos/liquefaction-model-evalkit` is a worktree on the organiser's machine, not a public
place. N_BOOT (2000) and SEED (20261106) of the v2 card are printed in its header
(`docs/scorecard-v2-2026-11-19/SCORECARD.md`, "Every interval").

**Two manifest forms.** `intake.read_entry` reads both and says which one it read.

| | Rule book form (B3) | Kit form |
|---|---|---|
| Known by | the field `rule_book_version` | no `rule_book_version` |
| Fields | every B3 field: `rule_book_version`, `entrant_id`, `task_sha256`, `model`, `model_version`, `primary`, `output_kind`, `conversion_sha256`, `division`, `package_sha256`, `inputs`, `exposure`, `training_cases_sha256`, `choices_after_outcomes`, `text_read`, `statement`, `house`. Only `conversion_sha256` and `package_sha256` may be null (`package_sha256` null on the open division, a hash on the closed). Extra fields are allowed | required `model`, `output_kind`, `statement`, `exposure`, `house`; optional `conversion_id`, `division`, `feature_allowlist`, `declared_target`; no other field |
| Values | every value in [0, 1], for every output kind (R4) | probabilities in [0, 1]; a `score` may take any finite value |
| Entry hash | the B3 recipe on the manifest exactly as submitted. The B3 example files hash to `913254e4...` (tested) | the same recipe on the manifest after the kit fills its defaults: `conversion_id` and `division` written as null when absent; `feature_allowlist` and `declared_target` left out when unset |
| Entrant | `entrant_id` in the manifest, inside the hash; a different `--entrant` is refused | the `--entrant` flag; NOT inside the hash |
| Conversion | `conversion_sha256` (mapped onto the kit's `conversion_id`) | `conversion_id` |

Both forms: entry hash = sha256 of (canonical manifest JSON, a newline, the predictions text), with canonical JSON
and the predictions text as in rule book D3. The kit form manifest fields are listed in `intake.KIT_REQUIRED` and
`intake.KIT_OPTIONAL`. To see exactly what was hashed:

```
python -m scoreboard.evalkit.register hash ENTRY.json [--csv ENTRY.csv]
```

It prints the form, the entry hash and `hashed_manifest`, the canonical manifest text that went into the hash. The
card builder (`card.build_card`) still hashes the kit form. A rule book form entry's receipt hash is not yet bound on a
card; that waits for the v3 card (B8).

**Submit, check against a task, verify.**

```
python -m scoreboard.evalkit.register submit --register R.jsonl ENTRY.json [--csv ENTRY.csv] [--task TASK.json] \
    [--entrant ID] [--receipt-out receipt.json]
python -m scoreboard.evalkit.register verify --register R.jsonl --receipt receipt.json [--entry ENTRY.json]
python -m scoreboard.evalkit.register check  --register R.jsonl
```

With `--task`, the intake refuses an entry whose site ids do not hash to the task's site list (one row per task site,
no other row), whose exposure names no exposure for the task earthquake, whose division differs from the task's, or
(rule book form) whose `task_sha256` differs from the task hash. Without `--task` the submit output says "task:
not checked". The intake checks that an exposure key is a non-empty string; it does not check that it is a ComCat id.

**Task file.** A task file is one JSON object (`spec.Task`). These 11 fields are required:

| Field | Value |
|---|---|
| `event_id` | non-empty text |
| `target` | non-empty text |
| `label_source` | non-empty text |
| `label_rule_sha256` | 64 lowercase hex characters |
| `site_list_sha256` | 64 lowercase hex characters: `spec.site_list_sha256`, the sha256 of the SORTED site ids, each followed by `\n` (so the text ends with `\n`). Row order in the entry CSV does not change this hash. It does change the entry hash: D3 hashes the rows in file order |
| `track` | `forward` or `record` |
| `division` | `closed` or `open` |
| `tile_m` | metres, 100 to 100000 |
| `clip` | one of 1e-6, 1e-4, 1e-3, 1e-2 (`metrics.CLIPS`) |
| `seed` | integer, 0 or more |
| `label_caveats` | non-empty text |

These 9 fields take a default when the file leaves them out: `n_boot` 2000, `min_sites` 200, `min_per_class` 30,
`min_tiles` 20, `census` false, `earlier_base_rate` null, `label_error` null (or `{"sensitivity": x,
"specificity": y}`), `tile_rule` `fixed` (or `variogram`), `earlier_label_sets` 0. Any other field is refused. An empty
task is refused with the list of the 11 missing fields. Rule book v1 R2 describes the task in prose, with other
names. R2 and R9 also say a task names no clip, but the kit requires `clip`, `tile_m` and `seed`. That conflict waits
for the founder (dry run 2 point 8). Until then, a kit task file carries all three.

**Task hash.** The task hash is not the hash of the task file's bytes. It is not the canonical JSON of the file as
written either. It is `spec.Task.sha256()`: sha256 of the canonical JSON of the task after the kit reads it. The kit
fills each optional field the file leaves out with its default (`n_boot`, `min_sites`, `min_per_class`, `min_tiles`,
`census`, `earlier_base_rate`, `label_error`, `tile_rule`, `earlier_label_sets`), so all 20 fields are hashed. It
writes `tile_m`, `clip` and (when set) `earlier_base_rate` as floats (`2000` becomes `2000.0`). The scorer checks
the same hash against a registration. To print the hash and the exact text hashed:

```
python -m scoreboard.evalkit.register hash --task TASK.json [ENTRY.json]
```

A refused `task_sha256` check prints the hash the manifest gave and the task hash it needed. Rule book v1 R2 does not
define this hash yet; that waits for the founder.

**Register line hash.** `line_sha256` = sha256 of the UTF-8 bytes of the line's canonical JSON text, without its
trailing newline. Canonical JSON = `json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
allow_nan=False)`. Line 1 has `prev_sha256` = 64 zeros; every later line has the sha256 of the line before.

**What a receipt proves today.** No public register, intake address, mirror or organiser key exists yet. The
receipt is not signed. A receipt from `register submit` proves intake only against the register file it was written
to. The submit output says so (`receipt_signed` false, `register_public` false). No task is open, so there is no
real site list to predict on.

**Rebuilding a card number.** The sealed v2 card (`docs/scorecard-v2-2026-11-19/`, pillar, frozen) prints no rebuild
class and no rebuild command. Its builder is `build_v2.py` in that folder (named in its REGISTRATION.md section 9).
It builds the whole card in one run; it cannot rebuild one number. It expects to run from an evalkit worktree two
levels under the pillar (`PILLAR = KIT.parents[1]`), reads a private exposure ledger under `~/.geoliquefy-runs/`, and
reads NZGD Canterbury data under restricted terms. The world cells come from `world_population()` on
`data/world-case-histories-2026-11-19/cases.parquet` and `traces.npz`, which never enter git and whose R2 copy is
private. So no per-earthquake v2 cell (an AUC, a log loss, a paired difference) can be rebuilt by an outsider from
public files. Under R13 those cells are at best TESTER REBUILD ONLY and read "not rebuilt outside the organiser". The
first D5 dry run rebuilt the 15 world "baseline: shaking only" AUC cells (point and interval) exactly, but only on the
organiser's machine.

The pooled lines are different. They are second-stage numbers: each one pools the card's own per-earthquake paired
differences d_i and their standard errors s_i. So they rebuild from `scorecard.json` alone, with the kit or by hand.
Each block under `pooled_lines.<line>` (for example `C1.primary_kit_random_effects` or `W1 P1.primary_equal_weight`)
holds `per_earthquake` (`event_id`, `diff`, `se`), `pooled` and `leave_one_out`. The `method` field names the
weighting:

- `hk`: rule book R1 (P1). DerSimonian-Laird tau^2, weights 1 / (s_i^2 + tau^2), Hartung-Knapp variance, Student t
  on k - 1 degrees of freedom.
- `equal`: equal weight per earthquake. Mean of d_i; variance sum (d_i - mean)^2 / (k (k - 1)); Student t on k - 1
  degrees of freedom. The rule book does not define it, and SCORECARD.md "Pooled lines" does not name it. It is kit
  option K1 in the sealed v2 `REGISTRATION.md` (section 2), the file SCORECARD.md names in its header.

Each fit records its own interval `level`. The pooled lines use 0.95. Each leave-one-out fit uses R1's 1 - alpha_j,
which is 0.975 on C2 and 0.95 elsewhere on the v2 card. That level is in the JSON only, not in SCORECARD.md. W2 lines
print NOT ALLOWED (fewer than 3 earthquakes) and have no fit. The kit call, from the repo root:

```python
import json
from scoreboard.evalkit import pool
card = json.load(open("PILLAR/docs/scorecard-v2-2026-11-19/scorecard.json"))
b = card["pooled_lines"]["C1"]["primary_kit_random_effects"]
d = [x["diff"] for x in b["per_earthquake"]]
s = [x["se"] for x in b["per_earthquake"]]
r = pool.random_effects(d, s, method=b["pooled"]["method"], level=b["pooled"]["level"])
# r["mean"], r["se"], r["lo"], r["hi"], r["p"] equal b["pooled"]; for a leave_one_out item, drop its
# left_out earthquake from d and s and pass that item's own method and level
```

On kit d38757a this rebuilds all 8 pooled lines and all 44 leave-one-out fits with a largest absolute difference of 0
(dry run 2 and its review 1 also did it by hand). These are second-stage numbers only: they do not rebuild any
per-earthquake cell.

## Rule to module to test

| Rule | What the kit does | Module | Tests (`tests/test_evalkit_*.py`) |
|---|---|---|---|
| R1 | pooled paired differences, DerSimonian-Laird tau^2 with a Hartung-Knapp t interval, at least 3 earthquakes past the gate, Holm across the pooled lines of a release, leave-one-out at 1 - alpha_j, words HOLDS / FRAGILE / POOLED CANNOT TELL / NOT ALLOWED, registered flag | `pool.py` | `pool`: all (HK formula, HK coverage calibration, Holm across lines); `card::test_card_item_reads_a_paired_difference` |
| R2 | task spec and hash; refuses a hash that differs from the registered one, and a label site list that differs from the task's hash | `spec.Task`, `card` | `spec::test_task_*`, `card::test_task_and_site_list_hashes_are_checked` |
| R3 | `output_kind`; only `target_probability`, or another kind with a `conversion_id`, gets probability metrics | `spec.probability_eligible`, `card` | `card::test_only_target_probabilities_*`, `card::test_kind_cannot_be_faked`, `metrics::test_triggering_probability_*` |
| R4 | canonical entry hash: values at 1e-9 through `g5_amend.canon_cell`, manifest bound in | `spec.entry_sha256` | `spec::test_entry_hash_matches_g5_*` |
| R5 | forward card names a best entry only if it separates after Holm; record card reads RETROSPECTIVE, names none | `card._best` | `card::test_forward_card_*`, `card::test_record_card_never_names_one` |
| R6 | exposure per earthquake required; mark printed (NOT CHECKED for `unknown`); only UNSEEN can be named | `spec`, `card` | `card::test_exposure_and_division_are_required`, `card::test_house_and_exposed_*` |
| R7 | labels 1/0/blank, duplicate ids stop the run, join by id only, caveats and rule hash printed | `spec.Labels`, `spec.align` | `spec::test_labels_*`, `card::test_join_is_by_site_id_only` |
| R8 | label hash, label error or NOT MEASURED, AUC ceiling from `label_ceiling.ceiling` | `card._label_block` | `card::test_label_error_and_ceiling` |
| R9 | AUC; log loss and Brier; CITL and slope, each also without exact 0 and 1; clips 1e-6 to 1e-2; zero shares; CORP curve and Brier decomposition; level miss on a census only | `metrics.py` | `metrics`: all; `card::test_level_miss_only_on_a_census`, `card::test_every_metric_carries_an_interval` |
| R10 | coin always; earlier-earthquake rate and shaking-only scored as entries; with `earlier_label_sets > 0` a missing one refuses the card (scorer fault); with none, NOT AVAILABLE; no pair of two baselines is tested | `card._rows`, `card._check_inputs` | `card::test_baselines_are_entries_on_every_card`, `card::test_a_required_baseline_that_is_missing_refuses_the_card` |
| R11 | tile bootstrap intervals on every metric, paired differences with 95% and 90% intervals, MDD, two Holm families (ranking, probability), words SEPARATES BETTER/WORSE, EQUIVALENT, CANNOT TELL, count gate (UNDERPOWERED, counts only), words at 1/2/4 km and every clip (TILE-SENSITIVE only at sizes with 20 or more tiles, CLIP-SENSITIVE), tile chosen from a residual variogram before scoring (`tile_rule` "variogram", refused on mismatch, range and variogram hash on the card) | `resample.py`, `card` | `resample`: all; `card::test_underpowered_*`, `card::test_words_are_holm_corrected_within_each_family`, `card::test_two_families_are_not_one`, `card::test_equivalent_*`, `card::test_tile_size_comes_from_the_variogram_before_scoring`, `card::test_sensitivity_flags_*` |
| R12 (a)-(c) | gates: planted entry found, null silent, defects caught (one-tile shuffle, all-label shuffle, join order, monotone copy, kind) | `gates.py` | `gates`: all; `card::test_failed_gates_block_every_card` |
| R15 | an entry stating another division is refused | `card` | `card::test_exposure_and_division_are_required` |
| R16 (b), (c), (f) | house marked and first, house losses print first, house never named best, "scored by the organiser" on every card | `card` | `card::test_house_*` |
| R3, R16 (b) (B1, eval M4) | a house `other_probability` entry may set `declared_target` in its manifest: the house stands behind the number as a probability of the task target, so Level 0 and M0 get log loss, Brier and calibration instead of ranking only (numbers unchanged; the entry list marks the row `DECLARED TARGET PROBABILITY (house)`; the flag is bound in the entry hash and omitted when unset). `build_card(..., losses_first=True)` (CLI `--losses-first`) puts a `losses` block first: one LOSS line per house row whose log loss at the card's clip or Brier is higher than a baseline's (coin, earlier-earthquake rate, shaking only), with both scores and the paired log loss interval; house rows with no proper score are listed under `no_proper_score`; a tie prints nothing. Point estimates, not a word. `require_house_proper_score=True` (CLI `--require-house-proper-score`) refuses a card with a house `other_probability` row that has no declaration and no conversion. Correction 8: both LSN fragility rows (`losses.PUBLISHED_OURS`, any chain id) carry one mark, and their LOSS lines read "loss of our run of a published model (Maurer, engrXiv 121, Table 3)", never a house loss; B2 (`outside.py`) owns the matching pattern. All off by default: older cards, the sealed v2 card among them, keep their bytes. `losses.loss_block(card)` reads an existing card | `losses.py`, `spec`, `card` | `losses`: all |
| R17 | output scan refuses arrays, lists over 32 items, `nzgd:`/`rer:`/`usgs:` strings and any known site id | `card.scan_aggregates` | `card::test_scan_refuses_*`, `card::test_site_id_shaped_names_*` |
| R18 | the five "does not claim" lines on every card (citations of prior work are not) | `card.CLAIMS` | (printed only) |
| R25 (D6, eval M1) | sequences: `attribution_window` (previous survey end, else origin minus 30 days, to this survey's end; read (start, end]; with the 30-day fallback, `sequence_shock_list` re-anchors the window on the first shock until it is stable, a kit reading, and `plan_tasks` refuses a fallback window whose first shock is before its anchor); `shock_list` from ComCat rows only, shocks sorted by time: every earthquake in the window with census magnitude >= 5.0 (D6: largest Mw-family, not mwp) whose hypocentral radius min(200, 30 x 10^(0.5 (Mw - 6))) km covers a task site; a summary-CSV Mw-family magnitude is marked a lower bound, a detail record (`census_from_detail`) wins; a row with a non-Mw preferred magnitude >= 5.0 is UNRESOLVED and `plan_tasks` refuses until the caller resolves it; complete only for events whose ComCat preferred magnitude is >= 5.0 (the on-disk source is a M >= 5 query; `unresolved_from` flags rows below 5.0 when a broader catalogue is given); `plan_tasks`: SINGLE, PER SHOCK (source attributes, S3.2) or SEQUENCE (first shock's origin, target "manifestation seen after any shock in the window"); `after_earlier_shock` prints the mark; `combine_runs` is the old two_shock p = max (scripts/run_g6b_p3v2.py `p_of`, copied as `p_max_over_shocks` with a parity test against the script's source; the script stays as the registered source, unchanged; its `gm_o` variant has no kit counterpart); `cross_check` matches GeoNet and lists GeoNet-only M >= 5 rows as notes. Not wired into `card.py` yet: a card caller prints `TaskPlan.summary()`. Fixture cut from the public catalogues by `scripts/cut_sequence_fixture.py` | `sequences.py` | `sequences`: all |
| (review) | duplicate detector: one model counted once | `gates.duplicate_pairs` | `gates::test_duplicate_*`, `card::test_duplicates_*`, `card::test_a_monotone_copy_*` |
| (step 1c) | "baseline: effort only": pre-event effort features only (`EFFORT_FEATURES`: LCDB 2008 built-up and tree share, aerial footprint, CPTs pushed before 2010-09-04; anything else refused); exposure mark SEEN PLACE (other earthquakes' labels at the same sites); leave one earthquake out (`fit_predict_loeo` drops the scored labels first); fixed L2 logistic, lambda 1.0; `stratified_auc`; `build_card(..., effort_only=effort.entry(...))` adds a ranking-only baseline row. The 2d1bfc6 set (post-event CPT density, pushed-after) is `PROXY_FEATURES`, "diagnostic: investigation-density proxy": an outcome descendant, never a baseline; the card refuses it. Registrations `docs/evals/EFFORT-BASELINE-PREEVENT-REGISTRATION-2026-11-19.md` and 2d1bfc6 (pillar root) | `effort.py`, `card` | `effort`: all |
| (B2, eval M4) | outside-model rows (not house and not a baseline, plus LSN + CES fragility and LSN + global fragility under an `[authors-...]` chain id) carry `author_confirmed` under the card option `build_card(..., author_confirmed={name: bool})`; a missing name is unconfirmed; the two LSN fragility rows always get the same mark, since both are our run of a published model (Maurer, engrXiv 121, Table 3, which our chain fails parity with; ruling 45 correction 8): the match keys on the chain id only, a map that confirms one without its twin is refused, and the entry carries a `published_run` note; an unconfirmed row prints NOT RUN BY AUTHOR SETTINGS and enters no metric, comparison, Holm family, best or duplicate record; `check_card` refuses a card that prints its score (`build_card`, `descriptive`, `write_card` call it). Off by default, so the sealed v2 card is unchanged. `zhu_cells.read_cells` reads a Zhu 2017 grid with empty or outside cells marked OFF-GRID (NaN, never 0); `drop_off_grid` drops them from that cell and `drop_note` prints the count; `cell_input` is the one-call hook for a card builder (scorecard v2 deviation 3; which masks count as off-grid awaits a founder ruling before v3) | `outside.py`, `zhu_cells.py`, `card` | `outside`: all |
| (B4, eval M4) | continuous shaking baselines, replacing the binned and tied shaking-only row: "baseline: shaking score" = PGA / MSF(M) (MSF form `MSF-M-ONLY-1.3`: Boulanger and Idriss 2014, UCD/CGM-14/01, eq. 2.19 on page 14, with MSFmax fixed at 1.3 for every site, as `g5_entry.MSFMAX`; CPT-free, the MSF depends on M alone), output kind score, ranking only, no fit, exposure UNSEEN, on any task; "baseline: shaking only" = expit(a + b ln(PGA / MSF(M))), a and b fitted by maximum likelihood on the pooled rows of earlier earthquakes only (each with its own M), refused if the fit set names the scored earthquake; "baseline: shaking only (binned, old)" = the scorecard v1/v2 bins, kept beside for one release. Correction B4 (founder check 2026-11-23): the B&I 2014 MSF as published needs qc1Ncs, because eq. 2.20 (page 14) sets MSFmax = 1.09 + (qc1Ncs / 180)^3, capped at 2.2; a shaking-only baseline needs a CPT-free form, so MSFmax is one fixed number (1.3, which eq. 2.20 gives at qc1Ncs about 107), an array of per-site values is refused, every row statement and fit record names the form (`shaking.MSF_FORM`, `shaking.MSF_SOURCE`), and another fixed MSFmax prints as a VARIANT. The v3 registration (B8) quotes `MSF_SOURCE`. Earlier B4 text cited eq. 2.17 and 2.18; that was wrong (eq. 2.17, page 12, is Idriss 1999, MSF = 6.9 exp(-M/4) - 0.058 <= 1.8, CPT-free too but not used). No bins means no ties of their own; ties come only from equal input PGA, counted by `tie_count` and never broken. Review 1: the fit is refused (never kept) with fewer than 10 earlier yes or 10 earlier no, separation by shaking alone, no convergence in 50 Newton steps, or slope b <= 0; the row statement carries a, b, MSFmax, the earlier ids and magnitudes, n_rows and n_yes. With the old row beside, the card requires the continuous fit in the shaking-only slot (name plus statement marker) and refuses an old row equal to it; without the old row the slot still takes any target probability (CLI CSV path, demo), so full enforcement starts with scorecard v3 (B8). `build_card(..., shaking_only=probability_entry(...)[0], shaking_score=score_entry(...), shaking_binned=binned_entry(...)[0])` | `shaking.py`, `card` | `shaking`: all |
| (B3) | "baseline: site memory" (did this site liquefy last time?): each task site takes the label of the same site id, or else the nearest labelled site within 20 m (ties to the smaller id), in the most recent earlier earthquake. The 20 m is not settled: pillar ruling 16 set 20 m for the twin test and the display, and its use here is a new choice that the v3 registration (B8) must fix before any real run (correction B3, founder check 2026-11-23). On the same label source (which the card enforces) the row also measures label persistence, not only ground behaviour: a label error at a place can repeat there, and aerial specificity is 0.47 [0.30, 0.63] in Feb 2011 (pillar `docs/evals/LABEL-ERROR-RATES-2026-11-19.md`). The row statement, `site_memory.limits`, `site_memory.label_persistence`, `site_memory.radius_status` and the card notes all say both. Built from `Sites` (ids and coordinates, never the scored labels); `load_history` drops the scored label cells on read, refuses duplicate (event, site) rows in any earthquake, and skips later earthquakes. Every `Earlier` carries its `label_source`; a mix is refused. Two earlier earthquakes: probability version, the memory earthquake's own yes shares by its memory in the earthquake before it (memory yes, memory no, no memory), Jeffreys (k + 0.5) / (n + 1), a cell under 20 sites takes the Jeffreys estimate over all labelled sites of the memory earthquake. One earlier earthquake: ranking-only score (yes 1, no 0, no memory = the yes share). No-memory sites are counted (`n_no_memory`) and kept. Exposure SEEN PLACE. `build_card(..., site_memory=site_memory.build(...))` adds the row and prints rule, counts, rates and limits under `site_memory`; the slot takes only an object `build` returned (a weak provenance registry checks the object, its entry hash and its summary; a hand-built SiteMemory with the right name and allowlist is refused), for the task earthquake and the task label source, never with more earlier earthquakes than the task names (the probability version needs two). The row joins both Holm families (paired AUC for every entry, paired log loss for eligible ones), so it changes `holm_family_sizes` and can change other entries' adjusted words: B8 (scorecard v3) must register whether site memory is passed. Canterbury is not scored here (wave 3, under a registration) | `site_memory.py`, `card` | `site_memory`: all |
| R23 tooling (D7, eval M6) | sealed custody for an independent tester: labels and entries are encrypted to the tester's certificate (OpenSSL CMS, RSA-OAEP-SHA256 key transport, AES-256-GCM; no crypto written here), so the organiser holds sealed files but no key that opens them, and the tester role is one private key away. `custody.json` is written once and pins the tester certificate's SHA-256 and the freeze time; its own SHA-256 sits in the first log line, and every seal, open and verify refuses an edited record. Sealing to another certificate, sealing an entry at or after the freeze, and replacing a sealed object are refused; each seal line names the recipient certificate. Every seal and every open attempt (opened, refused before the freeze, refused for a swapped file, failed to decrypt) writes one line to a hash-chained `custody_log.jsonl` (actor, time, object, key id, hashes; never content) before any plaintext returns; a broken chain refuses every new line. The lines are unkeyed: a rewrite that recomputes every later hash, or a cut tail, still verifies until the head is pinned outside the custody folder (committed or stamped) and checked with `verify --expected-head`. The seal line's plaintext hash lets a reader confirm a guessed file for a very small label set. The freeze check is policy; the key is the barrier. Custody adds nothing for label sets we built and already hold or read. CLI `python -m scoreboard.evalkit.custody keygen|init|seal|open|verify`. R23 itself (a named tester, acceptance, pledge) stays a rule that waits for people (correction 3) | `custody.py` | `custody`: all |
| R4, B3 (D5, eval M5) | intake in two manifest forms: a rule book form manifest (B3, known by `rule_book_version`) is checked field by field (only `conversion_sha256` and `package_sha256` may be null; extra fields allowed and hashed; values in [0, 1] for every kind) and hashed exactly as submitted by the B3 recipe; the entrant is its `entrant_id`. A kit form manifest keeps its pre-D5 hash. `register hash` prints the form, the hash and the manifest text hashed; `register submit --task` checks site list, exposure key, division and task hash. See "Entering and checking an entry" above | `intake.py`, `register.py` | `intake`: all |
| R20 (D3, eval M6) | register and receipt: `python -m scoreboard.evalkit.register submit --register R.jsonl NAME.json` checks the entry through the kit's entry spec (`spec.load_entry`), takes its R4 hash (`spec.entry_sha256`), appends one canonical JSON line to an append-only JSONL register (seq, intake time UTC, item type, item sha256, previous line's sha256, entrant) under a file lock, and prints a receipt (entry hash, line hash, previous-line hash, intake time, seq, entrant). `verify --receipt` checks a receipt against the register (changed, removed, renumbered line, entry files that hash differently: REGISTER FAULT, exit 1); `check` reads the whole chain (seq gaps, broken links, non-canonical lines, a cut last line, intake time that decreases). The intake refuses a broken register and a clock earlier than the head. `--stamp` writes the head to `R.jsonl.heads/` and runs `ots stamp` on it: a publication, founder go only; tests mock it. A stamp that fails after the append still prints the receipt, with `stamp.stamped` false and the error (exit 1, not REFUSED). Not built: the organiser's receipt signature, the public mirror, block time from upgraded proofs; a register line for a refused intake (R20 "prints REFUSED"); receipts appended as register lines; a hash-first intake on the command line (only `append_item` takes a bare hash). One line with a wrong far-future clock blocks later intakes; the recovery is an R14 erratum and a new register that records the old head. Lines removed after the newest receipt are found only by a stamped head | `register.py` | `register`: all |
| R21 (D4, eval M6) | forward runner: `flagged_events` reads the G5 watcher state.json (read only; CANDIDATE, ALERT, ALERT UNKNOWN or CLASS B); `freeze_inputs` copies the ShakeMap grid and the CPT list into the run folder, keeps sites inside the grid, writes an input manifest and logs its sha256 in an append-only, hash-chained `run_log.jsonl`; `publish_input_hash` writes the anchor text and returns the OpenTimestamps command, run only with `run=True` and a founder go (never in the kit); `freeze_inputs` refuses a record the watcher did not flag unless an override reason is given (logged), and freezes the register prefix (its byte length `n_bytes` and sha256) in the manifest; `predict` runs every counted entry of a D3-shape register (builtin `shaking_score`, `pga_logistic`, or a `package.module:function` hook whose module file hash `code_sha256` is checked before import) on the frozen inputs, takes no labels, refuses once any label file was opened, refuses a register whose frozen prefix changed (lines appended after the freeze are not read: D3 is append only), and logs a predictions sha256. An entry counts only if its intake time is at least 24 h before the origin; a LATE entry is not run and is listed. Each entry runs twice; a second run that differs by more than 1e-9 is RUN FAILED. Types A and B must run the same counted entries. A register written by D3 `evalkit.register` holds hashes only and is read only after D3 `check_register` passes: a recipe counts when an `entry` line's hash equals its `recipe_sha256` (put on by `submit_recipe`: `check_recipe`, then D3 `append_item`, then the D3 receipt; not `submit_entry`, whose hash is the R4 hash of a manifest plus predictions); other entry lines are skipped, a recipe on a line after the frozen prefix is skipped, and a recipe file whose hash is on no line is refused. `score` refuses a type already scored (a card is written once, never replaced), re-checks log, manifest and every prediction file, and only then opens the label file, logs its hash (a refused open too), refuses a label file other than the first one opened in the run, and builds one card per input type with a `forward` block. Type A (frozen inputs, first ShakeMap version) and type B (final inputs) are separate cards, never pooled, and must score the same label file. `scripts/forward_replay.py` replays Feb 2011 (usp000huvq) end to end with a PLANTED label file (corrections 1 and 6), output local only. NOT enforced from R21 (a done D4 does not mean R21 is enforced): OCI images, digests and environment hashes; the sandbox (no network, read-only inputs, 8 CPU, 32 GB, 12 h); the 48 h output limit and the label join only after it; the ShakeMap choice (R21: first version updated at least 24 h after the origin; type A takes the version given, and the manifest records only whether it meets the 24 h rule); the census read at origin plus 7 days; standing items, primaries, retirement, runner and tester rules; a D3 item type for forward recipes | `forward.py`, `scripts/forward_replay.py` | `forward`: all (the blind order is enforced with a Python audit hook on file opens); `d34_joint`: all (real D3 intake, receipt and verify with the runner, blind, types A and B apart, three seams) |
| (B5, eval M4) | two optional baselines, off by default (`build_card(..., pgv=None, geo_null=None)` keeps the default card bytes). "baseline: pgv score" = ShakeMap PGV (cm/s, bilinear on the grid), ranking only; "baseline: pgv only" = expit(a + b ln PGV) fitted on earlier earthquakes only with the B4 guards. No fill: a site off the grid or next to an empty node, a grid with no PGV field, no grid, or an earlier earthquake with a missing PGV gives a NOT AVAILABLE object; the card adds no row and the `pgv` block and a note say why. `read_grid` reads grid.xml or a zip, records sha256 and ShakeMap version, refuses Kaikoura (us1000778i, ruling 45 correction 6). The task id (for example feb2011) and the grid ComCat id may differ: pass `shakemap_id`; both ids print. "baseline: geographic null" = minus the distance to the nearest labelled yes site of the earlier earthquakes, each dated strictly before the `scored_date` that `geo_null.build` takes (the eval design R2.2 item 2, Hijmans 2012, quoted in `geo_null.RULE_QUOTE`); exposure SEEN PLACE; the `geographic_null` block adds calibrated AUC = AUC - null AUC + 0.5 per entry with 0.5 plus the paired AUC difference interval. `geo_null.matched_sites` (matched-distance test set, TR 0.33) is built but not wired. Interpretation and TR are not settled: the v3 registration (B8) fixes them. Both rows join the Holm families when passed, so B8 must register whether they are passed. Canterbury is not scored here | `pgv.py`, `geo_null.py`, `card` | `b5_baselines`: all |
| (B6, eval M4) | breakdowns the field reads, as card tables that name no winner: by PGA bin (g; edges 0.1 to 0.5), by Ic class (Robertson SBTn zone edges 1.31, 2.05, 2.60, 2.95, 3.60), by crust thickness (m; edges 1, 2, 3, 5) and by severity class ("any ejecta" = class 1 or worse, "moderate or worse" = class `moderate_at` or worse, default 2). Each splitting value is a `SiteValues` (one value per site, nan = no value, joined by site id only, with a statement). Every cell prints n; a cell under `floor_n` (30) sites or `floor_per_class` (5) yes or no sites prints "too few" and nothing else. A shown cell prints yes and no counts, tiles, the observed share and each scored row's AUC (plus log loss at the card clip, Brier and mean p for probability rows), each with a 95% tile-bootstrap interval inside the cell. Severity cells are AUC only (R3). Rows keep the card order; `check_no_winner` refuses a best, rank, word, favours or diff key. `build_card(..., breakdowns=breakdowns.Breakdowns(...))` adds the `breakdowns` section; off by default, so older cards (the sealed v2 card among them) keep their bytes. Correction 1: synthetic and planted data only until the v3 registration (B8) is committed alone | `breakdowns.py`, `card` | `breakdowns`: all |
| (B7, eval M4) | Placement control, off by default. (a) `placement.build(site_ids, x, y, cpt_x, cpt_y, basis=..., proxy=None)` puts every task site (labelled or not; no label read) in a CPT-density stratum: CPTs per km2 within 1000 m, own record within 2 m left out, cut at the 1/3 and 2/3 quantiles over all task sites (a site at a cut goes to the lower stratum; empty strata dropped). `basis` must be named: `pre_event` (the clean control) or `all_index` (descends from the outcome; can explain the proxy, can also remove true skill). Review 1: `pre_event` needs `cpt_dates` and `first_event_date`; `build` keeps only CPTs pushed strictly before that date, leaves undated CPTs out, and prints both counts. `build_card(..., placement=...)` adds a `placement` block: for every non-constant row, AUC overall, the pooled within-strata AUC (weights n_yes,k n_no,k over the sum, as `effort.stratified_auc`) and their paired difference on the card's tile bootstrap; with the investigation-density proxy supplied (allowlist `proxy`, diagnostic only) it prints EXPLAINED BY DENSITY STRATA (90% interval of the within-strata proxy AUC inside 0.5 +/- 0.05), OPEN BIAS (95% interval excludes 0.5; size = within-strata AUC - 0.5 with its interval) or CANNOT TELL, and the share of the proxy's lift the strata remove (NaN with a note when the lift is below 0.05). EXPLAINED is checked first (an equivalence test); `residual_excludes_0.5` flags a residual inside the margin whose 95% interval still excludes 0.5. Limit: quantile strata remove only placement that steps with the strata, so an OPEN BIAS can be placement left inside a stratum (a planted world stepping at 5 levels gives about 0.59 inside 3 strata and about 0.5 inside 5); the stratum count is a choice B8 must register; the strata control density only, and the proxy's `pushed_after_event` stays in the residual. The block does not enter the Holm families. (b) Known limit N4: `placement.imagery_footprint_feature` builds `in_imagery_footprint` from the imagery extent and refuses a footprint equal to the mapped-class union or one that leaves a mapped class outside; `effort.FOOTPRINT_FEATURES` (manifest `pre-event-imagery`) swaps it for `in_aerial_footprint`; `build_card(..., effort_imagery_footprint=True)` makes the effort slot take only that allowlist and adds a card note naming it (refused without an effort row). The union refusal is weak (a slightly buffered union passes); B8 must check the polygon comes from the ECan imagery boundary layer. Tests use planted worlds only (correction 1): placement stepping on the strata cuts gives proxy AUC about 0.72 and about 0.5 inside strata (built in by the alignment). The real proxy AUC (0.72) is not recomputed here; B8 registers the basis and runs it under v3 | `placement.py`, `effort`, `card` | `placement`: all |

## Proposals that need calibration (not settled)

1. **Tile chooser.** Fit `gamma(h) = c0 + c1 (1 - exp(-3h/r))` to the binned residual semivariogram (24 bins to
   12 km, at most 1,500 sites) by weighted least squares over a grid of r; range = 0 if `c1 < 0.05 (c0 + c1)`;
   tile = smallest of 0.5, 1, 2, 4, 8 km at least the range (blocks at least the residual range, after Roberts et
   al. 2017). The 5% structure share, the candidates and the "at least the range" rule are mine. R11 also leaves
   open whose residuals: a shaking-only baseline fitted on earlier earthquakes is the natural choice.
2. **Duplicate threshold.** Spearman above 0.99 marks one model (v0's M0 against Level 0 was 0.9995).
3. **Pooling estimator.** DerSimonian-Laird tau^2, Hartung-Knapp t interval on k - 1 degrees of freedom (R1 P1).
   Calibrated on synthetic meta-analyses (k 3 to 5): plain HK covers 0 at 0.94 to 0.95; the normal DL interval
   falls to 0.85 to 0.90 when tau > 0. The floored HK variant (`hk_floor=True`) covers 0.96 to 1.00 and is not the
   default. Known edge: plain HK has zero width when every difference is equal.
4. **Gate sizes.** Null gate: at most 3 of 20 replicates separate (P = 0.016 under a correct 5% test). The planted
   gate uses one seeded case. Neither is the null calibration at the card's shape that R11 asks for.
5. **Holm families.** Two per card (R11 d): ranking (paired AUC) and probability (paired log loss). No pair of two
   baselines. EQUIVALENT margins AUC 0.05 and log loss 0.05 nats (R27 P8).
6. **Interval method.** Percentile tile bootstrap. A quick study (120 null replicates, 400 sites, 2 km tiles)
   gave 0.95 coverage of 0 for the paired AUC interval; the test accepts 0.85 to 1.0.
7. **Reliability curve.** Read at 21 fixed forecasts (0, 0.05, ..., 1) so the card holds no per-site points.
8. **Site-memory level.** The probability version carries one earthquake pair's conditional yes shares to the next
   pair. Shaking differs between earthquakes, so its level can be off; the ranking part (yes above no) is the robust
   part. The 20-site cell floor and the Jeffreys estimate are mine. The 20 m radius is the CPT same-site distance;
   a label source with coarser locations may need its own radius. Repeat labels share the label maker's habits, so
   a high site-memory AUC can partly measure label persistence.
9. **Candidate intervals (own-work item C1, not certified, not on any card).** `intervals.py`, each behind a flag
   whose default is the current method (`within_event_draws(method="tile")`, `pooled_interval(method="hartung-knapp")`):
   (a) `spatial-block`: blocks of side = the measured paired-difference range rounded up to 500 m (LAB-CERTIFICATES
   section 11.1: AUC 5.5 km, log loss 2.7 km, so 3.0 km blocks); (b) `survey-cluster`: whole survey units drawn with
   replacement, one unit id per site with `n_sites` required, so a short list, a missing id (any NA) or mixed id
   kinds (7 and "7") is refused; (c) `event-resample`: earthquakes drawn with replacement (exact enumeration up to
   k = 6), optional second stage with one within-earthquake draw per pick, and a small-k caveat on every result.
   Only the two-stage form of (c) is a new candidate for C2: its one-stage t form is `pool.equal_weight` (tested),
   and its one-stage percentile form undercovers at k = 3 to 4. Three equal differences give a zero-width interval
   with p = 0; that is not a decisive result at small k, and the result carries `zero_width` and says so. Interval
   forms "percentile" (kit default) and "t" (G - 1 degrees of freedom). Known-truth toys in
   `tests/test_evalkit_intervals.py` (deterministic; rates as the committed tests give them): 5.5 km blocks 0.91
   against site-level 0.63; units 0.93 against 2 km tiles 0.46; one-stage percentile 0.68 at k = 3 and 0.78 at
   k = 4 (the caveat), one-stage t 0.96 and 0.97, two-stage 0.86 and 0.89. The toys check the marginal truth only
   (field and unit offsets random, truth 0); certificates section 12.2 leaves the choice of truth open, and C2 runs
   the candidates in the lab under both truths (marginal and models-fixed). Item C4 certifies one.

10. **Interval by card version (own-work item C4, ruling 47).** `interval_policy.py`; `build_card(card_version=...)`
   and `interval_policy.pooled_lines(card_version=...)`. "v2" (the default) runs no new code, so the sealed v2 card
   keeps its hash. "v3" prints the registered C4 outcome as the only interval. C4 result (pillar
   `docs/evals/C4-CERTIFICATION-2026-11-23.md`, registration b844a93, harness `lab/c4_certify.py` at 92933f0):
   no candidate certifies under "models fixed" (tile4 per-earthquake UNDER, pooled lines OVER, block AUC under the
   20-block gate, cluster NOT TESTABLE). So a v3 card withholds every within-earthquake word (DESCRIPTIVE, with the
   measured coverage in the note) and keeps the R1 word on the pooled line, marked not certified.
   Review 1 (wave 3): a v3 card needs fixed 4 km tiles (the tile4 interval C4 measured; any other tile is refused),
   and it also withholds the two words outside `comparisons` that rest on the per-earthquake tile bootstrap: the B5
   geographic null's `word_vs_null` and the B7 placement proxy word. A final scan refuses a v3 card that still holds
   any word value (`interval_policy.PER_EARTHQUAKE_WORDS`).
11. **Scorecard v3 pieces (own-work item B8, rulings 46 and 48).** Registration pillar
   `docs/evals/SCORECARD-V3-REGISTRATION-2026-11-23.md`. Hash recipe v2 in `spec.py`: `card_sha256(recipe="v2")`
   (9 significant digits; within 1e-6 of zero, an absolute 1e-12 grid) and `entry_sha256(recipe="v2", refit=True)`
   (refit rows at 6 decimals). `build_card(card_version="v3")` uses it for refit rows and records `hash_recipe`;
   the default recipe stays "v1", so the sealed v2 card keeps its hash. `zhu_cells.cell_input_ruled`: the ruling 46
   reader (empty cell 0 only by a failed PGA, PGV or Vs30 limit from our inputs; else OFF-GRID or NOT CHECKABLE,
   dropped), for both readers. `pgv.score_baseline(..., sampling="nearest")` reads the nearest grid node (NaN
   outside by more than half a cell or on an empty node; no fill), as v3 registration section 5 names for the B5 PGV
   score; `pgv.sample_nearest` is the reader. The default stays bilinear and its output is unchanged.

## What the kit does not do yet

- **External parity, R12 (e).** Reproducing a published AUC or LSN distribution needs real data and its own
  registration; R12 (d) (reproduce the last release) needs a release.
- **Label error rates.** The card prints sensitivity and specificity from the task; measuring them is
  `label_ceiling.py` and the G3 work, not this kit.
- **Not enforced:** R1 (a) beyond a `registered` flag the caller passes; R4 OpenTimestamps, block-before-join, one
  primary entry; R5 anchor time against origin time (the task declares the track) and the SEALED mark; R6 per-site
  exposure (exposed sites leaving the primary score) and twin checks; R7 label rule commit order and protocol
  statement; R10 building the baselines itself from committed items (the caller builds them); R11 the variogram
  hash in the registration (the card prints it) and the sign-flip small test of `scorer.py`; R13 container rerun; R14 versions and errata;
  R15 input hash against the package; R16 (a), (c) tester rerun path, (d), (e); R17 NLM sanction file; R18 citations.

## Worked example: `python -m scoreboard.evalkit demo`

A synthetic earthquake of 1,500 sites (1,360 labelled, 305 yes, 100 tiles of 2 km), forward track, 2,000 draws.
Labels are manifestation: triggering times a fragility of 0.6. Gates pass. Trimmed card:

```
entry                              mark  exposure                   AUC                   log loss (1e-6)       slope (1e-6)          slope no 0/1
House manifestation model          HOUSE UNSEEN                     0.666 [0.632, 0.696]  0.584 [0.523, 0.652]  0.169 [0.105, 0.304]  0.689 [0.518, 0.870]
House model, steeper copy          HOUSE FITTED ON THIS EARTHQUAKE  0.666 [0.632, 0.696]  0.654 [0.578, 0.741]  0.200 [0.138, 0.284]  0.345 [0.259, 0.435]
Published triggering model               NOT CHECKED                0.648 [0.610, 0.683]  ranking only (R3)
Published index                          SEEN INPUTS                0.671 [0.637, 0.704]  ranking only (R3)
baseline: coin                           UNSEEN                     0.500                 0.693
baseline: earlier-earthquake rate        UNSEEN                     0.500                 0.534 [0.502, 0.566]
baseline: shaking only                   UNSEEN                     0.572 [0.533, 0.609]  0.540 [0.503, 0.576]  0.418 [0.233, 0.615]

first comparisons (house losses first; 30 in all, Holm across them):
log_loss  House model, steeper copy vs baseline: earlier-earthquake rate  +0.120 [+0.063, +0.188] MDD 0.090  SEPARATES  favours the baseline
log_loss  House model, steeper copy vs baseline: shaking only             +0.113 [+0.055, +0.183] MDD 0.093  SEPARATES  CLIP-SENSITIVE
auc       Published triggering model vs baseline: shaking only            +0.076 [+0.020, +0.134] MDD 0.080  CANNOT TELL  TILE-SENSITIVE

best ranking: none (candidate Published index: not separated from three entries; exposure SEEN INPUTS)
best probability: none (candidate baseline: earlier-earthquake rate: not separated from two entries)
duplicates: House manifestation model = House model, steeper copy (Spearman 1.0); 4 distinct models
```

The house model has 9.3% exact zeros, so its slope reads 0.17 at the 1e-6 clip and 0.69 without the zeros: the v0
clip artefact, now printed beside the registered value instead of hidden by it.
