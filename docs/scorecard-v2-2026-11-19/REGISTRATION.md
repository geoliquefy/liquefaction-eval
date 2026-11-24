# Liquefaction model scorecard v2 (record track): registration

G6 + G1 + G5, eval milestone M4 (`docs/evals/EVAL-GOAL.md`), eval plan step 3a
(`docs/evals/EVAL-PLAN-STEPS-1-3-2026-11-19.md`). Tier 2. Written and committed ALONE before `build_v2.py`
exists and before any v2 score is computed. Design: `docs/evals/EVAL-SYSTEM-DESIGN-2026-11-19.md` REVISION 2.
Rule book: `docs/evals/RULEBOOK-V1-DRAFT-2026-11-15.md`, DRAFT, not adopted. No synthetic data on the card.
No em dashes.

```
 POPULATION                   ENTRIES + BASELINES                          WHAT PRINTS
 Canterbury, 4 quakes,   -->  Level 0, M0, LSN+global, LSN+CES          -> per quake: numbers, 4 km tile interval,
 4,542 sites, sets B, A       (authors' id; default id beside),             "not certified", NO word
   beside: CPTs pushed        Zhu 2017, GEM TS22 (stand-in)              -> exposure SEEN / POSSIBLE / PLACE / UNSEEN,
   before each quake          coin, earlier rate, shaking only,             UNSEEN-only read beside
 World, 6 quakes (P1),   -->  pre-event effort (Canterbury only)         -> 2 named pooled lines per population:
   union P2 beside, AUC       density proxy: diagnostic table only          DESCRIPTIVE (no certificate covers them)
```

## 1. What this card is, and what it is not

1. RECORD TRACK, RETROSPECTIVE. Every label on it was read before (v0, v1, G1, G5, step 1). It is not a skill test.
2. It names no winner and no best model. It holds no damage or loss number. Atlas, the T+T NLM and the T+T 2019 map
   are not on it.
3. External parity is NOT ACHIEVED: two registered attempts (pillar cd473b9 then dea3183; dd92d10 then 53aa93c)
   both read DISAGREES (`docs/evals/PARITY-M3-2026-11-19.md`). The card says so in its first lines.
4. The audit (step 1d) is designed and hashed but NOT GRADED. The rule book is a draft, not adopted.
5. The card prints the kit commit and its canonical card hash (`spec.card_sha256`).

## 2. Kit and one scorer

- Kit: `repos/liquefaction-model-evalkit`, branch `evalkit-2026-11-15`, base commit 16a4c63 (pushed). Every number on
  the card comes from kit functions (`card.build_card`, `metrics`, `resample`, `pool`, `effort`, `chain_settings`).
- Kit options this card needs, to be added AFTER this commit and BEFORE the build, each with tests on synthetic data
  only, no default changed, so no sealed card moves:
  - K1 `pool`: `weighting="equal"` (equal weight per earthquake: mean of d_i; variance sum (d_i - mean)^2 /
    (k (k - 1)); Student t on k - 1 degrees of freedom), and leave-one-group-out (a region left out in turn).
  - K2 `card.descriptive(card, note)`: returns the card with every within-earthquake word and every best-entry field
    replaced by DESCRIPTIVE, keeping the point values, intervals, differences and standard errors, and adding the
    interval note of section 5.
- The kit commit that adds K1 and K2 is recorded in SEAL.txt and in the card JSON. It is a planned step of this
  registration, not a deviation. If K1 or K2 needs any other kit change, that change is logged under Deviations.

## 3. Populations

### 3.1 Canterbury (primary population)

- Sites: the 4,542 bench v2 train and val CPTs of scorecard v1 (`label_sets.load_rows()`, rows npz sha256
  a55ed6c4...d25bda; `check_no_test` stops on any test row). The bench v2 test split is SPENT and is never read.
- Earthquakes: 4 Sep 2010, 22 Feb 2011, 13 Jun 2011, 23 Dec 2011.
- Label set B is primary, set A beside, exactly as v1 section 3 (`scoreboard.label_sets.SETS`, same G6b rule).
- Per-source AUC ceiling (Menon C, `docs/evals/LABEL-ERROR-RATES-2026-11-19.json`) prints beside each card: Sep and
  Feb as measured (Feb identified, Sep assumption-driven); Jun and Dec as the admissible scenarios only.
- Beside read, placement bias: each card is also built on the sites whose CPT was pushed BEFORE that earthquake
  (bench v2 `push_date` earlier than the event date). Sep 2010 has no such site: NOT AVAILABLE. These CPTs were
  still placed after earlier earthquakes; the restriction removes only placement that follows the scored earthquake.

### 3.2 World case histories (second population)

- Data: `data/world-case-histories-2026-11-19/` (`cases.parquet` sha256 c42d2e0d...b931, `traces.npz` sha256
  2b5e3fcd...9faf; step 3c, pillar fe4ddc4). Licences: NGL open; PRJ-3012 ODC-By (Geyin and Maurer, DesignSafe,
  doi:10.17603/ds2-wftt-mv37), attributed on the card.
- Primary: P1, the source rule (`pop_g1_source_rule`). Beside: P2, the union (`pop_union`), no word.
- Record gate per earthquake: at least 3 yes, 3 no and 5 tiles of 1 km (local projection with origin at the mean
  case position, as G1 curated and step 3c). On P1 six earthquakes pass: Tangshan, Westmorland, Loma Prieta,
  Hyogoken-Nambu, Chi-Chi, Emilia. Earthquakes under the gate print counts only.
- Metric: AUC only (curated case sets, not a census: no log loss, calibration, level or base rate).
- Regions (statement of method from 3c): each of the six P1 earthquakes is its own region, so leave one region out
  equals leave one earthquake out on P1; on P2, Imperial Valley and Westmorland share the Imperial-Mexicali Valley.

## 4. Entries and baselines

### 4.1 Canterbury

| Entry | Mark | Output kind | Values |
|---|---|---|---|
| Level 0 | HOUSE | other_probability (ranking only) | as v1: `g5.p_level0(lnk, sdlnpga)` |
| M0 (Level 0 + earthquake band) | HOUSE | other_probability (ranking only) | as v1: `R.mixture(p0, Q)` |
| LSN + global fragility [authors-maurer-2026-11-19] | HOUSE | score with conversion | LSN from `chain_settings.run(..., AUTHORS_ID)` with u2 from the corpus, the G6 event water table and PGA, the pre-drill mask, through C-GLOBAL (x_m 8.314, beta 1.113) |
| LSN + CES fragility [authors-maurer-2026-11-19] | HOUSE | score with conversion | same LSN through C-CES (x_m 15.910, beta 1.131) |
| LSN + global fragility [default-2026-11] (beside) | HOUSE | score with conversion | LSN from `chain_settings.run(..., DEFAULT_ID)`; gate: equals v1 `lsn_rows.npz` to 1e-9 |
| LSN + CES fragility [default-2026-11] (beside) | HOUSE | score with conversion | as above, C-CES |
| USGS Zhu 2017 | ORGANISER-RUN | other_probability (ranking only) | as v1 |
| GEM TS22 | ORGANISER-RUN, STAND-IN INPUTS, NOT MONOTONE (step 2c) | target_probability | as v1 (`gem_p.npz`) |

Fragility rows use the authors' settings id because the CES and global fragilities were fitted with Ic < 2.50 and
the authors' fines rule (`docs/evals/FORMULA-CHECKS-2026-11-19.md` deviation 4). The authors' id itself rests on two
ASSUMED items (a = 0.80; the 2016 and 2019 Christchurch correlations are the same). Round 2 parity found that the
authors' id does not reproduce their CES LPI fragility median (DISAGREES). Both ids print; neither is called right.

Baselines (as v1 section 7, R10 departures as v1): coin; earlier-earthquake rate; shaking only (PGA bins, same
merge rule); and the pre-event effort baseline (`evalkit.effort`, allowlist "pre-event", fitted leave one earthquake
out within the label set, features as `EFFORT-BASELINE-PREEVENT-REGISTRATION-2026-11-19.md`, 22188c0; through the
card's effort slot). Known limit (step 1 review 2, N4): `in_aerial_footprint` is the union of the mapped CGD0200
classes, not the imagery footprint; if mappers chose the extent by the damage, the feature leaks a little toward
the baseline. The investigation-density proxy prints in a diagnostic table only, never on a card, never a baseline.

### 4.2 World (AUC only, default id)

| Row | Values | Note |
|---|---|---|
| Level 0 | `g5.p_level0(ln k*, 0)`: ln k* from the step 2c chain on each trace (0.05 m grid, as G1 curated), sigma_e = 0 because no per-case PGA spread exists | ranking equals ln k* |
| M0 | equals Level 0 in ranking within an earthquake (one band for all sites); not printed separately | |
| BI14-LSN (both LSN fragility rows) | `chain_settings.run(..., DEFAULT_ID)` | a fragility is monotone, so both LSN rows share this AUC. The authors' Canterbury id is not used outside Canterbury |
| BI14-LPI (beside) | the same chain, LPI | the G1 reference index |
| USGS Zhu 2017 | the USGS ground-failure `zhu_2017_general.hdf5` grid value at the nearest cell, where on disk (Hyogoken-Nambu, Chi-Chi) | elsewhere NOT RUN |
| GEM TS22 | NOT RUN (inputs not on disk) | |
| baseline: shaking only | the case PGA (rank) | |
| baseline: coin | AUC 0.5 by definition | |

No earlier-earthquake rate and no effort baseline exist outside Canterbury.

## 5. What prints, and the words

- Per earthquake (both populations): AUC for every row; for Canterbury probability rows also log loss (clip 1e-6;
  other clips beside), Brier, calibration intercept and slope and CORP, as the kit prints them; paired differences
  for every pair the kit forms. Canterbury log loss and calibration on inventory "no" labels measure agreement with
  the label source, and the card says so.
- Interval: the 95% tile bootstrap on fixed 4 km tiles of the NZTM2000 grid anchored at the grid origin (multiples of
  4,000 m), 2,000 draws, seed 20261106 (world: 1 km local tiles, same draws and seed). Every interval is labelled
  "not certified; lab coverage 0.80 to 0.96 depending on an unidentified range" (`LAB-CERTIFICATES-2026-11-19.md`
  sections 2, 3, 11, 12).
- NO word within an earthquake: no SEPARATES, BETTER, WORSE, EQUIVALENT or CANNOT TELL. Every such field reads
  DESCRIPTIVE (kit K2).
- Canterbury count gate for printing numbers: 200 sites, 30 per class, 10 tiles of 4 km. Set after seeing the tile
  counts of v1 (not any score), so that every full card prints: 4 km tiles per card are set B 25 / 27 / 24 / 17,
  set A 13 / 21 / 16 / 17. Certificate 5 found that the tile count does not protect coverage, so the gate only stops
  numbers on near-empty sets.
- Exposure per cell: for every entry, the counts of labelled sites that are CASE SEEN, POSSIBLE, PLACE (CASE unseen
  at this earthquake but SEEN or POSSIBLE at another) and UNSEEN (unseen at every earthquake), from
  `~/.geoliquefy-runs/exposure-ledger-2026-11-19/site_ledger.csv` (sha256 b20c02d3...f9ce). Beside each entry's AUC,
  an UNSEEN-only read (that entry and the four baselines on its UNSEEN sites, same tile rule) where the subset passes
  the count gate; else counts only. GEM TS22 is NOT CHECKED everywhere, so it has no UNSEEN read. The effort,
  shaking-only and earlier-rate baselines are fitted on other earthquakes at the same sites: PLACE exposed by design.
- World exposure: per earthquake from step 3c section 3 (B&I chain SEEN or POSSIBLE except Emilia; LSN global
  fragility SEEN everywhere; Zhu SEEN on Hyogoken-Nambu and Chi-Chi). No UNSEEN pooled line is possible (k = 1).

## 6. Pooled lines (at most 2 per population, named here before any join)

| Id | Population | A | B | Metric | Earthquakes | Weighting |
|---|---|---|---|---|---|---|
| C1 | Canterbury, set B | Level 0 | baseline: shaking only | AUC | Feb, Jun, Dec 2011 (Sep has no shaking-only) | kit random effects, Hartung-Knapp; equal weight beside |
| C2 | Canterbury, set B | Level 0 | baseline: effort only | AUC | all 4 | as C1 |
| W1 | World P1 | Level 0 | baseline: shaking only | AUC | the P1 earthquakes that pass the record gate | equal weight (K1); kit weights beside |
| W2 | World P1 | Level 0 | USGS Zhu 2017 | AUC | P1 earthquakes where both run and pass the gate | equal weight (K1) |

- Per-earthquake inputs: the paired AUC difference and its tile-bootstrap SE from the card (section 5 tiles).
- Each line prints the mean, the 95% interval, tau^2, each d_i and s_i, leave one earthquake out, and (W1, W2) leave
  one region out. A line with fewer than 3 earthquakes prints NOT ALLOWED (R1 c). W2 has at most 2 (Zhu runs on
  Hyogoken-Nambu and Chi-Chi only), so W2 is expected to print NOT ALLOWED; it is named so that no other line can
  take its place after the join.
- Words: a pooled word (HOLDS, FRAGILE, POOLED CANNOT TELL) prints only inside a certificate's stated working range.
  Certificate 4 (Hartung-Knapp, fixed 4 km inputs, k = 3 or 4) was certified at the first lab's 2 km model-error
  range; amendment 1 and review 1 (sections 11.4 and 12.1) found the real model-difference range unidentified
  between 2.7 and more than 12 km and said "no pooled-line word". So C1 and C2 print DESCRIPTIVE. No certificate
  covers equal weighting or world earthquakes, so W1 and W2 print DESCRIPTIVE. Every pooled line is descriptive.
- Founder ruling pending (`LAB-CERTIFICATES` 12.2): which truth a word covers (models fixed or averaged). A
  models-fixed ruling could later allow per-earthquake words under a NEW registration. This card does not assume it.

## 7. Verdict rule for any future word (not used on this card)

The audit (step 1d) is primary once graded; it is not graded. Until then a word needs agreement across label
sources: set A and set B (and the bench, road and aerial sources inside them) must agree in sign. The card prints,
per paired difference on Sep and Dec, whether set A and set B agree in sign, as a descriptive count only.

## 8. Gates (all must pass before the card is written)

- G-k `gates.run_gates()` passes; the kit test suite passes at the build commit (count printed).
- G-lsn the default id reproduces `lsn_rows.npz` (sha256 6aa251f0...9d1f) for all four earthquakes to 1e-9, with the
  same finite pattern; else stop.
- G-v1 with the v1 tile rule replaced by fixed 4 km only, the kit's AUC for the v1 entries on every v1 card that printed AUC
  equals the v1 JSON AUC point values to 1e-12 (same inputs, same metric).
- G-world the world dataset hashes match section 3.2.
- G-c no per-site output: `card.scan_aggregates` on the whole JSON against all CPT names, row ids and world case
  ids; SCORECARD.md scanned for the same.
- G-t the test split stays unread (`check_no_test`).

## 9. Outputs and seal

In this folder: `build_v2.py`, `scorecard.json` (aggregates only), `SCORECARD.md`, `SEAL.txt` (SHA-256 of each
file, the canonical card hash, kit commit, registration commit). OpenTimestamps is a founder step (it sends hashes
to public calendars); this session does not stamp. One independent review (at most 2 rounds) follows in a later
session. Any change after this commit is logged under Deviations.

## Deviations

All logged 2026-11-19, after the registration commit (0c736f2).

Planned step, not a deviation: kit options K1 and K2 were committed on `evalkit-2026-11-15` as 6638d75 (8 new tests on
synthetic data; 235 eval tests pass; 2 unrelated failures in `tests/test_g5_watch_failures.py` exist at that commit).
The build ran at kit 6638d75.

1. Run attempts. Attempt 1 scored every card in memory and then stopped at the G-c scan: the world `site_id` column
   holds empty strings for PRJ-3012 rows, and the scan treated "" as a site id. No file was written and no score was
   printed (the log holds statuses only). Fix: empty and null ids are dropped from the scan list. Attempt 2 is the
   card. No rule, input or entry changed between the attempts.
2. Renderer. `render_v2.py` (presentation only, reads the JSON) was added beside `build_v2.py`. Section 9 listed only
   the build script. After the first render it was fixed twice: the count row reads the kit's `count_gate` keys, and
   the Dec 2011 S_feb ceiling (C 0.50) is marked inadmissible, as step 1a says.
3. Zhu 2017 on world earthquakes: grid cells that are empty (NaN) are read as 0 (gfail masks to 0). Not registered.
   Reworded after review 1 (M1): this is not a small change. On Hyogoken-Nambu it sets 22 of 26 P1 inputs (30 of 36
   P2), including 15 of 19 "yes" and all 7 "no" cases, so that Zhu cell rests mostly on tied zeros. Chi-Chi has no
   empty cell. W2 is NOT ALLOWED, so no pooled line moves. The page prints the count beside each world Zhu cell.
4. Extra check, not registered: world BI14-LPI and BI14-LSN under the default id were compared with the registered
   G1 curated path (`run_g1_curated.grid_values`). Largest differences 6.2e-7 (LPI) and 3.0e-6 (LSN), same NaN pattern;
   the cause of these small differences was not isolated in the time box. The card uses the `chain_settings`
   values, as registered.
5. "Pushed before the earthquake" is read as bench v2 `push_date` strictly earlier than the event date.
6. The LSN rows under the default id carry the same exposure ledger columns as the authors' id rows (the ledger has
   one column per fragility, not per chain setting).
7. Each kit comparison keeps `family_size` (the count of pairs the kit formed). No Holm result prints.
8. A post hoc check after the run, not on the card: on world P1, `p_level0(ln k*, 0)` has no ties at 1.0, so its AUC
   equals the AUC of ln k* on all six earthquakes (registration 4.2 "ranking equals ln k*" holds).
9. The per-card caveat text (reused from v1) still says "label error NOT MEASURED"; the page says that step 1a has
   measured it and prints the Menon ceilings instead. The JSON text is unchanged.

### Post-review fixes (review 1, `REVIEW-1-2026-11-19.md`, pillar 2834fab; verdict CARD STANDS WITH FIXES)

Renderer, notes, logs and this section only. No new score was computed. No population, entry or rule changed.
`build_v2.py` and `scorecard.json` are unchanged, so the canonical card hash is unchanged (cc9f7b74...847d).

10. Post-review fix (M1): each world Zhu cell prints the count of cases on empty grid cells read as 0 (Kobe P1: 22 of
    26, 15 yes and 7 no; P2: 30 of 36). The counts come from the same nearest-cell lookup as `build_v2.world()`, run
    once on `cases.parquet` (no score). Deviation 3 is reworded.
11. Post-review fix (M2): a Kobe note on the world rows and on W1: 25 of 26 P1 PGAs carry the NGL tag "site response
    analysis"; such a PGA may depend on the outcome; this is not tested. W1 P1 without Kobe (+0.108, equal weight,
    from the JSON leave-one-out record) prints as a descriptive side line. Kobe stays in every population.
12. Post-review fix (M3): the world tables add the Zhu 2017 and LSN global fragility exposure columns from
    `world.populations.*.exposure`. The LSN global column reads "in sample (fit)".
13. Post-review fix (m1): world intervals from fewer than 10 tiles of 1 km are marked "uninformative".
14. Post-review fix (m2): the page now prints the set A Menon ceilings, the baseline UNSEEN-only reads, W1 P2 leave
    one region out, and the counts of the world earthquakes under the record gate. All come from the JSON.
15. Post-review fix (m3): Sep 2010 log-loss cells of the earlier rate and shaking only read NOT AVAILABLE.
16. Post-review fix (m4): a note says that C2 has tau^2 0 (Q 2.42 below k - 1) and that with `hk_floor` off its
    Hartung-Knapp interval is narrower than a fixed-effect interval.
17. Post-review fix (m5): a note says that the Dec 2011 sign agreement does not test label sources (sets A and B
    share the Dec labels).
18. Post-review fix (m6): a note says that the Jun 2011 S_sep ceiling 0.64 lies below the observed Level 0 (0.683)
    and Zhu (0.696) AUC, so the data contradict that scenario.
19. Post-review fix (m7): no change; deviation 9 and the page already cover it.
20. Post-review fix (m8): the build logs exist in the session scratchpad and are now committed as
    `build_v2-attempt1.log` and `build_v2-attempt2.log` and hashed in SEAL.txt. Attempt 1 shows statuses, gate
    values and timings only, then the scan refusal, as deviation 1 says.
21. Post-review fix (m9), correction of the 0f93274 commit message: it says that the pre-event-CPT read "raises
    Level 0 to 0.80". That wording compares AUC across different site sets, which design R2.2 item 6 does not allow.
    Read it as: on the pre-event-CPT sites, Level 0 AUC is 0.80 (Feb), 0.72 (Jun) and 0.72 (Dec), and shaking only
    is 0.81 (Feb). The card text does not make the comparison.
