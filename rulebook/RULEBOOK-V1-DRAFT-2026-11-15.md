G1+G5+G6 (M1): the standing rule book of the GeoLiquefy liquefaction-model eval (the scorecard).

# Liquefaction model eval: rule book v1.0

**DRAFT. Not adopted. Not for sending.** Revision 4, 2026-11-15. It serves `docs/evals/EVAL-GOAL.md` milestone M1
("Rule book v1.0 is unambiguous"). It answers every finding of independent review 2
(`docs/evals/RULEBOOK-V1-REVIEW-2-2026-11-15.md`); each answer is in
`docs/evals/RULEBOOK-V1-REVIEW-2-RESPONSE-2026-11-15.md`. What the kit does today is in
`docs/evals/RULEBOOK-V1-STATUS-2026-11-15.md`, not here. No em dashes.

**What binds.** The summary (`docs/evals/RULEBOOK-V1-SUMMARY-2026-11-15.md`) and Parts A to D bind. Where the
summary and this text differ, or a rule here reads two ways, the reading that favours the outside entrant concerned
applies until an erratum fixes it. Where the readings favour different entrants, the reading that leaves the
organiser less choice applies. Such an erratum may only confirm the reading that applied; any other reading is a new
major version (R14).

How to read it. Part A holds the rules, each with **Rule** and **Check** (how code or a named human check tests it);
the failure each rule prevents is listed in the status file. Part B tells an entrant what to do. Part C defines
every term and verdict word; a defined term has one meaning everywhere. Part D quotes the rule text an outsider
needs. Part E holds proposals, open items and the change log. Every number in Parts A to D is fixed; R27 collects the
ones a task might be tempted to vary. A method marked P1 to P12 is a proposal until milestone M2 calibrates it (E1).
`[from memory, verify]` marks an unchecked claim.

## The eval in one picture

```
 PUBLIC REGISTER  append-only, hash-chained, OpenTimestamps; the clock is register block time (R20)
 PUBLIC MIRROR    a second log, run by a third party, for entry hashes (R20)
   |
 [0] STANDING ITEMS   rule-book version (>= 30 days before use); per region: site universe, region
   |                  table row; per label source: footprint rule, label rule, target, window rule;
   |                  each counts only if committed >= 24 h before the origin (R21)
 [1] ENTRIES          record: values file + manifest; task open >= 30 days (R2, R4)
   |                  forward: frozen image + manifest; entry time >= 24 h before origin (R5, R21)
   |                  signed receipt <= 1 h; every hash sent is shown (R20)
 [2] EARTHQUAKE       day 7: census read, ShakeMap version, run list; inputs <= 6 h; outputs <= 48 h (R21)
 [3] RELEASE REGISTRATION   <= 30 days after origin: a card or a reason per census earthquake x
   |                  label source (R24); pooled lines (R1)
 [4] LABELS JOINED    after the 48 h limit; label set hash first; seed from the next Bitcoin block
   |
 [5] SCORER GATES     planted entry, null gate, defect, last release, published number (R12)
   |
 [6] CARDS            primaries + baselines; two Holm families; LOSS check; beside table (R9 to R19)
   |
 [7] RELEASE          <= 90 days after registration; never edited; errata <= 14 days (R14)

 BEST: forward, tester rerun, >= 2 outside entrants, family representative, separates (R5)
```

# Part A. The rules

### R1. Unit of scoring, and the pooled line

- **Rule.** Every word, interval and MDD is computed on one card, except a pooled line. A pooled line is allowed only
  if all of (a) to (e) hold.
  - (a) The organiser or the tester, and no one else, commits a pooled-line registration. It names the earthquake
    set, one card key per earthquake, the metric (paired AUC or paired log loss), and entries A and B, each with one
    model id and one model version on every earthquake, and lists their entry hashes on each earthquake.
  - (b) The set holds at least 3 earthquakes.
  - (c) The registration's register block time precedes the first release of any card in the set.
  - (d) No organiser person joined any set earthquake's labels to A or B before that block time.
  - (e) Every named earthquake enters: each card passes the count gate, both entries are on it, and the
    intersection of their kept sites (R6) passes the count gate.
- **Estimator (P1).** For earthquake i, d_i is the paired difference A minus B on the intersection of kept sites, and
  s_i its tile-bootstrap SE (R11). Between-earthquake variance tau^2 by DerSimonian-Laird; weights
  w_i = 1 / (s_i^2 + tau^2); pooled mean m = sum w_i d_i / sum w_i; Hartung-Knapp variance
  V = sum w_i (d_i - m)^2 / ((k - 1) sum w_i). The line's p is the two-sided p of t = m / sqrt(V) on k - 1 degrees
  of freedom. Leave-one-out repeats the estimate with each earthquake removed in turn.
- **Words.** Holm runs across all pooled lines of one release: with L lines, the line with the j-th smallest p is
  tested at alpha_j = 0.05 / (L - j + 1). NOT ALLOWED: any of (a) to (e) fails. POOLED CANNOT TELL: Holm does not
  reject. HOLDS: Holm rejects, and every leave-one-out interval at level 1 - alpha_j excludes 0 with the sign of m.
  FRAGILE: Holm rejects, and some leave-one-out interval covers 0 or has the other sign.
- **Place.** Under the full cards, as "across these N earthquakes", with each d_i, s_i and tau^2. It never names a
  best entry. It prints RETROSPECTIVE if any set earthquake is on the record track.
- **Check.** All signal in one earthquake gives no HOLDS. Each failed condition gives NOT ALLOWED.

### R2. Task definition

- **Rule.** A task is one card. Its card key is earthquake x label source x target x division. A task file names:
  the card key; the ComCat id, origin time (UTC), census magnitude and type, hypocentre and census read hash (D6);
  the track; the site list hash; the label source version and label rule hash (R7); the attribution window and its
  shocks (R25); the ShakeMap version and its hash (R21); the baselines and their input hashes (R10); on the closed
  division, the package hash (R15); the rule-book version; and on the record track, the deadline (UTC). It names no
  R27 parameter; the scorer reads those from the named version.
- The kit's public task builder makes every task file from committed items only: the site universe, the label
  source's standing items, the census read, the ShakeMap rule and the rule-book version. No person chooses a site.
  The record site list is the universe sites inside the label source's footprint for this earthquake. The forward
  site list is the run list (R21).
- The version is fixed by rule: the latest major version whose register block time is at least 30 days before the
  task's register block time (record) or before the origin (forward).
- A record deadline is at least 30 days after the task's register block time. Every task file is committed before
  any label join.
- A record card prints "task fixed after the organiser saw labels" if any organiser person had access to any label
  of this source on this earthquake before the task's register block time. Access means the label was public, or an
  organiser person held a copy.
- **Check.** The scorer refuses a task that breaks any clause above. The builder, rerun from the committed items,
  gives the same task hash.

### R3. Each entry declares what its number is

- **Rule.** The manifest gives the output kind: (a) `target_probability`, a probability of the task target;
  (b) `other_probability`, a probability of another event (triggering, areal extent, index exceedance); (c) `score`,
  any number used only for its order.
- Probability metrics apply only to probability-eligible entries: kind (a), or kind (b) or (c) with a conversion. A
  conversion is frozen code that maps an entry's values to probabilities of the task target. An entry has at most one
  conversion; the same model with another conversion is another entry. The conversion's hash is in the manifest, and
  the card names its author. When a published model's authors commit an entry with their own conversion, every entry
  of that model with anyone else's conversion moves to the beside table on every later task.
- The R26 checker confirms the declared kind before the join. Only it can dispute a kind, with written reasons, and
  R26 sets when its finding takes effect. A disputed kind prints "kind disputed" and gets ranking only.
- A ranking-only entry is kind (b) or (c) with no conversion, or a disputed kind. It gets AUC and ranking-family words
  only, and can be named best only in the ranking family, labelled "ranking only, calibration not tested".
- **Check.** No log loss or calibration prints for a ranking-only entry. A second conversion is refused. Only an R26
  finding makes a kind disputed.

### R4. Entry file, freeze, and one primary entry

- **Rule.** A record entry is an entry file plus a manifest (B3). The entry file is UTF-8 text with the header
  `site_id,p` and exactly one row for every task site and no other row. A value is any string Python `float()`
  accepts that gives a finite number in [0, 1]. Scoring uses the canonical values (D3). Any breach makes the entry
  REFUSED, with the reason. A kind (c) entrant rescales its scores into [0, 1]. A forward output meets the same rules
  on the run list (R21).
- The entry hash (D3) is the entry's identity. R20 sets when an entry counts.
- Each entrant has at most one primary entry per task. Record track: the entry whose manifest says
  `"primary": true` with the latest intake time before the deadline, or on the hash-first path the hash named primary
  in clear at intake (B3). Forward track: R21. An entrant with exactly one entry on a task and no flag: that entry is
  primary. More than one and none flagged: no primary.
- Every non-primary entry prints in the beside table, outside every Holm family, never named best. At most 5 entries
  per entrant per task; a sixth is REFUSED. A card shows at most 5 REFUSED rows per entrant and the count of the rest.
- No withdrawal at any time. A corrected model is a new entry. Retiring a standing entry (R21) is not a withdrawal.
- **Check.** The kit refuses each file breach, finds the hash in the register or the mirror, and finds at most one
  primary per entrant.

### R5. Tracks, and when a card names a best entry

- **Rule.** A forward task covers an earthquake whose entries are frozen code (R21). An entry is FORWARD if its entry
  time (R20) is at least 24 h before the origin; for a sequence task, the first shock's origin (R25). A forward entry
  that misses this is LATE: not run, not scored, and its hash prints in the LATE list with its entry time. The card
  prints each entry time and the origin time.
- A register fault never changes the track. The card prints REGISTER FAULT, and each entry counts at the time its own
  proof shows (R20).
- Every other task is on the record track and prints RETROSPECTIVE. A record row prints SEALED if its entry time
  precedes the origin plus 24 h. Otherwise, if its entry time precedes the holder's first public release of the
  label set, it prints "before label release" with that release and its UTC date, confirmed by the R26 checker.
- A card names a best entry, once per Holm family (R11 d), only if all of these hold:
  1. The card is on the forward track.
  2. A tester exists (R23) and has rerun the card, with the rerun hash in the register.
  3. The card holds primary entries from at least two outside entrants.
  4. The entry is a primary. It is not a baseline, NEAR COPY, LATE, SELF-RUN NOT RERUN, or scored as the coin (R21).
     A HOUSE entry is not barred.
  5. It is the representative of its model family (R19).
  6. It SEPARATES BETTER than every other row in that Holm family, baselines and barred rows included, except a
     later entry of another entrant above the R27 blocking threshold (R19).
  7. In the ranking family, a ranking-only entry prints "ranking only, calibration not tested".
- If the only entry that meets point 6 fails another point, no entry is named, and the card says why. A card with no
  tester prints "no independent tester: no best entry is named". A card failing point 3 prints "fewer than two
  outside entrants: no best entry is named".
- **Check.** A synthetic card failing any one point names no entry. A late entry is not run.

### R6. Exposure: inputs, declaration, check and site removal

- **Rule.** Each entry declares one class per task earthquake: UNSEEN; SEEN INPUTS (met this earthquake's inputs,
  not its outcomes); FITTED ON THIS EARTHQUAKE (used any outcome of it to fit or to make any choice); UNKNOWN. The
  manifest lists training cases (D3), choices made after outcomes, text read, and inputs used.
- Inputs are site data plus ground motion. Site data are soundings and site properties (CPT, borehole, water table,
  geology), whatever their date, and observations of ground failure made before the origin. A sounding made after
  the origin prints POST-EVENT SOUNDING. Ground motion is the PGA, PGV and spectral acceleration grids, with their
  standard deviations, of the task's ShakeMap version (even where it uses felt reports), and the census read's
  origin parameters. Any other observation made after the origin is an outcome: for example imagery, InSAR, field or
  damage reports, macroseismic intensity, the USGS ground-failure product. An entry that uses an outcome as an input
  is REFUSED on both tracks.
- We check training-case tables against the task site list with the twin rules (D1). Each row prints CHECKED,
  citing the check, or DECLARED. If the check contradicts the declaration, the check wins: the row prints "DECLARED
  X, CHECKED Y" and the class is Y; on the forward track the entry is REFUSED.
- Removal. A site leaves an entry's main score if it is one of that entry's training cases from the task earthquake,
  or a twin of one. The sites left are its kept sites. Each pair is scored on the intersection of the two entries'
  kept sites. An entry or pair whose sites fail the count gate prints UNDERPOWERED. If the task's score list passes
  the count gate, such an entry instead prints "FITTED, UNDERPOWERED BY OWN DECLARATION", ranks last (R16 d), and, if
  HOUSE, prints LOSS. Scores on removed sites print beside only if 20 or more sites were removed.
- Forward track: the class is UNSEEN, no site is removed, and the inputs print. Record track: no best entry is named,
  so the class marks the row and blocks nothing.
- **Check.** Every row shows a class and CHECKED, DECLARED, or both. A planted training case leaves the main score. A
  pair uses one site set.

### R7. Label source and label rule

- **Rule.** A label source enters a card only if it meets clauses S1, S2, S3, S4 and S7 (D2). A source that meets S1
  and S2 is a census label source, so every source on a card is one.
- The label rule maps the source's native classes to yes, no and not labelled, and gives one target, the footprint
  rule and the attribution window rule (R25). Polygon labels become site labels by the G6b rule (D2). The label rule
  is part of the task file; on the forward track it is a standing item (R21). The holder draws each footprint without
  seeing any entry (S4).
- No holder or grader enters a model on a card whose labels it graded or whose footprint it drew. If an organiser
  person graded or drew for a card, no HOUSE entry is on it.
- Join by site id only; on the forward track, after the R21 limit. A duplicate id stops the run. The score list is
  the task sites the label set grades yes or no. Each card prints its label caveats: method, not-graded share, and
  PROTOCOL DIFFERS where S3 applies.
- **Check.** The label rule hash is in the register in time (R2, R21). Duplicate ids stop the run.

### R8. Label error, label ceiling, versions, hidden labels

- **Rule.** The task gives sensitivity and specificity for this source on this earthquake, estimated by Dawid and
  Skene (D5) from either (i) three or more independent gradings of the same sites, or (ii) two independent gradings
  on two or more earthquakes of different prevalence, with rates shared across them (Hui and Walter). Each rate
  prints with a 95% tile-bootstrap interval (P12). With neither design, the task says NOT MEASURED.
- The card prints the label ceiling (D5) only if 0 < sensitivity <= 1, 0 < specificity <= 1, their sum exceeds 1, and
  the estimated prevalence (q + specificity - 1) / (sensitivity + specificity - 1) lies strictly between 0 and 1,
  with q the observed yes share. The ceiling interval is the 2.5 and 97.5 percentiles over the bootstrap draws. Rates
  estimated without this earthquake in the data, or any failed bound, print NOT IDENTIFIED and no ceiling.
- Each label set has a version and a sha256, committed before the join. A correction gets a new name.
- Forward labels never reach an entrant before scoring. Scoring runs on the tester's, holder's or organiser's
  machine. After scoring, per-site labels are released to everyone at once if the licence allows, never to one
  entrant alone.
- **Check.** No card renders without an error-rate field and a committed label hash. Two gradings on one earthquake
  give NOT MEASURED. A holder bundle matches the frozen scorer with 0 fields different.

### R9. Metrics on every card

- **Rule.** All entries: AUC with its interval. Probability-eligible entries add: log loss (the probability-family
  words use it), Brier score and its decomposition, mean p against the observed rate, the shares of exact 0 and exact
  1 (canonical values), an isotonic (CORP) reliability curve read at 0, 0.05, ..., 1, and:
  - calibration-in-the-large: a in logit P(y = 1) = a + logit(p), slope fixed at 1;
  - calibration slope: b in logit P(y = 1) = a' + b logit(p);
  - both again without the sites where p is exactly 0 or 1;
  - log loss and probability-family words at every R27 clip, with p set to [c, 1 - c] and Holm rerun at each clip.
    The main clip gives the word and the calibration fits. A word that differs at another clip is CLIP-SENSITIVE.
- **Check.** Honest synthetic p gives a slope interval holding 1 and an intercept interval holding 0; logit x 3 gives
  a slope below 0.5. A task file naming a clip is refused (R2).

### R10. Baselines are entries

- **Rule.** Every card carries the coin (p = 0.5). An earlier earthquake of a label source is one in the same region,
  graded by any version of the source with the same yes map, whose label set hash was committed before this task. If
  one exists, every card, on every division, also carries:
  - the earlier-earthquake rate: total yes over total labelled sites, pooled over those label sets;
  - the shaking-only baseline: the pooled yes rate in each R27 PGA bin over those label sets, read at each task
    site's PGA from the task's ShakeMap version. Bins merge from the top down: a bin with fewer than 30 labelled sites
    merges into the next lower bin, repeatedly; if the lowest bin then has fewer than 30, it merges upward.
- With no earlier earthquake, both are NOT AVAILABLE and not required.
- The kit builds each required baseline from committed items. One not built is a scorer fault: the card prints
  "scorer fault: baseline not built" and carries (R24).
- Baselines are never HOUSE, never named best, never in a model family, and sit in both Holm families. A pair of two
  baselines is not tested. An entry above the near-copy threshold with a baseline prints "NEAR COPY of" it (R19).
- **Check.** Remove a required baseline: the card does not render and prints the fault. A monotone copy of the
  shaking-only baseline is NEAR COPY.

### R11. Intervals, power, tiles, multiplicity

- **(a) Intervals.** A percentile tile block bootstrap (D4), with the R27 draws and seed, gives every metric and
  paired difference a 95% interval (2.5 and 97.5 percentiles) and a 90% interval (5 and 95, same draws). The
  bootstrap p of a paired difference d is min(1, 2 min(share of d <= 0, share of d >= 0)). MDD = (z 0.975 + z 0.80)
  times the bootstrap SE of the paired difference, not adjusted for Holm.
- **(b) Power.** A card that fails the count gate reads UNDERPOWERED and prints counts only.
- **(c) Tiles.** Residuals are label minus the shaking-only p; without it, minus the earlier-earthquake rate; with
  neither, minus 0.5. Fit the R27 semivariogram; its practical range is r = 3a. The main tile size is the smallest R27
  candidate at least r; if r exceeds them all, the largest, and the card prints "range exceeds the largest tile".
  Tiles are squares in the WGS84 UTM zone of the epicentre, index (floor(E / s), floor(N / s)). A tile counts only if
  it holds a labelled site; the count gate and the draws use those tiles. The card records s, r and the variogram
  sha256.
- **(d) Multiplicity.** Two Holm families per card. Probability family: paired log-loss differences for every pair of
  probability-eligible primaries and baselines. Ranking family: paired AUC differences for every pair of primaries
  and baselines, except two constant entries. Neither tests two baselines. Beside, LATE and NEAR COPY entries are
  outside both, and a row scored as the coin adds no pair. Family size is the number of pairs that pass the count
  gate. Holm runs at the R27 level. The card prints each family's size.
- **(e) Words.** Part C2. SEPARATES is tested before EQUIVALENT.
- **(f) Sensitivity tiles.** Metrics and words also print at the R27 sensitivity sizes, Holm rerun at each. A word
  that differs at a size with 20 or more tiles is TILE-SENSITIVE. A size with fewer prints counts and "too few tiles
  at this size". No sensitivity row changes the main word.
- **Check.** The variogram hash recomputes. The null gate (R12) passes at the card's shape.

### R12. Scorer validation

- **Rule.** Before each release the scorer must (a) find a planted entry of known skill; (b) pass the null gate;
  (c) fail on an injected defect; (d) reproduce the last release under its own code and version; (e) reproduce the R27
  published number on its database within the R27 tolerance.
- Null gate (P10). A null replicate keeps the card's sites, tiles and labels, and replaces every row of a Holm family
  by a synthetic entry: one seeded latent score plus independent noise of one common variance, set so the synthetic
  AUC equals the family's median AUC. No pair then differs in truth. A replicate is a false positive if any pair reads
  SEPARATES. The scorer fails the gate only if the 95% Clopper-Pearson lower bound of the false-positive rate, over
  at least 1,000 replicates, exceeds 0.05. The card prints the observed rate and its 95% interval.
- A failed gate stops every card, not the release: each card prints "scorer gate failed" with the gate outputs and
  carries (R24).
- **Check.** Gate outputs go in the register with the release.

### R13. Reproducibility

- **Rule.** Each release lists the hashes of every entry, input, label set, code version and container. Each card
  prints one rebuild class. PUBLIC REBUILD: every entry, input, label and code file is public or obtainable under
  public terms, and a clean-room rerun matches every number to 1e-9 and every word. TESTER REBUILD ONLY: some file is
  restricted; the tester reruns in the holder's environment and commits the rerun hash; until a tester exists, the
  card prints "not rebuilt outside the organiser".
- A row prints OWN-ROW REBUILD when the labels are public under R8. Its entrant can rebuild every number of its row
  and every word of its pairs with baselines from its own values and the public labels, inputs, code, tiles and seed.
  Words of pairs with other entrants need the tester. Per-draw values of other rows are not published, because tiles
  holding one site would leak per-site values.
- EVAL-GOAL done item 4 holds for PUBLIC REBUILD cards and OWN-ROW REBUILD rows.
- **Check.** Container rerun before release; every card has a class; each OWN-ROW REBUILD row rebuilds from public
  files alone.

### R14. Versions, errata, withdrawals

- **Rule.** Each task is always scored under its R2 version. Any change to the summary, to Parts A to D, or to an R27
  value is a new major version; a change to Part E alone is minor.
- A release is published within 90 days of its registration; R24 governs carried cards. Released cards are never
  edited. Every reported error goes in the public errata log, in the register, within 14 days of the report: the
  report, its date, the decision and the cards it touches. Errata decisions follow the R26 performer and effect rules.
- A release is withdrawn only for a scorer defect: a kit defect that changes a number or a word, shown before a tester
  exists by a public failing test on public synthetic inputs that trigger the same code path, and after by the
  tester. A withdrawn release stays public with its reason. The rerun keeps the tasks, entries, labels and versions;
  only the fix changes, and its diff is listed.
- A wrong label set is not a scorer defect. Its correction is a new label source version and a new card in a later
  release; the old card stays, with an erratum.
- **Check.** Every past release hash verifies. Every task's version matches R2. Every withdrawal cites a failing test
  or a tester confirmation.

### R15. Divisions

- **Rule.** The division is part of the card key, so no card ranks across divisions. The closed package is the site
  id list plus a table of the sha256 of every input file; the package hash is the sha256 of that table. An entrant who
  fetches files itself (as NZGD terms require) checks each against the table. A closed record entry names the package
  hash. A closed forward entry takes each task's package, and the runner checks every file. An open entry lists its
  inputs, which must be site data plus ground motion (R6).
- **Check.** An entry naming another division, or a closed record entry with another package hash, is refused.

### R16. Governance and conflict of interest

- **Rule.** We referee and we enter, so:
  - (a) HOUSE and ORGANISER-RUN entries meet every deadline through the register. A HOUSE record entry counts only if
    its register block time precedes the deadline.
  - (b) HOUSE is any entry the organiser built, or whose conversion or run settings it wrote. ORGANISER-RUN is a
    published model the organiser runs with its authors' published settings and no conversion. An ORGANISER-RUN row
    sits in the beside table until its authors confirm the settings in writing, committed. The R26 checker confirms
    each classification. The Atlas product never enters a card (R17).
  - (c) LOSS. Each HOUSE entry, primary or beside, is compared with every non-HOUSE row: outside primaries,
    ORGANISER-RUN rows (primary or beside) and baselines. A pair already in a Holm family uses that family's word.
    Every other such pair enters the card's LOSS family for its metric, Holm at the R27 level. A HOUSE entry WORSE in
    any of these pairs prints LOSS. The LOSS family prints only LOSS and changes no other word.
  - (d) Row order: LOSS rows first; then by AUC point value, highest first, ties by earlier intake time; rows marked
    "FITTED, UNDERPOWERED BY OWN DECLARATION" last. The beside table uses the same order.
  - (e) An organiser-run entry's authors are invited in writing at least 30 days before the deadline (record) or
    before its commit (forward), and the card prints the date. An author entry of the same model moves the
    organiser-run row to the beside table on that and every later card.
  - (f) A paid link is a payment, equity, employment, contract or advisory role, either way, in the last 3 years.
    Each card lists every paid link between the organiser and an entrant, against the code name if one is used.
    Payment buys no rule change, early label or hidden row.
  - (g) Each card prints, for each HOUSE entry, the share of task sites that are its training sites from any
    earthquake, or their twins (D1).
  - (h) Until a tester exists, every card prints "scored by the organiser, who also enters models".
- **Check.** Rows follow the order rule. A planted house loss to an ORGANISER-RUN beside row prints LOSS.

### R17. Never shown

- **Rule.** Never on any card or release: per-site values of any kind, except per-site labels released under R8;
  values derived from NZGD data at the site level (terms clause 6.2); the NLM without written T+T or NHC sanction;
  T+T labels, ids, LSN or results without T+T's written OK; damage or loss numbers; the Atlas product (it makes no
  prediction claim); beside scores on fewer than 20 sites.
- **Check.** The build refuses site ids (`nzgd:`, `rer:`, `usgs:`) outside a label release, an NLM row without a
  sanction file hash, an Atlas row, and a beside row under 20 sites.

### R18. What the eval does not claim, and prior work

- **Rule.** Every card prints five lines: scores rate models on named label sources, not ground; a record score is
  not a test of skill; one city is not the world; UNDERPOWERED is not a tie; no score is design advice. Each release
  cites prior evaluations of the same models and explains conflicts (an R26 check).
- **Check.** The five lines print on every card. Citations pass the cold read and the comms gate.

### R19. Model families and near copies

- **Rule.** A model family forms only among one entrant's entries that apply to this card, beside entries included;
  retired entries and entries for other cards do not count. The representative is the earliest of them by intake
  time. Another such entry joins if its Spearman correlation with the representative, over the card's score list,
  exceeds the near-copy threshold. No chaining. Constant entries are never linked. Only the representative can be
  named best; if it cannot be, no member is. A family counts once in the card's count of models.
- NEAR COPY. An entry prints "NEAR COPY of X" if its Spearman correlation with X over the score list exceeds the
  near-copy threshold, where X is a baseline or an earlier primary of another entrant on the card. For a published
  model, its authors' entry counts as earlier than anyone else's entry of it; otherwise earlier means earlier intake
  time. A NEAR COPY moves to the beside table and is never named best.
- Blocking. Under R5 point 6, a candidate need not separate from a later entry of another entrant whose Spearman
  correlation with it, over the score list, exceeds the R27 blocking threshold.
- **Check.** A chain A-B-C with A and C below the threshold does not join C to A. A later cross-entrant monotone copy
  is NEAR COPY. A later 0.98 variant does not block. An authors' entry made after a copy leaves the copy NEAR COPY.

### R20. The public register, the mirror and the clock

- **Rule.** One append-only public register holds every item this book commits, every entry hash received with its
  intake time, and every receipt.
- Each line holds a sequence number, the intake time (UTC), the item type, the item's sha256 and the previous line's
  sha256. Intake times never decrease along the chain; a decrease is a REGISTER FAULT. The head is stamped with
  OpenTimestamps at each intake and at least hourly, and upgraded proofs are published.
- "Committed" means "hash in the register". An item's register block time is the block time of the lowest-height
  Bitcoin block whose stamp covers a register head at or after its line. It is the only clock for every limit in this
  book, except where the mirror is named.
- The mirror is one public append-only log run by a party other than the organiser, named in R27. An entrant may post
  there the line "entrant id, task hash or `standing`, entry hash, `primary` or blank", signed with its registered
  key (R22). The mirror time is the log's own time for that line; a post with another key does not count. An entry's
  entry time is the earlier of its register block time and its mirror time.
- An outside record entry counts if its intake time precedes the deadline and its register block time is at most 3 h
  after it, or if its mirror time precedes the deadline. HOUSE: R16 (a). Forward: R5, R21. A record entry that does
  not count prints LATE.
- A hash in the mirror but missing from the register still counts, with REGISTER FAULT. Every hash an entrant sends
  or posts stays in the register and prints on the card, within the R4 cap on REFUSED rows; one that cannot be scored
  prints REFUSED, LATE, NOT SUPPLIED or RUN FAILED. A private stamp never counts, and email is never evidence of time.
- Each entrant gets a receipt signed with the organiser's key within 1 hour of intake: entry hash, entrant id, intake
  time, line number and previous-line hash. A receipted hash missing from the register counts at its receipt time,
  with REGISTER FAULT and an erratum. If no receipt comes, the remedy is the mirror.
- The register verifies for a card if every line from the first standing item the card uses to its release line
  chains, and every proof the card relies on checks.
- **Check.** The kit verifies the chain, every proof, the intake-time order and every mirror line's signature, and
  finds every scored item committed before its limit.

### R21. Forward entries are frozen code

- **Rule.** A forward entry is a standing entry: an OCI image (linux/amd64) sent by the R27 transport, its digest, an
  environment hash, a declared input list (B4), the output kind, any conversion, and a manifest; its hash (D3) is
  committed. It applies to every earthquake that triggers a run in each region it names, if the origin is at least
  24 h after its entry time. Values-only entries are record track only.
- Standing items. Per region, the organiser commits a site universe (site id, longitude, latitude, hash of each
  site's input files) and a region table row (name, universe hash, run-input URL). Per label source, it commits the
  footprint rule, label rule, target and window rule. A standing item, designation or retirement counts for an
  earthquake only if its block time is at least 24 h before the origin; the latest such version applies. With none in
  time, the earthquake prints "no standing item in time".
- Primary and retirement. One designation item per region and division names the primary; the latest one in time
  applies. An entrant with exactly one standing entry for a region and division and no designation: that entry is
  primary. Otherwise no primary. Forward manifests have no `primary` field. A retirement item stops an entry for
  later earthquakes.
- Census read. At origin plus 7 days the kit reads ComCat and fixes the census class, magnitude and hypocentre (D6),
  and the ShakeMap version: the first USGS ShakeMap version of the ComCat preferred event updated at least 24 h after
  the origin. With none by then, the earthquake is NOT RUN. The ComCat response hash and frozen `event.json` hash are
  committed within 24 h of the read, or the release prints "census read late". Every event with a census magnitude of
  5.0 or more and a universe site within its radius R triggers a run.
- Run list: the universe sites within the D6 radius R of the hypocentre, built by the kit.
- Inputs. Within 6 h of the census read, the organiser publishes at the region's URL, and commits the hashes of,
  `event.json`, the ShakeMap grid stripped to PGA, PGV, spectral acceleration and their standard deviations,
  `sites.csv` (the run list), and the package (closed division). If late, every later limit here runs from their
  publication.
- Run. No network, inputs read-only, 8 CPU, 32 GB memory, 12 h. The output meets R4 on the run list. Its output hash
  (D3) is committed within 48 h of the inputs. A second run must match the canonical hash or differ by at most 1e-9;
  otherwise the entry is RUN FAILED.
- Join. Labels are joined only after the 48 h limit has passed. A run hash missing then is NOT SUPPLIED on that card.
  Exception: a runner-held image whose hash the runner missed is still run, scored when committed, and marked "run
  late by runner".
- Scored as the coin. A forward entry REFUSED, RUN FAILED or NOT SUPPLIED on a card prints its reason and the coin's
  metrics and words in its row, and adds no pair to any Holm family.
- Runner. The tester, once one exists. Before then the entrant chooses: (i) the organiser runs the image under the
  pledge (R23); or (ii) self-run: run the committed image on the published inputs and send the output hash and file
  within the 48 h. A self-run row prints SELF-RUN NOT RERUN until the tester reruns the image and matches the hash.
  The entrant gives the image to the tester within 30 days of appointment. A mismatch, or no image in 30 days, makes
  the entry RUN FAILED on every card it was on, by erratum.
- **Check.** The census read, ShakeMap choice and run list rebuild from the committed ComCat response. A rerun matches
  the output hash. A mounted grid holds only the allowed fields.

### R22. Entrants

- **Rule.** An entrant is one group with a checked identity (named contact, members, institution and a public key,
  confirmed by an R26 check) and disclosed affiliations and paid links. Groups from one institution, or with a member
  in common in the last 12 months, are one entrant. A code name is allowed if fixed at intake. The tester holds the
  identity, or the organiser until a tester exists; its hash, the code name and the public key are committed. The
  organiser is one entrant, owning every HOUSE entry. Each ORGANISER-RUN model holds its authors' place as one entrant
  until they enter.
- One group found entering under two names: all its entries on those tasks move beside, and an erratum is logged.
- **Check.** One identity per entrant; the merge rule applied at intake; entrant counts match the identity register.

### R23. The tester, the outside check performer, and the organiser's pledge

- **Rule.** The tester is a named third party that accepts the role in writing, has had no paid link (R16 f) with any
  entrant or the organiser in the last 3 years, and has no entry. Its acceptance and independence statement are
  committed. It holds outside entries until the join; runs forward images; reruns every release and commits the rerun
  hash; approves errata decisions and withdrawals; decides appeals; performs R26 checks; holds code-name identities.
- The outside check performer is a named person with the same independence who accepts in writing, committed. It
  performs R26 checks and decides appeals while no tester exists.
- Until a tester exists, no card names a best entry. An outside entrant may use the hash-first path (B3).
- Before the first outside entry, the organiser commits a signed pledge: it will not use any outside entry's values,
  code or image to build, fit, tune or choose its own models, or to choose any standing item, task, organiser-run
  model or rule-book change, and will use them only to score, rerun and publish aggregates under this book. A breach
  found is an erratum, and every house entry made after it prints "pledge breach".
- **Check.** No best entry without a committed tester acceptance; the pledge line precedes the first outside intake.

### R24. The release registration: which cards exist

- **Rule.** A covered region has a committed site universe. A release registration lists each census-class
  earthquake (D6) of a covered region within 30 days of its origin. For an earthquake older than its region's
  universe, the census read is made within 7 days after the universe's block time, and the listing within 30 days of
  that read. Every later registration lists every census earthquake with a card not yet published. A registration is
  committed before any label join for its release.
- For each listed earthquake, every label source that meets R7 is listed. Each earthquake x label source x target x
  division with at least one primary gets a card. Every other combination, and every card not ready, prints one
  reason: no label source meets R7 (naming the clause); no entry in this division; NOT RUN; no standing item in time;
  R17 forbids it; labels not yet delivered; scorer gate failed; scorer fault: baseline not built; scoring not done. A
  card failing the count gate prints as UNDERPOWERED. No listed card is dropped.
- A card with one of the last four reasons carries. The next registration is committed within 90 days of the release
  it was carried from. A card carried from two releases prints "carried past limit" in every later release until it
  is published.
- **Check.** Every registration row has a card or a reason; listing times and carry counts recompute.

### R25. Sequences

- **Rule.** Each task names its attribution window: from the end of this source's previous survey of the region (or
  30 days before the origin, if none) to the end of this survey. The sequence is every ComCat event of census
  magnitude 5.0 or more in the window whose radius R (D6) covers a task site.
  - One event: a single-shock task.
  - More, and the source attributes each yes to one shock (S3.2): each shock is its own task, with its own labels and
    origin.
  - Otherwise a sequence task. The card prints SEQUENCE, the target is "manifestation seen after any shock in the
    window", and the origin is the first shock's (R5). An image runs once per shock on that shock's inputs; its value
    at a site is its largest over the runs that include the site. An image with `sequence_input` runs once on every
    shock's inputs, within 48 h after the window enters the task file and before the join.
- An entry whose entry time follows an earlier shock in the window, or whose inputs include observations of one,
  prints "after an earlier shock".
- The window rule is a standing item on the forward track; on the record track the window and shocks print in the
  task. Canterbury 13 Jun 2011 had two shocks on one day (the larger at 02:20:49 UTC), so it is a sequence task unless
  the source attributes.
- **Check.** The kit rebuilds the window and shock list from ComCat and the survey dates.

### R26. Named human checks and appeals

- **Rule.** Checks with no code: (1) output kind (R3); (2) conversion authorship, published-model authorship and
  "same model" (R3, R16, R19); (3) text read, choices after outcomes, and outcome inputs code cannot see (R6); (4) twin
  rule C pairs (D1); (5) label independence S4 and grader entries (R7); (6) entrant identity, the merge rule, and one
  group under two names (R22); (7) HOUSE and ORGANISER-RUN classification (R16); (8) prior-work conflicts (R18); (9)
  the "before label release" citation (R5); (10) pledge breach (R23).
- Performer: the tester; if none, the outside check performer; if none, an organiser person named in the release
  registration who built, tuned or converted no entry on the card. The card prints each check's outcome, performer
  name and role, and date.
- Effect. A check not done counts for the entrant: the declared value stands and the card prints "check not done";
  but check (7) not done counts an entry the organiser built, converted or ran as HOUSE, and check (10) not done
  prints "pledge breach reported, not decided". An adverse finding by the tester or the outside check performer takes effect. An adverse finding by an
  organiser person takes effect on a HOUSE entry; on an outside entry it prints "organiser check:" and the finding and
  changes nothing else.
- Appeals. An entrant may appeal a refusal, a check finding, an exposure check, a NEAR COPY mark or a twin decision
  within 14 days of the card. The tester, or else the outside check performer, decides within 30 days with written
  reasons; an appeal not decided by then is granted. With neither, a public rerun of the kit decides an appeal on a
  code-checked rule, and an appeal on a human check is granted. Decisions go in the errata log. A successful appeal
  gives an erratum with the corrected result, under the same task and version.
- **Check.** Every release lists every check with an outcome, a performer and a date.

### R27. Fixed parameters

A task cannot change these; a change is a new major version (R14). Each card that uses a method marked P1 to P12
prints "method proposal, not yet calibrated" until M2 calibrates it.

| Parameter | Value | Rule | Proposal |
|---|---|---|---|
| Count gate | at least 200 labelled sites, at least 30 in each class, at least 20 tiles holding a labelled site at the main tile size | R11 b, R6, R1 | P4 |
| Clips | 1e-6, 1e-4, 1e-3, 1e-2; main clip 1e-6 | R9 | P9 |
| EQUIVALENT margins | AUC 0.05; log loss 0.05 nats | R11 e | P8 |
| Tile candidates | 0.5, 1, 2, 4, 8 km; sensitivity sizes 1, 2, 4 km | R11 c, f | P2 |
| Semivariogram | gamma(h) = c0 + c1 (1 - exp(-3 h / r)); 24 equal bins to 12 km; at most 1,500 sites by seeded draw; r on 120 equal steps from the centre of the first non-empty bin to twice the centre of the last non-empty bin; at each r, c0, c1 >= 0 by weighted non-negative least squares, weights = pairs per bin; r = the step with the least weighted squared error; r = 0 if c1 / (c0 + c1) < 0.05 | R11 c | P2 |
| Tile grid | WGS84 UTM zone of the epicentre; origin at multiples of the tile size | R11 c | none |
| Bootstrap | 2,000 draws; seed = the integer of the last 16 hex digits of the hash (usual big-endian display) of the Bitcoin block one height above the block that sets the label set's register block time; the same seed draws the semivariogram sites | R11 a | P5 |
| Holm level | 0.05 in each Holm family, in each LOSS family, and across the pooled lines of a release | R11 d, R16 c, R1 | P7 |
| MDD power | 0.80 | R11 a | P6 |
| Near-copy; blocking threshold | Spearman 0.99; Spearman 0.95 | R19, R10, R5 | P3 |
| Pooled line | at least 3 earthquakes; DerSimonian-Laird with Hartung-Knapp | R1 | P1 |
| Null gate | at least 1,000 replicates; fail only if the 95% Clopper-Pearson lower bound of the familywise false SEPARATES rate exceeds 0.05 | R12 | P10 |
| Shaking-only bins | PGA edges 0, 0.1, 0.2, 0.3, 0.4, 0.5 g, open top; merge under 30 sites, top down | R10 | P11 |
| Census read; listing | origin + 7 days; hashes within 24 h; listed within 30 days of origin | R21, R24 | none |
| Forward times | entry and standing items at least 24 h before origin; ShakeMap first version at least 24 h after origin; inputs within 6 h of the census read; output hash within 48 h of the inputs | R21 | none |
| Forward run | no network, 8 CPU, 32 GB, 12 h; image at most 20 GB | R21 | none |
| Record times | task open at least 30 days; outside block-time grace 3 h; HOUSE grace none; hash-first files within 14 days | R2, R20, R16 | none |
| Version lead | 30 days, by block time | R2 | none |
| Receipt; errata; appeal | 1 h; 14 days; 14 days to appeal, 30 to decide | R20, R14, R26 | none |
| Paid-link window; invitation lead; merge window | 3 years; 30 days; 12 months | R16, R22 | none |
| Beside minimum; entries per task; REFUSED rows shown | 20 sites; 5 per entrant; 5 per entrant | R6, R17, R4 | none |
| Not graded share | at most 25% of footprint sites | R7 (S2) | none |
| Release; carry | within 90 days of registration; next registration within 90 days; "carried past limit" after two carries | R14, R24 | none |
| Public mirror; intake address; image transport | to be named before M6 (E2) | R20, R21 | none |
| R12 (e) published number, database, tolerance | to be named before M3 (E2) | R12 | none |

# Part B. For entrants

This part tells you what to do. Part C defines the words.

### B1. What you can enter

| | Record track | Forward track |
|---|---|---|
| Earthquake | already happened | not yet happened |
| You send | a values file and a manifest | a container image and a manifest |
| When | hash before the task's deadline; the task is open at least 30 days | hash at any time; it applies to earthquakes at least 24 h after its entry time |
| Card says | RETROSPECTIVE | FORWARD |
| Can name a best entry | never | only with a tester and at least two outside entrants |

### B2. Register as an entrant (once)

1. Email the intake address with: group name; contact person; members; institution; affiliations; every paid link
   with the organiser or any entrant in the last 3 years; a public signing key; and an optional code name. Groups from
   one institution, or sharing a member in the last 12 months, register as one entrant.
2. Answer the identity check (an institutional email reply or a document check).
3. Receive your entrant id and its register line. A code name is fixed now.

### B3. Record track: build and send an entry

1. Read the task file in the register: site list, target, division, package hash (closed) and deadline (UTC).
2. Closed division: get each input file by its id and check it against the package table. Report any mismatch before
   the deadline.
3. Write `entry.csv`: UTF-8; header exactly `site_id,p`; one row per task site; no other row; comma separator; no
   quotes. Each `p` is any string Python `float()` accepts, in [0, 1]. The scorer uses the canonical value (nine
   decimals). A score model rescales into [0, 1].
4. Write `manifest.json` with every field below, in any order and spacing. Only `conversion_sha256` and
   `package_sha256` may be null. Extra fields are allowed and are hashed.
5. Compute the entry hash (recipe below).
6. Before the deadline, email the hash to the intake address, saying whether it is your primary, with both files or
   by the hash-first path. Post the same line to the mirror, signed with your key.
7. Keep the receipt.

Example `entry.csv` (made-up sites):

```
site_id,p
ex:0001,0.412
ex:0002,0.03
ex:0003,1
```

Example `manifest.json`:

```
{
 "rule_book_version": "1.0",
 "entrant_id": "E0007",
 "task_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
 "model": "example-model",
 "model_version": "2.1",
 "primary": true,
 "output_kind": "target_probability",
 "conversion_sha256": null,
 "division": "closed",
 "package_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
 "inputs": ["package", "shakemap"],
 "exposure": {"us7000abcd": "unseen"},
 "training_cases_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
 "choices_after_outcomes": "none",
 "text_read": "none about this earthquake",
 "statement": "Probability of minor or worse ejecta at the site after this earthquake.",
 "house": false
}
```

Fields: `output_kind` is `target_probability`, `other_probability` or `score`. `conversion_sha256` is the hash of
your conversion code, or null. `division` is `closed` or `open`. `package_sha256` is null on the open division.
`exposure` maps the task's ComCat id to `unseen`, `seen_inputs`, `fitted_on` or `unknown`. `training_cases_sha256` is
the hash of your training-case table in the D3 form, sent with the entry. `house` is false for every outside entrant.

Hash recipe (Python 3, standard library):

```
import csv, hashlib, json, math
def cell(s):
    x = float(s)
    if math.isnan(x): return "nan"
    if math.isinf(x): return "inf" if x > 0 else "-inf"
    t = format(x, ".9f")
    return format(0.0, ".9f") if float(t) == 0.0 else t
rows = list(csv.reader(open("entry.csv", newline="", encoding="utf-8")))
text = "\n".join([",".join(rows[0])] + [r[0] + "," + cell(r[1]) for r in rows[1:]]) + "\n"
m = json.load(open("manifest.json", encoding="utf-8"))
mj = json.dumps(m, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
print(hashlib.sha256((mj + "\n" + text).encode("utf-8")).hexdigest())
```

The two example files above hash to
`913254e4277b730bf805c5f41ab2b5fa32cc7cdba13819d75b026fe531385cc9`. Check your copy of the recipe against it.

Hash-first path. Send only the hash before the deadline, naming in clear which hash is your primary. When the label
set hash is in the register, send the files within 14 days; they must give the committed hash. A hash with no files
prints NOT SUPPLIED, and then you have no primary on that task.

### B4. Forward track: build and send a standing entry

1. Build a linux/amd64 OCI image of at most 20 GB. It runs with no network, 8 CPU, 32 GB memory and 12 h.
2. It reads only what your manifest declares from `/inputs/`:
   - `event.json`: `comcat_id`, `origin_utc`, `lon`, `lat`, `depth_km`, `mw`, `mw_type` (frozen at the census read);
   - `shakemap/grid.xml`: the R21 version, holding only PGA, PGV, spectral acceleration and their standard deviations;
   - `sites.csv`: `site_id,lon,lat` for the run list;
   - `package/`: the files of the package table (closed division only).
3. It writes `/outputs/entry.csv` by the B3 file rules, one row per site in `sites.csv`. A second run must give the
   same hash, or differ by at most 1e-9 in every value.
4. Write a forward manifest: the B3 fields except `task_sha256`, `package_sha256`, `exposure`, `primary` and
   `rule_book_version`, plus `regions` (region table names), `image_digest`, `environment_sha256`, `sequence_input`
   (true or false) and `run_by` (`runner` or `self`).
5. The entry hash is the sha256 of the canonical manifest JSON alone (`mj` in the recipe).
6. Email the hash and post it to the mirror. Send the image by the R27 transport unless `run_by` is `self`.
7. To name your primary for a region and division, or to retire an entry, send a register item; it applies to
   earthquakes at least 24 h after its block time. A sole standing entry for a region and division is your primary.
8. After an earthquake, inputs appear at the region's URL within 6 h of the census read (origin plus 7 days). Your
   output hash is due within 48 h of them. If you self-run, send the output hash and `entry.csv` in that time.
9. If your image crashes, times out, writes a bad file or sends nothing, your row is scored as the coin.

### B5. Your receipt and the mirror

Within 1 hour you get a receipt signed by the organiser: entry hash, entrant id, intake time (UTC), register line and
the previous line's hash. Check that line in the public register. Your signed mirror line is your proof if no receipt
comes. Email is never proof of time.

### B6. Deadlines

| What | When |
|---|---|
| Record entry hash | intake before the deadline and block time at most 3 h after it, or mirror time before it |
| Record files, hash-first path | within 14 days of the label set hash entering the register |
| Forward entry hash | entry time at least 24 h before an earthquake's origin (the first shock's, for a sequence) |
| Forward output hash | within 48 h of the inputs' publication |
| Primary designation or retirement | block time at least 24 h before the origin |
| Appeal | within 14 days of the card |

### B7. What comes back

1. The card for each task you entered: aggregates only, with your rows, the baselines, the words and the marks.
2. Your hashes, register lines, the rebuild class and, where labels are public, OWN-ROW REBUILD.
3. No per-site values. Per-site labels go to everyone at once, only where the licence allows.

### B8. What you may publish

1. Your card in full, or your rows quoted with the card's link and rule-book version.
2. Call your entry "best" only if a card names it best. Never present a record card as a forecast.
3. Your predictions are yours. Third-party data terms (for example NZGD) bind you for your own outputs.

### B9. Your rights and limits

1. You cannot withdraw any entry. You can retire a forward entry for future earthquakes.
2. The organiser uses your values, code and image only to score, rerun and publish aggregates (R23), and publishes
   no per-site value of yours. You keep every right in your model.
3. Every hash you send or post is recorded and shown (R20). Do not send test hashes.
4. Your primary can be named best only if it is your earliest entry on that card (R19). Send it first.
5. A human check not done counts for you; an organiser check against you changes nothing. Appeals: R26.

### B10. Which rule-book version binds your entry

The latest major version whose block time is at least 30 days before the task's block time (record) or the origin
(forward).

# Part C. Definitions and words

### C1. Definitions

- **Access** (to labels): R2.
- **Authors**: a published model's listed authors. **Published model**: one in a peer-reviewed paper or a public
  code release.
- **Baseline**: the coin, the earlier-earthquake rate or the shaking-only baseline (R10).
- **Beside table**: rows under a card, outside every Holm family, never named best.
- **Card**: the scored table of one task. **Card key**: earthquake x label source x target x division.
- **Census class**, **census magnitude**: D6. **Census read**: R21. **Census label source**: meets S1 and S2 (D2).
- **Closed division**: entries use the organiser's package. **Open division**: entries list their own inputs.
- **Committed**: hash in the register.
- **Conversion**: R3. **Count gate**: the R27 minimum counts.
- **Covered region**: a region with a committed site universe.
- **Deadline**: record track, the UTC time in the task file; forward track, 24 h before the origin.
- **Designation item**: a register item naming a forward primary for one region and division (R21).
- **Entrant**: R22. **Entry**: one values file and manifest (record), or one image and manifest (forward).
- **Entry hash**: D3. **Entry time**: the earlier of register block time and mirror time (R20).
- **Exposure**: any route by which an entry met the task earthquake's inputs or outcomes (R6).
- **Footprint**: the ground a label source surveyed for one earthquake, as polygons. **Footprint rule**: the committed
  rule that gives it.
- **Holder**: the body that holds a label set (for example T+T).
- **HOUSE**, **ORGANISER-RUN**: R16 (b). **Outside entry**, **outside entrant**: neither HOUSE nor ORGANISER-RUN.
- **Inputs**, **site data**: R6. **Outcome**: any other observation made after the origin.
- **Intake time**: the UTC time on an entry's register line.
- **Join**: matching a label set to entries by site id. **Kept sites**: R6.
- **Label rule**: R7. **Label set**: one versioned delivery of labels. **Label source**: a named, versioned grading
  protocol. **Label ceiling**: the best AUC any model could reach against noisy labels (D5).
- **LOSS family**: R16 (c).
- **Main clip**, **main tile size**: R27, R11 (c). **Main score**: scores on kept sites. **Main word**: the word at
  the main clip and main tile size.
- **Mirror**, **mirror time**: R20.
- **Model family**, **representative**: R19.
- **Organiser**: GeoLiquefy and every person working for it.
- **Outside check performer**, **tester**: R23.
- **Package**: R15. **Paid link**: R16 (f).
- **Primary entry**: R4, R21. **Probability-eligible**, **ranking-only**: R3.
- **Record track**, **forward track**: R5.
- **Region table**: committed rows of region name, universe hash and run-input URL (R21).
- **Register**, **register block time**: R20.
- **Release**: the cards of one release registration, with its hash committed. **Release registration**: R24.
- **Rule-book version**: major.minor (R14).
- **Run list**: R21. **Score list**: the task sites the label set grades yes or no (R7).
- **Sequence task**: R25. **ShakeMap version**: R21.
- **Site universe**: the committed sites of a region (R21).
- **Standing entry**: a forward entry that applies to later earthquakes until retired. **Standing item**: a site
  universe, region table row, footprint rule, label rule, target or window rule (R21).
- **Target**: the yes or no question the label source answers at a site, for example "minor or worse ejecta seen at
  the CPT, inside the survey footprint, in the attribution window".
- **Task**: one card's definition (R2). **Task site**: a site on the task's site list.
- **Tile**: one square of the R11 (c) grid holding a labelled site.
- **Twin**: D1.

### C2. Words

The eval uses these words and no others as verdicts.

| Applies to | Word | Meaning |
|---|---|---|
| A pair, A against B, in one Holm family on one card | SEPARATES, with BETTER or WORSE | Holm rejects "no difference" (R11 d). BETTER: A has the higher AUC, or the lower log loss. |
| | EQUIVALENT | Not SEPARATES, and the 90% paired interval lies strictly inside plus or minus the R27 margin. |
| | CANNOT TELL | Neither. The card prints the MDD. |
| | UNDERPOWERED | The pair's kept-site intersection fails the count gate. No word is tested. |
| An entry row | UNDERPOWERED | Its kept sites fail the count gate. Counts only. |
| | BEST | Named under R5 only. |
| | LOSS | A HOUSE entry WORSE than at least one non-HOUSE row (R16 c), or underpowered by its own declaration (R6). |
| | REFUSED, NOT SUPPLIED, RUN FAILED | Cannot be scored; the reason prints (R4, R20, R21). A forward entry so marked is scored as the coin. |
| | LATE | Not run, not scored; the hash prints in the LATE list (R5, R20). |
| A whole card | UNDERPOWERED | The count gate fails. Counts only. |
| An earthquake | NOT RUN | No ShakeMap version by the census read (R21). |
| A pooled line | HOLDS, FRAGILE, POOLED CANNOT TELL, NOT ALLOWED | R1. |
| An exposure class | UNSEEN, SEEN INPUTS, FITTED ON THIS EARTHQUAKE, UNKNOWN | R6. |

**Marks** describe and never change a word: FORWARD, RETROSPECTIVE, SEALED, "before label release", HOUSE,
ORGANISER-RUN, "NEAR COPY of X", CHECKED, DECLARED, "DECLARED X, CHECKED Y", "FITTED, UNDERPOWERED BY OWN
DECLARATION", POST-EVENT SOUNDING, NOT MEASURED, NOT IDENTIFIED, NOT AVAILABLE, TILE-SENSITIVE, CLIP-SENSITIVE,
SEQUENCE, "after an earlier shock", "ranking only, calibration not tested", "kind disputed", "too few tiles at this
size", "range exceeds the largest tile", "method proposal, not yet calibrated", "task fixed after the organiser saw
labels", "no independent tester: no best entry is named", "fewer than two outside entrants: no best entry is named",
"scored by the organiser, who also enters models", REGISTER FAULT, SELF-RUN NOT RERUN, "run late by runner", "census
read late", PUBLIC REBUILD, TESTER REBUILD ONLY, OWN-ROW REBUILD, "not rebuilt outside the organiser", "pledge
breach", "pledge breach reported, not decided", "paid link", "check not done", "organiser check:", "carried past
limit", PROTOCOL DIFFERS. R24 lists the card reasons.

# Part D. Quoted rule text

Square brackets mark this rule book's own words inside a quote.

### D1. Twin rules

From `repos/liquefaction-model/docs/REGISTRATION-G5-BLIND-PREDICT-2026-11-05-ADDENDUM-1.md` and its predecessor. Two
rows are the same test (twins) if any of these holds:

- **Same site (20 m).** The two soundings lie within 20 m of each other.
- **Trace twin (200 m).** They lie within 200 m and "median relative qc difference below 2 percent".
- **A, test code.** "The rows come from different sources, lie within 2 km, and their normalized test codes are
  equal." The code is the test code in a Regione Emilia-Romagna id; the PRJ-3012 case ID without its file extension;
  the NGL `TEST_NAME` with a trailing "CPT" label removed; the USGS sounding name; the NZGD investigation id.
  "Normalized: upper case, letters and digits only, and the letter O read as the digit 0."
- **B, readings.** "The rows come from different sources, lie within 2 km, their soundings overlap by at least 3 m,
  and on a 5 cm grid the median relative qc difference is at most 0.05 and the Pearson r of qc is at least 0.95."
- **C, by hand.** A pair whose names differ only in a site prefix, checked against the site name. Each such pair is an
  R26 check and is listed.
- **W, repeated upload.** "Same source, within 2 km, final depths within 0.2 m, median relative qc difference at most
  0.005 and r at least 0.999."

"The rows joined by these links form a test group. Only names, coordinates and readings are read. No outcome ...
column is read." A group is exposed if any member is exposed, and "every member of an exposed group is exposed".
Rule A "is counted as the same test" even when readings disagree. [Rules B and W run only for a training site whose
readings are in the site universe or the package.]

### D2. Label source clauses

From `repos/liquefaction-model/docs/SPEC-STANDING-GRADING-PROTOCOL-2026-11-05.md` section 1.

- **S1 footprint.** "A polygon (or polygons) of the surveyed land, in lon/lat (WGS84), delivered as a file. A text
  description alone ... does not count." Also what the survey covered inside it, and for aerial grading the image
  dates, ground resolution and extent.
- **S2 every site graded.** "Every package CPT inside the footprint carries one of: yes (ejecta seen), no (surveyed
  and none seen), or not graded with a stated reason." "'Not reported' is never 'no'." Not graded is "at most 25% of
  the footprint CPTs", and "the reasons must not depend on the outcome". Polygons graded by class become site grades
  by the G6b rule (below).
- **S3 method.** "Method per CPT or per survey zone: aerial, ground, or both." Survey start and end dates per zone;
  the native classes with written definitions; "the map from classes to yes and no, fixed before delivery". A yes
  map other than "minor ejecta = yes", or grading by area, is "still scored but tagged PROTOCOL DIFFERS". S3.2: if an
  Mw 5 or larger shock falls [in the R25 attribution window], "the labels name the shock each 'yes' belongs to, or
  the earthquake carries a sequence flag".
- **S4 independence.** "The graders do not see our predictions ... before they grade." Labels reach the scorer only
  after the entries are committed. "A label set made before our predictions ... meets S4.1 by date. It must arrive
  unchanged: the holder states the file's creation date and sends its sha256."
- **S7 licence.** "Must allow: computing the scores and publishing them per earthquake (the words, [metrics],
  intervals, MDD) with the yes and no counts."

**G6b rule** (from `repos/liquefaction-model/scoreboard/labels.py`), in metres in the R11 (c) UTM zone:
1. Repair each polygon (`make_valid`) and keep only its polygon parts.
2. For each class, take the union of its polygons and close it: grow by 0.5 m, then shrink by 0.5 m. This fills gaps
   narrower than 1 m between polygons of one class.
3. For each site: inside no class, drop it ("unmapped"); inside two or more classes, drop it ("two classes"); inside
   one class, keep that class's yes or no if the site is at least 10 m from that class's boundary, else drop it
   ("edge").
4. A dropped site is not graded, with its reason, and counts toward the S2 25% limit.

Worked example. "Minor ejecta" maps to yes; "none seen" maps to no. Site 1 is 25 m inside a minor ejecta polygon:
yes. Site 2 is 6 m inside it: dropped (edge). Site 3 is where a minor ejecta polygon overlaps a none seen polygon:
dropped (two classes). Site 4 is 40 m inside a none seen polygon: no. Site 5 is in no polygon: dropped (unmapped).
Two minor ejecta polygons 0.8 m apart merge in step 2, so a site 0.4 m from the old gap and 12 m from the outer edge
is yes.

### D3. Canonical hash

From `repos/liquefaction-model/docs/AMENDMENT-G5-RULINGS-2026-11-05.md` section 7.

- **Predictions text.** Read the file "with Python's `csv` module as strings"; header row unchanged; `site_id`
  unchanged; each value "parsed with Python `float()` and written as `format(x, ".9f")`, that is, rounded to an
  absolute 1e-9; `inf`, `-inf` and `nan` written as such; a value that rounds to zero is written `0.000000000` with
  no sign"; "cells joined by `,`, rows by `\n`, a final `\n`, UTF-8". Rows stay in file order.
- **Canonical JSON.** `json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)`.
- **Entry hash.** sha256 of the canonical manifest JSON, a newline, then the predictions text.
- **Forward entry hash.** sha256 of the canonical forward manifest JSON (it holds the image digest).
- **Forward output hash.** [sha256 of the canonical forward manifest JSON, a newline, then the output's predictions
  text.]
- **Training-case table.** [UTF-8; header exactly `site_id,earthquake_id,lon,lat`; lon and lat as `format(x, ".6f")`;
  rows sorted by `site_id`, then `earthquake_id`; rows joined by `\n` with a final `\n`; hashed as those bytes. Site
  ids use the universe's prefixes; a site outside every universe uses its source's id with the source as prefix.]
- **Tolerance.** "If two platforms give different canonical hashes, the tolerance check decides": the largest
  absolute difference over all value cells, "at most 1e-9 with the same rows and columns is 'same prediction'".

### D4. Statistics text

- **Tile bootstrap.** Each draw takes as many tiles as the card has, with replacement, and keeps every site of each
  drawn tile. Both entries of a pair use the same draws. A draw where a statistic is undefined gives no value for
  it; intervals use the finite draws.
- **Percentiles.** Linear interpolation between order statistics (numpy default).
- **SE.** The sample standard deviation of the finite draws, divisor n - 1.
- **Holm.** Sort the m p values. Reject in order while p_(k) < 0.05 / (m - k + 1), for k = 1, 2, ...; stop at the
  first that fails.
- **MDD.** (1.959963984540054 + 0.8416212335729143) times the SE of the paired draws.
- **Semivariogram.** gamma for a bin is 0.5 times the mean of (r_i - r_j)^2 over the site pairs whose distance falls
  in that bin. The fit and its grid are in R27.

### D5. Label ceiling

From `repos/liquefaction-model/scoreboard/label_ceiling.py`. With prevalence pi, sensitivity Se and specificity Sp:
q = Se pi + (1 - Sp)(1 - pi); PPV = Se pi / q; NPV = Sp (1 - pi) / (1 - q); ceiling C = 0.5 (1 + PPV + NPV - 1).
The prevalence estimate from an observed yes share q is pi = (q - (1 - Sp)) / (Se + Sp - 1). Se and Sp come from
Dawid and Skene (1979) by EM, with one prevalence per earthquake, on the R8 designs only. A grading is usable only if
Se + Sp > 1.

### D6. Census class of an earthquake

From the spec's S5, with the read fixed by this rule book. At the census read (origin plus 7 days, R21) the kit takes
from ComCat the largest Mw-family magnitude listed for the event ("mw, mww, mwc, mwb, mwr; not mwp"), preferred or
not, as the census magnitude, and the preferred hypocentre at that time. An earthquake is in the census class of a
covered region if its census magnitude is at least 5.0 and at least 200 universe sites lie within
"R = min(200, 30 x 10^(0.5 (Mw - 6))) km, hypocentral", with Mw the census magnitude. An earthquake where nothing
liquefied stays in the class.

# Part E. Not settled

### E1. Proposals awaiting calibration (milestone M2)

M2 calibrates each proposal on synthetic data at the card's shape and states its pass bar before the run. A changed
value is a new major version.

| # | Proposal | Rule | M2 question |
|---|---|---|---|
| P1 | DerSimonian-Laird with Hartung-Knapp; 3 earthquakes; Holm across lines | R1 | False HOLDS or FRAGILE on nulls of 3 to 8 earthquakes; how often HOLDS is reachable at k = 3 to 5 (t at 2 degrees of freedom is 4.30) |
| P2 | Semivariogram fit and candidates | R11 c | Interval coverage at the chosen size |
| P3 | Near-copy threshold 0.99; blocking threshold 0.95 | R19, R5 | Distinct models above each; copies below each |
| P4 | Count gate 200, 30, 20 tiles | R11 b | False words at 20 to 60 tiles |
| P5 | Percentile bootstrap, 2,000 draws, bootstrap p | R11 a | Coverage |
| P6 | MDD at 80% power, unadjusted | R11 a | Realised power at the MDD |
| P7 | Holm at 0.05 in two families and the LOSS family | R11 d, R16 c | Familywise false SEPARATES against the R12 target |
| P8 | EQUIVALENT from the 90% interval, margins 0.05 | R11 e | False EQUIVALENT rate |
| P9 | Clips, main clip 1e-6 | R9 | CLIP-SENSITIVE rate on honest entries |
| P10 | Null gate: at least 1,000 exchangeable-entry replicates; fail only if the lower bound exceeds 0.05 | R12 | False-fail rate at true rates 0.04 and 0.05; false-pass rate at 0.08 and 0.10; run time |
| P11 | Shaking-only PGA bins | R10 | Calibration on held-out earthquakes |
| P12 | Label rates by Dawid and Skene on three gradings, or by Hui and Walter; tile-bootstrap intervals | R8 | Coverage of the rate and ceiling intervals |

### E2. Still open

These are not ambiguities: each rule reads one way. They are gaps or risks no rule closes today.

1. **No tester and no outside check performer.** Until one accepts, no card names a best entry, organiser checks on
   outside entries are marks only, and appeals on human checks are granted.
2. **Values not yet named.** The mirror, intake address and image transport (before M6); the R12 (e) published
   number, database and tolerance (before M3). Each is an R27 value, so naming it is a major version.
3. **Region choice.** The organiser chooses which regions get a universe, and on the record track it knows the
   outcomes. "Task fixed after the organiser saw labels" and R16 (g) disclose this; no rule removes it.
4. **Merge rule cost.** Two independent labs at one institution count as one entrant (R22).
5. **Own-row rebuild is partial.** Pairs with other entrants need the tester (R13).
6. **Regions without a USGS ShakeMap** by the census read print NOT RUN. No second source is named.
7. **Volume.** Perhaps one forward census earthquake in one to two decades. No rule adds earthquakes.

### E3. Founder decisions

1. **Before M6:** publish the register and intake address; name the mirror and image transport; sign the pledge.
2. **Before adoption:** adopt revision 4 as v1.0 after review 3.
3. **Pooled line first?** (a) Keep it under the cards (as written). (b) Put it first and change GOALS rule 4.
4. **Before the first outside entry: who is the tester or outside check performer?** (a) Recommended: offer the role
   in writing to one outside academic. (b) Ask after the first buyer replies. (c) Do not ask.
5. **After M6:** may a paying company appear (recommended: yes, paid link printed)? When may we call this a standard
   (recommended: after two or three outside specialists review it in writing)?

### E4. Not verified

- Facts about CASP and CSEP: model memory.
- NZGD terms clause 6.2 needs an NZGD login; whether outside code may run on NZGD files is not checked.
- Bitcoin block-time error is assumed to be a few hours at most (24 h margin, 3 h grace) `[from memory, verify]`.
- ShakeMap `grid.xml` field names, and its intensity column `[from memory, verify]`.
- Hui and Walter identifiability needs different prevalences and conditionally independent gradings
  `[from memory, verify]`.
- The 20 m and 200 m twin rules are quoted from Addendum 1's description of step 11a, not its code. The G6b code works
  in the runner's projection, not the UTM zone named here.
- Canterbury 13 Jun 2011: the smaller shock's time and magnitude are not checked here.
- Geyin and Maurer 2020, Van Calster et al. 2016, CORP diagrams: named by the v0 review; not read.
- The B3 example hash was computed with the B3 recipe and rechecked by review 2, not by the kit.

### E5. Change log

Revision 4, 2026-11-15: answers all 66 findings and 45 attempts of review 2
(`RULEBOOK-V1-REVIEW-2-RESPONSE-2026-11-15.md`); binding summary and precedence rule; status moved to its own file;
one clock; the mirror; a runnable forward contract; checks not done count for the entrant; failure map cut.

Revision 3, 2026-11-15: answered review 1's 61 findings (`RULEBOOK-V1-REVIEW-1-RESPONSE-2026-11-15.md`).

Revision 2, 2026-11-15: settled the twelve unclear places from the kit build.
