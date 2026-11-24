# Liquefaction model scorecard v2 (record track)

G6 + G1 + G5, eval milestone M4. Registration: `docs/scorecard-v2-2026-11-19/REGISTRATION.md (pillar 0c736f2)`. Kit evalkit-2026-11-15 at 6638d75. Canonical card hash (spec.card_sha256 of scorecard.json without this field): `cc9f7b744d6bd6b3b4d5ce0956f04e3a17539ef9014de0277223033c08c8847d`. No em dashes.

## Read this first

- every label was read before.
- no skill test.
- no winner named.
- external parity NOT ACHIEVED (2 registered attempts DISAGREE: PARITY-M3-2026-11-19.md).
- audit (step 1d) not graded.
- rule book draft, not adopted.
- no word within an earthquake; every pooled line DESCRIPTIVE.
- founder ruling on the coverage truth (models fixed or averaged) pending; a models-fixed ruling could allow per-earthquake words only under a NEW registration.
- Every interval: 95% tile bootstrap, fixed 4 km tiles on the NZTM2000 grid anchored at the origin, 2000 draws, seed 20261106: **not certified; lab coverage 0.80 to 0.96 depending on an unidentified range**.
- Log loss and calibration on inventory "no" labels measure agreement with the label source, not with the ground. GEM TS22 ran on our open stand-in input layers and is not monotone (step 2c).
- The v1 label caveat text inside each card still says "label error NOT MEASURED"; step 1a has since measured it, and the Menon ceilings below replace that phrase.
- Pre-event effort baseline known limit (step 1 review 2, N4): `in_aerial_footprint` is the union of the mapped CGD0200 classes, not the imagery footprint.
- Aggregates only (NZGD terms clause 6.2). Bench v2 test split never read. No synthetic data.

## Gates

- G-k kit self-checks: PASS
- G-lsn default id reproduces v1 LSN: PASS
- G-v1 kit AUC equals v1 card AUC: PASS
- G-world dataset hashes: PASS
- G-t test split unread: PASS
- G-c no per-site output: PASS

## Canterbury, set B (primary): AUC per earthquake, DESCRIPTIVE

Point [95% tile interval, not certified]. UNSEEN-only = the entry and the baselines on the sites where that entry saw no case at any earthquake (exposure ledger). No word is printed within an earthquake.

### All labelled sites

| Row | Sep 2010 | Feb 2011 | Jun 2011 | Dec 2011 |
|---|---|---|---|---|
| n (yes), 4 km tiles | 4415 (1035), 25 | 4504 (3751), 27 | 3003 (1926), 24 | 3265 (1447), 17 |
| status | SCORED | SCORED | SCORED | SCORED |
| Level 0 (HOUSE) | 0.631 [0.565, 0.748]; UNSEEN-only 0.627 [0.564, 0.741] (n 4388) | 0.682 [0.653, 0.725]; UNSEEN-only 0.681 [0.653, 0.724] (n 4477) | 0.683 [0.629, 0.752]; UNSEEN-only 0.682 [0.628, 0.751] (n 2987) | 0.731 [0.693, 0.809]; UNSEEN-only 0.731 [0.692, 0.811] (n 3252) |
| M0 (HOUSE) | 0.631 [0.564, 0.747]; UNSEEN-only 0.727 [0.593, 0.850] (n 330) | 0.681 [0.652, 0.724]; UNSEEN-only 0.667 [0.525, 0.766] (n 343) | 0.682 [0.628, 0.751]; UNSEEN-only 0.733 [0.607, 0.815] (n 312) | 0.731 [0.693, 0.809] |
| LSN + global frag., authors id (HOUSE) | 0.639 [0.578, 0.743]; UNSEEN-only 0.635 [0.578, 0.736] (n 4388) | 0.699 [0.644, 0.750]; UNSEEN-only 0.699 [0.644, 0.749] (n 4477) | 0.689 [0.628, 0.756]; UNSEEN-only 0.688 [0.626, 0.755] (n 2987) | 0.739 [0.706, 0.804]; UNSEEN-only 0.738 [0.706, 0.805] (n 3252) |
| LSN + CES frag., authors id (HOUSE) | 0.639 [0.578, 0.743] | 0.699 [0.644, 0.750] | 0.689 [0.628, 0.756] | 0.739 [0.706, 0.804] |
| LSN + global frag., default id (beside) | 0.647 [0.587, 0.745]; UNSEEN-only 0.644 [0.587, 0.739] (n 4388) | 0.682 [0.629, 0.731]; UNSEEN-only 0.682 [0.629, 0.730] (n 4477) | 0.690 [0.628, 0.753]; UNSEEN-only 0.689 [0.627, 0.752] (n 2987) | 0.732 [0.696, 0.794]; UNSEEN-only 0.732 [0.695, 0.794] (n 3252) |
| LSN + CES frag., default id (beside) | 0.647 [0.587, 0.745] | 0.682 [0.629, 0.731] | 0.690 [0.628, 0.753] | 0.732 [0.696, 0.794] |
| USGS Zhu 2017 (ORGANISER-RUN) | 0.559 [0.395, 0.670]; UNSEEN-only 0.590 [0.453, 0.699] (n 715) | 0.659 [0.592, 0.729]; UNSEEN-only 0.616 [0.512, 0.731] (n 737) | 0.696 [0.616, 0.782]; UNSEEN-only 0.683 [0.548, 0.817] (n 659) | 0.736 [0.657, 0.831]; UNSEEN-only 0.821 [0.662, 0.921] (n 378) |
| GEM TS22 (ORGANISER-RUN, STAND-IN INPUTS, NOT MONOTONE) | 0.527 [0.445, 0.603] | 0.498 [0.417, 0.589] | 0.477 [0.365, 0.586] | 0.507 [0.365, 0.614] |
| coin | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| earlier-earthquake rate | NOT AVAILABLE | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| shaking only | NOT AVAILABLE | 0.635 [0.570, 0.708] | 0.573 [0.507, 0.689] | 0.685 [0.573, 0.781] |
| pre-event effort | 0.427 [0.217, 0.540] | 0.564 [0.521, 0.598] | 0.475 [0.373, 0.528] | 0.598 [0.486, 0.691] |

### Beside: only CPTs pushed before the earthquake

| Row | Sep 2010 | Feb 2011 | Jun 2011 | Dec 2011 |
|---|---|---|---|---|
| n (yes), 4 km tiles | 0 (0), 0 | 422 (322), 19 | 670 (390), 22 | 545 (209), 17 |
| status | NOT AVAILABLE | SCORED | SCORED | SCORED |
| Level 0 (HOUSE) | counts only | 0.800 [0.611, 0.900] | 0.716 [0.600, 0.814] | 0.717 [0.614, 0.786] |
| M0 (HOUSE) | counts only | 0.798 [0.608, 0.899] | 0.716 [0.597, 0.814] | 0.716 [0.615, 0.786] |
| LSN + global frag., authors id (HOUSE) | counts only | 0.746 [0.582, 0.856] | 0.730 [0.619, 0.823] | 0.713 [0.609, 0.789] |
| LSN + CES frag., authors id (HOUSE) | counts only | 0.746 [0.582, 0.856] | 0.730 [0.619, 0.823] | 0.713 [0.609, 0.789] |
| LSN + global frag., default id (beside) | counts only | 0.735 [0.570, 0.848] | 0.736 [0.617, 0.832] | 0.705 [0.600, 0.786] |
| LSN + CES frag., default id (beside) | counts only | 0.735 [0.570, 0.848] | 0.736 [0.617, 0.832] | 0.705 [0.600, 0.786] |
| USGS Zhu 2017 (ORGANISER-RUN) | counts only | 0.797 [0.606, 0.861] | 0.670 [0.549, 0.769] | 0.733 [0.656, 0.818] |
| GEM TS22 (ORGANISER-RUN, STAND-IN INPUTS, NOT MONOTONE) | counts only | 0.619 [0.459, 0.717] | 0.478 [0.380, 0.570] | 0.468 [0.350, 0.527] |
| coin | counts only | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| earlier-earthquake rate | counts only | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| shaking only | counts only | 0.805 [0.621, 0.854] | 0.561 [0.488, 0.683] | 0.701 [0.591, 0.761] |
| pre-event effort | counts only | 0.392 [0.303, 0.567] | 0.449 [0.345, 0.549] | 0.627 [0.558, 0.669] |

Menon AUC ceilings for set B (step 1a): Sep 2010 bench measured 0.98 (ASSUMPTION-DRIVEN (bench as reference)); Feb 2011 bench measured 0.90 (IDENTIFIED); Jun 2011 road25 S_sep 0.64, S_feb 0.77, S_pool 0.66 (SCENARIOS ONLY); Dec 2011 aerial S_sep 0.85, S_feb 0.50 (inadmissible, step 1a), S_pool 0.85 (SCENARIOS ONLY).

Jun 2011 note (review 1 m6): the S_sep ceiling 0.64 lies below the observed set B AUC of Level 0 (0.683) and Zhu 2017 (0.696), so the data contradict that scenario. S_feb (0.77) and S_pool (0.66) remain.

Menon AUC ceilings for set A (step 1a): Sep 2010 aerial measured 0.79 (ASSUMPTION-DRIVEN (bench as reference)); Feb 2011 aerial measured 0.85 (IDENTIFIED); Jun 2011 aerial S_sep 0.71, S_feb 0.86, S_pool 0.87 (SCENARIOS ONLY); Dec 2011 aerial S_sep 0.85, S_feb 0.50 (inadmissible, step 1a), S_pool 0.85 (SCENARIOS ONLY).

### Exposure per cell (labelled sites, set B): SEEN / POSSIBLE / PLACE / UNSEEN

| Entry | Sep 2010 | Feb 2011 | Jun 2011 | Dec 2011 |
|---|---|---|---|---|
| Level 0 (HOUSE) | 27 / 0 / 0 / 4388 | 27 / 0 / 0 / 4477 | 0 / 0 / 16 / 2987 | 0 / 0 / 13 / 3252 |
| M0 (HOUSE) | 2687 / 0 / 1398 / 330 | 1426 / 0 / 2735 / 343 | 1992 / 0 / 699 / 312 | 3265 / 0 / 0 / 0 |
| LSN + global frag., authors id (HOUSE) | 27 / 0 / 0 / 4388 | 27 / 0 / 0 / 4477 | 0 / 0 / 16 / 2987 | 0 / 0 / 13 / 3252 |
| LSN + CES frag., authors id (HOUSE) | 27 / 4386 / 2 / 0 | 27 / 4473 / 4 / 0 | 0 / 0 / 2989 / 14 | 0 / 0 / 3248 / 17 |
| LSN + global frag., default id (beside) | 27 / 0 / 0 / 4388 | 27 / 0 / 0 / 4477 | 0 / 0 / 16 / 2987 | 0 / 0 / 13 / 3252 |
| LSN + CES frag., default id (beside) | 27 / 4386 / 2 / 0 | 27 / 4473 / 4 / 0 | 0 / 0 / 2989 / 14 | 0 / 0 / 3248 / 17 |
| USGS Zhu 2017 (ORGANISER-RUN) | 1067 / 2207 / 426 / 715 | 2584 / 0 / 1183 / 737 | 0 / 0 / 2344 / 659 | 0 / 0 / 2887 / 378 |
| GEM TS22 (ORGANISER-RUN, STAND-IN INPUTS, NOT MONOTONE) | 0 / 4415 / 0 / 0 | 0 / 4504 / 0 / 0 | 0 / 3003 / 0 / 0 | 0 / 3265 / 0 / 0 |

### UNSEEN-only reads with the baselines on the same sites (set B), DESCRIPTIVE

| Entry | Earthquake | n (yes), 4 km tiles | entry AUC | shaking only | pre-event effort |
|---|---|---|---|---|---|
| Level 0 (HOUSE) | Sep 2010 | 4388 (1015), 25 | 0.627 [0.564, 0.741] | NOT AVAILABLE | 0.432 [0.225, 0.542] |
| Level 0 (HOUSE) | Feb 2011 | 4477 (3727), 27 | 0.681 [0.653, 0.724] | 0.635 [0.570, 0.709] | 0.564 [0.522, 0.599] |
| Level 0 (HOUSE) | Jun 2011 | 2987 (1919), 24 | 0.682 [0.628, 0.751] | 0.571 [0.505, 0.689] | 0.477 [0.377, 0.531] |
| Level 0 (HOUSE) | Dec 2011 | 3252 (1442), 17 | 0.731 [0.692, 0.811] | 0.686 [0.573, 0.784] | 0.597 [0.487, 0.692] |
| M0 (HOUSE) | Sep 2010 | 330 (80), 23 | 0.727 [0.593, 0.850] | NOT AVAILABLE | 0.231 [0.170, 0.346] |
| M0 (HOUSE) | Feb 2011 | 343 (234), 25 | 0.667 [0.525, 0.766] | 0.701 [0.594, 0.761] | 0.481 [0.400, 0.579] |
| M0 (HOUSE) | Jun 2011 | 312 (165), 23 | 0.733 [0.607, 0.815] | 0.673 [0.539, 0.777] | 0.461 [0.343, 0.552] |
| LSN + global frag., authors id (HOUSE) | Sep 2010 | 4388 (1015), 25 | 0.635 [0.578, 0.736] | NOT AVAILABLE | 0.432 [0.225, 0.542] |
| LSN + global frag., authors id (HOUSE) | Feb 2011 | 4477 (3727), 27 | 0.699 [0.644, 0.749] | 0.635 [0.570, 0.709] | 0.564 [0.522, 0.599] |
| LSN + global frag., authors id (HOUSE) | Jun 2011 | 2987 (1919), 24 | 0.688 [0.626, 0.755] | 0.571 [0.505, 0.689] | 0.477 [0.377, 0.531] |
| LSN + global frag., authors id (HOUSE) | Dec 2011 | 3252 (1442), 17 | 0.738 [0.706, 0.805] | 0.686 [0.573, 0.784] | 0.597 [0.487, 0.692] |
| LSN + global frag., default id (beside) | Sep 2010 | 4388 (1015), 25 | 0.644 [0.587, 0.739] | NOT AVAILABLE | 0.432 [0.225, 0.542] |
| LSN + global frag., default id (beside) | Feb 2011 | 4477 (3727), 27 | 0.682 [0.629, 0.730] | 0.635 [0.570, 0.709] | 0.564 [0.522, 0.599] |
| LSN + global frag., default id (beside) | Jun 2011 | 2987 (1919), 24 | 0.689 [0.627, 0.752] | 0.571 [0.505, 0.689] | 0.477 [0.377, 0.531] |
| LSN + global frag., default id (beside) | Dec 2011 | 3252 (1442), 17 | 0.732 [0.695, 0.794] | 0.686 [0.573, 0.784] | 0.597 [0.487, 0.692] |
| USGS Zhu 2017 (ORGANISER-RUN) | Sep 2010 | 715 (191), 23 | 0.590 [0.453, 0.699] | NOT AVAILABLE | 0.426 [0.217, 0.605] |
| USGS Zhu 2017 (ORGANISER-RUN) | Feb 2011 | 737 (519), 25 | 0.616 [0.512, 0.731] | 0.677 [0.562, 0.724] | 0.526 [0.439, 0.588] |
| USGS Zhu 2017 (ORGANISER-RUN) | Jun 2011 | 659 (310), 23 | 0.683 [0.548, 0.817] | 0.643 [0.516, 0.765] | 0.475 [0.338, 0.576] |
| USGS Zhu 2017 (ORGANISER-RUN) | Dec 2011 | 378 (50), 13 | 0.821 [0.662, 0.921] | 0.854 [0.694, 0.960] | 0.795 [0.621, 0.888] |

Cells absent here have too few UNSEEN sites (counts in the exposure table above).


### Probability rows, set B: log loss (clip 1e-6), DESCRIPTIVE

| Row | Sep 2010 | Feb 2011 | Jun 2011 | Dec 2011 |
|---|---|---|---|---|
| LSN + global frag., authors id (HOUSE) | 0.852 [0.659, 1.050] | 1.162 [0.711, 1.703] | 1.276 [0.949, 1.521] | 0.727 [0.606, 0.808] |
| LSN + CES frag., authors id (HOUSE) | 0.749 [0.567, 0.894] | 1.438 [1.001, 1.952] | 1.609 [1.200, 1.941] | 0.780 [0.549, 0.910] |
| LSN + global frag., default id (beside) | 0.970 [0.757, 1.206] | 1.089 [0.647, 1.638] | 1.062 [0.811, 1.241] | 0.761 [0.655, 0.897] |
| LSN + CES frag., default id (beside) | 0.764 [0.591, 0.928] | 1.306 [0.877, 1.826] | 1.282 [0.979, 1.505] | 0.712 [0.583, 0.784] |
| GEM TS22 (ORGANISER-RUN, STAND-IN INPUTS, NOT MONOTONE) | 0.578 [0.491, 0.702] | 1.338 [1.092, 1.587] | 1.308 [1.034, 1.459] | 0.881 [0.641, 1.003] |
| coin | 0.693 [0.693, 0.693] | 0.693 [0.693, 0.693] | 0.693 [0.693, 0.693] | 0.693 [0.693, 0.693] |
| earlier-earthquake rate | NOT AVAILABLE | 1.253 [1.140, 1.300] | 0.675 [0.662, 0.699] | 0.716 [0.690, 0.758] |
| shaking only | NOT AVAILABLE | 0.547 [0.455, 0.729] | 0.856 [0.706, 0.943] | 0.659 [0.593, 0.733] |

## Canterbury, set A (beside)

### All labelled sites

| Row | Sep 2010 | Feb 2011 | Jun 2011 | Dec 2011 |
|---|---|---|---|---|
| n (yes), 4 km tiles | 2700 (902), 13 | 1429 (1309), 21 | 3351 (2864), 16 | 3265 (1447), 17 |
| status | SCORED | SCORED | SCORED | SCORED |
| Level 0 (HOUSE) | 0.569 [0.504, 0.717]; UNSEEN-only 0.563 [0.502, 0.707] (n 2679) | 0.685 [0.594, 0.784]; UNSEEN-only 0.686 [0.593, 0.790] (n 1417) | 0.614 [0.477, 0.733]; UNSEEN-only 0.612 [0.477, 0.729] (n 3337) | 0.731 [0.693, 0.809]; UNSEEN-only 0.731 [0.692, 0.811] (n 3252) |
| M0 (HOUSE) | 0.569 [0.504, 0.717] | 0.683 [0.591, 0.783] | 0.614 [0.477, 0.732] | 0.731 [0.693, 0.809] |
| LSN + global frag., authors id (HOUSE) | 0.585 [0.523, 0.723]; UNSEEN-only 0.579 [0.519, 0.711] (n 2679) | 0.705 [0.575, 0.825]; UNSEEN-only 0.713 [0.576, 0.839] (n 1417) | 0.628 [0.550, 0.706]; UNSEEN-only 0.627 [0.550, 0.705] (n 3337) | 0.739 [0.706, 0.804]; UNSEEN-only 0.738 [0.706, 0.805] (n 3252) |
| LSN + CES frag., authors id (HOUSE) | 0.585 [0.523, 0.723] | 0.705 [0.575, 0.825] | 0.628 [0.550, 0.706] | 0.739 [0.706, 0.804] |
| LSN + global frag., default id (beside) | 0.593 [0.529, 0.736]; UNSEEN-only 0.588 [0.526, 0.724] (n 2679) | 0.664 [0.536, 0.787]; UNSEEN-only 0.673 [0.540, 0.804] (n 1417) | 0.629 [0.564, 0.695]; UNSEEN-only 0.628 [0.563, 0.694] (n 3337) | 0.732 [0.696, 0.794]; UNSEEN-only 0.732 [0.695, 0.794] (n 3252) |
| LSN + CES frag., default id (beside) | 0.593 [0.529, 0.736] | 0.664 [0.536, 0.787] | 0.629 [0.564, 0.695] | 0.732 [0.696, 0.794] |
| USGS Zhu 2017 (ORGANISER-RUN) | 0.542 [0.327, 0.691] | 0.818 [0.643, 0.916] | 0.667 [0.445, 0.832]; UNSEEN-only 0.864 [0.684, 0.948] (n 336) | 0.736 [0.657, 0.831]; UNSEEN-only 0.821 [0.662, 0.921] (n 378) |
| GEM TS22 (ORGANISER-RUN, STAND-IN INPUTS, NOT MONOTONE) | 0.515 [0.400, 0.621] | 0.572 [0.357, 0.728] | 0.532 [0.345, 0.701] | 0.507 [0.365, 0.614] |
| coin | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| earlier-earthquake rate | NOT AVAILABLE | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| shaking only | NOT AVAILABLE | 0.566 [0.499, 0.692] | 0.640 [0.417, 0.828] | 0.685 [0.573, 0.781] |
| pre-event effort | 0.425 [0.155, 0.530] | 0.751 [0.600, 0.849] | 0.436 [0.261, 0.552] | 0.594 [0.493, 0.672] |

### Beside: only CPTs pushed before the earthquake

| Row | Sep 2010 | Feb 2011 | Jun 2011 | Dec 2011 |
|---|---|---|---|---|
| n (yes), 4 km tiles | 0 (0), 0 | 59 (41), 9 | 481 (362), 14 | 545 (209), 17 |
| status | NOT AVAILABLE | UNDERPOWERED | SCORED | SCORED |
| Level 0 (HOUSE) | counts only | counts only | 0.652 [0.487, 0.822] | 0.717 [0.614, 0.786] |
| M0 (HOUSE) | counts only | counts only | 0.651 [0.486, 0.819] | 0.716 [0.615, 0.786] |
| LSN + global frag., authors id (HOUSE) | counts only | counts only | 0.672 [0.528, 0.793] | 0.713 [0.609, 0.789] |
| LSN + CES frag., authors id (HOUSE) | counts only | counts only | 0.672 [0.528, 0.793] | 0.713 [0.609, 0.789] |
| LSN + global frag., default id (beside) | counts only | counts only | 0.671 [0.528, 0.789] | 0.705 [0.600, 0.786] |
| LSN + CES frag., default id (beside) | counts only | counts only | 0.671 [0.528, 0.789] | 0.705 [0.600, 0.786] |
| USGS Zhu 2017 (ORGANISER-RUN) | counts only | counts only | 0.728 [0.478, 0.874] | 0.733 [0.656, 0.818] |
| GEM TS22 (ORGANISER-RUN, STAND-IN INPUTS, NOT MONOTONE) | counts only | counts only | 0.584 [0.346, 0.762] | 0.468 [0.350, 0.527] |
| coin | counts only | counts only | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| earlier-earthquake rate | counts only | counts only | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| shaking only | counts only | counts only | 0.535 [0.377, 0.783] | 0.701 [0.591, 0.761] |
| pre-event effort | counts only | counts only | 0.313 [0.110, 0.491] | 0.598 [0.433, 0.666] |

Label-source sign agreement (set A against set B, every paired difference): Sep 2010: 52 same sign, 5 opposite; Dec 2011: 76 same sign, 7 opposite (Dec 2011 sets A and B use the same aerial labels). Descriptive count only.

Dec 2011 note (review 1 m5): sets A and B carry the same Dec labels, so the Dec count does not test label sources. Its opposite signs come from baselines fitted on different earlier label sets.

## Pooled lines (all DESCRIPTIVE)

| Line | Population | A minus B | Weighting | Earthquakes | Mean [95%] | Leave one out (min to max mean) |
|---|---|---|---|---|---|---|
| C1 | Canterbury, set B | Level 0 minus baseline: shaking only | primary_kit_random_effects | feb2011, jun2011, dec2011 | +0.074 [-0.022, +0.170] | +0.047 to +0.092 |
| C1 | Canterbury, set B | Level 0 minus baseline: shaking only | beside_equal_weight | feb2011, jun2011, dec2011 | +0.068 [-0.024, +0.159] | +0.047 to +0.078 |
| C2 | Canterbury, set B | Level 0 minus baseline: effort only | primary_kit_random_effects | sep2010, feb2011, jun2011, dec2011 | +0.131 [+0.077, +0.184] | +0.123 to +0.178 |
| C2 | Canterbury, set B | Level 0 minus baseline: effort only | beside_equal_weight | sep2010, feb2011, jun2011, dec2011 | +0.166 [+0.091, +0.241] | +0.152 to +0.182 |
| W1 P1 | World P1 | Level 0 minus baseline: shaking only | primary_equal_weight | Chi-Chi, Emilia, Hyogoken-Nambu, Loma Prieta, Tangshan, Westmorland | +0.184 [-0.042, +0.409] | +0.108 to +0.219 |
| W1 P1 | World P1 | Level 0 minus baseline: shaking only | beside_kit_weights | Chi-Chi, Emilia, Hyogoken-Nambu, Loma Prieta, Tangshan, Westmorland | +0.162 [-0.067, +0.391] | +0.038 to +0.218 |
| W2 P1 | World P1 | Level 0 minus USGS Zhu 2017 | primary_equal_weight | Chi-Chi, Hyogoken-Nambu | NOT ALLOWED (R1 (c): 2 earthquake(s) pass the count gate, 3 needed) | |
| W2 P1 | World P1 | Level 0 minus USGS Zhu 2017 | beside_kit_weights | Chi-Chi, Hyogoken-Nambu | NOT ALLOWED (R1 (c): 2 earthquake(s) pass the count gate, 3 needed) | |
| W1 P2 | World P2 | Level 0 minus baseline: shaking only | primary_equal_weight | Chi-Chi, Edgecumbe, Emilia, Hyogoken-Nambu, Imperial Valley, Loma Prieta, Tangshan, Tohoku, Westmorland | +0.076 [-0.095, +0.247] | +0.016 to +0.112 |
| W1 P2 | World P2 | Level 0 minus baseline: shaking only | beside_kit_weights | Chi-Chi, Edgecumbe, Emilia, Hyogoken-Nambu, Imperial Valley, Loma Prieta, Tangshan, Tohoku, Westmorland | +0.078 [-0.097, +0.252] | +0.018 to +0.115 |
| W2 P2 | World P2 | Level 0 minus USGS Zhu 2017 | primary_equal_weight | Chi-Chi, Hyogoken-Nambu | NOT ALLOWED (R1 (c): 2 earthquake(s) pass the count gate, 3 needed) | |
| W2 P2 | World P2 | Level 0 minus USGS Zhu 2017 | beside_kit_weights | Chi-Chi, Hyogoken-Nambu | NOT ALLOWED (R1 (c): 2 earthquake(s) pass the count gate, 3 needed) | |

C2 note (review 1 m4): Q = 2.42 is below k - 1 = 3, so tau^2 = 0. With hk_floor off (the registered kit default) the Hartung-Knapp SE (0.0169) is below the fixed-effect SE, so the C2 primary interval is narrower than a fixed-effect interval.

W1 note (review 1 M2): W1 P1 without Hyogoken-Nambu (Kobe) is +0.108 (equal weight; source: scorecard.json pooled_lines, 'W1 P1', primary_equal_weight, leave_one_out, left_out Hyogoken-Nambu). This is a descriptive side line, not a replacement. Hyogoken-Nambu PGA: 25 of 26 P1 cases carry the NGL tag 'site response analysis' (P2: the same 25, plus 10 PRJ-3012 cases with no per-case method and 1 ShakeMap). A site-response PGA can be lower where the soil softened, so this PGA may depend on the outcome. This is not tested. The same PGA enters Level 0, LSN and LPI. Shaking only ranks below 0.5 here. Kobe stays in every population, because removing it now would follow the results.

W1 P2 leave one region out (equal weight, mean with that region left out): Bay of Plenty +0.098; Imperial-Mexicali Valley +0.097; Kanto and Pacific Tohoku +0.112; North China (Bohai) +0.074; Osaka Bay +0.016; Po Plain +0.073; San Francisco Bay and Monterey Bay +0.058; Taiwan west coast plain +0.086.


Why descriptive: Certificate 4 (Hartung-Knapp, 4 km inputs, k = 3 or 4) holds at the first lab's 2 km model-error range; the real model-difference range is not identified (LAB-CERTIFICATES 11.4, 12.1). No certificate covers equal weighting or world earthquakes.

## World case histories, P1 (primary): AUC, DESCRIPTIVE

1 km local tiles. Record gate: 3 yes, 3 no, 5 tiles.

| Earthquake | n (yes), tiles | Level 0 | BI14-LSN | BI14-LPI | Zhu 2017 | shaking only | Level 0 minus shaking [95%] | B&I exposure | Zhu 2017 exposure | LSN global fragility exposure |
|---|---|---|---|---|---|---|---|---|---|---|
| Chi-Chi | 54 (34), 17 | 0.487 [0.185, 0.996] | 0.744 [0.574, 0.922] | 0.641 [0.312, 0.945] | 0.450 [0.242, 0.582] (0 of 54 cases off the model grid) | 0.482 [0.190, 1.000] | +0.004 [-0.067, +0.034] | POSSIBLE 54 | EVENT IN TRAINING 54 | POSSIBLE 54: in sample (fit) |
| Emilia | 46 (18), 19 | 0.954 [0.875, 1.000] | 0.911 [0.818, 0.981] | 0.946 [0.862, 1.000] | NOT RUN | 0.848 [0.646, 0.962] | +0.106 [+0.005, +0.311] | UNSEEN 46 | UNSEEN 46 | SEEN (same compilation) 46: in sample (fit) |
| Hyogoken-Nambu (see Kobe note) | 26 (19), 21 | 0.722 [0.436, 0.954] | 0.805 [0.593, 0.971] | 0.857 [0.663, 0.979] | 0.605 [0.524, 0.731] (22 of 26 cases off the model grid, read as 0: 15 yes, 7 no) | 0.158 [0.000, 0.375] | +0.564 [+0.350, +0.791] | POSSIBLE 26 | EVENT IN TRAINING 26 | POSSIBLE 26: in sample (fit) |
| Loma Prieta | 122 (98), 26 | 0.731 [0.582, 0.948] | 0.772 [0.649, 0.926] | 0.725 [0.588, 0.922] | NOT RUN | 0.423 [0.199, 0.936] | +0.307 [-0.089, +0.560] | POSSIBLE 122 | EVENT IN TRAINING 122 | POSSIBLE 122: in sample (fit) |
| Tangshan | 10 (7), 10 | 0.667 [0.000, 1.000] | 0.619 [0.000, 1.000] | 0.667 [0.000, 1.000] | NOT RUN | 0.571 [0.222, 1.000] | +0.095 [-0.409, +0.688] | SEEN (name match) 10 | UNSEEN 10 | SEEN (same compilation) 10: in sample (fit) |
| Westmorland | 9 (4), 6 | 1.000 [1.000, 1.000] (uninformative: under 10 tiles) | 0.750 [0.251, 1.000] (uninformative: under 10 tiles) | 1.000 [1.000, 1.000] (uninformative: under 10 tiles) | NOT RUN | 0.975 [0.786, 1.000] (uninformative: under 10 tiles) | +0.025 [+0.000, +0.214] (uninformative: under 10 tiles) | POSSIBLE 9 | UNSEEN 9 | SEEN (same compilation) 9: in sample (fit) |

BI14-LSN is the ranking of both LSN rows. Its B&I column is the triggering chain's exposure. The LSN global fragility was fitted on this same compilation, so it is in sample (a fit) on every earthquake.

Kobe note (review 1 M2): Hyogoken-Nambu PGA: 25 of 26 P1 cases carry the NGL tag 'site response analysis' (P2: the same 25, plus 10 PRJ-3012 cases with no per-case method and 1 ShakeMap). A site-response PGA can be lower where the soil softened, so this PGA may depend on the outcome. This is not tested. The same PGA enters Level 0, LSN and LPI. Shaking only ranks below 0.5 here. Kobe stays in every population, because removing it now would follow the results.

Under the record gate (counts only): 16 earthquakes.

| Earthquake | n (yes), 1 km tiles |
|---|---|
| Achaia-Ilia | 2 (1), 2 |
| Baja | 4 (4), 3 |
| Borah Peak | 3 (3), 2 |
| Edgecumbe | 36 (34), 14 |
| Elmore Ranch | 2 (0), 2 |
| Haicheng | 4 (3), 4 |
| Imperial Valley | 7 (3), 4 |
| Kocaeli | 16 (15), 10 |
| Near East Coast of Honshu | 10 (0), 3 |
| Nihonkai-Chubu | 3 (2), 3 |
| Niigata | 3 (2), 3 |
| Northridge | 35 (35), 7 |
| San Fernando | 6 (6), 3 |
| Superstition Hills | 8 (1), 6 |
| Tohoku | 15 (15), 7 |
| Victoria (Mexicali) | 5 (4), 3 |

## World case histories, P2 (beside, union): AUC, DESCRIPTIVE

1 km local tiles. Record gate: 3 yes, 3 no, 5 tiles.

| Earthquake | n (yes), tiles | Level 0 | BI14-LSN | BI14-LPI | Zhu 2017 | shaking only | Level 0 minus shaking [95%] | B&I exposure | Zhu 2017 exposure | LSN global fragility exposure |
|---|---|---|---|---|---|---|---|---|---|---|
| Chi-Chi | 75 (43), 21 | 0.587 [0.350, 0.843] | 0.703 [0.519, 0.839] | 0.639 [0.509, 0.848] | 0.458 [0.341, 0.539] (0 of 75 cases off the model grid) | 0.585 [0.328, 0.820] | +0.002 [-0.090, +0.146] | POSSIBLE 71, SEEN (name match) 4 | EVENT IN TRAINING 75 | POSSIBLE 54, SEEN (same compilation) 21: in sample (fit) |
| Edgecumbe | 42 (39), 16 | 0.726 [0.323, 1.000] | 0.744 [0.431, 1.000] | 0.684 [0.318, 1.000] | NOT RUN | 0.825 [0.578, 1.000] | -0.098 [-0.338, +0.056] | POSSIBLE 41, SEEN (name match) 1 | UNSEEN 42 | POSSIBLE 36, SEEN (same compilation) 6: in sample (fit) |
| Emilia | 46 (18), 19 | 0.954 [0.875, 1.000] | 0.911 [0.818, 0.981] | 0.946 [0.862, 1.000] | NOT RUN | 0.848 [0.646, 0.962] | +0.106 [+0.005, +0.311] | UNSEEN 46 | UNSEEN 46 | SEEN (same compilation) 46: in sample (fit) |
| Hyogoken-Nambu (see Kobe note) | 36 (26), 25 | 0.812 [0.591, 0.963] | 0.865 [0.696, 0.978] | 0.904 [0.762, 0.987] | 0.615 [0.538, 0.725] (30 of 36 cases off the model grid, read as 0: 20 yes, 10 no) | 0.250 [0.059, 0.454] | +0.562 [+0.382, +0.753] | POSSIBLE 36 | EVENT IN TRAINING 36 | POSSIBLE 26, SEEN (same compilation) 10: in sample (fit) |
| Imperial Valley | 8 (3), 6 | 0.800 [0.333, 1.000] (uninformative: under 10 tiles) | 1.000 [1.000, 1.000] (uninformative: under 10 tiles) | 0.867 [0.500, 1.000] (uninformative: under 10 tiles) | NOT RUN | 0.733 [0.250, 1.000] (uninformative: under 10 tiles) | +0.067 [+0.000, +0.250] (uninformative: under 10 tiles) | POSSIBLE 8 | UNSEEN 8 | SEEN (same compilation) 7, POSSIBLE 1: in sample (fit) |
| Loma Prieta | 152 (114), 30 | 0.722 [0.614, 0.826] | 0.744 [0.647, 0.853] | 0.717 [0.608, 0.854] | NOT RUN | 0.497 [0.309, 0.763] | +0.225 [+0.007, +0.371] | POSSIBLE 130, SEEN (name match) 22 | EVENT IN TRAINING 152 | POSSIBLE 122, SEEN (same compilation) 30: in sample (fit) |
| Tangshan | 10 (7), 10 | 0.667 [0.000, 1.000] | 0.619 [0.000, 1.000] | 0.667 [0.000, 1.000] | NOT RUN | 0.571 [0.222, 1.000] | +0.095 [-0.409, +0.688] | SEEN (name match) 10 | UNSEEN 10 | SEEN (same compilation) 10: in sample (fit) |
| Tohoku | 22 (19), 13 | 0.684 [0.455, 0.909] | 0.544 [0.325, 0.835] | 0.649 [0.417, 0.877] | NOT RUN | 0.895 [0.600, 1.000] | -0.211 [-0.381, +0.000] | POSSIBLE 22 | EVENT IN TRAINING 22 | POSSIBLE 15, SEEN (same compilation) 7: in sample (fit) |
| Westmorland | 10 (5), 8 | 0.920 [0.550, 1.000] (uninformative: under 10 tiles) | 0.720 [0.167, 1.000] (uninformative: under 10 tiles) | 0.920 [0.550, 1.000] (uninformative: under 10 tiles) | NOT RUN | 0.980 [0.857, 1.000] (uninformative: under 10 tiles) | -0.060 [-0.444, +0.100] (uninformative: under 10 tiles) | POSSIBLE 10 | UNSEEN 10 | SEEN (same compilation) 9, POSSIBLE 1: in sample (fit) |

BI14-LSN is the ranking of both LSN rows. Its B&I column is the triggering chain's exposure. The LSN global fragility was fitted on this same compilation, so it is in sample (a fit) on every earthquake.

Kobe note (review 1 M2): Hyogoken-Nambu PGA: 25 of 26 P1 cases carry the NGL tag 'site response analysis' (P2: the same 25, plus 10 PRJ-3012 cases with no per-case method and 1 ShakeMap). A site-response PGA can be lower where the soil softened, so this PGA may depend on the outcome. This is not tested. The same PGA enters Level 0, LSN and LPI. Shaking only ranks below 0.5 here. Kobe stays in every population, because removing it now would follow the results.

Under the record gate (counts only): 13 earthquakes.

| Earthquake | n (yes), 1 km tiles |
|---|---|
| Achaia-Ilia | 2 (1), 2 |
| Baja | 5 (4), 2 |
| Borah Peak | 3 (3), 2 |
| Elmore Ranch | 2 (0), 2 |
| Haicheng | 6 (4), 5 |
| Kocaeli | 16 (15), 10 |
| Near East Coast of Honshu | 10 (0), 3 |
| Nihonkai-Chubu | 3 (2), 3 |
| Niigata | 4 (3), 3 |
| Northridge | 35 (35), 7 |
| San Fernando | 8 (6), 3 |
| Superstition Hills | 9 (1), 6 |
| Victoria (Mexicali) | 6 (5), 3 |

World cross-check against the G1 curated chain: {'lpi_max_abs_diff_vs_g1_curated': 6.246000774368099e-07, 'lsn_max_abs_diff_vs_g1_curated': 3.0398449268886907e-06, 'nan_pattern_equal': True}. Zhu grids: Hyogoken-Nambu 36 cases in P1 and P2 together, 30 of them on empty cells read as 0; Chi-Chi 75 cases in P1 and P2 together, 0 of them on empty cells read as 0.

## Diagnostic only: investigation-density proxy AUC (never a baseline)

| Set | Sep 2010 | Feb 2011 | Jun 2011 | Dec 2011 |
|---|---|---|---|---|
| B | 0.493 [0.291, 0.601] | 0.724 [0.665, 0.769] | 0.677 [0.622, 0.737] | 0.720 [0.650, 0.779] |
| A | 0.562 [0.369, 0.664] | 0.829 [0.679, 0.926] | 0.661 [0.548, 0.804] | 0.690 [0.628, 0.753] |

## Attribution

- PRJ-3012: Geyin and Maurer, global CPT liquefaction case histories, DesignSafe, doi:10.17603/ds2-wftt-mv37, ODC-By
- NGL: Next Generation Liquefaction database (nextgenerationliquefaction.org)
- NZGD (Canterbury CPTs), terms clause 6.2: aggregates only
- PRJ-2937: Geyin, Maurer, Bradley, Green and van Ballegooy 2020, ODC-By

## Deviations

Logged in `REGISTRATION.md` (Deviations). Items 1 to 9 come from the build. Item 3 (Zhu empty cells read as 0) sets 22 of 26 inputs of the Kobe P1 Zhu cell (see the world tables). Items 10 onward are post-review fixes from review 1: renderer, notes and logs only, no new score, no change to any population, entry or rule.

Build logs: `build_v2-attempt1.log` (stopped at the output scan) and `build_v2-attempt2.log` (the card), committed after review 1 and hashed in SEAL.txt. They hold statuses, gate values and timings only.

## Not done

- Independent review 1 is done (CARD STANDS WITH FIXES, REVIEW-1-2026-11-19.md). Its fixes are applied here. There is no round 2 (no blocker).
- OpenTimestamps: a founder step (it sends hashes to public calendars). Not stamped.
- Audit (step 1d) not graded; external parity not achieved; GEM TS22 training events NOT CHECKED; CES fragility CPT list unpublished (POSSIBLE everywhere).
