# The grading rule (one page)

Source: audit design `docs/evals/AUDIT-DESIGN-2026-11-19.md`, sections 4.1 and 4.3. This page restates it for graders.

## The question

At the marked point, inside the yellow 25 m radius disc, did THIS earthquake bring sand or silt ejecta to the
ground surface (sand boils, ejecta sheets, fissures with ejecta)?

Grade only ejecta that first appears after this earthquake and before the next one. The form shows the window.

## What you see, and what you do not see

1. You see a queue id, a point, an earthquake date and the evidence tiles.
2. You do not see any model output, score, map label or inventory label. Do not look them up.
3. Do not look up the point in published case histories or damage maps. Grade from the tiles.
4. Some queue items are known-answer checks. They look like every other item.
5. Work alone. Do not discuss any item with another grader until both of you have locked your grades.

## How to read the tiles

- **Imagery "after"**: an aerial photo taken inside the window. Ejecta looks grey or light brown, in sheets, cones
  or lines along cracks, on lawns, roads and driveways. Water ponding and fresh silt piles count as clues.
- **Imagery "before"**: an aerial photo from before the earthquake, for comparison. Grey ground in both photos is
  not new ejecta.
- **LiDAR change**: ground level after minus ground level before, 5 m cells. Blue is a rise, red is a fall,
  colour clipped at plus or minus 0.5 m. Grey is no data. A local fall inside the disc, against the 75 to 150 m
  ring, supports settlement. LiDAR alone does not show ejecta. Use it as support, not as the answer.
- **LiDAR caveats.** Tectonic movement is not removed: a broad rise or fall across the whole tile is the fault,
  not the site. Surveys sit several cm apart, so changes under about 0.10 m mean little. Each LiDAR caption gives
  a quality rating for that earthquake:
  - good: usable as support;
  - caution: weak support only;
  - unusable: do not lean on it (Sep 2010: mostly a 2003 baseline, 7 years old, +/-0.15 m);
  - unchecked: nobody has checked it.
- **Post grid flown after the window end.** The caption says so when it happens. The Dec 2011 post grid was flown
  17-18 Feb 2012, after the window closes on 2 Jan 2012, so its change also holds the 2 Jan 2012 aftershocks. Do
  not grade YES from such a tile alone.
- When LiDAR is the only evidence and it is rated unusable, or it was flown after the window end, CANNOT TELL or
  CANNOT GRADE (R1) is the honest answer unless the tile is very clear.

Each tile prints its source and licence. NZGD tiles also carry the EQC "Important notice" (shown under the tiles
and listed in `manifest.json`). It travels with any copy of a tile.

## The answers (all on the form)

| Field | Choose one |
|---|---|
| Ejecta | YES, NO, CANNOT TELL (evidence present but ambiguous) |
| Probability of ejecta | 0.05, 0.25, 0.50, 0.75, 0.95 (always required, also for YES and NO) |
| Severity | none, minor, moderate, severe, CANNOT TELL |
| CANNOT GRADE | only when there is no usable evidence; then give a reason code |
| Evidence used | tick each tile you used; confidence 1 (low) to 3 (high) |
| I recognise this site or its published record | yes or no |

Reason codes for CANNOT GRADE:

- R1 no imagery for the date window
- R2 site obscured (canopy, roof, shadow, cloud)
- R3 ground disturbed (works, clean-up before capture)
- R4 ejecta from an earlier earthquake not cleared
- R5 location doubtful
- R6 other (write the reason)

## Rules of thumb

1. The probability should agree with the class (YES with 0.75 or 0.95, NO with 0.05 or 0.25). The form warns
   when it does not, but it accepts your choice. CANNOT TELL goes with any of the five.
2. Ejecta just outside the disc does not count. Ejecta that crosses into the disc counts.
3. Spend about 12 minutes on each item. The form times each item; the time is used to price the audit.
4. Save often. You may change a grade until you press "Lock all".
