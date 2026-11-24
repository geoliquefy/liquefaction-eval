"""Simulation lab (eval plan step 2d): Canterbury-fitted synthetic worlds that certify the kit's statistical rules.

Protocol: docs/evals/LAB-PROTOCOL-2026-11-19.md (pillar), registered before any run. Synthetic worlds live only here;
no lab number goes on any card. The real CPT positions are read in memory only; nothing per CPT is written.
Modules: fields (Gaussian random fields at sites), engine (vectorised tile bootstrap equal to resample.bootstrap),
rules (tile rules under test), world (one world), fit (fit to the real data), run (conditions, aggregation).
No em dashes.
"""
