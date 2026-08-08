# Stage D-1 evidence provenance (faithful extracts of the real recorded decision trail)

CLAUDE_STAGE1_6_HISTORICAL_SUMMARY = RECORDED_BUT_NOT_ARTIFACT_REPLAYABLE
(the historical stage-1..6 summary exists in project records but no reachable workflow-run
 artifact bundle was found; it is NOT used as replay evidence and NOT reconstructed).

Source repo RES = research-sio2-allegro-simplenn-distillation. Evidence fixtures are
METRICS-ONLY (no verdict); historical verdicts live only in golden_decisions.json.

- d1-dft-cell_001: historical=PASS | source: scan_labeled_structures/manifest.csv row cell_001 (judge_decision=PASS)
- d1-dft-clustered_cell_002: historical=PASS | source: scan_labeled_structures/manifest.csv row clustered_cell_002 (judge_decision=PASS)
- d1-dft-cc001: historical=FAIL | source: coordination_log.csv 2026-06-19 judge-gate-clustered clustered_cell_001 FAIL (0/0/3)
- d1-committee-v3: historical=REVISE | source: coordination_log.csv 2026-06-27 committee-reliability-gate v3 REJECT (evidence) / gates/coordination_votes.csv v3 rows
- d1-committee-v5: historical=PASS | source: coordination_log.csv 2026-07-16 committee-reliability-gate v5-committee-ADOPT PASS (3/0/0)
- d1-data-provenance: historical=REVISE | source: coordination_log.csv 2026-06-27 data-provenance-gate REVISE (0/3/0)
- d1-physical-validation: historical=PASS | source: coordination_log.csv 2026-06-27 physical-validation-gate PASS (3/0/0)
