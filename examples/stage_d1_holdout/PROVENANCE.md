# Stage D-1 HOLDOUT evidence provenance (faithful extracts of the real recorded trail)

UNSEEN AUDITABLE SCIENTIFIC DECISION REPLAY — 8 decisions disjoint from the 7
development checkpoints; not used to design the frozen architecture.

Evidence fixtures are METRICS-ONLY (no verdict); historical verdicts live only in
golden_decisions.json. Deterministic predictions were recorded (DETERMINISTIC_PREDICTIONS.json)
BEFORE any inference and are NOT tuned to historical agreement.

- hd-dft-cell_016: historical=PASS | det=PASS | source: scan_labeled_structures/manifest.csv row cell_016 (judge_decision=PASS)
- hd-dft-cell_011: historical=PASS | det=PASS | source: scan_labeled_structures/manifest.csv row cell_011 (judge_decision=PASS)
- hd-dft-clustered_cell_005: historical=PASS | det=PASS | source: scan_labeled_structures/manifest.csv row clustered_cell_005 (judge_decision=PASS)
- hd-committee-v3final: historical=REVISE | det=REVISE | source: gates/coordination_votes.csv 2026-06-29 v3-final-committee-ADOPT (3x REVISE); coordination_log.csv 2026-06-29 v3-final-committee-ADOPT REVISE
- hd-committee-v3final-v2: historical=REVISE | det=REVISE | source: gates/coordination_votes.csv 2026-07-03 v3-final-v2-committee-ADOPT (3x REVISE); coordination_log.csv 2026-07-03 v3-final-v2-committee-ADOPT REVISE
- hd-production-sizing-revote: historical=REVISE | det=REVISE | source: coordination_log.csv 2026-06-27 production-sizing-gate production-cell-12288-cubic-REVOTE REVISE (formula extra *2 -> effective 2 K/ps not 1; total ~2.9ns not 4.75ns)
- hd-error-decomposition: historical=REVISE | det=REVISE | source: coordination_log.csv 2026-06-27 error-decomposition-gate REVISE (broad c=0.233 vs AL-cell c=0.368 conflation; frame-mean vs atom-weighted unlabeled); teacher_diag/ERROR_SCOPE.md numbers
- hd-committee-uncertainty: historical=REVISE | det=REVISE | source: coordination_log.csv 2026-06-27 committee-uncertainty-gate REVISE ('3.5x' unsupported; artifacts give 2.97x; chain not traceable)
