# Stage D-1 HOLDOUT V2 provenance (faithful extracts of the real recorded trail)

UNSEEN AUDITABLE DECISION REPLAY (architecture v2). 7 decisions selected DETERMINISTICALLY
by sha256(gate|target) rank (SELECTION_MANIFEST.json) — not by verdict/difficulty. Evidence
is METRICS-ONLY; historical verdicts live only in golden_decisions.json; deterministic
predictions recorded BEFORE inference and NOT tuned. Frozen operators only.

- hv2-er-finetune: family=committee-model-selection hist=REVISE det=REVISE | source: coordination_log.csv 2026-06-27 er-finetune-gate teacher-ER-finetune-AB REVISE (eval logs wrong checkpoint; naive no eval log; E-MAE offset unproven)
- hv2-dft-clustered_cell_003: family=dft-physical hist=PASS det=PASS | source: scan_labeled_structures/manifest.csv row clustered_cell_003 (judge_decision=PASS)
- hv2-dft-cell_009: family=dft-physical hist=PASS det=PASS | source: scan_labeled_structures/manifest.csv row cell_009 (judge_decision=PASS)
- hv2-meltquench-protocol: family=production-protocol hist=REVISE det=REVISE | source: coordination_log.csv 2026-06-29 meltquench-protocol-gate production_12288-meltquench-protocol REVISE (protocol standard; only gap = no in-run MSD at melt)
- hv2-cristobalite-seed: family=production-protocol hist=PASS det=PASS | source: coordination_log.csv 2026-06-27 cristobalite-build-gate cristobalite-12288-seed PASS (12288 cubic 57.04A NSi4096/NO8192 rho0.0662 mindist1.54A O1/Si2 reproducible)
- hv2-ph-pipeline: family=science-analysis hist=PASS det=PASS | source: coordination_log.csv 2026-06-27 ph-pipeline-gate persistent-homology-pipeline PASS (weighted-alpha periodic rO1.275/rSi0.375 PD0/1/2 calibrated void radius~1.74A)
- hv2-paper2-findings: family=science-analysis hist=REVISE det=REVISE | source: coordination_log.csv 2026-06-27 production-science-gate paper2-production-findings REVISE (Spearman sign wrong: artifact rho=+0.505 not -0.14)
