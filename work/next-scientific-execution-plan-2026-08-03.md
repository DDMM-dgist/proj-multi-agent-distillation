# Next Scientific Execution Plan — 2026-08-03

Ordering only. **No calculation, training, DFT/MD submission, API call, or run-state change is
performed by this document.** Framework V1 is COMPLETE (software) and the SiO2 scientific
demonstration is PARTIAL; the missing piece that would move it toward COMPLETE is a single
*executed* recovery iteration on SiO2 (see `project-status-snapshot-2026-08-03.md` §13, §17).
Terms follow that snapshot's glossary.

## 1. Immediate housekeeping (no compute)

- Decide PR #2 (Draft): keep as an optional harness or leave until a provider key exists.
- Confirm which GPU-server state is authoritative for the SiO2 pilot: the fresh
  `sio2-allegro-simplenn-pilot` run (teacher_baseline was queued) vs. starting clean from the
  bundle. The earlier "stages 1–5 PASS" run used pre-integration code and is diagnostic only.
- Verify GPU/SLURM availability before starting (the earlier pilot remained queued because the
  GPU resource was not available during the audit session, with no completed manifest). The
  precise scheduling cause and current availability are a researcher/scheduler-side check.
- Software fix item — **RESOLVED** by merged PR #3 (`45d2dab`): the RDF symbol-selector and
  teacher-MD FixCom/extxyz paths now work across ASE 3.23.0–3.29.0; the `pyproject` floor is
  `ase>=3.23` and a `min-ase` CI job guards it. No longer an open item. See
  `project-status-snapshot-2026-08-03.md` §5a.

## 2. SiO2 recovery-cycle objective

Produce ONE completed, auditable iteration:

    gate REVISE/FAIL → RecoveryPlan (human-approved) → data/labeling change →
    teacher relabel (if declared) → 4-seed Student retrain → same-profile revalidation →
    verify-recovery (changed-artifact hashes) → before/after metric table

This is the single artifact that turns "recovery implemented" into "recovery cycle
demonstrated". It does not, by itself, establish closed-loop autonomy or transferability — those
require the further steps in §11 and a controlled comparison.

## 3. Candidate recovery trigger (do NOT fix in advance)

The trigger is selected only after inspecting the authoritative SiO2 run, its failed/revised
gate, the available artifacts, and retraining feasibility. Candidates, in rough order of
controllability:

- **student_fidelity** — student–teacher force MAE above threshold on the held-out set → data
  coverage remedy (data-curator). Cheapest, most controllable, needs no new DFT.
- **dataset_coverage** — a distribution region under-covered vs. the deployment target → add
  structures in that region.
- **physical_validation** — a required observable (density / rdf_peak:O-Si / nve_drift) outside
  tolerance → MD/protocol or data remedy.
- **simulation_protocol** — an MD-stability/protocol issue (timestep, thermostat, cell) → protocol fix.

**Preferred initial candidate: `student_fidelity` or `dataset_coverage`.** Final trigger
selection requires inspection of the real gate evidence; this plan does not pre-commit to a
specific numeric threshold as the trigger.

## 4. Required RecoveryPlan (declared, human-approved before compute)

Fields per `controller.propose_recovery` (controller.py:687): category, responsible_agent
(RECOVERY_AGENTS), return stage, proposed data/config change, teacher/DFT labeling flag,
retraining flag, revalidation profile, estimated cost. Approved via `approve_recovery`
(controller.py:780) BEFORE any expensive stage.

## 5. Data changes

Return to data-curation: add structures addressing the triggered gap (e.g. top-σ_F production
frames for fidelity, or under-covered regions for coverage), preserving `parent_structure_id`
lineage; record a `DataCoverageReport` delta. No replay unless the plan declares it.

## 6. Teacher relabeling

Only if the RecoveryPlan declares it: relabel the new/changed structures with the SAME teacher
checkpoint (hash-bound), producing a labeling manifest (teacher_model_sha256). New DFT is a
separate declaration and is NOT required for a fidelity- or coverage-driven cycle.

## 7. Four-seed Student retraining

Retrain the 4-seed committee on the revised set. **Preconditions:** SIMPLE-NN driver/runner
resolved, num_workers set (avoid the earlier I/O-bound epoch), and submission via SLURM sbatch
(not a Claude background shell — the earlier training was killed by background-task lifecycle).
Approval-gated (costly_training).

## 8. Same-profile revalidation

Re-run the frozen validation profile (density, rdf_peak:O-Si, nve_drift,
student_teacher_force_fidelity, uncertainty_ranking) on the retrained committee; re-run the gate
(3 lenses). Must use the SAME profile as the failing gate for a valid before/after.

## 9. GO / STOP decision points

**GO to §3 only if ALL of the following are confirmed (server/researcher-side):**

1. Authoritative SiO2 run directory identified.
2. Current run manifest present.
3. Teacher checkpoint accessible.
4. Student training dataset accessible.
5. Four seed checkpoints or training configs accessible.
6. SIMPLE-NN execution environment confirmed (driver resolved, num_workers set).
7. SLURM or local resource available (queue not blocked).
8. Existing validation profile located.
9. Real gate evidence for the chosen recovery trigger present (a genuine REVISE/FAIL, not a
   synthetic one).

**STOP and report missing inputs** if ANY of the above is absent — do not start real compute.
**Approval gate** before every costly stage (training, production MD, new DFT).

## 10. Before/after comparison table (fill only with measured values)

Do not estimate. Leave unmeasured cells as `not evaluated`.

| Metric | Iteration 1 | Iteration 2 |
|---|---|---|
| Dataset frames | not evaluated | not evaluated |
| New parent structures | not evaluated | not evaluated |
| New labeled structures | not evaluated | not evaluated |
| Student–Teacher energy MAE | not evaluated | not evaluated |
| Student–Teacher force MAE | not evaluated | not evaluated |
| Teacher–DFT force MAE | not evaluated | not evaluated |
| Student–DFT force MAE | not evaluated | not evaluated |
| Student-MD–DFT force MAE | not evaluated | not evaluated |
| Committee disagreement (σ_F) | not evaluated | not evaluated |
| High-risk/OOD subset error | not evaluated | not evaluated |
| RDF deviation | not evaluated | not evaluated |
| ADF deviation | not evaluated | not evaluated |
| MD stability | not evaluated | not evaluated |
| Throughput | not evaluated | not evaluated |
| Training cost | not evaluated | not evaluated |
| Teacher-labeling cost | not evaluated | not evaluated |
| Gate verdict | not evaluated | not evaluated |

Plus `verify-recovery` confirming the relevant artifact hashes changed (controller.py:825).

## 11. Second-system entry criteria (do NOT start until §2–§10 done)

Only after one SiO2 recovery cycle is demonstrated: pick a second material/model (per 회의록
2.9, battery/ion-conductor favored for clean diffusivity) to support the transferability claim.
Paper-scope-critical but sequenced AFTER the first real cycle.

## 12. Manuscript update gate

Do NOT edit the manuscript until real before/after data exists (회의록 3.x principle: workflow +
data first, prose second). When it does, sync Methods to the merged framework and frame claims
to exactly the demonstrated scope.

**Methods items to update:**
- the agent-exchange contract is no longer a plain file convention — typed
  `AgentTask`/`AgentResult`/`JudgeVote` contracts are implemented;
- the existing repository validator (`validate_agent_response`) is the final acceptance
  authority — successful PydanticAI parsing alone does NOT accept a result;
- the recovery path is explicitly implemented in the controller (categories, RecoveryPlan,
  approval, iteration, verify-recovery).

**Must NOT be written into the manuscript** (no executed evidence):
- that a real external provider was validated;
- that a real closed-loop recovery cycle was performed;
- that a second system was completed;
- that multi-agent superiority over a single-agent/manual baseline was demonstrated.

**Figure 1 items to reflect:** teacher baseline first → coverage-aware data curation → student
distillation → application-specific validation → judge gate → failure-type-specific return
stage → teacher relabel → student retrain → same-profile revalidation → human approval.

## 13. Terminology

Use the glossary in `project-status-snapshot-2026-08-03.md` §18. `closed-loop`, `active
learning`, and `autonomous` are not used for the recovery cycle until it is actually executed.
