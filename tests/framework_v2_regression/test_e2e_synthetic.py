"""Synthetic end-to-end V2 dry run (Section 21).

Threads a synthetic scientific plan through every V2 contract in order,
demonstrating that:

  * a downstream contract can bind (via SHA256) to an upstream contract
  * cross-stage scope consistency holds
  * DecisionLedger accumulates ScientificDecisionRecords per stage
  * ConvergencePolicy classifier accepts the training log's evidence
  * scope-aware EvaluationReport separates PRIMARY_DEPLOYMENT from
    the rest and does NOT let mixed-scope aggregate become primary
  * BlindTestBoundary + guard_blind_access prevent an out-of-order stage
    from touching blind artifacts, but do allow the final evaluation
    stage.

The dry run does NOT execute expensive Teacher/Student compute; it
threads contracts through a synthetic dataset with pre-computed metric
values.
"""
from __future__ import annotations

import pytest

from framework_v2 import (
    AugmentationPlan,
    BlindTestAccessLog,
    BlindTestAccessViolation,
    BlindTestBoundary,
    ConvergencePolicy,
    CoveragePlan,
    DatasetPartitionPlan,
    DecisionLedger,
    DeploymentMDPolicy,
    DeploymentScopeContract,
    DeterministicFact,
    DomainRegime,
    DomainRepresentation,
    EvaluationPolicy,
    ExecutorCapabilities,
    FactVerdict,
    ParentSelectionPlan,
    PartitionRole,
    PerParentAugPolicy,
    PhysicalValidationPolicy,
    ProvenanceClass,
    RecipeParameter,
    ScientificDecisionRecord,
    ScopeCategory,
    ScopeRegion,
    StudentRecipePlan,
    UncertaintyPolicy,
    augmentation_capability_requirements,
    build_convergence_report,
    build_evaluation_report,
    check_capabilities,
    cross_stage_scope_consistent,
    guard_blind_access,
    partition_frames,
    validate_recipe_provenance,
)
from framework_v2.convergence import CONVERGED_EARLY


class TestSyntheticE2E:
    """One combined test that drives an end-to-end V2 workflow. This is
    a smoke/contract-chain test, not a scientific correctness test."""

    def test_full_chain_pass(self, tmp_path):
        ledger = DecisionLedger(tmp_path / "ledger")

        # ---- Stage 1..2  (scope + reference contract) --------------
        scope = DeploymentScopeContract(
            contract_id="r32-scope-v1",
            objective="reliable Student potential for bulk amorphous "
                      "SiO2-x production MD",
            regions=[
                ScopeRegion(
                    region_id="amorphous_target",
                    category=ScopeCategory.PRIMARY_DEPLOYMENT,
                    membership_rule="structure_class == amorphous_SiO2-x",
                    membership_evidence=["evidence/objective_note.md"],
                    rationale="the human-declared campaign target",
                ),
                ScopeRegion(
                    region_id="crystal_stabiliser",
                    category=ScopeCategory.AUXILIARY_SUPPORT,
                    membership_rule="structure_class == alpha-quartz",
                    rationale="known to stabilise amorphous forces training",
                ),
                ScopeRegion(
                    region_id="pure_si",
                    category=ScopeCategory.OUT_OF_SCOPE,
                    membership_rule="composition.O == 0",
                    rationale="not in deployment envelope",
                ),
                ScopeRegion(
                    region_id="blind_set_v1",
                    category=ScopeCategory.BLIND_TEST,
                    membership_rule="frame_id in R32_BLIND_SET_V1",
                    rationale="fresh R32 blind test; not R31's heldout",
                ),
            ],
        )
        scope_sha = scope.content_sha256()

        # Record a decision citing a deterministic fact
        fact_objective = DeterministicFact(
            fact_id="fact-obj-1",
            kind="human_objective_declared",
            observed="bulk amorphous SiO2-x production MD",
            verdict=FactVerdict.PASS,
            validator="human/directive",
        )
        ledger.append_fact(fact_objective)
        ledger.append_decision(ScientificDecisionRecord(
            decision_id="scope-1", stage="reference_validation",
            decision="declare 5-way deployment scope",
            selected="4 regions (1 primary, 1 aux, 1 OOS, 1 blind)",
            deterministic_facts=["fact-obj-1"],
            evidence_refs=[f"scope_sha:{scope_sha}"],
            provenance_class=ProvenanceClass.HUMAN_FIXED,
            rationale="Human directive fixed the primary; auxiliary derived "
                     "from Teacher applicability evidence; blind is fresh.",
            actor="human",
        ))

        # ---- Stage 3a  domain representation -----------------------
        domain = DomainRepresentation(
            representation_id="dom-v1",
            kind="hybrid",
            descriptor="composition + coordination + density (hybrid)",
            regimes=[
                DomainRegime(
                    regime_id="amorph_low_p",
                    label="amorphous low-pressure",
                    membership_rule="amorphous & p<5GPa",
                    within_scope_categories=[ScopeCategory.PRIMARY_DEPLOYMENT],
                ),
                DomainRegime(
                    regime_id="cryst_reference",
                    label="crystalline reference",
                    membership_rule="alpha-quartz",
                    within_scope_categories=[ScopeCategory.AUXILIARY_SUPPORT],
                ),
            ],
            linked_scope_contract_sha256=scope_sha,
        )

        # ---- Stage 3b  coverage + parents ---------------------------
        coverage = CoveragePlan(
            plan_id="cov-1",
            representation_sha256=domain.content_sha256(),
            distance_metric="euclidean_over_composition_and_density",
            stopping_criterion="marginal-coverage-benefit < 0.01",
        )
        parents = ParentSelectionPlan(
            plan_id="par-1",
            coverage_plan_sha256=coverage.content_sha256(),
            selector="FPS+boundary-first",
            selector_config={"boundary_bias": 0.3},
            selected_ids=["P1", "P2", "P3"],
        )

        # ---- Stage 3c  augmentation: HETEROGENEOUS ------------------
        aug = AugmentationPlan(
            plan_id="aug-1",
            parent_selection_plan_sha256=parents.content_sha256(),
            per_parent=[
                PerParentAugPolicy(parent_id="P1", n_samples=4,
                                   method="gaussian_displacement",
                                   amplitude_range=(0.03, 0.10)),
                PerParentAugPolicy(parent_id="P2", n_samples=12,
                                   method="gaussian_displacement",
                                   amplitude_range=(0.03, 0.15)),
                PerParentAugPolicy(parent_id="P3", n_samples=8,
                                   method="gaussian_displacement",
                                   amplitude_range=(0.03, 0.12)),
            ],
            required_capabilities=[
                "acquisition.per_parent_augmentation_count",
                "acquisition.per_parent_amplitude_range",
                "acquisition.per_parent_method",
            ],
        )
        assert aug.is_heterogeneous() is True

        # ---- Capability negotiation MUST PASS before dispatch -------
        reqs = augmentation_capability_requirements(
            plan_id=aug.plan_id, is_heterogeneous=True)
        capable_executor = ExecutorCapabilities(
            executor_id="augment_atoms_v2_per_parent",
            supported=list(reqs.required),
        )
        assert check_capabilities(reqs, capable_executor) is None
        # And an OLD executor is blocked (regression against R31)
        old_executor = ExecutorCapabilities(
            executor_id="augment_atoms_v1_global_only",
            supported=["acquisition.global_n_per_structure"],
        )
        assert check_capabilities(reqs, old_executor) is not None

        # ---- Stage 6  dataset partition -----------------------------
        split = DatasetPartitionPlan(
            plan_id="split-1",
            scope_contract_sha256=scope_sha,
            lineage_key="parent_structure_id",
            stratification_variables=["regime", "composition_bin"],
            fractions={PartitionRole.TRAIN: 0.8,
                       PartitionRole.VALIDATION: 0.1,
                       PartitionRole.BLIND_TEST: 0.1},
            representativeness_requirement=(
                "every discovered regime with n>=10 must appear in each "
                "partition with >= 0.5 * expected share"
            ),
        )

        # ---- Stage 7  Student recipe --------------------------------
        recipe = StudentRecipePlan(
            plan_id="rec-1",
            descriptor=RecipeParameter(
                name="descriptor", value="17-D SiO PES descriptor",
                provenance_class=ProvenanceClass.EVIDENCE_DERIVED,
                evidence=["r31_pilot_descriptor_report.json"],
                rationale="carried over from R31 with explicit re-validation",
            ),
            architecture=RecipeParameter(
                name="architecture", value="30-30",
                provenance_class=ProvenanceClass.LEGACY_REUSED,
                rationale="30-30 arch retained from R27 Recipe-B pilot "
                          "pending R32 pilot comparison; new architecture "
                          "pilots deferred to a follow-up decision",
            ),
            optimizer=RecipeParameter(
                name="optimizer", value="Adam",
                provenance_class=ProvenanceClass.TOOL_DEFAULT,
                rationale="SIMPLE-NN's default optimiser; no evidence a "
                          "swap would improve amorphous MD fidelity",
            ),
            learning_rate=RecipeParameter(
                name="learning_rate", value=1e-4,
                provenance_class=ProvenanceClass.AGENT_HEURISTIC,
                rationale="lr=1e-4 for a small SIMPLE-NN with double "
                          "precision; verified stable in R27-R31",
            ),
            batch_size=RecipeParameter(
                name="batch_size", value=32,
                provenance_class=ProvenanceClass.AGENT_HEURISTIC,
                rationale="matches gradient-noise budget for ~350-frame "
                          "training set",
            ),
            energy_force_loss_weighting=RecipeParameter(
                name="energy_force_loss_weighting", value=(1.0, 1.0),
                provenance_class=ProvenanceClass.AGENT_HEURISTIC,
                rationale="balanced weighting; force focus deferred to pilot",
            ),
            normalization=RecipeParameter(
                name="normalization", value="per-species-centered",
                provenance_class=ProvenanceClass.EVIDENCE_DERIVED,
                evidence=["r27_recipe_B_report.json"],
                rationale="Recipe B ablation confirmed E-alignment benefit",
            ),
            initial_training_budget=RecipeParameter(
                name="initial_training_budget", value=800,
                provenance_class=ProvenanceClass.EVIDENCE_DERIVED,
                evidence=["r31_epoch_pilot_seed202634.json"],
                rationale="R31 epoch pilot showed val loss still improving "
                          "at 200 and 400; 800 provisional per convergence "
                          "policy (may be extended)",
            ),
            numerical_precision=RecipeParameter(
                name="numerical_precision", value="float64",
                provenance_class=ProvenanceClass.EVIDENCE_DERIVED,
                evidence=["r27_precision_note.md"],
                rationale="fp64 matches Teacher to reduce E offset",
            ),
        )
        # Provenance validator MUST accept this recipe (every LEGACY_REUSED /
        # TOOL_DEFAULT / AGENT_HEURISTIC has a non-empty rationale; every
        # EVIDENCE_DERIVED has evidence)
        assert validate_recipe_provenance(recipe) == []

        # ---- Stage 7b  convergence policy ---------------------------
        conv_policy = ConvergencePolicy(
            policy_id="conv-1",
            trailing_window=50, projection_window=50,
            min_relative_improvement=0.05, boundary_tolerance=5,
            metrics=["valid_energy_rmse", "valid_force_rmse"],
            provenance_class=ProvenanceClass.AGENT_HEURISTIC,
            provenance_source="R31 epoch pilot informed default; may be "
                              "revised after R32 pilot",
        )

        # Synthetic committee log (all seeds converged early -- passes gate)
        synthetic_log = "\n".join([
            "Total traning epoch: 800",
            "Best loss lammps potential written at 300 epoch",
            *[f"Epoch     {(i+1)*10} E RMSE(T V) 1.0e+00 1.0e+00 "
              f"F RMSE(T V) 1.5e+00 1.5e+00 learning_rate: 1.0000e-04"
              for i in range(80)],
        ]) + "\n"
        conv_report = build_convergence_report(
            conv_policy, seed_logs={"seed-1": synthetic_log,
                                    "seed-2": synthetic_log})
        assert conv_report["committee_status"] == CONVERGED_EARLY

        # ---- Stage 8  evaluation policy + scope-aware report --------
        eval_policy = EvaluationPolicy(
            policy_id="eval-1",
            scope_contract_sha256=scope_sha,
            primary_metrics=["E_MAE", "F_R2"],
            diagnostic_metrics=["E_R2_aligned"],
            reject_mixed_aggregate_as_primary=True,
        )
        # Synthetic eval frames: 4 amorphous (primary), 2 aux, 1 out-of-scope
        frames = [{"frame_id": f"f{i}"} for i in range(7)]
        cls = {"f0": "amorphous_target", "f1": "amorphous_target",
               "f2": "amorphous_target", "f3": "amorphous_target",
               "f4": "crystal_stabiliser", "f5": "crystal_stabiliser",
               "f6": "pure_si"}

        def classifier(f):
            return cls[f["frame_id"]]

        classifications = partition_frames(frames, scope, classifier)
        # Toy metric_fn: return the frame count and a fake E_MAE proxy
        def metric_fn(indices):
            return {"E_MAE": 100.0 - len(indices), "F_R2": 0.7 + 0.02*len(indices)}

        eval_report = build_evaluation_report(
            report_id="eval-r1", policy=eval_policy, scope=scope,
            classifications=classifications, metric_fn=metric_fn,
            include_mixed_aggregate=True,
        )
        # Primary partition has ONLY 4 amorphous frames
        assert eval_report.primary_partition.n_frames == 4
        assert eval_report.primary_partition.category == ScopeCategory.PRIMARY_DEPLOYMENT
        # Diagnostic partitions cover aux (2) and OOS (1)
        cats = {p.category: p.n_frames for p in eval_report.diagnostic_partitions}
        assert cats.get(ScopeCategory.AUXILIARY_SUPPORT) == 2
        assert cats.get(ScopeCategory.OUT_OF_SCOPE) == 1
        # Mixed aggregate exists but is not primary
        assert eval_report.mixed_aggregate is not None
        assert eval_report.mixed_aggregate.is_primary_partition is False

        # ---- Stage 9/10/11  policies -------------------------------
        _ = UncertaintyPolicy(
            policy_id="unc-1", method="committee_std",
            metrics=["force_std_per_atom"],
        )
        _ = DeploymentMDPolicy(
            policy_id="md-1", scope_contract_sha256=scope_sha,
            ensembles=[{"kind": "NVE", "T": 300}, {"kind": "NVT", "T": 300}],
            stability_checks=["nve_drift", "collapse_detector"],
            max_wall_time_s=1200.0,
        )
        _ = PhysicalValidationPolicy(
            policy_id="pv-1", scope_contract_sha256=scope_sha,
            observables=["rdf", "coordination", "density"],
        )

        # ---- Cross-stage scope consistency --------------------------
        assert cross_stage_scope_consistent(
            split.scope_contract_sha256,
            eval_policy.scope_contract_sha256,
        )

        # ---- Blind-test enforcement ---------------------------------
        blind_boundary = BlindTestBoundary(
            boundary_id="blind-r32",
            blind_artifact_sha256s=["deadbeef" * 8],
            allowlist={"final_evaluation": ["compute_primary_metric"]},
            rationale="R32 fresh blind test",
        )
        access_log = BlindTestAccessLog(tmp_path / "blind_access.jsonl")
        # Acquisition MUST NOT touch blind
        with pytest.raises(BlindTestAccessViolation):
            guard_blind_access(
                boundary=blind_boundary, stage="acquisition",
                purpose="parent_selection",
                artifact_sha256="deadbeef" * 8,
                log=access_log,
            )
        # Final eval IS allowed
        outcome = guard_blind_access(
            boundary=blind_boundary, stage="final_evaluation",
            purpose="compute_primary_metric",
            artifact_sha256="deadbeef" * 8,
            log=access_log,
        )
        assert outcome == "ALLOW"

        # ---- Ledger has records; auditor can answer "why this?" ----
        assert ledger.why("scope-1") is not None
        assert ledger.why("scope-1").selected.startswith("4 regions")
