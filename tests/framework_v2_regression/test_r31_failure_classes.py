"""Section 20 regression matrix -- the R31 failure classes that V2 must
structurally prevent.

Covered here (V2 code exists):

  CASE A (also L): heterogeneous augmentation plan meeting an executor
                   that lacks per-parent capability -> FRAMEWORK_CAPABILITY_BLOCKER
  CASE D: primary evaluation metric must not aggregate mixed-scope frames
  CASE E: LLM Judge contradicting a deterministic fact ->
          JUDGE_CONTRADICTION, and it is not usable as REVISE evidence
  CASE F: disallowed stage attempts to read a blind-test artifact ->
          BlindTestAccessViolation (fail-closed)
  CASE G: acquisition scope-SHA != evaluation scope-SHA -> cross-stage
          scope-consistency check fails
  CASE H: scientific-critical Student parameter with LEGACY_REUSED / TOOL_DEFAULT
          provenance and no rationale -> validate_recipe_provenance flags it
  CASE K: decision cites a fact_id not in the ledger -> append_decision
          fails closed

Not covered here (V2 code not yet implemented -- listed in the final
report as remaining blockers):
  CASE B (representative split), CASE I (bounded evidence -- covered by
  pre-existing tests/test_bounded_evidence_directory_summary.py),
  CASE J (hallucinated artifact path -- covered by pre-existing
  tests/test_pydantic_ai_root_cause.py).
"""
from __future__ import annotations

import pytest

from framework_v2 import (
    ALLOW,
    AugmentationPlan,
    BlindTestAccessLog,
    BlindTestAccessViolation,
    BlindTestBoundary,
    DecisionLedger,
    DecisionLedgerError,
    DeploymentScopeContract,
    DeterministicFact,
    EvaluationPolicy,
    EvaluationReport,
    ExecutorCapabilities,
    FactVerdict,
    FrameClassification,
    JudgeClaim,
    PerParentAugPolicy,
    ProvenanceClass,
    RecipeParameter,
    ScientificDecisionRecord,
    ScientificJudgment,
    ScopeCategory,
    ScopeRegion,
    StudentRecipePlan,
    augmentation_capability_requirements,
    build_evaluation_report,
    check_capabilities,
    cross_stage_scope_consistent,
    guard_blind_access,
    judgment_usability,
    partition_frames,
    validate_recipe_provenance,
)
from framework_v2.capability import FRAMEWORK_CAPABILITY_BLOCKER


# --------------------------- fixtures ---------------------------
def _mk_scope():
    return DeploymentScopeContract(
        contract_id="scope-1",
        objective="bulk amorphous SiO2-x MD",
        regions=[
            ScopeRegion(region_id="amorphous",
                        category=ScopeCategory.PRIMARY_DEPLOYMENT,
                        membership_rule="amorphous_SiO2-x",
                        rationale="primary target"),
            ScopeRegion(region_id="crystal",
                        category=ScopeCategory.AUXILIARY_SUPPORT,
                        membership_rule="alpha-quartz",
                        rationale="auxiliary stabiliser"),
            ScopeRegion(region_id="pure_si",
                        category=ScopeCategory.OUT_OF_SCOPE,
                        membership_rule="pure_Si",
                        rationale="not in deployment envelope"),
        ],
    )


def _mk_recipe(*, arch_pc=ProvenanceClass.HUMAN_FIXED,
               arch_rationale="human-fixed after pilot comparison"):
    """Build a legal recipe; caller may override architecture provenance."""
    p = lambda name: RecipeParameter(
        name=name, value=42, provenance_class=ProvenanceClass.HUMAN_FIXED,
        rationale="fixed by campaign objective",
    )
    return StudentRecipePlan(
        plan_id="recipe-1",
        descriptor=p("descriptor"),
        architecture=RecipeParameter(
            name="architecture", value="30-30",
            provenance_class=arch_pc, rationale=arch_rationale,
        ),
        optimizer=p("optimizer"),
        learning_rate=p("learning_rate"),
        batch_size=p("batch_size"),
        energy_force_loss_weighting=p("energy_force_loss_weighting"),
        normalization=p("normalization"),
        initial_training_budget=p("initial_training_budget"),
        numerical_precision=p("numerical_precision"),
    )


# =====================================================================
# CASE A (also L): heterogeneous augmentation plan + incapable executor
# =====================================================================
class TestCaseA_CapabilityBlocker:
    def _hetero_plan(self):
        return AugmentationPlan(
            plan_id="aug-1", parent_selection_plan_sha256="p" * 64,
            per_parent=[
                PerParentAugPolicy(parent_id="P1", n_samples=6,
                                   method="gaussian_displacement",
                                   amplitude_range=(0.03, 0.10)),
                PerParentAugPolicy(parent_id="P2", n_samples=12,
                                   method="gaussian_displacement",
                                   amplitude_range=(0.03, 0.15)),
            ],
            required_capabilities=[
                "acquisition.per_parent_augmentation_count",
                "acquisition.per_parent_amplitude_range",
                "acquisition.per_parent_method",
            ],
        )

    def test_hetero_plan_is_heterogeneous(self):
        assert self._hetero_plan().is_heterogeneous() is True

    def test_hetero_plan_capability_requirements(self):
        plan = self._hetero_plan()
        reqs = augmentation_capability_requirements(
            plan_id=plan.plan_id, is_heterogeneous=True)
        assert "acquisition.per_parent_augmentation_count" in reqs.required

    def test_incapable_executor_produces_blocker(self):
        plan = self._hetero_plan()
        reqs = augmentation_capability_requirements(
            plan_id=plan.plan_id, is_heterogeneous=True)
        executor = ExecutorCapabilities(
            executor_id="augment_atoms_v1",
            supported=["acquisition.global_n_per_structure"],  # R31-era
        )
        blocker = check_capabilities(reqs, executor)
        assert blocker is not None
        assert blocker.status == FRAMEWORK_CAPABILITY_BLOCKER
        assert set(blocker.unmet_requirements) == set(reqs.required)

    def test_capable_executor_returns_none(self):
        plan = self._hetero_plan()
        reqs = augmentation_capability_requirements(
            plan_id=plan.plan_id, is_heterogeneous=True)
        executor = ExecutorCapabilities(
            executor_id="augment_atoms_v2",
            supported=list(reqs.required),
        )
        assert check_capabilities(reqs, executor) is None

    def test_homogeneous_plan_has_no_per_parent_requirement(self):
        plan = AugmentationPlan(
            plan_id="aug-2", parent_selection_plan_sha256="p" * 64,
            per_parent=[
                PerParentAugPolicy(parent_id="P1", n_samples=6,
                                   method="gaussian_displacement",
                                   amplitude_range=(0.03, 0.10)),
                PerParentAugPolicy(parent_id="P2", n_samples=6,
                                   method="gaussian_displacement",
                                   amplitude_range=(0.03, 0.10)),
            ],
        )
        assert plan.is_heterogeneous() is False
        reqs = augmentation_capability_requirements(
            plan_id=plan.plan_id, is_heterogeneous=False)
        # No per-parent capabilities needed for a homogeneous plan
        assert reqs.required == []


# =====================================================================
# CASE D: mixed-scope evaluation aggregate must not be the primary
# =====================================================================
class TestCaseD_MixedScopeAggregate:
    def test_primary_partition_has_only_primary_frames(self):
        scope = _mk_scope()
        # 3 primary, 1 aux, 1 out-of-scope frames
        frames = [
            {"frame_id": "f1"}, {"frame_id": "f2"}, {"frame_id": "f3"},
            {"frame_id": "f4"}, {"frame_id": "f5"},
        ]

        def classifier(f):
            return {"f1": "amorphous", "f2": "amorphous", "f3": "amorphous",
                    "f4": "crystal", "f5": "pure_si"}[f["frame_id"]]

        classifications = partition_frames(frames, scope, classifier)
        policy = EvaluationPolicy(
            policy_id="e", scope_contract_sha256=scope.content_sha256(),
            primary_metrics=["E_MAE"], reject_mixed_aggregate_as_primary=True,
        )

        def mean_metric(indices):
            # Toy metric: number of frames in group (per-frame "score" = 1)
            return {"E_MAE": float(len(indices))}

        report = build_evaluation_report(
            report_id="r1", policy=policy, scope=scope,
            classifications=classifications, metric_fn=mean_metric,
            include_mixed_aggregate=True,
        )
        assert report.primary_partition.n_frames == 3
        # mixed aggregate exists but is not primary
        assert report.mixed_aggregate is not None
        assert report.mixed_aggregate.is_primary_partition is False
        assert report.mixed_aggregate.category == ScopeCategory.OUT_OF_SCOPE
        # And structurally cannot be promoted to primary role
        with pytest.raises(Exception):
            EvaluationReport(
                report_id="r2", policy_sha256="a" * 64, scope_sha256="a" * 64,
                primary_partition=report.mixed_aggregate,  # wrong category
                diagnostic_partitions=[], frame_classifications=classifications,
            )

    def test_diagnostic_partitions_may_not_be_promoted(self):
        # Manually constructing an EvaluationReport where the diagnostic
        # partition tries to set is_primary_partition=True must fail
        # closed.
        scope = _mk_scope()
        with pytest.raises(Exception):
            from framework_v2.evaluation import PartitionMetrics
            EvaluationReport(
                report_id="r3", policy_sha256="a" * 64,
                scope_sha256=scope.content_sha256(),
                primary_partition=PartitionMetrics(
                    category=ScopeCategory.PRIMARY_DEPLOYMENT,
                    n_frames=1, metrics={"E_MAE": 1.0},
                    is_primary_partition=True,
                ),
                diagnostic_partitions=[PartitionMetrics(
                    category=ScopeCategory.OUT_OF_SCOPE,
                    n_frames=1, metrics={"E_MAE": 2.0},
                    is_primary_partition=True,  # illegal
                )],
                frame_classifications=[],
            )


# =====================================================================
# CASE E: LLM Judge contradicts a deterministic fact
# =====================================================================
class TestCaseE_JudgeContradiction:
    def test_contradiction_produces_JUDGE_CONTRADICTION_status(self):
        fact = DeterministicFact(
            fact_id="fact-1",
            kind="protected_reference_overlap",
            artifact_sha256="a" * 64,
            observed="PASS",
            expected="PASS",
            verdict=FactVerdict.PASS,
            validator="workflow.validation.protected_reference",
        )
        # Judge claims FAIL for the same fact kind against the same artifact
        judgment = ScientificJudgment(
            judgment_id="j-1", judge_role="judge-2",
            cited_fact_ids=["fact-1"],
            claims=[JudgeClaim(
                claim_id="c-1",
                about_kind="protected_reference_overlap",
                about_artifact_sha256="a" * 64,
                asserted_verdict=FactVerdict.FAIL,
                quote="protected-reference criterion ok=false",
            )],
            interpretation="I claim the protected-reference criterion failed",
            verdict_advice="REVISE",
        )
        status, contradictions = judgment_usability(
            judgment, cited_facts={"fact-1": fact})
        assert status == "JUDGE_CONTRADICTION"
        assert len(contradictions) == 1
        c = contradictions[0]
        assert c.fact_verdict == FactVerdict.PASS
        assert c.claimed_verdict == FactVerdict.FAIL
        assert c.kind == "protected_reference_overlap"

    def test_agreeing_judgment_is_usable(self):
        fact = DeterministicFact(
            fact_id="fact-2", kind="split_leakage", observed=0, expected=0,
            verdict=FactVerdict.PASS,
            validator="workflow.validation.split",
        )
        judgment = ScientificJudgment(
            judgment_id="j-2", judge_role="judge-1",
            cited_fact_ids=["fact-2"],
            claims=[JudgeClaim(
                claim_id="c-2", about_kind="split_leakage",
                asserted_verdict=FactVerdict.PASS,
            )],
            interpretation="Split leakage is zero as expected",
            verdict_advice="PASS",
        )
        status, contradictions = judgment_usability(
            judgment, cited_facts={"fact-2": fact})
        assert status == "USABLE"
        assert contradictions == []


# =====================================================================
# CASE F: blind-test access fails closed
# =====================================================================
class TestCaseF_BlindTestAccess:
    def _boundary(self):
        return BlindTestBoundary(
            boundary_id="r32-blind",
            blind_artifact_sha256s=["b" * 64, "c" * 64],
            allowlist={
                "final_evaluation": ["compute_primary_metric"],
                "physical_validation": ["compute_observables"],
            },
            rationale="fresh R32 blind set; only final eval/PV may touch",
        )

    def test_disallowed_stage_denies_access(self, tmp_path):
        boundary = self._boundary()
        log = BlindTestAccessLog(tmp_path / "access.jsonl")
        with pytest.raises(BlindTestAccessViolation):
            guard_blind_access(
                boundary=boundary, stage="acquisition",
                purpose="parent_selection",
                artifact_sha256="b" * 64, log=log,
            )
        # DENY was logged
        attempts = list(log.iter_attempts())
        assert len(attempts) == 1
        assert attempts[0].outcome == "DENY"
        assert attempts[0].stage == "acquisition"

    def test_allowlisted_stage_allows(self, tmp_path):
        boundary = self._boundary()
        log = BlindTestAccessLog(tmp_path / "access.jsonl")
        outcome = guard_blind_access(
            boundary=boundary, stage="final_evaluation",
            purpose="compute_primary_metric",
            artifact_sha256="b" * 64, log=log,
        )
        assert outcome == "ALLOW"
        assert list(log.iter_attempts())[0].outcome == "ALLOW"

    def test_non_blind_artifact_always_allowed(self, tmp_path):
        boundary = self._boundary()
        log = BlindTestAccessLog(tmp_path / "access.jsonl")
        # d..d is not on the blind list -> access unconditionally allowed
        outcome = guard_blind_access(
            boundary=boundary, stage="acquisition",
            purpose="anything", artifact_sha256="d" * 64, log=log,
        )
        assert outcome == "ALLOW"
        # And should not have written a log entry (non-blind is public)
        assert list(log.iter_attempts()) == []


# =====================================================================
# CASE G: acquisition scope-SHA != evaluation scope-SHA
# =====================================================================
class TestCaseG_ScopeMismatch:
    def test_two_matching_shas_pass(self):
        s = _mk_scope()
        assert cross_stage_scope_consistent(
            s.content_sha256(), s.content_sha256(), s.content_sha256()
        ) is True

    def test_scope_disagreement_fails(self):
        s1 = _mk_scope()
        s2 = DeploymentScopeContract(
            contract_id="scope-2", objective=s1.objective,
            regions=[
                ScopeRegion(region_id="amorphous",
                            category=ScopeCategory.PRIMARY_DEPLOYMENT,
                            membership_rule="amorphous_SiO2-x-DIFFERENT",
                            rationale="silently reinterpreted"),
            ],
        )
        assert cross_stage_scope_consistent(
            s1.content_sha256(), s2.content_sha256()
        ) is False


# =====================================================================
# CASE H: legacy recipe param without rationale
# =====================================================================
class TestCaseH_RecipeProvenance:
    def test_legacy_reused_without_rationale_flags(self):
        recipe = _mk_recipe(
            arch_pc=ProvenanceClass.LEGACY_REUSED,
            arch_rationale="   ",  # whitespace-only
        )
        violations = validate_recipe_provenance(recipe)
        assert len(violations) == 1
        v = violations[0]
        assert v.parameter_name == "architecture"
        assert v.provenance_class == ProvenanceClass.LEGACY_REUSED

    def test_tool_default_without_rationale_flags(self):
        recipe = _mk_recipe(
            arch_pc=ProvenanceClass.TOOL_DEFAULT,
            arch_rationale="",
        )
        violations = validate_recipe_provenance(recipe)
        assert any(v.parameter_name == "architecture" for v in violations)

    def test_legacy_reused_with_rationale_ok(self):
        recipe = _mk_recipe(
            arch_pc=ProvenanceClass.LEGACY_REUSED,
            arch_rationale="Reusing the 30-30 arch from R27 Recipe-B pilot; "
                           "no evidence yet that a different architecture "
                           "would help for amorphous SiO2-x.",
        )
        violations = validate_recipe_provenance(recipe)
        assert violations == []

    def test_evidence_derived_requires_evidence_ref(self):
        p = lambda name: RecipeParameter(
            name=name, value=42,
            provenance_class=ProvenanceClass.HUMAN_FIXED,
            rationale="human-fixed",
        )
        recipe = StudentRecipePlan(
            plan_id="r-e",
            descriptor=p("descriptor"),
            architecture=RecipeParameter(
                name="architecture", value="30-30",
                provenance_class=ProvenanceClass.EVIDENCE_DERIVED,
                evidence=[],  # empty!
                rationale="claims evidence but lists none",
            ),
            optimizer=p("optimizer"), learning_rate=p("learning_rate"),
            batch_size=p("batch_size"),
            energy_force_loss_weighting=p("energy_force_loss_weighting"),
            normalization=p("normalization"),
            initial_training_budget=p("initial_training_budget"),
            numerical_precision=p("numerical_precision"),
        )
        violations = validate_recipe_provenance(recipe)
        assert any(v.parameter_name == "architecture"
                   and v.provenance_class == ProvenanceClass.EVIDENCE_DERIVED
                   for v in violations)


# =====================================================================
# CASE K: decision cannot be traced to evidence (fact not in ledger)
# =====================================================================
class TestCaseK_DecisionTraceability:
    def test_unknown_fact_ref_fails_closed(self, tmp_path):
        ledger = DecisionLedger(tmp_path / "ledger")
        decision = ScientificDecisionRecord(
            decision_id="d-1", stage="acquisition",
            decision="choose 77 parents",
            selected=77,
            deterministic_facts=["fact-missing"],  # never appended
            provenance_class=ProvenanceClass.EVIDENCE_DERIVED,
            rationale="knee analysis suggests 77",
            actor="data-curator",
        )
        with pytest.raises(DecisionLedgerError):
            ledger.append_decision(decision)

    def test_decision_with_known_fact_ref_ok(self, tmp_path):
        ledger = DecisionLedger(tmp_path / "ledger")
        fact = DeterministicFact(
            fact_id="fact-77", kind="parent_count_evidence",
            observed=77, verdict=FactVerdict.PASS,
            validator="framework_v2.tests.synthetic",
        )
        ledger.append_fact(fact)
        decision = ScientificDecisionRecord(
            decision_id="d-2", stage="acquisition",
            decision="choose 77 parents",
            selected=77,
            deterministic_facts=["fact-77"],
            provenance_class=ProvenanceClass.EVIDENCE_DERIVED,
            rationale="knee analysis",
            actor="data-curator",
        )
        ledger.append_decision(decision)
        assert ledger.why("d-2") is not None
        assert ledger.why("d-2").selected == 77

    def test_legacy_provenance_without_rationale_rejected(self, tmp_path):
        ledger = DecisionLedger(tmp_path / "ledger")
        decision = ScientificDecisionRecord(
            decision_id="d-3", stage="training",
            decision="use 30-30 architecture",
            selected="30-30",
            provenance_class=ProvenanceClass.LEGACY_REUSED,
            rationale="",  # empty -> fail closed
            actor="ml-trainer",
        )
        with pytest.raises(DecisionLedgerError):
            ledger.append_decision(decision)

    def test_duplicate_decision_id_rejected(self, tmp_path):
        ledger = DecisionLedger(tmp_path / "ledger")
        d = ScientificDecisionRecord(
            decision_id="d-dup", stage="training", decision="x",
            selected=1, provenance_class=ProvenanceClass.HUMAN_FIXED,
            rationale="obj-fixed", actor="human",
        )
        ledger.append_decision(d)
        with pytest.raises(DecisionLedgerError):
            ledger.append_decision(d)
