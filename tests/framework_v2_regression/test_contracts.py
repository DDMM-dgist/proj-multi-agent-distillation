"""Contract shape, validation, and content-address determinism."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from framework_v2.contracts import (
    AugmentationPlan,
    DatasetPartitionPlan,
    DeploymentScopeContract,
    EvaluationPolicy,
    PartitionRole,
    PerParentAugPolicy,
    ProvenanceClass,
    RecipeParameter,
    ScopeCategory,
    ScopeRegion,
    StudentRecipePlan,
)


def _mk_scope(*extra_regions):
    return DeploymentScopeContract(
        contract_id="scope-x",
        objective="bulk amorphous SiO2-x MD",
        regions=[
            ScopeRegion(
                region_id="primary",
                category=ScopeCategory.PRIMARY_DEPLOYMENT,
                membership_rule="amorphous_SiO2-x",
                rationale="primary deployment target",
            ),
            *extra_regions,
        ],
    )


class TestDeploymentScopeContract:
    def test_requires_at_least_one_primary_region(self):
        with pytest.raises(ValidationError):
            DeploymentScopeContract(
                contract_id="scope-y",
                objective="obj",
                regions=[ScopeRegion(
                    region_id="aux", category=ScopeCategory.AUXILIARY_SUPPORT,
                    membership_rule="rule", rationale="",
                )],
            )

    def test_rejects_duplicate_region_ids(self):
        with pytest.raises(ValidationError):
            DeploymentScopeContract(
                contract_id="scope-z", objective="obj",
                regions=[
                    ScopeRegion(region_id="p", category=ScopeCategory.PRIMARY_DEPLOYMENT,
                                membership_rule="a", rationale=""),
                    ScopeRegion(region_id="p", category=ScopeCategory.AUXILIARY_SUPPORT,
                                membership_rule="b", rationale=""),
                ],
            )

    def test_sha_is_deterministic_across_construction(self):
        s1 = _mk_scope()
        # Reconstruct from serialized form: SHA must be identical
        s2 = DeploymentScopeContract.model_validate(s1.model_dump(mode="json"))
        assert s1.content_sha256() == s2.content_sha256()

    def test_sha_changes_with_content(self):
        s1 = _mk_scope()
        s2 = _mk_scope(ScopeRegion(
            region_id="aux", category=ScopeCategory.AUXILIARY_SUPPORT,
            membership_rule="crystal", rationale="stabiliser",
        ))
        assert s1.content_sha256() != s2.content_sha256()

    def test_regions_of_filters_correctly(self):
        s = _mk_scope(ScopeRegion(
            region_id="oos", category=ScopeCategory.OUT_OF_SCOPE,
            membership_rule="pure_Si", rationale="",
        ))
        assert [r.region_id for r in s.regions_of(ScopeCategory.PRIMARY_DEPLOYMENT)] == ["primary"]
        assert [r.region_id for r in s.regions_of(ScopeCategory.OUT_OF_SCOPE)] == ["oos"]


class TestAugmentationPlan:
    def _one_policy(self, pid="p1"):
        return PerParentAugPolicy(
            parent_id=pid, n_samples=6, method="gaussian_displacement",
            amplitude_range=(0.03, 0.10),
        )

    def test_homogeneous_plan_reports_homogeneous(self):
        plan = AugmentationPlan(
            plan_id="a1", parent_selection_plan_sha256="abc",
            per_parent=[self._one_policy("p1"), self._one_policy("p2")],
        )
        assert plan.is_heterogeneous() is False

    def test_heterogeneous_plan_reports_heterogeneous(self):
        p1 = self._one_policy("p1")
        p2 = PerParentAugPolicy(
            parent_id="p2", n_samples=12, method="gaussian_displacement",
            amplitude_range=(0.03, 0.15),
        )
        plan = AugmentationPlan(
            plan_id="a2", parent_selection_plan_sha256="abc",
            per_parent=[p1, p2],
        )
        assert plan.is_heterogeneous() is True

    def test_rejects_duplicate_parent_ids(self):
        with pytest.raises(ValidationError):
            AugmentationPlan(
                plan_id="a3", parent_selection_plan_sha256="abc",
                per_parent=[self._one_policy("p1"), self._one_policy("p1")],
            )


class TestDatasetPartitionPlan:
    def test_fractions_must_sum_to_one(self):
        with pytest.raises(ValidationError):
            DatasetPartitionPlan(
                plan_id="split", scope_contract_sha256="a" * 64,
                lineage_key="parent_id", stratification_variables=["regime"],
                fractions={PartitionRole.TRAIN: 0.5, PartitionRole.VALIDATION: 0.2},
                representativeness_requirement="every regime with n>=10 in each split",
            )

    def test_valid_split(self):
        p = DatasetPartitionPlan(
            plan_id="split", scope_contract_sha256="a" * 64,
            lineage_key="parent_id", stratification_variables=["regime"],
            fractions={PartitionRole.TRAIN: 0.8, PartitionRole.VALIDATION: 0.1,
                       PartitionRole.BLIND_TEST: 0.1},
            representativeness_requirement="every regime with n>=10 in each split",
        )
        assert p.fractions[PartitionRole.TRAIN] == 0.8


class TestEvaluationPolicy:
    def test_primary_metrics_required(self):
        with pytest.raises(ValidationError):
            EvaluationPolicy(
                policy_id="e", scope_contract_sha256="a" * 64,
                primary_metrics=[],
            )

    def test_rejects_mixed_aggregate_default(self):
        p = EvaluationPolicy(
            policy_id="e", scope_contract_sha256="a" * 64,
            primary_metrics=["E_MAE", "F_R2"],
        )
        assert p.reject_mixed_aggregate_as_primary is True


class TestStudentRecipePlan:
    def _rp(self, name, pc=ProvenanceClass.HUMAN_FIXED, rationale="obj-fixed"):
        return RecipeParameter(
            name=name, value=42, provenance_class=pc, rationale=rationale,
        )

    def test_all_parameters_returns_core_plus_additional(self):
        core_names = ["descriptor", "architecture", "optimizer", "learning_rate",
                      "batch_size", "energy_force_loss_weighting", "normalization",
                      "initial_training_budget", "numerical_precision"]
        recipe = StudentRecipePlan(
            plan_id="r1",
            **{n: self._rp(n) for n in core_names},
            additional=[self._rp("extra")],
        )
        names = [p.name for p in recipe.all_parameters()]
        assert names == core_names + ["extra"]
