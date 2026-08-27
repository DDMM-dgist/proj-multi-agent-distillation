"""V2-H06: replay and supercell bridges obey the single eligibility invariant.

DFT replay frames and supercell/augmentation parents may only be things a
validated :class:`EligibilityCheckResult` already cleared as TRAIN-eligible.
Protected evaluation/test data can never leak in, and a supercell training
strategy is a plan that may be authored before Teacher labels exist -- its
generated candidates enter training only once labeled.
"""
import pytest

from framework_v2.contracts import PartitionRole
from framework_v2.experiment_controls import (
    DFTReplayPolicy,
    SupercellExecutionRecord,
    SupercellStrategy,
    SupercellUse,
    build_replay_training_plan,
)
from framework_v2.protected_eligibility import derive_training_eligible_candidates


def _eligibility(frame_ids):
    return derive_training_eligible_candidates(
        candidate_ids=list(frame_ids),
        candidate_roles={f: PartitionRole.TRAIN for f in frame_ids},
        protected_ids=set(),
        training_split_sha256="split",
        expected_training_split_sha256="split",
        protected_evidence_sha256="prot",
        expected_protected_evidence_sha256="prot",
    )


def test_replay_disabled_by_default():
    policy = DFTReplayPolicy(policy_id="replay")
    assert policy.enabled is False
    assert policy.ratio == 0.0
    with pytest.raises(ValueError, match="replay disabled"):
        build_replay_training_plan(policy, _eligibility(["f1"]))


def test_enabled_replay_requires_matching_eligibility():
    policy = DFTReplayPolicy(
        policy_id="replay",
        enabled=True,
        ratio=0.1,
        selected_frame_ids=["f1", "f2"],
        provenance_refs=["split"],
    )
    plan = build_replay_training_plan(policy, _eligibility(["f1", "f2"]))
    assert set(plan.selected_frame_ids) == {"f1", "f2"}
    assert plan.ratio == 0.1
    assert plan.eligibility_result_sha256


def test_enabled_replay_rejects_ids_outside_eligibility():
    policy = DFTReplayPolicy(
        policy_id="replay",
        enabled=True,
        ratio=0.1,
        selected_frame_ids=["f1", "leak"],
        provenance_refs=["split"],
    )
    with pytest.raises(ValueError, match="do not match validated eligibility"):
        build_replay_training_plan(policy, _eligibility(["f1", "f2"]))


def test_protected_candidate_never_becomes_replay_eligible():
    with pytest.raises(ValueError, match="protected"):
        derive_training_eligible_candidates(
            candidate_ids=["f1", "p1"],
            candidate_roles={
                "f1": PartitionRole.TRAIN,
                "p1": PartitionRole.TRAIN,
            },
            protected_ids={"p1"},
            training_split_sha256="split",
            expected_training_split_sha256="split",
            protected_evidence_sha256="prot",
            expected_protected_evidence_sha256="prot",
        )


def test_non_train_parent_rejected_for_augmentation():
    with pytest.raises(ValueError, match="non-training role"):
        derive_training_eligible_candidates(
            candidate_ids=["f1", "t1"],
            candidate_roles={
                "f1": PartitionRole.TRAIN,
                "t1": PartitionRole.BLIND_TEST,
            },
            protected_ids=set(),
            training_split_sha256="split",
            expected_training_split_sha256="split",
            protected_evidence_sha256="prot",
            expected_protected_evidence_sha256="prot",
        )


def test_supercell_training_strategy_plannable_without_labels():
    strategy = SupercellStrategy(
        strategy_id="supercell",
        use=SupercellUse.TRAINING_STRATEGY,
        parent_ids=["f1"],
        replication_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    )
    assert strategy.parent_ids == ["f1"]
    assert strategy.teacher_labeling_provenance == []


def test_supercell_generated_candidates_require_labels_before_training():
    strategy = SupercellStrategy(
        strategy_id="supercell",
        use=SupercellUse.TRAINING_STRATEGY,
        parent_ids=["f1"],
        replication_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    )
    unlabeled = SupercellExecutionRecord(
        strategy_sha256=strategy.content_sha256(),
        generated_candidate_ids=["f1_2x2x2"],
        parent_ids=["f1"],
        replication_matrix=strategy.replication_matrix,
    )
    assert unlabeled.admits_to_training() is False

    labeled = SupercellExecutionRecord(
        strategy_sha256=strategy.content_sha256(),
        generated_candidate_ids=["f1_2x2x2"],
        parent_ids=["f1"],
        replication_matrix=strategy.replication_matrix,
        teacher_labeling_provenance=["teacher_label_run"],
    )
    assert labeled.admits_to_training() is True
