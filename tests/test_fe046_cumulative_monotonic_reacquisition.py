"""FE-046 -- cumulative + monotonic coverage-gap reacquisition.

Two invariants, plus the user-directed fail-closed-immediately boundary:

  1. CUMULATIVE + MONOTONIC selection: a targeted reacquisition (return_stage=acquisition after a
     Stage-4 COVERAGE_INSUFFICIENT gate) ACCUMULATES the prior-accepted frames (never supersede-and-
     drop) and adds a per-declared-class occupancy FLOOR so every still-unsupported class is covered
     in ONE cycle -- so coverage can only grow, never oscillate.
  2. STRICT-REDUCTION-OR-FAIL-CLOSED: each successful coverage-gap recovery cycle must strictly
     shrink the unsupported declared-class set; if it does not, the campaign fails closed BEFORE any
     (paid) LLM diagnosis/proposal rather than looping a non-convergent reacquisition.

  Boundary: an UNREMEDIABLE unsupported class (pool physically has no source family the frozen
  label_map maps to it) fails closed IMMEDIATELY (no partial-progress-then-fail).

The tests exercise the selection helper, the cumulative-floor composition (pure), the persisted-fact
detection helpers, the realize-level immediate unremediable fail, and the invariant-2 progress gate.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from framework_v2.acquisition.contracts import (
    AcquisitionStrategyKind,
    CandidateGenerationResult,
    GenerationProvenance,
    ProtectedDisjointnessReport,
)
from framework_v2.acquisition.descriptor_plugins import AcquisitionCapabilityGap
from framework_v2.acquisition.selection import select_candidates_from_indices
from runtimes.pydantic_ai.default_acquisition_provider import (
    FrameworkDefaultAcquisitionProvider,
)
from validation.coverage_gap_assessment import build_label_index


# ---------------------------------------------------------------------------- helpers / fixtures
def _provider():
    return FrameworkDefaultAcquisitionProvider(
        backend_probe=lambda *a, **k: [], teacher_probe=lambda *a, **k: None)


def _gen_result(candidate_ids):
    prov = [
        GenerationProvenance(
            candidate_id=cid, strategy_kind=AcquisitionStrategyKind.EXISTING_POOL_SELECTION,
            backend_id="b0", parent_id=cid, exploration_only=True)
        for cid in candidate_ids]
    return CandidateGenerationResult(
        result_id="gen", strategy_sha256="s0", backend_id="b0",
        candidate_ids=list(candidate_ids), provenance=prov,
        n_requested=len(candidate_ids), n_generated=len(candidate_ids), n_rejected=0)


def _pass_checker(selected_ids):
    return ProtectedDisjointnessReport(
        status="PASS", n_checked=len(selected_ids), n_overlaps=0,
        dft_labels_used_as_selection_scores=False)


def _label_map():
    """raw source-category label -> canonical declared structure class (all primary_claim)."""
    pairs = {
        "bulk_cryst": "bulk_crystalline_SiO2",
        "surfaces": "surface_SiO2",
        "bulk_amo": "amorphous_bulk_SiO2",
        "liquid": "liquid_or_melt_SiO2",
        "vacancy": "oxygen_vacancy_SiO2",
    }
    return [
        {"raw_label": raw, "canonical_domain": dom, "claim_role": "primary_claim"}
        for raw, dom in pairs.items()]


def _compose_fixture(item_ids):
    """Eligible-subset fixture: locals 0..n-1 at globals 10..10+n-1, 1-D descriptors 0..n-1."""
    n = len(item_ids)
    vectors = [[float(i)] for i in range(n)]
    eligible_positions = [10 + i for i in range(n)]
    label_index = build_label_index(_label_map())
    return vectors, eligible_positions, label_index


# =========================================================== Class A -- selection helper contract
class TestSelectCandidatesFromIndices:
    def test_builds_from_explicit_indices_and_dedups_preserving_order(self):
        gen = _gen_result(["a", "b", "c", "d"])
        res = select_candidates_from_indices(
            selection_id="sel", generation_result=gen,
            selected_indices=[2, 0, 2, 3, 0],  # dupes 2,0 dropped after first occurrence
            disjointness_checker=_pass_checker)
        assert res.selected_candidate_ids == ["c", "a", "d"]
        assert res.diversity_evidence["selection_order_indices"] == [2, 0, 3]
        assert res.disjointness_report.status == "PASS"

    def test_fails_closed_on_non_pass_disjointness(self):
        gen = _gen_result(["a", "b", "c"])

        def _fail_checker(selected_ids):
            return ProtectedDisjointnessReport(
                status="FAIL", n_checked=len(selected_ids), n_overlaps=1,
                dft_labels_used_as_selection_scores=False)

        with pytest.raises(ValueError):
            select_candidates_from_indices(
                selection_id="sel", generation_result=gen, selected_indices=[0, 1],
                disjointness_checker=_fail_checker)

    def test_rejects_out_of_range_index(self):
        gen = _gen_result(["a", "b"])
        with pytest.raises(ValueError):
            select_candidates_from_indices(
                selection_id="sel", generation_result=gen, selected_indices=[0, 5],
                disjointness_checker=_pass_checker)


# ================================================ Class B -- cumulative-floor composition (pure)
class TestCumulativeFloorComposition:
    ITEM_IDS = [
        "bulk_cryst#0", "bulk_cryst#1", "surfaces#0", "bulk_amo#0",
        "liquid#0", "vacancy#0", "bulk_cryst#2", "bulk_cryst#3"]
    UNSUPPORTED = ["amorphous_bulk_SiO2", "liquid_or_melt_SiO2", "oxygen_vacancy_SiO2"]
    FLOOR = {"amorphous_bulk_SiO2": ["bulk_amo"],
             "liquid_or_melt_SiO2": ["liquid"],
             "oxygen_vacancy_SiO2": ["vacancy"]}

    def _fe046(self, label_index, prior_globals=(10, 11, 12)):
        return {"unsupported": list(self.UNSUPPORTED), "floor_targets": dict(self.FLOOR),
                "prior_accepted_globals": set(prior_globals), "label_index": label_index}

    def test_retains_prior_and_covers_every_unsupported_class_no_growth(self):
        prov = _provider()
        vectors, eligible_positions, label_index = _compose_fixture(self.ITEM_IDS)
        selected, comp = prov._compose_cumulative_coverage_selection(
            vectors=vectors, item_ids=self.ITEM_IDS, eligible_positions=eligible_positions,
            knee_k=2, fe046=self._fe046(label_index))
        # prior-accepted locals {0,1,2} retained + floor picks for the 3 uncovered classes {3,4,5}.
        assert set(selected) == {0, 1, 2, 3, 4, 5}
        # cumulative core is never dropped
        assert {0, 1, 2}.issubset(set(selected))
        # each unsupported class family is represented among selected
        cats = {self.ITEM_IDS[i].rsplit("#", 1)[0] for i in selected}
        assert {"bulk_amo", "liquid", "vacancy"}.issubset(cats)
        assert comp["cumulative_prior_accepted"] == 3
        assert comp["n_floor_added"] == 3
        assert sorted(comp["floor_added_classes"]) == sorted(self.UNSUPPORTED)

    def test_knee_topup_grows_but_never_below_core(self):
        prov = _provider()
        vectors, eligible_positions, label_index = _compose_fixture(self.ITEM_IDS)
        selected, comp = prov._compose_cumulative_coverage_selection(
            vectors=vectors, item_ids=self.ITEM_IDS, eligible_positions=eligible_positions,
            knee_k=8, fe046=self._fe046(label_index))
        assert len(selected) == 8
        assert {0, 1, 2, 3, 4, 5}.issubset(set(selected))
        assert comp["final_n_selected"] == 8

    def test_accumulation_alone_can_cover_a_class_without_extra_floor_frame(self):
        # If a prior-accepted frame already covers an unsupported class, no floor frame is added
        # for it (covered set includes it). Make vacancy already prior-accepted (global 15 -> local 5).
        prov = _provider()
        vectors, eligible_positions, label_index = _compose_fixture(self.ITEM_IDS)
        selected, comp = prov._compose_cumulative_coverage_selection(
            vectors=vectors, item_ids=self.ITEM_IDS, eligible_positions=eligible_positions,
            knee_k=2, fe046=self._fe046(label_index, prior_globals=(10, 15)))
        # prior locals {0,5}: 5=vacancy already covers oxygen_vacancy_SiO2 -> only amo+liquid floored
        assert {0, 5}.issubset(set(selected))
        assert "oxygen_vacancy_SiO2" not in comp["floor_added_classes"]
        assert set(comp["floor_added_classes"]) == {"amorphous_bulk_SiO2", "liquid_or_melt_SiO2"}

    def test_fail_closed_when_pool_lacks_unsupported_class_family(self):
        prov = _provider()
        # replace the only bulk_amo frame with another bulk_cryst -> amorphous_bulk_SiO2 unremediable
        item_ids = list(self.ITEM_IDS)
        item_ids[3] = "bulk_cryst#9"
        vectors, eligible_positions, label_index = _compose_fixture(item_ids)
        with pytest.raises(AcquisitionCapabilityGap) as ei:
            prov._compose_cumulative_coverage_selection(
                vectors=vectors, item_ids=item_ids, eligible_positions=eligible_positions,
                knee_k=2, fe046=self._fe046(label_index))
        assert ei.value.gap_kind == "POOL_LACKS_STRUCTURE_CLASS"

    def test_fail_closed_when_prior_accepted_frame_no_longer_eligible(self):
        prov = _provider()
        vectors, eligible_positions, label_index = _compose_fixture(self.ITEM_IDS)
        with pytest.raises(AcquisitionCapabilityGap) as ei:
            prov._compose_cumulative_coverage_selection(
                vectors=vectors, item_ids=self.ITEM_IDS, eligible_positions=eligible_positions,
                knee_k=2, fe046=self._fe046(label_index, prior_globals=(10, 999)))
        assert ei.value.gap_kind == "CUMULATIVE_FRAME_DROPPED"


# ===================================================== Class C -- persisted-fact detection helpers
class TestDetectionHelpers:
    def test_latest_coverage_gap_unsupported_reads_most_recent_gate(self):
        prov = _provider()
        controller = SimpleNamespace(state={"events": [
            {"type": "gate", "stage": "s4",
             "coverage_adequacy": {"unsupported_structure_classes": ["a", "b", "c"]}},
            {"type": "other"},
            {"type": "gate", "stage": "s4",
             "coverage_adequacy": {"unsupported_structure_classes": ["a", "b"]}},
        ]})
        assert prov._latest_coverage_gap_unsupported(controller) == ["a", "b"]

    def test_latest_coverage_gap_returns_none_when_no_gate(self):
        prov = _provider()
        controller = SimpleNamespace(state={"events": [{"type": "other"}]})
        assert prov._latest_coverage_gap_unsupported(controller) is None

    def test_prior_accepted_globals_unions_superseded_plans_only(self, tmp_path):
        prov = _provider()
        plan_a = tmp_path / "a_acquisition_plan.json"
        plan_a.write_text(json.dumps({"selected_source_global_indices": [1, 2, 3]}))
        plan_b = tmp_path / "b_acquisition_plan.json"
        plan_b.write_text(json.dumps({"selected_source_global_indices": [3, 4]}))
        active = tmp_path / "c_acquisition_plan.json"
        active.write_text(json.dumps({"selected_source_global_indices": [99]}))
        other = tmp_path / "not_a_plan.json"
        other.write_text(json.dumps({"selected_source_global_indices": [77]}))
        controller = SimpleNamespace(state={"inputs": [
            {"superseded": True, "snapshot": str(plan_a)},
            {"superseded": True, "snapshot": str(plan_b)},
            {"superseded": False, "snapshot": str(active)},   # active -> excluded
            {"superseded": True, "snapshot": str(other)},     # not an acquisition_plan -> excluded
        ]})
        assert prov._prior_accepted_source_globals(controller) == {1, 2, 3, 4}


# ============================================ Class D -- realize-level immediate unremediable fail
class TestRealizeUnremediableFailsClosed:
    def test_unremediable_class_fails_closed_immediately(self, tmp_path, monkeypatch):
        prov = _provider()

        # Pool physically has NO liquid family, but the coverage gate declares liquid_or_melt_SiO2
        # unsupported -> unremediable -> fail closed immediately.
        full_item_ids = ["bulk_cryst#0", "bulk_cryst#1", "surfaces#0"]
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        prov._pool_cache[str(run_dir)] = {
            "vectors": [[0.0], [1.0], [2.0]],
            "item_ids": list(full_item_ids),
            "manifest_path": str(tmp_path / "pool_manifest.json"),
            "duplicate_handling": "none",
        }

        # No protected rows.
        monkeypatch.setattr(
            FrameworkDefaultAcquisitionProvider, "_resolve_protected",
            staticmethod(lambda controller: (None, set())))

        # Fake frozen scope evidence exposing the label_map via .model_dump().
        fake_scope = SimpleNamespace(model_dump=lambda: {"label_map": _label_map()})
        import runtimes.pydantic_ai.acquisition_readiness as ar
        monkeypatch.setattr(
            ar, "_load_scope_classification_evidence",
            lambda controller, project_dir: (fake_scope, {"reason": "fake"}))

        plan = tmp_path / "prior_acquisition_plan.json"
        plan.write_text(json.dumps({"selected_source_global_indices": [0]}))
        controller = SimpleNamespace(
            run_dir=run_dir,
            state={
                "project_dir": str(tmp_path),
                "events": [{"type": "gate", "stage": "data_coverage",
                            "coverage_adequacy": {
                                "unsupported_structure_classes": ["liquid_or_melt_SiO2"]}}],
                "inputs": [{"superseded": True, "snapshot": str(plan)}],
            })
        proposal = SimpleNamespace(
            selected_parent_ids=["bulk_cryst#0"], selected_source_global_indices=[0])

        with pytest.raises(AcquisitionCapabilityGap) as ei:
            prov._realize_existing_pool(
                controller, context=None, m=None, strategy=None,
                backend_id=None, proposal=proposal)
        assert ei.value.gap_kind == "POOL_LACKS_STRUCTURE_CLASS"


# ================================================= Class E -- invariant 2 strict-reduction gate
class TestRecoveryProgressGate:
    def _gate(self, sha, unsupported, stage="data_coverage"):
        return {"type": "gate", "stage": stage,
                "coverage_adequacy": {"report_sha256": sha,
                                      "unsupported_structure_classes": list(unsupported)}}

    def _run(self, events, current_cov, stage="data_coverage"):
        from runtimes.pydantic_ai.cli import _fe046_recovery_progress_check
        controller = SimpleNamespace(state={"events": events})
        return _fe046_recovery_progress_check(controller, stage, current_cov)

    def test_first_cycle_has_no_prior_returns_none(self):
        cur = {"report_sha256": "sha1", "unsupported_structure_classes": ["a", "b"]}
        # only the current cycle's own gate is on record (same sha) -> no distinct prior
        res = self._run([self._gate("sha1", ["a", "b"])], cur)
        assert res is None

    def test_strict_reduction_proceeds(self):
        cur = {"report_sha256": "sha2", "unsupported_structure_classes": ["a", "b"]}
        res = self._run([self._gate("sha1", ["a", "b", "c"])], cur)
        assert res is None

    def test_no_progress_fails_closed(self):
        from runtimes.pydantic_ai.cli import CAMPAIGN_FAILED, EXIT_BLOCKED_POLICY
        cur = {"report_sha256": "sha2", "unsupported_structure_classes": ["a", "b"]}
        res = self._run([self._gate("sha1", ["a", "b"])], cur)
        assert res is not None
        assert res.outcome == CAMPAIGN_FAILED
        assert res.exit_code == EXIT_BLOCKED_POLICY
        assert "RECOVERY_NO_PROGRESS" in res.message

    def test_oscillation_fails_closed(self):
        from runtimes.pydantic_ai.cli import EXIT_BLOCKED_POLICY
        cur = {"report_sha256": "sha2", "unsupported_structure_classes": ["a", "c"]}
        res = self._run([self._gate("sha1", ["a", "b"])], cur)
        assert res is not None
        assert res.exit_code == EXIT_BLOCKED_POLICY

    def test_same_report_sha_is_skipped_compares_earlier_distinct(self):
        # Events (oldest->newest): distinct prior {a,b,c}@sha1, then current's own gate @sha2.
        # The current cov is @sha2; the gate @sha2 is skipped, so it compares against @sha1 -> strict
        # reduction {a,b,c}->{a,b} => proceed.
        cur = {"report_sha256": "sha2", "unsupported_structure_classes": ["a", "b"]}
        events = [self._gate("sha1", ["a", "b", "c"]), self._gate("sha2", ["a", "b"])]
        assert self._run(events, cur) is None
