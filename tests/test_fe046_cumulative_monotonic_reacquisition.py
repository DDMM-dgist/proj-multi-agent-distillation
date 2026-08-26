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


# ================================= Class F -- FE-046 AMENDMENT: domain-local autonomous sizing
# The remediation for an unsupported class is NOT one guaranteed frame + a global top-up. For every
# still-uncovered unsupported class we build that class's OWN candidate population (its admissible
# source families, protected+already-acquired frames excluded) and run the SAME generic FPS
# marginal-novelty saturation sizer WITHIN THAT CLASS ONLY. The class-local knee -- never a human N,
# never a fixed per-class quota -- decides how many NEW frames that class contributes. Presence
# (occupancy>0) is reached at the first pick; acquisition continues until the class's own diversity
# saturates. These tests prove: a diverse pool yields >1, classes size independently and can differ,
# the count is data-driven (not a constant), and the FE-046 cumulative/monotonic invariants hold.
def _compose(prov, *, item_ids, vectors, prior_globals, unsupported, floor_targets, label_map,
             knee_k=1):
    """Run _compose_cumulative_coverage_selection over an explicit eligible-subset fixture
    (locals 0..n-1 at globals 10..10+n-1). ``knee_k=1`` neutralizes the OPTIONAL secondary global
    top-up (max(knee_k, len)=len) so the assertions isolate the DOMAIN-LOCAL additions."""
    eligible_positions = [10 + i for i in range(len(item_ids))]
    fe046 = {
        "unsupported": list(unsupported), "floor_targets": dict(floor_targets),
        "prior_accepted_globals": set(prior_globals),
        "label_index": build_label_index(label_map),
        "coverage_gap_sha256": "sha-test", "sizing_id_prefix": "fe046-test"}
    return prov._compose_cumulative_coverage_selection(
        vectors=vectors, item_ids=item_ids, eligible_positions=eligible_positions,
        knee_k=knee_k, fe046=fe046)


class TestDomainLocalAutonomousSizing:
    # A hermetic fixture: 2 prior bulk_cryst frames (cover bulk_crystalline_SiO2) + three unsupported
    # classes with DIFFERENT class-local geometries:
    #   liquid  -> [0,100,101,102]  (diverse; class-local knee k=2)
    #   vacancy -> [7,7,7]          (descriptor-degenerate; k=1 -- presence, no over-sampling)
    #   bulk_amo-> [3,4]            (only 2 candidates; conservative fallback k=2)
    ITEM_IDS = [
        "bulk_cryst#0", "bulk_cryst#1",
        "liquid#0", "liquid#1", "liquid#2", "liquid#3",
        "vacancy#0", "vacancy#1", "vacancy#2",
        "bulk_amo#0", "bulk_amo#1"]
    VECTORS = [[0.0], [0.5],
               [0.0], [100.0], [101.0], [102.0],
               [7.0], [7.0], [7.0],
               [3.0], [4.0]]
    UNSUPPORTED = ["liquid_or_melt_SiO2", "oxygen_vacancy_SiO2", "amorphous_bulk_SiO2"]
    FLOOR = {"liquid_or_melt_SiO2": ["liquid"],
             "oxygen_vacancy_SiO2": ["vacancy"],
             "amorphous_bulk_SiO2": ["bulk_amo"]}
    PRIOR = (10, 11)

    def _run(self, **over):
        prov = _provider()
        kw = dict(item_ids=self.ITEM_IDS, vectors=self.VECTORS, prior_globals=self.PRIOR,
                  unsupported=self.UNSUPPORTED, floor_targets=self.FLOOR, label_map=_label_map())
        kw.update(over)
        return _compose(prov, **kw)

    def test_A_diverse_class_pool_is_not_limited_to_one_frame(self):
        selected, comp = self._run()
        # liquid has a diverse 4-frame pool: its class-local knee selects TWO frames, not one.
        assert comp["domain_local_added_by_class"]["liquid_or_melt_SiO2"] == 2
        # both liquid picks (locals 2 and 5, the FPS endpoints) are present
        assert {2, 5}.issubset(set(selected))

    def test_B_classes_autonomously_produce_different_additional_N(self):
        _, comp = self._run()
        by_class = comp["domain_local_added_by_class"]
        assert by_class["liquid_or_melt_SiO2"] == 2      # diverse -> knee 2
        assert by_class["oxygen_vacancy_SiO2"] == 1      # degenerate -> presence only
        assert by_class["amorphous_bulk_SiO2"] == 2      # 2 candidates -> fallback full
        # the per-class N genuinely differs (not one uniform number)
        assert len(set(by_class.values())) > 1

    def test_C_class_local_sizing_is_independent_across_classes(self):
        _, base = self._run()
        base_vac = base["domain_local_added_by_class"]["oxygen_vacancy_SiO2"]
        base_liq = base["domain_local_added_by_class"]["liquid_or_melt_SiO2"]
        # Diversify ONLY the liquid pool ([0,100,200,201,202] -> class-local knee 3); leave vacancy
        # untouched. vacancy's autonomous N must be unchanged -> its sizing depends on its own pool.
        item_ids = ["bulk_cryst#0", "bulk_cryst#1",
                    "liquid#0", "liquid#1", "liquid#2", "liquid#3", "liquid#4",
                    "vacancy#0", "vacancy#1", "vacancy#2",
                    "bulk_amo#0", "bulk_amo#1"]
        vectors = [[0.0], [0.5],
                   [0.0], [100.0], [200.0], [201.0], [202.0],
                   [7.0], [7.0], [7.0],
                   [3.0], [4.0]]
        _, alt = self._run(item_ids=item_ids, vectors=vectors)
        assert alt["domain_local_added_by_class"]["liquid_or_melt_SiO2"] == 3  # changed
        assert alt["domain_local_added_by_class"]["oxygen_vacancy_SiO2"] == base_vac  # unchanged
        assert base_liq == 2 and base_vac == 1

    def test_D_count_is_data_driven_not_a_fixed_per_class_quota(self):
        # SAME class, two different diverse pools -> two different autonomous N (2 vs 3): the count is
        # produced by the class-local knee, never a hard-coded per-class number.
        prov = _provider()
        base_ids = ["bulk_cryst#0", "bulk_cryst#1"]
        base_vec = [[0.0], [0.5]]
        lm = _label_map()
        _, c4 = _compose(
            prov, item_ids=base_ids + ["liquid#0", "liquid#1", "liquid#2", "liquid#3"],
            vectors=base_vec + [[0.0], [100.0], [101.0], [102.0]],
            prior_globals=(10, 11), unsupported=["liquid_or_melt_SiO2"],
            floor_targets={"liquid_or_melt_SiO2": ["liquid"]}, label_map=lm)
        _, c5 = _compose(
            prov, item_ids=base_ids + ["liquid#0", "liquid#1", "liquid#2", "liquid#3", "liquid#4"],
            vectors=base_vec + [[0.0], [100.0], [200.0], [201.0], [202.0]],
            prior_globals=(10, 11), unsupported=["liquid_or_melt_SiO2"],
            floor_targets={"liquid_or_melt_SiO2": ["liquid"]}, label_map=lm)
        assert c4["domain_local_added_by_class"]["liquid_or_melt_SiO2"] == 2
        assert c5["domain_local_added_by_class"]["liquid_or_melt_SiO2"] == 3

    def test_E_fe046_cumulative_and_monotonic_invariants_intact(self):
        selected, comp = self._run()
        # (1) cumulative core (prior-accepted locals 0,1) is never dropped
        assert {0, 1}.issubset(set(selected))
        assert comp["cumulative_prior_accepted"] == 2
        # (2) every unsupported class is now covered (occupancy>=1 for each)
        cats = {self.ITEM_IDS[i].rsplit("#", 1)[0] for i in selected}
        assert {"liquid", "vacancy", "bulk_amo"}.issubset(cats)
        # (3) the optional global top-up only GROWS the set, never below the domain-local core
        core = set(selected)
        grown, _ = self._run(knee_k=len(self.ITEM_IDS))
        assert core.issubset(set(grown))
        # provenance keeps FE-046 back-compat fields alongside the new per-class breakdown
        assert comp["n_domain_local_frames"] == sum(
            comp["domain_local_added_by_class"].values())
        assert comp["n_floor_added"] == 3  # one presence marker per newly-covered class

    def test_F_ffv4t_eng2_gap_classes_get_multiframe_domain_local_populations(self):
        # Faithful fixture for the REAL ffv4t-eng2 coverage gap: the three still-unsupported declared
        # classes and their PRIMARY-claim source families + representative pool sizes (from the frozen
        # deployment_scope_v2 label_map and the sanitized_pool manifest). Descriptors are STRUCTURED
        # synthetic (per-family clusters separated by 1000, unit grid within) -- a stand-in for the
        # real material descriptors, which are only available by running the descriptor plugin over
        # ~11k frames at eng3. The point proven here is structural: each gap class receives a
        # data-driven multi-frame domain-local population sized by its OWN pool, NOT exactly one frame
        # by construction. Real run-time counts will emerge from the real descriptor knee at eng3.
        real_families = {
            "liquid_or_melt_SiO2": [("liquid", 313)],
            "oxygen_vacancy_SiO2": [("vacancy", 278), ("vacancy_int_AL", 780)],
            "condensed_pure_Si_boundary": [
                ("silicon_bulk_amo", 159), ("silicon_crystalline_main", 1257),
                ("silicon_liquid", 76), ("silicon_surfaces", 214), ("silicon_defects", 423)]}
        label_map = [{"raw_label": "bulk_cryst", "canonical_domain": "bulk_crystalline_SiO2",
                      "claim_role": "primary_claim"}]
        item_ids = ["bulk_cryst#0", "bulk_cryst#1"]
        vectors = [[0.0], [0.5]]
        floor_targets = {}
        for cls, fams in real_families.items():
            floor_targets[cls] = [f for f, _ in fams]
            for fi, (fam, count) in enumerate(fams):
                label_map.append({"raw_label": fam, "canonical_domain": cls,
                                  "claim_role": "primary_claim"})
                for j in range(count):
                    item_ids.append(f"{fam}#{j}")
                    vectors.append([fi * 1000.0 + j])
        prov = _provider()
        _, comp = _compose(
            prov, item_ids=item_ids, vectors=vectors, prior_globals=(10, 11),
            unsupported=list(real_families), floor_targets=floor_targets, label_map=label_map)
        by_class = comp["domain_local_added_by_class"]
        # every gap class gets a MULTI-frame population (never exactly 1 by construction)
        for cls in real_families:
            assert by_class[cls] > 1, (cls, by_class)
        # and the classes are sized INDEPENDENTLY -> distinct autonomous counts on this fixture
        assert by_class["liquid_or_melt_SiO2"] == 9
        assert by_class["oxygen_vacancy_SiO2"] == 7
        assert by_class["condensed_pure_Si_boundary"] == 6
