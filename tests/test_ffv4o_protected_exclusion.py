"""ffv4o Stage-3 defect -- protected-reference exclusion regression suite.

Pins the exact ffv4o failure and its fix. ffv4o's autonomous EXISTING_POOL_SELECTION planner
fabricated a ``protected_reference_exclusion_report`` with ``status=PASS`` /
``protected_excluded_count=0`` while never loading the run-bound protected population; the executor's
independent ``assert_source_indices_allowed`` guard then fail-closed on leaked seed-pool global rows
3536 and 5044. The fix routes BOTH the planner and the executor through the ONE canonical resolver
``validation.protected_reference.resolve_protected_population`` and EXCLUDES the protected rows from
the eligible pool BEFORE descriptor/FPS selection and marginal-novelty sizing.

These tests prove, deterministically and without GPU/Teacher/network:

  * the executor guard still rejects the exact ffv4o indices (3536, 5044) -- defense in depth intact;
  * the planner and executor resolve the IDENTICAL protected set from the same reference.yaml;
  * the real ``_realize_existing_pool`` EXCLUDES protected rows before FPS/sizing, and the autonomous
    N is ALLOWED to change when a protected row was the FPS knee (the essential N-changes case);
  * the plan's exclusion report is AUTHORITATIVE (names the reference, real counts) and an anonymous
    or mismatched report is rejected;
  * edge cases: no protected reference, one protected, all-candidates-protected -> fail closed.

Optional-dep-heavy paths (ase + the pydantic-ai runtime) skip cleanly on a core-only install, like
every other pydantic-ai test in this suite.
"""
import hashlib
import os
import tempfile
import types
import unittest
from pathlib import Path


# Exact seed-pool global rows the ffv4o executor guard fail-closed on. Pinned so a regression that
# silently stops excluding them is caught here.
FFV4O_LEAKED_INDICES = (3536, 5044)


def _sha(name):
    return hashlib.sha256(name.encode()).hexdigest()


def _write_protection_reference(root, protected_indices, *, reference_id="ffv4o-test-protection"):
    """Write a valid ``protected-structure-identity`` reference.yaml (+ its geometry-only structures
    file and protected-source-index file) for a synthetic pool, and return its path.

    ``protected-structure-identity`` (FE-023) is the PROTECTION-ONLY reference kind: identity +
    geometry, never DFT labels, and -- unlike ``protected-existing-dft`` -- it hardcodes no specific
    duplicate rows, so it is the correct kind for a synthetic regression fixture."""
    import yaml
    from ase import Atoms
    from ase.io import write as ase_write
    from workflow.integrity import sha256_file
    from validation.protected_reference import (
        PROTECTION_ONLY_STRUCTURE_IDENTITY_REFERENCE_CLASS,
        RECOVERED_HOLDOUT_REQUIRED_PROHIBITIONS,
    )

    root = Path(root)
    # Two geometry-only frames with distinct positions -> unique fingerprints, no label truth.
    frames = [
        Atoms("Si2", positions=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], cell=[8, 8, 8], pbc=True),
        Atoms("Si2", positions=[[0.0, 0.0, 0.0], [2.2, 0.0, 0.0]], cell=[8, 8, 8], pbc=True),
    ]
    struct_path = root / "protection_structures.xyz"
    ase_write(str(struct_path), frames, format="extxyz")

    idx_path = root / "protected_source_indices.txt"
    idx_path.write_text("\n".join(str(int(i)) for i in protected_indices) + "\n", encoding="utf-8")

    cfg = {
        "kind": "protected-structure-identity",
        "reference_class": PROTECTION_ONLY_STRUCTURE_IDENTITY_REFERENCE_CLASS,
        "status": "IDENTITY_AVAILABLE_AND_PROTECTED",
        "reference_id": reference_id,
        "protected_source_indices_file": str(idx_path),
        "protected_source_indices_sha256": sha256_file(idx_path),
        "protected_source_rows": len(protected_indices),
        "structures": {
            "path": str(struct_path),
            "sha256": sha256_file(struct_path),
            "logical_frames": len(frames),
        },
        "prohibited_uses": sorted(RECOVERED_HOLDOUT_REQUIRED_PROHIBITIONS),
    }
    ref_path = root / "reference.yaml"
    ref_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    return ref_path, reference_id


def _controller(run_dir, reference_yaml):
    """A minimal controller whose ``acquire_structures`` stage declares ``reference_yaml`` as the
    protection reference -- exactly the resolution path ``_acquisition_protection_reference_yaml``
    (and thus the executor) uses. ``reference_yaml=None`` models an explicitly unprotected campaign."""
    import yaml
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    params = {}
    if reference_yaml is not None:
        params["reference_yaml"] = str(Path(reference_yaml).resolve())
    wf = {"stages": [
        {"name": "acquisition",
         "pydantic_ai": {"action": "acquire_structures", "parameters": params}}]}
    wf_path = run_dir / "workflow.yaml"
    wf_path.write_text(yaml.safe_dump(wf), encoding="utf-8")
    state = {"inputs": [], "workflow_config": str(wf_path), "project_dir": str(run_dir)}
    return types.SimpleNamespace(run_dir=run_dir, state=state)


class _FakeShaObj:
    """A content-addressed evidence stand-in: a stable ``content_sha256`` derived from a label."""

    def __init__(self, label, **extra):
        self._label = label
        self.__dict__.update(extra)

    def content_sha256(self):
        return _sha(self._label)


def _fake_m():
    """The materialized-evidence bundle ``_realize_existing_pool`` threads into the assembled plan --
    only the attributes that path actually reads, each a stable content-SHA stand-in."""
    from framework_v2.acquisition.contracts import AcquisitionPhase
    return types.SimpleNamespace(
        frozen_artifact=types.SimpleNamespace(evidence_id="ffv4o-ev"),
        objective=_FakeShaObj("objective", phase=AcquisitionPhase.INITIAL),
        inventory=_FakeShaObj("inventory"),
        target_regime_model=_FakeShaObj("trm"),
        region_resolution=_FakeShaObj("rr"),
        coverage=_FakeShaObj("coverage"),
        teacher_identity_sha256=_sha("teacher"))


def _provider():
    from runtimes.pydantic_ai.default_acquisition_provider import FrameworkDefaultAcquisitionProvider
    # The probes are never invoked on the direct _realize_existing_pool path (build_context is
    # bypassed); pass inert callables so construction succeeds.
    return FrameworkDefaultAcquisitionProvider(
        backend_probe=lambda *a, **k: [], teacher_probe=lambda *a, **k: None)


def _strategy():
    from framework_v2.acquisition.contracts import AcquisitionStrategyKind
    return _FakeShaObj("strategy", kind=AcquisitionStrategyKind.EXISTING_POOL_SELECTION)


class ExecutorGuardTests(unittest.TestCase):
    """Defense in depth: the acquisition executor independently rejects a plan that names a protected
    row, and rejects an exclusion report that cannot be authoritative. Directive step 6."""

    def _deps(self):
        try:
            import ase  # noqa: F401
            import yaml  # noqa: F401
            from runtimes.pydantic_ai.executors import _validate_existing_pool_plan  # noqa: F401
        except ModuleNotFoundError as e:
            self.skipTest(f"optional dep not installed: {e}")

    def _plan(self, selected, *, report_reference_id="ffv4o-test-protection", omit_reference_id=False):
        report = {"status": "PASS", "n_checked": len(selected), "n_overlaps": 0,
                  "dft_labels_used_as_selection_scores": False,
                  "protected_candidate_count": 2, "protected_excluded_count": 2}
        if not omit_reference_id:
            report["reference_id"] = report_reference_id
        n = len(selected)
        return {
            "schema_version": 1,
            "pool_path": "pool.json",
            "eligible_source_categories": ["bulk"],
            "selected_parent_structure_ids": [f"bulk#{i}" for i in range(n)],
            "selected_source_global_indices": list(selected),
            "n_selected": n,
            "expected_output_count": n,
            "duplicate_handling": "reject",
            "labeling_population_sizing": {"recommended_population_size": n},
            "protected_reference_exclusion_report": report,
        }

    def test_exact_ffv4o_indices_rejected(self):
        """A plan selecting the exact ffv4o leaked row 3536 is rejected with the leakage error naming
        it -- the guard that fail-closed in ffv4o still works. Pins FFV4O_LEAKED_INDICES."""
        self._deps()
        from runtimes.pydantic_ai.executors import _validate_existing_pool_plan
        d = tempfile.mkdtemp(prefix="ffv4o_guard_")
        ref, _ = _write_protection_reference(d, FFV4O_LEAKED_INDICES)
        plan = self._plan([FFV4O_LEAKED_INDICES[0], 7])
        with self.assertRaises(Exception) as ctx:
            _validate_existing_pool_plan(plan, reference_yaml=str(ref))
        self.assertIn(str(FFV4O_LEAKED_INDICES[0]), str(ctx.exception))
        self.assertIn("leakage", str(ctx.exception).lower())

    def test_disjoint_selection_passes(self):
        """A selection disjoint from the protected set, with an authoritative (named) report,
        validates and echoes the selected rows back."""
        self._deps()
        from runtimes.pydantic_ai.executors import _validate_existing_pool_plan
        d = tempfile.mkdtemp(prefix="ffv4o_guard_")
        ref, ref_id = _write_protection_reference(d, FFV4O_LEAKED_INDICES)
        plan = self._plan([1, 2], report_reference_id=ref_id)
        validated = _validate_existing_pool_plan(plan, reference_yaml=str(ref))
        self.assertEqual(validated["selected_source_global_indices"], [1, 2])
        self.assertEqual(validated["n_selected"], 2)

    def test_anonymous_report_rejected(self):
        """A protected run whose exclusion report omits reference_id is rejected: an anonymous PASS
        cannot masquerade as an authoritative exclusion (the shape ffv4o emitted)."""
        self._deps()
        from runtimes.pydantic_ai.executors import _validate_existing_pool_plan
        d = tempfile.mkdtemp(prefix="ffv4o_guard_")
        ref, _ = _write_protection_reference(d, FFV4O_LEAKED_INDICES)
        plan = self._plan([1, 2], omit_reference_id=True)
        with self.assertRaises(Exception) as ctx:
            _validate_existing_pool_plan(plan, reference_yaml=str(ref))
        self.assertIn("anonymous", str(ctx.exception).lower())

    def test_wrong_reference_id_rejected(self):
        """A report naming a DIFFERENT reference than the run binds is rejected -- the exclusion must
        attest against the run-bound protected identity, not an arbitrary one."""
        self._deps()
        from runtimes.pydantic_ai.executors import _validate_existing_pool_plan
        d = tempfile.mkdtemp(prefix="ffv4o_guard_")
        ref, _ = _write_protection_reference(d, FFV4O_LEAKED_INDICES)
        plan = self._plan([1, 2], report_reference_id="some-other-reference")
        with self.assertRaises(Exception) as ctx:
            _validate_existing_pool_plan(plan, reference_yaml=str(ref))
        self.assertIn("does not match", str(ctx.exception).lower())


class ResolverParityTests(unittest.TestCase):
    """Directive step 9: PROVE planner-resolved protected set == executor-resolved protected set. Both
    consume the ONE canonical ``resolve_protected_population`` over the same reference.yaml."""

    def _deps(self):
        try:
            import ase  # noqa: F401
            import yaml  # noqa: F401
            from runtimes.pydantic_ai.default_acquisition_provider import (  # noqa: F401
                FrameworkDefaultAcquisitionProvider)
        except ModuleNotFoundError as e:
            self.skipTest(f"optional dep not installed: {e}")

    def test_planner_and_executor_resolve_identical_set(self):
        self._deps()
        from validation.protected_reference import resolve_protected_population
        d = tempfile.mkdtemp(prefix="ffv4o_parity_")
        protected = [3, 3536, 5044, 9]
        ref, ref_id = _write_protection_reference(d, protected)
        controller = _controller(os.path.join(d, "run"), ref)

        planner_ref_id, planner_set = _provider()._resolve_protected(controller)
        executor = resolve_protected_population(str(ref))

        self.assertEqual(planner_ref_id, ref_id)
        self.assertEqual(planner_set, set(protected))
        self.assertEqual(set(executor["protected_source_indices"]), planner_set)
        self.assertEqual(executor["reference_id"], planner_ref_id)

    def test_no_reference_resolves_to_empty(self):
        """An explicitly unprotected campaign resolves to (None, empty set) -- never a fabricated
        protected population, never a silent failure."""
        self._deps()
        d = tempfile.mkdtemp(prefix="ffv4o_parity_")
        controller = _controller(os.path.join(d, "run"), None)
        ref_id, protected_set = _provider()._resolve_protected(controller)
        self.assertIsNone(ref_id)
        self.assertEqual(protected_set, set())


class RealizeExclusionTests(unittest.TestCase):
    """The core ffv4o fix: the REAL ``_realize_existing_pool`` excludes the canonically-resolved
    protected rows BEFORE FPS/sizing, emits an authoritative report, and lets autonomous N change."""

    def _deps(self):
        try:
            import ase  # noqa: F401
            import numpy  # noqa: F401
            import yaml  # noqa: F401
            from framework_v2.acquisition.selection import select_candidates  # noqa: F401
            from runtimes.pydantic_ai.default_acquisition_provider import (  # noqa: F401
                FrameworkDefaultAcquisitionProvider)
        except ModuleNotFoundError as e:
            self.skipTest(f"optional dep not installed: {e}")

    def _seed_pool(self, provider, controller, values):
        """Cache a single-category ('bulk') eligible pool of 1-D descriptor vectors, so a protected
        seed-pool global index ``g`` names position ``g`` here (the same space the executor uses)."""
        item_ids = [f"bulk#{i}" for i in range(len(values))]
        vectors = [[float(v)] for v in values]
        provider._pool_cache[str(controller.run_dir)] = {
            "vectors": vectors, "item_ids": item_ids,
            "manifest_path": "pool.json", "duplicate_handling": "reject"}
        return item_ids

    def _realize(self, provider, controller, proposal_parent_ids=("bulk#0",)):
        proposal = types.SimpleNamespace(selected_parent_ids=list(proposal_parent_ids))
        return provider._realize_existing_pool(
            controller, None, _fake_m(), _strategy(), "existing_pool_selection.ase", proposal)

    # Full pool [0,10,10.1,10.2,10.3,50]: sizing knee -> N=3. Excluding the protected far point at
    # global index 5 (value 50) collapses the far cluster -> N=2. Empirically pinned; proves the
    # autonomous N is ALLOWED to change after a legitimate protected exclusion.
    KNEE_VALUES = [0.0, 10.0, 10.1, 10.2, 10.3, 50.0]

    def test_protected_excluded_before_fps_and_N_changes(self):
        self._deps()
        d = tempfile.mkdtemp(prefix="ffv4o_realize_")
        ref, ref_id = _write_protection_reference(d, [5])  # protect the FPS-knee far point
        controller = _controller(os.path.join(d, "run"), ref)
        provider = _provider()
        self._seed_pool(provider, controller, self.KNEE_VALUES)

        realized = self._realize(provider, controller)
        proj = realized.plan.existing_pool_projection

        # N changed: the full-pool knee is 3, the protected-excluded knee is 2.
        self.assertEqual(proj["n_selected"], 2)
        # The protected global row 5 never appears in the selected population.
        self.assertNotIn(5, proj["selected_source_global_indices"])
        # Authoritative report: names the reference and carries REAL counts (never a fabricated 0).
        report = proj["protected_reference_exclusion_report"]
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["reference_id"], ref_id)
        self.assertEqual(report["protected_candidate_count"], 1)
        self.assertEqual(report["protected_excluded_count"], 1)
        self.assertEqual(report["eligible_population_after_exclusion"], 5)
        self.assertIs(report["dft_labels_used_as_selection_scores"], False)

    def test_full_pool_knee_is_three_without_protection(self):
        """Control: with NO protection the same pool sizes to N=3 -- so the N=2 above is caused by the
        exclusion, not by an unrelated sizing artifact."""
        self._deps()
        d = tempfile.mkdtemp(prefix="ffv4o_realize_")
        controller = _controller(os.path.join(d, "run"), None)
        provider = _provider()
        self._seed_pool(provider, controller, self.KNEE_VALUES)

        realized = self._realize(provider, controller)
        proj = realized.plan.existing_pool_projection
        self.assertEqual(proj["n_selected"], 3)
        report = proj["protected_reference_exclusion_report"]
        self.assertEqual(report["protected_excluded_count"], 0)
        # An unprotected run emits no anonymous reference_id (nothing to attest against).
        self.assertNotIn("reference_id", report)

    def test_one_protected_row_excluded(self):
        """One protected row inside the pool is excluded; the rest remain eligible and are selected
        from. Zero protected overlap reaches acquisition (directive step 7F)."""
        self._deps()
        d = tempfile.mkdtemp(prefix="ffv4o_realize_")
        ref, _ = _write_protection_reference(d, [2])
        controller = _controller(os.path.join(d, "run"), ref)
        provider = _provider()
        self._seed_pool(provider, controller, self.KNEE_VALUES)
        realized = self._realize(provider, controller)
        proj = realized.plan.existing_pool_projection
        self.assertNotIn(2, proj["selected_source_global_indices"])
        self.assertEqual(proj["protected_reference_exclusion_report"]["protected_excluded_count"], 1)

    def test_all_candidates_protected_fails_closed(self):
        """When every admissible in-scope frame is protected, the eligible pool is empty after
        exclusion -> typed fail-closed gap, never a silent fallback to the full (protected) pool."""
        self._deps()
        from runtimes.pydantic_ai.default_acquisition_provider import AcquisitionCapabilityGap
        d = tempfile.mkdtemp(prefix="ffv4o_realize_")
        ref, _ = _write_protection_reference(d, list(range(len(self.KNEE_VALUES))))
        controller = _controller(os.path.join(d, "run"), ref)
        provider = _provider()
        self._seed_pool(provider, controller, self.KNEE_VALUES)
        with self.assertRaises(AcquisitionCapabilityGap) as ctx:
            self._realize(provider, controller)
        self.assertEqual(ctx.exception.gap_kind, "EMPTY_ELIGIBLE_POOL")

    def test_out_of_range_protected_indices_ignored(self):
        """Protected rows that lie outside this pool's index range (e.g. a reference from a larger
        source dataset) simply do not intersect it -- they are not treated as in-pool exclusions."""
        self._deps()
        d = tempfile.mkdtemp(prefix="ffv4o_realize_")
        ref, _ = _write_protection_reference(d, [3536, 5044])  # far beyond a 6-frame pool
        controller = _controller(os.path.join(d, "run"), ref)
        provider = _provider()
        self._seed_pool(provider, controller, self.KNEE_VALUES)
        realized = self._realize(provider, controller)
        proj = realized.plan.existing_pool_projection
        report = proj["protected_reference_exclusion_report"]
        self.assertEqual(report["protected_excluded_count"], 0)
        self.assertEqual(report["protected_candidate_count"], 0)
        self.assertEqual(proj["n_selected"], 3)  # unchanged: nothing in-range was excluded


class GenerationBackendParentGuardTests(unittest.TestCase):
    """Directive step 8 cross-check: a protected-reference source row must never become a selected
    PARENT/seed for the generation backends (LOCAL_PERTURBATION / TEACHER_DRIVEN_MD), resolved through
    the SAME canonical protected set. The shared ``realize`` guard fails closed, never a fabricated
    PASS."""

    def _deps(self):
        try:
            import ase  # noqa: F401
            import yaml  # noqa: F401
            from framework_v2.acquisition.selection import select_candidates  # noqa: F401
            from runtimes.pydantic_ai.default_acquisition_provider import (  # noqa: F401
                FrameworkDefaultAcquisitionProvider)
        except ModuleNotFoundError as e:
            self.skipTest(f"optional dep not installed: {e}")

    def _run(self, kind, protected_indices, selected_source_global_indices):
        from framework_v2.acquisition.contracts import AcquisitionPhase
        d = tempfile.mkdtemp(prefix="ffv4o_parent_")
        ref, _ = _write_protection_reference(d, protected_indices)
        controller = _controller(os.path.join(d, "run"), ref)
        provider = _provider()

        m = _fake_m()
        m.eligible_source_categories = ["bulk"]
        m.descriptor_evidence = types.SimpleNamespace(duplicate_handling="reject")
        provider._cache[str(controller.run_dir)] = m

        strategy = _FakeShaObj("strategy", kind=kind, selected_backend_ids=["gen.ase"])
        context = types.SimpleNamespace(
            strategy=strategy,
            coverage=types.SimpleNamespace(unsaturated_core_gaps=lambda: []))
        proposal = types.SimpleNamespace(
            selected_parent_ids=["bulk#0"], n_per_structure=1, params={},
            selected_source_global_indices=list(selected_source_global_indices))
        return provider.realize(controller, context, proposal)

    def test_local_perturbation_protected_parent_fails_closed(self):
        self._deps()
        from framework_v2.acquisition.contracts import AcquisitionStrategyKind
        from runtimes.pydantic_ai.default_acquisition_provider import AcquisitionCapabilityGap
        with self.assertRaises(AcquisitionCapabilityGap) as ctx:
            self._run(AcquisitionStrategyKind.LOCAL_PERTURBATION, [5], [5])
        self.assertEqual(ctx.exception.gap_kind, "PROTECTED_PARENT_SELECTED")

    def test_teacher_driven_md_protected_parent_fails_closed(self):
        self._deps()
        from framework_v2.acquisition.contracts import AcquisitionStrategyKind
        from runtimes.pydantic_ai.default_acquisition_provider import AcquisitionCapabilityGap
        with self.assertRaises(AcquisitionCapabilityGap) as ctx:
            self._run(AcquisitionStrategyKind.TEACHER_DRIVEN_MD, [7], [7])
        self.assertEqual(ctx.exception.gap_kind, "PROTECTED_PARENT_SELECTED")


if __name__ == "__main__":
    unittest.main()
