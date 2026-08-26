"""Framework-DEFAULT autonomous acquisition planning provider.

This is the framework-owned ``AcquisitionPlanningProvider`` that removes the old ``_PROVIDER=None``
hard no-op: a new campaign no longer has to hand-author run-specific Python to get autonomous
acquisition planning. Entry needs only the run's already-frozen inputs (the bound
DeploymentScopeContract + registered descriptor/domain plugins + the environment's backend/Teacher
capability probes); the framework then AUTO-MATERIALIZES the typed coverage-evidence artifact and
composes the existing deterministic pipeline around it.

Architecture (the user-mandated plugin auto-materialization flow):

    raw source structures
      -> a registered ``StructuralDescriptorProvider`` (material-specific descriptor-space work)
      -> ``DescriptorSpaceEvidence`` (pure descriptor-space FACTS + lazy representation builders)
      -> ``materialize_acquisition_evidence`` freezes the generic pipeline
         (inventory / target-regime / region / coverage / strategy) into a typed,
         content-addressed ``AutoMaterializedAcquisitionEvidence`` artifact
      -> THIS provider maps that onto ``AcquisitionPlanningContext`` so the dispatched Agent
         autonomously synthesizes the low-level recipe (the ONE genuinely-scientific choice),
         which ``realize`` deterministically projects into the full ``AcquisitionPlanV2`` chain.

The provider is fully material-agnostic: all descriptor-space work lives in the registered plugin
(``framework_v2.acquisition.descriptor_plugins``), and the environment-specific backend/Teacher
capability probes are injected seams (so tests inject fakes and a real campaign injects real
probes). If no admissible descriptor/domain capability exists for the campaign, the pipeline FAILS
CLOSED with a typed ``AcquisitionCapabilityGap`` -- it never falls back to prompting a human for
n_parents / percentages / sigma, and never fabricates descriptor-space values.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from framework_v2.acquisition.contracts import (
    AcquisitionPhase,
    AcquisitionStrategyKind,
    BackendCapabilityRecord,
    CandidateGenerationResult,
    GenerationProvenance,
    ProtectedDisjointnessReport,
    TeacherCapabilityRecord,
)
from framework_v2.acquisition.descriptor_plugins import (
    AcquisitionCapabilityGap,
    DescriptorSpaceEvidence,
    resolve_descriptor_provider,
)
from framework_v2.acquisition.evidence_materializer import (
    MaterializedAcquisitionEvidence,
    materialize_acquisition_evidence,
)
from framework_v2.acquisition.generators.base import GenerationProtocol
from framework_v2.acquisition.generic_coverage import (
    FrameworkSizingParams,
    recommend_labeling_population_sizing,
)
from framework_v2.acquisition.labeling import build_labeling_request
from framework_v2.acquisition.plan_assembly import (
    assemble_plan_v2,
    build_existing_pool_projection,
    build_legacy_projection,
)
from framework_v2.acquisition.selection import (
    farthest_point_selection,
    select_candidates,
    select_candidates_from_indices,
)

from .acquisition_planner import (
    AcquisitionPlanningContext,
    RealizedAcquisition,
    get_acquisition_planning_provider,
    set_acquisition_planning_provider,
)

# The environment-specific capability probes. They are genuinely run/environment evidence (which
# generation backends are importable + feasible here, what the frozen Teacher can do), NOT a
# scientific recipe choice, so they stay injected seams rather than baked into the material plugin.
BackendProbe = Callable[["object", DescriptorSpaceEvidence], Sequence[BackendCapabilityRecord]]
TeacherProbe = Callable[["object", DescriptorSpaceEvidence], TeacherCapabilityRecord]


def default_backend_probe(controller, evidence: DescriptorSpaceEvidence):
    """Framework-default backend capability probe: enumerate the shipped candidate-generation
    backends and return each one's deterministic feasibility probe.

    Fully material-agnostic -- it asks each backend whether its dependency imports in THIS
    environment (``augment_atoms`` for local perturbation, an ASE MD integrator for Teacher-driven
    dynamics). An infeasible backend is recorded, not hidden (absence is evidence the strategy
    planner must see). Nothing here is fabricated: a backend that cannot import reports
    ``feasible=False`` with a reason, and the strategy planner simply cannot select it."""
    from framework_v2.acquisition.generators.base import CandidateGeneratorRegistry
    from framework_v2.acquisition.generators.existing_pool import ExistingPoolSelectionGenerator
    from framework_v2.acquisition.generators.local_perturbation import LocalPerturbationGenerator
    from framework_v2.acquisition.generators.teacher_dynamics import TeacherDynamicsGenerator

    registry = CandidateGeneratorRegistry()
    registry.register(ExistingPoolSelectionGenerator())
    registry.register(LocalPerturbationGenerator())
    registry.register(TeacherDynamicsGenerator())
    return registry.probe_all()


def _sha256_file(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_bound_teacher_calculator_config(controller):
    """Deterministically locate the run's bound Teacher-calculator config from its OWN frozen
    workflow declaration.

    The Teacher calculator this run drives inference/dynamics with is declared by the stage-param
    key ``teacher_config`` in the run's ``workflow_config`` (the exact same file the labeling and
    augment executors load to construct the ASE calculator -- ``executors._write_executable_augment_config``).
    Reading it here derives the capability from the run's own bound evidence rather than assuming it.
    Returns ``(config_path, config_dict)`` or ``(None, None)`` when no such config is declared."""
    from pathlib import Path

    import yaml

    wf = controller.state.get("workflow_config")
    if not wf or not Path(wf).exists():
        return None, None
    run_dir = str(Path(wf).resolve().parent)
    try:
        doc = yaml.safe_load(Path(wf).read_text(encoding="utf-8")) or {}
    except Exception:
        return None, None

    found: list[str] = []

    def _walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "teacher_config" and isinstance(v, str):
                    found.append(v)
                else:
                    _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(doc)
    for raw in found:
        candidate = Path(raw.replace("{run_dir}", run_dir))
        if candidate.exists():
            try:
                cfg = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            if isinstance(cfg, dict):
                return str(candidate.resolve()), cfg
    return None, None


def _probe_teacher_dynamics_capability(controller) -> tuple[bool, str]:
    """Deterministically decide whether the run's bound Teacher can drive MD.

    TEACHER_DRIVEN_MD (``TeacherDynamicsGenerator``) drives ASE velocity-Verlet (NVE) or Langevin
    (NVT) integrators, which need only an ASE-compatible calculator that produces FORCES. The
    capability is therefore derived, not assumed, from the run's OWN bound Teacher-calculator config:
    a config that declares a constructible ASE calculator (``calculator.module`` + ``calculator.class``)
    is dynamics-capable; a config that binds no ASE calculator is not (its Teacher can still label,
    just not drive dynamics). Returns ``(can_drive_dynamics, rationale)`` -- never a hardcoded flag."""
    from pathlib import Path

    cfg_path, cfg = _resolve_bound_teacher_calculator_config(controller)
    if not cfg:
        return (False,
                "no bound Teacher-calculator config (workflow declares no stage-param "
                "'teacher_config'); TEACHER_DRIVEN_MD not admissible for this run")
    calc = cfg.get("calculator")
    if not isinstance(calc, dict) or not calc:
        return (False,
                f"bound Teacher config {Path(cfg_path).name!r} declares no ASE calculator "
                "(calculator block absent/empty); cannot drive ASE MD, so TEACHER_DRIVEN_MD "
                "is not admissible")
    module, klass = calc.get("module"), calc.get("class")
    if not module or not klass:
        return (False,
                f"bound Teacher calculator in {Path(cfg_path).name!r} is underspecified "
                f"(module={module!r}, class={klass!r}); cannot construct an ASE calculator for MD")
    return (True,
            f"bound Teacher declares a constructible ASE calculator ({module}.{klass}, "
            f"constructor={calc.get('constructor')!r}) that produces forces; ASE NVE/NVT_LANGEVIN "
            f"integrators can be driven under it (config {Path(cfg_path).name!r} "
            f"sha256={_sha256_file(cfg_path)[:12]})")


def default_teacher_probe(controller, evidence: DescriptorSpaceEvidence) -> TeacherCapabilityRecord:
    """Framework-default Teacher capability probe derived from the run's OWN bound Teacher.

    The frozen Teacher's identity is pinned to the content of the Teacher model this run declared in
    its ``teacher_evidence_sources`` -- never fabricated. A run that autonomously plans acquisition
    but bound no Teacher model cannot label candidates, so this FAILS CLOSED with a typed
    ``AcquisitionCapabilityGap`` rather than inventing a Teacher.

    ``can_label`` is True (a bound Teacher model produces canonical energy/force labels).
    ``can_drive_dynamics`` is DERIVED deterministically from the run's own bound Teacher-calculator
    config by ``_probe_teacher_dynamics_capability`` (does the config declare a constructible ASE
    calculator that produces forces) -- never hardcoded and never merely assumed True. A Teacher that
    binds no ASE calculator is labeling-only, so TEACHER_DRIVEN_MD is simply removed from the
    admissible set (the strategy planner then falls to another backend or fails closed)."""
    sources = controller.state.get("teacher_evidence_sources") or {}
    model_path = sources.get("teacher_model_path")
    if not model_path:
        raise AcquisitionCapabilityGap(
            "autonomous acquisition needs a labeling Teacher but this run bound no "
            "teacher_evidence_sources.teacher_model_path; the framework will not fabricate a Teacher",
            gap_kind="NO_LABELING_TEACHER")
    from pathlib import Path

    p = Path(model_path)
    if not p.exists():
        raise AcquisitionCapabilityGap(
            f"bound Teacher model {model_path!r} does not exist; cannot pin Teacher identity",
            gap_kind="NO_LABELING_TEACHER")
    identity_sha256 = _sha256_file(p)
    can_drive_dynamics, dynamics_rationale = _probe_teacher_dynamics_capability(controller)
    return TeacherCapabilityRecord(
        teacher_id=f"{controller.state['run_id']}-teacher",
        can_label=True,
        can_drive_dynamics=can_drive_dynamics,
        identity_sha256=identity_sha256,
        rationale=(f"identity pinned to content of bound Teacher model {p.name!r}; "
                   f"dynamics capability: {dynamics_rationale}"))


def _has_acquisition_stage(controller) -> bool:
    for stage in controller.state.get("stages", []) or []:
        name = stage.get("name") if isinstance(stage, dict) else getattr(stage, "name", None)
        if name == "acquisition":
            return True
    return False


def _acquisition_plan_already_bound(controller) -> bool:
    for inp in controller.state.get("inputs", []) or []:
        if not isinstance(inp, dict) or inp.get("superseded"):
            continue
        source = inp.get("source", "")
        if source.endswith("acquisition_plan.json"):
            return True
    return False


class FrameworkDefaultAcquisitionProvider:
    """The framework's default, material-agnostic ``AcquisitionPlanningProvider``.

    ``applies`` gates the autonomous path from the run's own frozen inputs alone (V2 closure bound,
    an ``acquisition`` stage present, and no acquisition plan already supplied as a human input) --
    keeping runs that supply their own plan, and the broad regression suite, entirely unperturbed.
    ``build_context`` auto-materializes the typed coverage evidence via the registered descriptor
    plugin and maps it onto the deterministic decision space (failing closed if no admissible
    provider exists). ``realize`` deterministically projects an accepted recipe proposal into the
    full assembled ``AcquisitionPlanV2`` evidence chain."""

    def __init__(
        self, *, backend_probe: BackendProbe, teacher_probe: TeacherProbe,
        phase: AcquisitionPhase = AcquisitionPhase.INITIAL,
    ) -> None:
        self._backend_probe = backend_probe
        self._teacher_probe = teacher_probe
        self._phase = phase
        self._cache: dict[str, MaterializedAcquisitionEvidence] = {}
        # Per-run EXISTING_POOL_SELECTION realization inputs: the global-ordered eligible-pool
        # descriptor vectors + their item ids + the source manifest path. Cached from build_context
        # so realize() sizes the labeling population deterministically without a human N.
        self._pool_cache: dict[str, dict] = {}

    # -- gating ------------------------------------------------------------------------------
    def applies(self, controller) -> bool:
        if not controller.v2_enabled():
            return False
        if not _has_acquisition_stage(controller):
            return False
        if _acquisition_plan_already_bound(controller):
            return False
        return True

    # -- evidence auto-materialization -------------------------------------------------------
    def build_context(self, controller) -> AcquisitionPlanningContext:
        from framework_v2.acquisition.contracts import CampaignObjective
        from framework_v2.contracts import DeploymentScopeContract

        run_id = controller.state["run_id"]
        scope_sha = controller._v2_state().get("scope_contract_sha256")
        if not scope_sha:
            raise AcquisitionCapabilityGap(
                "autonomous acquisition requires a bound V2 DeploymentScopeContract but none is "
                f"bound to run {run_id!r}",
                gap_kind="NO_SCOPE_CONTRACT")
        scope_dict = controller.v2_contract(scope_sha)
        if scope_dict is None:
            raise AcquisitionCapabilityGap(
                f"scope contract {scope_sha!r} is not resolvable in run {run_id!r} state",
                gap_kind="NO_SCOPE_CONTRACT")
        scope_contract = DeploymentScopeContract.model_validate(scope_dict)

        # Build the objective ONCE: CampaignObjective carries an ``established_at`` default that
        # would otherwise drift the content-SHA on reconstruction. The same object is threaded
        # through the materializer and returned in the context so every downstream SHA is stable.
        objective = CampaignObjective(
            objective_id=f"{run_id}-objective",
            primary_target=scope_contract.objective,
            claim_scope=scope_contract.objective,
            scope_contract_sha256=scope_sha,
            phase=self._phase)

        provider = resolve_descriptor_provider(
            controller=controller, objective=objective, scope_contract=scope_contract)
        evidence = provider.build_descriptor_space_evidence(
            controller=controller, objective=objective, scope_contract=scope_contract)

        backend_records = list(self._backend_probe(controller, evidence))
        teacher_record = self._teacher_probe(controller, evidence)

        materialized = materialize_acquisition_evidence(
            id_prefix=run_id, material_id=provider.material_id, objective=objective,
            scope_contract=scope_contract, descriptor_evidence=evidence,
            backend_records=backend_records, teacher_record=teacher_record)
        self._cache[str(controller.run_dir)] = materialized

        # When the deterministic strategy is EXISTING_POOL_SELECTION, cache the global-ordered
        # eligible-pool descriptor vectors so realize() can compute the labeling-population size
        # from the pool's own diminishing-novelty curve (never a human-supplied N). The global
        # index of a selected frame is its position in ``result.pool.frames`` -- exactly the order
        # the Stage-3 existing-pool executor reproduces by concatenating the manifest's category
        # files in manifest order.
        from framework_v2.acquisition.contracts import AcquisitionStrategyKind as _ASK
        if materialized.strategy.kind == _ASK.EXISTING_POOL_SELECTION:
            if not hasattr(provider, "build_representation_result"):
                raise AcquisitionCapabilityGap(
                    "EXISTING_POOL_SELECTION requires a descriptor provider that can expose its "
                    "raw pool for deterministic labeling-population sizing",
                    gap_kind="POOL_NOT_EXPOSED")
            from framework_v2.acquisition.generic_representation import (
                derive_admissible_sizing_representation)

            rep = provider.build_representation_result(
                controller=controller, objective=objective, scope_contract=scope_contract)
            # FE-029: a candidate axis may be genuinely uncomputable for some eligible frame (e.g. a
            # geometry axis on an isolated non-periodic single atom). Rather than fail closed on the
            # first such frame, drop the frame, or drop an axis silently, derive -- deterministically
            # and with a first-class RepresentationAdequacyEvidence check -- the admissible dense
            # sizing representation that KEEPS every eligible frame. Reduction is admitted only when
            # adequacy still holds; if no adequate dense representation exists this raises the typed
            # REPRESENTATION_INSUFFICIENT gap (never REPRESENTATION_INCOMPLETE-by-first-frame).
            sizing_rep = derive_admissible_sizing_representation(
                rep.pool, candidate_spec=rep.spec, scope_contract=scope_contract,
                deployment_claim=objective.claim_scope, id_prefix=run_id)
            axes = list(sizing_rep.axes)
            vectors: list[list[float]] = [
                [float(frame.features[k]) for k in axes] for frame in rep.pool.frames]
            item_ids: list[str] = [frame.item_id for frame in rep.pool.frames]
            self._pool_cache[str(controller.run_dir)] = {
                "axes": axes, "vectors": vectors, "item_ids": item_ids,
                "manifest_path": rep.pool.manifest_path,
                "manifest_sha256": rep.pool.manifest_sha256,
                "duplicate_handling": evidence.duplicate_handling,
                "eligible_source_categories": list(evidence.eligible_source_categories),
                "sizing_axes_reduced": sizing_rep.reduced,
                "sizing_representation_sha256": sizing_rep.spec.content_sha256(),
                "sizing_adequacy_verdict": sizing_rep.adequacy.verdict.value,
                "sizing_per_axis_available_count": dict(sizing_rep.per_axis_available_count)}

        m = materialized
        return AcquisitionPlanningContext(
            objective=m.objective, inventory=m.inventory,
            target_regime_model=m.target_regime_model, region_resolution=m.region_resolution,
            coverage=m.coverage, strategy=m.strategy,
            admissible_parent_ids=m.admissible_parent_ids,
            teacher_identity_sha256=m.teacher_identity_sha256,
            required_param_keys=m.required_param_keys,
            param_bounds=dict(m.param_bounds))

    # -- deterministic realization of an accepted recipe -------------------------------------
    def realize(self, controller, context, proposal) -> RealizedAcquisition:
        m = self._cache.get(str(controller.run_dir))
        if m is None:  # pragma: no cover - build_context always runs first in the planner
            raise AcquisitionCapabilityGap(
                "realize called before build_context materialized the evidence chain",
                gap_kind="EVIDENCE_NOT_MATERIALIZED")

        strategy = context.strategy
        backend_id = strategy.selected_backend_ids[0]

        # EXISTING_POOL_SELECTION does NOT generate frames: it deterministically SIZES and SELECTS a
        # representative subset of the already-eligible pool for canonical Teacher labeling. The
        # subset size is an OUTPUT of descriptor-space sizing (never a human/LLM N), so this branch
        # ignores the proposal's perturbation params entirely.
        if strategy.kind == AcquisitionStrategyKind.EXISTING_POOL_SELECTION:
            return self._realize_existing_pool(
                controller, context, m, strategy, backend_id, proposal)

        parent_ids = list(proposal.selected_parent_ids)
        n_per = int(proposal.n_per_structure)

        # Deterministic candidate-generation projection: one candidate id per (parent, index),
        # each with exploration_only provenance (generation PES is NEVER a training label; those
        # come only from the canonical Teacher relabeling request below). Descriptors are trivially
        # distinct 1-D coordinates so farthest-point selection over them is well-defined and stable.
        candidate_ids: list[str] = []
        provenance: list[GenerationProvenance] = []
        descriptors: list[list[float]] = []
        for pid in parent_ids:
            for j in range(n_per):
                cid = f"{pid}#cand{j}"
                candidate_ids.append(cid)
                provenance.append(GenerationProvenance(
                    candidate_id=cid, strategy_kind=strategy.kind, backend_id=backend_id,
                    parent_id=pid, exploration_only=True))
                descriptors.append([float(len(descriptors))])

        generation_result = CandidateGenerationResult(
            result_id=f"{m.frozen_artifact.evidence_id}-generation",
            strategy_sha256=strategy.content_sha256(), backend_id=backend_id,
            candidate_ids=candidate_ids, provenance=provenance,
            n_requested=len(candidate_ids), n_generated=len(candidate_ids), n_rejected=0)

        # The protocol carries n_per_structure inside params so the legacy projection reads it
        # from a single content-addressed source (its SHA is each candidate's generation provenance).
        protocol = GenerationProtocol(
            protocol_id=f"{m.frozen_artifact.evidence_id}-protocol",
            backend_id=backend_id, strategy_kind=strategy.kind,
            strategy_sha256=strategy.content_sha256(), n_requested=len(candidate_ids),
            target_regime_ids=[g.regime_id for g in context.coverage.unsaturated_core_gaps()],
            parent_ids=parent_ids,
            params={**dict(proposal.params), "n_per_structure": n_per})

        # Canonical protected-reference enforcement for the generation backends: a perturbation/
        # dynamics PARENT must never be a protected-reference source row (directive: "protected
        # reference structures must not become selected parents"). Resolve the SAME canonical set the
        # executor enforces and fail closed if any selected source-global parent is protected -- never
        # a fabricated PASS.
        protected_reference_id, protected_all = self._resolve_protected(controller)
        selected_source_globals = {int(g) for g in proposal.selected_source_global_indices}
        protected_parents = sorted(selected_source_globals & protected_all)

        def _disjointness_checker(selected_ids: list[str]) -> ProtectedDisjointnessReport:
            # Independent recompute (NOT fabricated): the selected source-global parents must be
            # disjoint from the canonically-resolved protected population; DFT labels are never used
            # as selection scores.
            if protected_parents:
                raise AcquisitionCapabilityGap(
                    "generation backend selected protected-reference source rows as parents: "
                    f"{protected_parents[:20]}", gap_kind="PROTECTED_PARENT_SELECTED")
            return ProtectedDisjointnessReport(
                status="PASS", n_checked=len(selected_ids), n_overlaps=0,
                dft_labels_used_as_selection_scores=False)

        selection_result = select_candidates(
            selection_id=f"{m.frozen_artifact.evidence_id}-selection",
            generation_result=generation_result, descriptors=descriptors,
            k=len(candidate_ids), disjointness_checker=_disjointness_checker,
            selector="farthest_point_sampling", seed_index=0)

        labeling_request = build_labeling_request(
            request_id=f"{m.frozen_artifact.evidence_id}-labeling",
            selection_result=selection_result,
            teacher_identity_sha256=m.teacher_identity_sha256)

        legacy_projection = None
        dynamics_protocol_sha256 = None
        if strategy.kind == AcquisitionStrategyKind.LOCAL_PERTURBATION:
            legacy_projection = build_legacy_projection(
                protocol=protocol, selection_result=selection_result,
                eligible_source_categories=list(m.eligible_source_categories),
                selected_source_global_indices=list(proposal.selected_source_global_indices),
                duplicate_handling=m.descriptor_evidence.duplicate_handling,
                protected_reference_id=protected_reference_id,
                protected_candidate_count=len(protected_all))
        else:
            dynamics_protocol_sha256 = protocol.content_sha256()

        plan = assemble_plan_v2(
            plan_id=f"{m.frozen_artifact.evidence_id}-plan",
            objective=m.objective, inventory=m.inventory,
            target_regime_model=m.target_regime_model, region_resolution=m.region_resolution,
            coverage=m.coverage, strategy=strategy, generation_result=generation_result,
            selection_result=selection_result, labeling_request=labeling_request,
            legacy_projection=legacy_projection,
            dynamics_protocol_sha256=dynamics_protocol_sha256)

        return RealizedAcquisition(
            generation_result=generation_result, selection_result=selection_result,
            labeling_request=labeling_request, plan=plan, legacy_projection=legacy_projection)

    # -- canonical protected-reference resolution (shared by ALL acquisition backends) ---------
    @staticmethod
    def _resolve_protected(controller):
        """Resolve the run's protected source population through the ONE canonical framework
        resolver the acquisition EXECUTOR also enforces against, so every acquisition backend
        (existing-pool, local-perturbation, teacher-driven MD) excludes/attests against the SAME
        set the executor independently re-checks. Returns ``(reference_id, protected_globals)`` --
        ``(None, set())`` only when the run declares no acquisition protection reference at all."""
        from validation.protected_reference import resolve_protected_population
        from .cli import _acquisition_protection_reference_yaml
        reference_yaml = _acquisition_protection_reference_yaml(controller)
        if not reference_yaml:
            return None, set()
        resolved = resolve_protected_population(reference_yaml)
        return resolved["reference_id"], {int(g) for g in resolved["protected_source_indices"]}

    # -- EXISTING_POOL_SELECTION deterministic realization ------------------------------------
    @staticmethod
    def _source_category_of(item_id: str) -> str:
        """The opaque source-family label an item_id carries (``"{category}#{index}"``).

        Never interpreted or matched against any material-specific name -- it is only used to group
        frames by the family the reasoning plane reasoned over. ``rsplit`` on the final ``#`` peels
        the frame index; a category string itself carries no ``#``."""
        return item_id.rsplit("#", 1)[0]

    # -- FE-046 cumulative + monotonic coverage-gap reacquisition -----------------------------
    @staticmethod
    def _latest_coverage_gap_unsupported(controller):
        """The declared structure classes the MOST RECENT Stage-4 coverage-adequacy gate found
        UNSUPPORTED, read from the persisted gate event (``coverage_adequacy`` block, FE-042) --
        never re-derived and never inferred from transient recovery state.

        Returns the ordered unsupported-class list (possibly empty if the latest coverage gate was
        satisfied), or ``None`` when no coverage-adequacy gate has ever fired (an INITIAL acquisition
        that must keep the pre-FE-046 behavior)."""
        for ev in reversed(controller.state.get("events", []) or []):
            if ev.get("type") == "gate" and isinstance(ev.get("coverage_adequacy"), dict):
                return list(ev["coverage_adequacy"].get("unsupported_structure_classes") or [])
        return None

    @staticmethod
    def _prior_accepted_source_globals(controller):
        """The UNION of every prior-accepted existing-pool global index across ALL superseded
        ``acquisition_plan`` inputs (cumulative acquired population). A recovery reacquisition
        SUPERSEDES its predecessor plan, so by the time the next plan is realized every previously
        accepted plan is a superseded input; unioning their ``selected_source_global_indices`` is the
        cumulative set FE-046 must never discard."""
        import json as _json
        from pathlib import Path as _Path
        globals_out: set[int] = set()
        for rec in controller.state.get("inputs", []) or []:
            if not rec.get("superseded"):
                continue
            snap = rec.get("snapshot")
            if not snap or not str(snap).endswith("acquisition_plan.json"):
                continue
            p = _Path(snap)
            if not p.is_file():
                continue
            try:
                plan = _json.loads(p.read_text())
            except (ValueError, OSError):
                continue
            for g in plan.get("selected_source_global_indices") or []:
                globals_out.add(int(g))
        return globals_out

    @staticmethod
    def _marginal_novelty_topup(vectors, initial_selected, target_size, *,
                                candidate_pool=None):
        """Deterministic farthest-point *marginal-novelty* extension of an already-chosen set.

        Given ``initial_selected`` local indices, greedily add the point whose min-Euclidean distance
        to the current selected set is largest (ties -> lowest index), restricted to ``candidate_pool``
        when given, until ``target_size`` is reached. This is the SAME raw-Euclidean FPS the selector
        and sizing use, generalized to seed from the cumulative/floor set instead of a single index, so
        the top-up frames are the most novel RELATIVE to everything already retained. Returns the full
        ordered selection (initial set first, in its given order, then the novelty-ordered additions)."""
        import numpy as np

        arr = np.asarray(vectors, dtype=float)
        n = arr.shape[0]
        selected = list(initial_selected)
        sel_set = set(selected)
        pool = set(range(n)) if candidate_pool is None else set(candidate_pool)
        target = min(int(target_size), len(sel_set | pool))
        if not selected:
            # No cumulative/floor seed: fall back to canonical FPS-from-0 over the candidate pool.
            base = farthest_point_selection(vectors, min(target, n), seed_index=0)
            return [i for i in base if i in pool][:target] if candidate_pool is not None else base

        min_dist = np.full(n, np.inf)
        for s in selected:
            min_dist = np.minimum(min_dist, np.linalg.norm(arr - arr[s], axis=1))
        while len(selected) < target:
            remaining = [i for i in pool if i not in sel_set]
            if not remaining:
                break
            nxt = max(remaining, key=lambda i: (float(min_dist[i]), -i))
            selected.append(nxt)
            sel_set.add(nxt)
            min_dist = np.minimum(min_dist, np.linalg.norm(arr - arr[nxt], axis=1))
        return selected

    def _compose_cumulative_coverage_selection(self, *, vectors, item_ids, eligible_positions,
                                               knee_k, fe046):
        """Compose the FE-046 coverage-gap reacquisition selection as
        ``cumulative_prior_accepted + per_unsupported_class_local_saturation + optional_fps_topup``.

        Returns ``(ordered_local_indices, composition_provenance)`` where the indices are into the
        eligible-subset ``item_ids``/``vectors``. The composed set is a SUPERSET of the cumulative
        prior-accepted frames plus, for every still-uncovered unsupported class, a DOMAIN-LOCAL
        population sized by that class's OWN farthest-point marginal-novelty saturation knee -- so
        coverage is monotonic by construction (an accepted frame is never dropped, a covered class
        never lost) AND each class's added N is determined by its own within-class diversity
        saturation, never a fixed per-class quota. Presence (Stage-4 occupancy>0) and adequacy
        (class-local novelty saturation) are kept separate: a class becomes PRESENT at its first
        selected frame, but its recovery acquisition continues until its own class-local knee
        terminates."""
        from validation.coverage_gap_assessment import resolve_config_type_domain

        label_index = fe046["label_index"]
        unsupported = fe046["unsupported"]
        floor_targets = fe046["floor_targets"]
        prior_globals = fe046["prior_accepted_globals"]
        coverage_gap_sha256 = fe046.get("coverage_gap_sha256", "fe046-classlocal")
        sizing_id_prefix = fe046.get("sizing_id_prefix", "fe046")

        local_of_global = {g: i for i, g in enumerate(eligible_positions)}

        # (a) CUMULATIVE core: retain every prior-accepted frame. Each is in-scope (admissible was
        #     widened to its family) and non-protected (it was when accepted), so it MUST resolve to
        #     an eligible local index; a missing one would silently DROP an accepted frame -> fail
        #     closed rather than violate the cumulative invariant.
        prior_local: list[int] = []
        dropped: list[int] = []
        for g in sorted(prior_globals):
            loc = local_of_global.get(g)
            if loc is None:
                dropped.append(g)
            else:
                prior_local.append(loc)
        if dropped:
            raise AcquisitionCapabilityGap(
                "FE-046 cumulative invariant violated: prior-accepted frame global(s) "
                f"{dropped[:20]} are no longer eligible; refusing to drop an accepted frame",
                gap_kind="CUMULATIVE_FRAME_DROPPED")
        selected: list[int] = list(dict.fromkeys(prior_local))

        # Classes ALREADY covered by the cumulative core -- accumulation alone can resolve a class
        # the prior superseding plan had dropped.
        covered: set[str] = set()
        for loc in selected:
            dom, _ = resolve_config_type_domain(self._source_category_of(item_ids[loc]), label_index)
            if dom is not None:
                covered.add(dom)

        # (b) per-class DOMAIN-LOCAL autonomous sizing: for each still-uncovered unsupported class,
        #     build that class's OWN candidate population (its admissible source families, minus
        #     protected frames -- already excluded from item_ids -- and already-acquired frames), then
        #     run the SAME generic FPS marginal-novelty saturation sizer WITHIN THAT CLASS ONLY. The
        #     class-local knee determines how many NEW frames that class contributes: presence
        #     (occupancy>0) is achieved at the first pick, but acquisition continues until the class's
        #     own diversity saturates. No fixed per-class quota, no human N -- a large/diverse class
        #     pool yields more than one frame; a descriptor-degenerate one yields exactly one.
        domain_local_added_by_class: dict[str, int] = {}
        floor_added: dict[str, int] = {}  # first pick per class (back-compat / presence marker)
        sel_set = set(selected)
        for c in unsupported:
            if c in covered:
                continue
            fams = set(floor_targets.get(c, []))
            class_local_positions = [
                loc for loc in range(len(item_ids))
                if loc not in sel_set and self._source_category_of(item_ids[loc]) in fams]
            if not class_local_positions:
                raise AcquisitionCapabilityGap(
                    f"FE-046 domain-local sizing cannot cover declared class {c!r}: no eligible "
                    "non-protected frame remains in its target families; failing closed",
                    gap_kind="POOL_LACKS_STRUCTURE_CLASS")
            class_vectors = [vectors[loc] for loc in class_local_positions]
            class_sizing = recommend_labeling_population_sizing(
                class_vectors, params=FrameworkSizingParams(),
                sizing_id=f"{sizing_id_prefix}-classlocal-{c}",
                coverage_gap_sha256=coverage_gap_sha256, protected_excluded_count=0,
                target_labeled_population=None, max_teacher_label_calls=None)
            class_k = int(class_sizing.recommended_population_size)
            class_picks = [class_local_positions[j]
                           for j in class_sizing.selected_positions[:class_k]]
            added = 0
            for pick in class_picks:
                if pick in sel_set:
                    continue
                selected.append(pick)
                sel_set.add(pick)
                if added == 0:
                    floor_added[c] = pick
                added += 1
            domain_local_added_by_class[c] = added
            covered.add(c)

        # (c) OPTIONAL SECONDARY global FPS top-up to the pool-wide novelty knee. This runs ONLY
        #     AFTER every unsupported class has independently satisfied its class-local sizing, and
        #     never substitutes for it: size can only grow (never below the cumulative+domain-local
        #     core), so a shrinking global knee can never force an accepted frame or a covered class
        #     out.
        selected = self._marginal_novelty_topup(vectors, selected, max(int(knee_k), len(selected)))

        composition = {
            "cumulative_prior_accepted": len(prior_local),
            "domain_local_added_by_class": dict(sorted(domain_local_added_by_class.items())),
            "n_domain_local_frames": sum(domain_local_added_by_class.values()),
            "floor_added_classes": sorted(floor_added),
            "n_floor_added": len(floor_added),
            "knee_k": int(knee_k),
            "final_n_selected": len(selected),
            "unsupported_at_reacquisition": list(unsupported),
        }
        return selected, composition

    def _realize_existing_pool(
        self, controller, context, m, strategy, backend_id, proposal,
    ) -> RealizedAcquisition:
        """Deterministically SIZE and SELECT an existing-pool labeling population WITHIN the
        objective-conditioned admissible candidate population.

        No frames are generated: the recommended population size is the knee of the pool's own
        farthest-point marginal-novelty curve (``recommend_labeling_population_sizing``), the SAME
        raw-Euclidean FPS the selector uses -- so sizing and selection agree by construction and no
        human/LLM N is ever consulted. The selected frames are named by their GLOBAL index (their
        position in the manifest-ordered pool), which the Stage-3 existing-pool executor reproduces
        exactly by concatenating the manifest's category files in manifest order.

        Objective-conditioned eligibility (GAP-2 fix): the deterministic FPS/sizing must optimize
        diversity ONLY within the scientifically admissible candidate population, and must never
        re-introduce a source family the canonical scientific scope decision classified as
        out-of-scope. In the generic path there is no deterministic scope->frame map (scope regions
        are free-text domain predicates the framework does not parse, and the raw features carry no
        scope semantics), so the sanctioned, gated bridge from scope to frames is the reasoning
        plane's own family-level decision: ``proposal.selected_parent_ids`` -- already validated to
        be a non-empty subset of the admissible parent ids and gated by the 3-Judge committee over
        the assembled plan. The admissible FAMILIES are the set of source categories those selected
        parents belong to; FPS then autonomously determines total N, family composition, and the
        concrete frames strictly WITHIN that eligible population. This is derived generically (opaque
        category strings, no material name is hard-coded) and fails closed with a typed
        scope-eligibility gap rather than ever silently falling back to the full pool."""
        pc = self._pool_cache.get(str(controller.run_dir))
        if pc is None:  # pragma: no cover - build_context populates it for this strategy
            raise AcquisitionCapabilityGap(
                "EXISTING_POOL_SELECTION realize called before build_context cached the eligible "
                "pool descriptor vectors",
                gap_kind="POOL_NOT_MATERIALIZED")

        full_vectors: list[list[float]] = pc["vectors"]
        full_item_ids: list[str] = pc["item_ids"]
        if not full_item_ids:
            raise AcquisitionCapabilityGap(
                "EXISTING_POOL_SELECTION has an empty eligible pool; nothing admissible to label",
                gap_kind="EMPTY_ELIGIBLE_POOL")

        # -- CANONICAL protected-reference exclusion (ffv4o Stage-3 defect fix) --------------------
        # Resolve the run's protected source population through the ONE canonical framework resolver
        # the acquisition EXECUTOR also enforces against, then EXCLUDE those seed-pool global rows
        # from the eligible pool BEFORE any descriptor/FPS selection or marginal-novelty sizing. A
        # seed-pool global index IS the row's position in manifest-concatenation order, which is
        # exactly ``full_item_ids`` order, so a protected global ``g`` names position ``g`` here.
        # Planner and executor thus agree by construction; the executor's independent
        # ``assert_source_indices_allowed`` stays as defense in depth and must now always find zero
        # overlap. This is what the ffv4o defect violated -- the planner never loaded the run-bound
        # 1143 protected rows and fabricated a PASS/protected_excluded_count=0 exclusion report.
        protected_reference_id, protected_all = self._resolve_protected(controller)
        protected_globals = {g for g in protected_all if 0 <= g < len(full_item_ids)}

        # Objective-conditioned admissible candidate population: restrict to the source families the
        # reasoning plane declared in-scope. selected_parent_ids is already gated (subset of the
        # admissible parents + deterministic plan validation + 3-Judge), so this honors -- rather
        # than overrides -- the scientific scope decision.
        admissible_categories = {
            self._source_category_of(pid) for pid in proposal.selected_parent_ids}
        if not admissible_categories:
            raise AcquisitionCapabilityGap(
                "the acquisition recipe declared no admissible source family; cannot derive an "
                "objective-conditioned eligible population and will not fall back to the full pool",
                gap_kind="SCOPE_ELIGIBILITY_UNDECIDABLE")

        # -- FE-046: cumulative + monotonic coverage-gap reacquisition ----------------------------
        # A targeted reacquisition (return_stage=acquisition after a Stage-4 COVERAGE_INSUFFICIENT
        # gate) must (1) ACCUMULATE the prior-accepted frames rather than supersede-and-replace them,
        # and (2) guarantee a per-declared-class occupancy FLOOR so every still-unsupported class is
        # covered in ONE cycle. Pre-FE-046 this branch re-selected a fresh ~knee-sized global-FPS set
        # with NO occupancy floor, so coverage oscillated (frames for one class were dropped to add
        # another) and never converged. This block detects the recovery context from PERSISTED facts
        # (a prior superseded acquisition_plan + the latest coverage gate's unsupported set), derives
        # the label_map target families, and FAILS CLOSED IMMEDIATELY if any unsupported class is
        # unremediable from the pool. INITIAL acquisition (no prior plan / no coverage gate) is
        # untouched: fe046 stays None and the legacy FPS-to-knee selection runs below.
        fe046 = None
        unsupported = self._latest_coverage_gap_unsupported(controller)
        prior_accepted_globals = self._prior_accepted_source_globals(controller)
        if prior_accepted_globals and unsupported:
            from collections import Counter as _Counter

            from validation.coverage_gap_assessment import (
                build_label_index as _build_label_index,
                derive_reacquisition_targets as _derive_reacquisition_targets,
            )
            from .acquisition_readiness import _load_scope_classification_evidence

            project_dir = controller.state.get("project_dir") or "."
            scope_v2, binding = _load_scope_classification_evidence(controller, project_dir)
            if scope_v2 is None:
                raise AcquisitionCapabilityGap(
                    "coverage-gap reacquisition requires the frozen config_type->structure-class "
                    f"label_map the Stage-4 gate used, but none is resolvable ({binding}); refusing "
                    "to reacquire without the occupancy floor's authoritative class map",
                    gap_kind="SCOPE_LABEL_MAP_UNRESOLVABLE")
            label_map = scope_v2.model_dump()["label_map"]
            pool_counts = _Counter(self._source_category_of(iid) for iid in full_item_ids)
            reacq = _derive_reacquisition_targets(
                unsupported, label_map, pool_counts,
                already_eligible_source_categories=admissible_categories)
            # USER-DIRECTED (FE-046): an unremediable unsupported class -> fail closed IMMEDIATELY.
            # The pool physically contains no source family the frozen label_map maps to that declared
            # class, so no reacquisition can ever cover it: a genuine scientific boundary, surfaced
            # rather than papered over with partial coverage.
            if reacq["unremediable_classes"]:
                raise AcquisitionCapabilityGap(
                    "coverage-gap reacquisition cannot remediate declared structure class(es) "
                    f"{sorted(reacq['unremediable_classes'])!r}: the pool contains NO source family "
                    "the frozen label_map maps to them (primary_claim). Failing closed rather than "
                    "acquiring partial coverage.",
                    gap_kind="POOL_LACKS_STRUCTURE_CLASS")
            floor_targets = reacq["target_config_types_by_class"]
            # Widen the admissible families so (a) every prior-accepted frame stays in scope (the
            # cumulative invariant: never drop an accepted frame) and (b) each still-unsupported
            # class's target families are selectable for the floor. This is deterministic remediation
            # driven by the FROZEN coverage gate's own label_map -- the classes it defines as MUST-be-
            # covered -- not an LLM scope escape.
            for g in prior_accepted_globals:
                if 0 <= g < len(full_item_ids):
                    admissible_categories.add(self._source_category_of(full_item_ids[g]))
            for fams in floor_targets.values():
                admissible_categories.update(fams)
            fe046 = {
                "unsupported": list(unsupported),
                "floor_targets": floor_targets,
                "prior_accepted_globals": prior_accepted_globals,
                "label_index": _build_label_index(label_map),
                "coverage_gap_sha256": m.coverage.content_sha256(),
                "sizing_id_prefix": f"{m.frozen_artifact.evidence_id}-fe046",
            }

        admissible_positions = [
            i for i, iid in enumerate(full_item_ids)
            if self._source_category_of(iid) in admissible_categories]
        if not admissible_positions:
            raise AcquisitionCapabilityGap(
                "the scientifically admissible source families "
                f"{sorted(admissible_categories)!r} map to zero frames in the eligible pool; "
                "refusing to silently fall back to the full pool of all source families",
                gap_kind="SCOPE_ELIGIBILITY_EMPTY")

        # Deterministic protected exclusion of the in-scope admissible frames. ``N`` is ALLOWED to
        # change: sizing/selection below run over whatever admissible non-protected pool remains.
        protected_excluded_count = sum(1 for i in admissible_positions if i in protected_globals)
        eligible_positions = [i for i in admissible_positions if i not in protected_globals]
        if not eligible_positions:
            raise AcquisitionCapabilityGap(
                "every admissible in-scope frame is a protected-reference row; the eligible pool is "
                f"empty after canonical protected exclusion ({protected_excluded_count} excluded); "
                "nothing admissible and non-protected to label",
                gap_kind="EMPTY_ELIGIBLE_POOL")

        # FPS/sizing operate ONLY over the admissible, in-scope, protected-EXCLUDED frames.
        vectors: list[list[float]] = [full_vectors[i] for i in eligible_positions]
        item_ids: list[str] = [full_item_ids[i] for i in eligible_positions]
        admissible_source_categories = sorted(admissible_categories)
        full_pos = {iid: i for i, iid in enumerate(full_item_ids)}

        # 1. Deterministic labeling-population sizing (size is an OUTPUT, no human N). The REAL
        #    protected_excluded_count is recorded (never the hardcoded 0 the ffv4o defect emitted).
        sizing = recommend_labeling_population_sizing(
            vectors, params=FrameworkSizingParams(),
            sizing_id=f"{m.frozen_artifact.evidence_id}-sizing",
            coverage_gap_sha256=m.coverage.content_sha256(),
            protected_excluded_count=protected_excluded_count,
            target_labeled_population=None, max_teacher_label_calls=None)
        k = int(sizing.recommended_population_size)

        # 2. Generation projection: every eligible pool frame is an exploration-only candidate whose
        #    "parent" is itself (an existing frame, never a newly generated configuration). The PES
        #    is never a training label; canonical labels come only from the Teacher relabel request.
        provenance = [
            GenerationProvenance(
                candidate_id=iid, strategy_kind=strategy.kind, backend_id=backend_id,
                parent_id=iid, exploration_only=True)
            for iid in item_ids]
        generation_result = CandidateGenerationResult(
            result_id=f"{m.frozen_artifact.evidence_id}-generation",
            strategy_sha256=strategy.content_sha256(), backend_id=backend_id,
            candidate_ids=list(item_ids), provenance=provenance,
            n_requested=len(item_ids), n_generated=len(item_ids), n_rejected=0)

        def _disjointness_checker(selected_ids: list[str]) -> ProtectedDisjointnessReport:
            # Independent recompute (NOT a fabricated PASS): map each selected candidate back to its
            # seed-pool global row and intersect with the canonically-resolved protected population.
            # The exclusion above guarantees zero overlap, so any hit here is a framework regression
            # and fails closed loudly rather than being silently reported as PASS. DFT labels are
            # never used as selection scores. This is the planner-side invariant that mirrors the
            # executor's assert_source_indices_allowed on the SAME canonical protected set.
            selected_globals = {full_pos[cid] for cid in selected_ids}
            overlaps = sorted(selected_globals & protected_globals)
            if overlaps:
                raise AcquisitionCapabilityGap(
                    "existing-pool selection produced protected-reference overlap AFTER canonical "
                    f"exclusion (regression): {overlaps[:20]}",
                    gap_kind="PROTECTED_EXCLUSION_REGRESSION")
            return ProtectedDisjointnessReport(
                status="PASS", n_checked=len(selected_ids), n_overlaps=0,
                dft_labels_used_as_selection_scores=False)

        # 3. Selection.
        if fe046 is None:
            # INITIAL acquisition (pre-FE-046 behavior, unchanged): diversity selection over the SAME
            # vectors with the SAME FPS/seed as sizing -> the chosen positions equal
            # sizing.selected_positions[:k], so selection and sizing never disagree.
            selection_result = select_candidates(
                selection_id=f"{m.frozen_artifact.evidence_id}-selection",
                generation_result=generation_result, descriptors=vectors,
                k=k, disjointness_checker=_disjointness_checker,
                selector="farthest_point_sampling", seed_index=0)
        else:
            # FE-046 coverage-gap reacquisition: compose CUMULATIVE (retain every prior-accepted
            # frame) + per-unsupported-class occupancy FLOOR (>=1 frame each, chosen by marginal
            # novelty) + FPS top-up to the novelty knee. The composed set can only GROW relative to
            # the cumulative+floor core, so coverage never regresses.
            ordered_local, composition = self._compose_cumulative_coverage_selection(
                vectors=vectors, item_ids=item_ids, eligible_positions=eligible_positions,
                knee_k=k, fe046=fe046)
            selection_result = select_candidates_from_indices(
                selection_id=f"{m.frozen_artifact.evidence_id}-selection",
                generation_result=generation_result, selected_indices=ordered_local,
                disjointness_checker=_disjointness_checker,
                selector="fe046_cumulative_floor_fps", seed_index=0, composition=composition)

        # 4. Map the selected candidate ids back to their GLOBAL pool positions (manifest order over
        #    the FULL pool, not the masked subset) so the Stage-3 executor -- which concatenates all
        #    manifest category files in order -- resolves the same frames.
        pos = {iid: i for i, iid in enumerate(full_item_ids)}
        selected_source_global_indices = [
            pos[cid] for cid in selection_result.selected_candidate_ids]
        selected_parent_structure_ids = [
            full_item_ids[i] for i in selected_source_global_indices]

        # Fail-closed post-condition (defense in depth): every selected frame must lie inside the
        # admissible in-scope population -- deterministic diversity selection may never re-introduce
        # an out-of-scope family. The masking above guarantees this; the check makes a regression
        # loud rather than silent.
        for cid in selected_parent_structure_ids:
            if self._source_category_of(cid) not in admissible_categories:
                raise AcquisitionCapabilityGap(
                    f"deterministic selection produced frame {cid!r} outside the admissible "
                    f"scope families {admissible_source_categories!r}; failing closed",
                    gap_kind="SCOPE_ELIGIBILITY_VIOLATION")

        labeling_request = build_labeling_request(
            request_id=f"{m.frozen_artifact.evidence_id}-labeling",
            selection_result=selection_result,
            teacher_identity_sha256=m.teacher_identity_sha256)

        existing_pool_projection = build_existing_pool_projection(
            pool_path=pc["manifest_path"],
            eligible_source_categories=admissible_source_categories,
            selected_parent_structure_ids=selected_parent_structure_ids,
            selected_source_global_indices=selected_source_global_indices,
            labeling_population_sizing=sizing.model_dump(mode="json"),
            selection_result=selection_result,
            duplicate_handling=pc["duplicate_handling"],
            protected_reference_id=protected_reference_id,
            protected_candidate_count=len(protected_globals),
            protected_excluded_count=protected_excluded_count,
            eligible_population_after_exclusion=len(eligible_positions))

        plan = assemble_plan_v2(
            plan_id=f"{m.frozen_artifact.evidence_id}-plan",
            objective=m.objective, inventory=m.inventory,
            target_regime_model=m.target_regime_model, region_resolution=m.region_resolution,
            coverage=m.coverage, strategy=strategy, generation_result=generation_result,
            selection_result=selection_result, labeling_request=labeling_request,
            existing_pool_projection=existing_pool_projection)

        return RealizedAcquisition(
            generation_result=generation_result, selection_result=selection_result,
            labeling_request=labeling_request, plan=plan,
            existing_pool_projection=existing_pool_projection)


def install_default_acquisition_provider(
    *, backend_probe: BackendProbe, teacher_probe: TeacherProbe,
    descriptor_providers: Sequence[object] = (),
    phase: AcquisitionPhase = AcquisitionPhase.INITIAL,
) -> FrameworkDefaultAcquisitionProvider:
    """Register the given descriptor plugins and install the framework-default provider globally.

    Idempotent-safe: descriptor registration replaces per ``material_id`` and the module-level
    provider is simply overwritten. Called by the run-campaign path at startup so a new user gets
    autonomous acquisition planning with no run-specific Python."""
    from framework_v2.acquisition.descriptor_plugins import register_descriptor_provider

    for dp in descriptor_providers:
        register_descriptor_provider(dp)
    provider = FrameworkDefaultAcquisitionProvider(
        backend_probe=backend_probe, teacher_probe=teacher_probe, phase=phase)
    set_acquisition_planning_provider(provider)
    return provider


def maybe_install_default_acquisition_provider(controller) -> Optional[FrameworkDefaultAcquisitionProvider]:
    """Auto-install the framework-default acquisition provider at run-campaign startup.

    This is the wiring that removes the old ``_PROVIDER=None`` hard no-op WITHOUT requiring any
    run-specific Python: the framework registers every descriptor plugin it ships and, if at least
    one exists, installs the default provider with the framework's own generic backend/Teacher
    probes. The provider still self-gates via ``applies`` (V2 closure + an ``acquisition`` stage +
    no human-supplied plan), so runs that supply their own plan and the broad regression suite are
    unperturbed.

    Returns the installed provider, or ``None`` when nothing was installed because a provider is
    already registered (an explicit registration wins -- e.g. a test's fake).

    Since FE-027 the framework always ships a material-agnostic generic fallback descriptor
    provider, so there is always at least one provider the default acquisition provider can compose
    around -- a brand-new material gets autonomous planning with no run-specific Python. The
    provider still self-gates via ``applies`` (V2 closure + an ``acquisition`` stage + no
    human-supplied plan), so runs that supply their own plan and the broad regression suite are
    unperturbed.

    Idempotent and side-effect-free beyond registration; safe to call once per run-campaign."""
    from framework_v2.acquisition.builtin import (
        register_builtin_descriptor_providers,
        register_builtin_generic_fallback,
    )

    if get_acquisition_planning_provider() is not None:
        return None
    register_builtin_descriptor_providers()
    register_builtin_generic_fallback()
    return install_default_acquisition_provider(
        backend_probe=default_backend_probe, teacher_probe=default_teacher_probe)
