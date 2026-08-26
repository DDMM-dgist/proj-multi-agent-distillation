"""FE-054 -- post-split TRAIN-only augmentation: one explicit, provenance-bound, approval-gated
lifecycle ACTION executed AFTER Stage 6 (``dataset_split``) has PASSed and BEFORE Stage 7
(``training``) is allowed to run.

This is NOT a new lifecycle stage and it does NOT mutate the frozen ``workflow/controller.py``. It
reuses the ALREADY-established autonomous acquisition reasoning capability -- the same descriptor
representation, coverage/strategy derivation, admissible-recipe decision space, LLM producer,
deterministic validators, provenance classes and stopping logic -- and only rebinds its evidence
context from the Stage-3 acquisition candidate pool onto the frozen Stage-6 TRAIN-parent population
plus the Student-distillation objective (see ``AugmentationAcquisitionProvider`` /
``build_augmentation_provider`` in ``default_acquisition_provider`` and the ``pool_manifest_path``
seam in ``generic_representation``). No perturbation number is hand-authored: the recipe's
admissible bounds come from the TRAIN pool's OWN nearest-neighbor scale via the same
``build_perturbation_envelope`` acquisition uses, and whether augmentation is even warranted is an
evidence-driven strategy outcome (a SATURATED TRAIN core deterministically selects
EXISTING_POOL_SELECTION -- i.e. "no augmentation warranted" -- which is a legitimate honest result,
not a fabrication).

Lifecycle contract implemented here:

    Stage 6 dataset_split PASS
      -> freeze TRAIN/validation/test parent membership (validation/test are never touched)
      -> build a schema-valid TRAIN-parent pool manifest, EXCLUDING any protected
         ``augmentation_parent`` structure (asserted)
      -> autonomous post-split AugmentationPlan over TRAIN parents only (reused producer)
      -> freeze the plan with full provenance
      -> [costly_teacher_labeling boundary] existing ``augment_atoms`` generation on TRAIN parents
      -> canonical Teacher labeling of the generated TRAIN children
      -> merge original labeled TRAIN parents + labeled augmented TRAIN children
         (``workflow.steps.prepare_student_distillation_dataset``)
      -> write ``artifacts/dataset/final_train.extxyz`` + a finalized provenance manifest
      -> Stage 7 training consumes the merged ``final_train`` as ``training.dataset``.

Stage 7 fails closed (``verify_finalized_augmentation``) if a run DECLARES post-split augmentation
(``training.pydantic_ai.requires_post_split_augmentation: true``) but the finalized output has not
been produced and routed as the training dataset.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------
# Run-local layout
# --------------------------------------------------------------------------------------------
def augmentation_dir(run_dir) -> Path:
    return Path(run_dir) / "augmentation"


def plan_path(run_dir, run_id) -> Path:
    return augmentation_dir(run_dir) / "plans" / f"{run_id}.augmentation_plan.json"


def final_train_path(run_dir) -> Path:
    # A NEW artifact -- never overwrite the frozen Stage-6 split output ``train.extxyz``.
    return Path(run_dir) / "artifacts" / "dataset" / "final_train.extxyz"


def finalized_manifest_path(run_dir) -> Path:
    return augmentation_dir(run_dir) / "augmentation_finalized.json"


def _split_manifest_path(run_dir) -> Path:
    return Path(run_dir) / "artifacts" / "dataset" / "split_manifest.json"


def derive_train_base_label_manifest(controller, *, train_dataset) -> Path:
    """Project the authoritative Stage-5 Teacher-label manifest onto the FROZEN TRAIN split.

    ``prepare_student_distillation_dataset`` proves same-run, same-Teacher binding by requiring a
    ``base_label_manifest`` whose recorded ``sha256`` equals the base dataset's bytes. The base
    dataset here is the Stage-6 split output ``train.extxyz`` (a byte-preserved SUBSET of the
    Stage-5 labeled pool), so NO Stage-5 manifest -- which records the FULL labeled pool's sha --
    matches it. This gap is closed HONESTLY, not by fabrication: the Stage-6 ``split_manifest``
    binds ``source_sha256`` to the exact Stage-5 labeled-pool output, and the TRAIN split's own
    recorded sha to ``train.extxyz``. We therefore locate the authoritative Stage-5 manifest by that
    source sha, copy its Teacher binding VERBATIM (teacher model/config SHAs, integrity, species
    mapping, units), and override ONLY the split-specific descriptors (output path, sha256,
    n_frames). Every override is an independently recomputed real value; nothing is invented.

    Fails closed if the split lineage cannot be proven (train bytes disagree with the split
    manifest, or no unique Stage-5 manifest carries the split's source sha)."""
    run_dir = controller.run_dir
    smp = _split_manifest_path(run_dir)
    if not smp.is_file():
        raise ValueError(f"no dataset split_manifest at {smp}; cannot derive the TRAIN base "
                         "label manifest")
    split = json.loads(smp.read_text())
    source_sha = split.get("source_sha256")
    train_split = (split.get("splits") or {}).get("train") or {}
    declared_train_sha = train_split.get("sha256")
    declared_train_n = train_split.get("n_frames")
    if not source_sha or not declared_train_sha:
        raise ValueError("split_manifest is missing source_sha256 or splits.train.sha256; the "
                         "TRAIN base label manifest lineage cannot be proven")
    actual_train_sha = _sha256_file(train_dataset)
    if actual_train_sha != declared_train_sha:
        raise ValueError(
            "TRAIN dataset bytes do not match the frozen split_manifest TRAIN sha "
            f"({actual_train_sha} != {declared_train_sha}); refusing to derive a base label "
            "manifest for an artifact the split did not produce")

    # Locate the authoritative Stage-5 manifest whose recorded output sha IS the split source.
    candidates = sorted((Path(run_dir) / "artifacts").glob("*.manifest.json"))
    matches = []
    for m in candidates:
        try:
            payload = json.loads(m.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("sha256") == source_sha and payload.get("teacher_model_sha256") \
                and payload.get("teacher_config_sha256"):
            matches.append((m, payload))
    if len(matches) != 1:
        raise ValueError(
            "could not uniquely resolve the authoritative Stage-5 Teacher-label manifest for the "
            f"split source sha {source_sha!r} (found {len(matches)} candidate(s)); refusing to "
            "guess the TRAIN parents' Teacher binding")
    source_manifest_path, source_payload = matches[0]

    projected = dict(source_payload)
    projected["schema_version"] = 1
    projected["output"] = str(Path(train_dataset).resolve())
    projected["sha256"] = actual_train_sha
    if declared_train_n is not None:
        projected["n_frames"] = int(declared_train_n)
    projected["derived_from"] = {
        "source_label_manifest": str(source_manifest_path.resolve()),
        "source_label_manifest_sha256": source_sha,
        "split_manifest": str(smp.resolve()),
        "split": "train",
        "note": ("Teacher binding copied verbatim from the Stage-5 labeled-pool manifest; only the "
                 "split-subset output/sha256/n_frames are recomputed. The split_manifest binds "
                 "source_sha256 to this Stage-5 output, proving train.extxyz is a byte-preserved "
                 "subset of it."),
    }
    out = augmentation_dir(run_dir) / "labeling" / "train_base_labels.manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(projected, indent=2, sort_keys=True) + "\n")
    return out


# --------------------------------------------------------------------------------------------
# Declaration + Stage-7 fail-closed guard
# --------------------------------------------------------------------------------------------
def _training_stage_config(controller) -> dict:
    import yaml
    wf = controller.state.get("workflow_config")
    if not wf or not Path(wf).exists():
        return {}
    cfg = yaml.safe_load(Path(wf).read_text()) or {}
    for stage in cfg.get("stages", []) or []:
        if stage.get("name") == "training":
            return stage
    return {}


def augmentation_required(controller) -> bool:
    """A run DECLARES post-split TRAIN augmentation by setting
    ``training.pydantic_ai.requires_post_split_augmentation: true`` in its frozen workflow.yaml.

    This is the single declaration switch the Stage-7 guard keys on; a run that does not declare it
    is entirely unaffected (the guard is a no-op and the whole action is opt-in)."""
    route = (_training_stage_config(controller).get("pydantic_ai") or {})
    return bool(route.get("requires_post_split_augmentation"))


def _resolved_training_dataset_param(controller) -> Optional[str]:
    route = (_training_stage_config(controller).get("pydantic_ai") or {})
    params = route.get("parameters") or {}
    ds = params.get("dataset")
    if not isinstance(ds, str):
        return None
    context = {"run_dir": str(controller.run_dir),
               "artifacts_dir": str(controller.run_dir / "artifacts"),
               "project_dir": controller.state.get("project_dir", "")}
    try:
        return str(Path(ds.format(**context)).resolve())
    except (KeyError, ValueError):
        return None


def verify_finalized_augmentation(controller) -> tuple[bool, Optional[str]]:
    """Deterministic Stage-7 precondition: the declared post-split augmentation MUST have been
    finalized and routed as the training dataset. Returns ``(True, None)`` when satisfied, else
    ``(False, reason)``. Never fabricates a PASS -- every check reads real bytes/hashes on disk."""
    run_dir = controller.run_dir
    fm_path = finalized_manifest_path(run_dir)
    if not fm_path.is_file():
        return (False,
                "post-split TRAIN augmentation is declared for this run "
                "(training.requires_post_split_augmentation) but no finalized augmentation "
                f"manifest exists at {fm_path}; run `augment-train --execute` before training")
    try:
        fm = json.loads(fm_path.read_text())
    except (ValueError, OSError) as exc:
        return (False, f"finalized augmentation manifest is unreadable/not JSON: {fm_path} ({exc})")

    recorded_sha = fm.get("final_train_sha256")
    recorded_path = fm.get("final_train_path")
    if not recorded_sha or not recorded_path:
        return (False,
                "finalized augmentation manifest is missing final_train_sha256/final_train_path; "
                "the augmentation action did not complete")
    ft = final_train_path(run_dir)
    if not ft.is_file():
        return (False, f"finalized training dataset does not exist on disk: {ft}")
    actual_sha = _sha256_file(ft)
    if actual_sha != recorded_sha:
        return (False,
                f"finalized training dataset hash mismatch: {ft} is {actual_sha[:12]} but the "
                f"finalized manifest records {str(recorded_sha)[:12]} -- the bound output does not "
                "match what the augmentation action produced")

    routed = _resolved_training_dataset_param(controller)
    if routed is None:
        return (False,
                "the training stage declares no resolvable dataset parameter to bind the finalized "
                "augmented dataset to")
    if routed != str(ft.resolve()):
        return (False,
                "the training stage dataset is not routed to the finalized augmented dataset "
                f"(training.dataset={routed} != final_train={ft.resolve()}); Stage 7 must consume "
                "the merged final_train, not the pre-augmentation split")
    return (True, None)


def stage7_augmentation_guard(controller, stage_name) -> Optional[str]:
    """The guard ``run_production_stage`` calls for the training stage: if this run declares
    post-split augmentation and it is not finalized+routed, return the fail-closed reason; else
    None. Keyed on the stage route action so it fires for the training stage regardless of the
    stage's declared name."""
    from .cli import _stage_route_action
    if _stage_route_action(controller, stage_name) != "train_committee":
        return None
    if not augmentation_required(controller):
        return None
    ok, reason = verify_finalized_augmentation(controller)
    return None if ok else reason


# --------------------------------------------------------------------------------------------
# TRAIN-parent pool manifest (protected augmentation_parent EXCLUDED)
# --------------------------------------------------------------------------------------------
def resolve_protected_source_indices(controller) -> tuple[Optional[str], set[int]]:
    """Resolve the run's protected source population through the ONE canonical framework resolver
    the acquisition executor/planner also enforce against. Returns ``(reference_id, globals)`` --
    ``(None, set())`` when the run declares no acquisition-protection reference. These global
    indices name protected rows in the ORIGINAL sanitized pool (manifest-concatenation order),
    which is exactly the ``source_global_index`` each split frame carries."""
    from .default_acquisition_provider import FrameworkDefaultAcquisitionProvider
    return FrameworkDefaultAcquisitionProvider._resolve_protected(controller)


def _frame_source_global_index(atoms, frame_index: int) -> int:
    """The frame's global index in the original sanitized pool. Prefer the explicit
    ``source_global_index`` info key; else parse it out of the ``parent_structure_id`` lineage
    tag ``seed-pool:<global_index>`` the split frames carry. Fails closed rather than guessing."""
    if "source_global_index" in atoms.info:
        return int(atoms.info["source_global_index"])
    psid = atoms.info.get("parent_structure_id")
    if isinstance(psid, str) and psid.startswith("seed-pool:"):
        return int(psid.split(":", 1)[1])
    raise ValueError(
        f"TRAIN parent frame {frame_index} carries neither source_global_index nor a "
        "seed-pool:<global_index> parent_structure_id lineage tag; cannot decide protected "
        "exclusion -- refusing to build an unprovenanced augmentation pool")


def build_train_pool_manifest(
    train_extxyz, out_dir, *, protected_source_indices=(), category: str = "train_parents",
) -> dict:
    """Build a schema-valid sanitized-pool manifest over the frozen Stage-6 TRAIN parents, with any
    protected ``augmentation_parent`` structure EXCLUDED (asserted). Writes the filtered parents
    file + manifest next to each other so ``locate_pool_manifest``/``load_pool`` resolve them via
    the ``pool_manifest_path`` seam, and returns a small provenance dict.

    Protected exclusion is mandatory: ``protected_reference.prohibited_uses`` includes
    ``augmentation_parent``, so a protected structure must never seed a perturbation. Every excluded
    frame is one whose original-pool ``source_global_index`` is in ``protected_source_indices``."""
    from ase.io import read as ase_read, write as ase_write

    train_extxyz = Path(train_extxyz)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    protected = {int(g) for g in protected_source_indices}

    frames = ase_read(str(train_extxyz), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    kept = []
    excluded_globals = []
    for i, atoms in enumerate(frames):
        g = _frame_source_global_index(atoms, i)
        if g in protected:
            excluded_globals.append(g)
            continue
        kept.append(atoms)
    if not kept:
        raise ValueError(
            "every TRAIN parent was excluded as a protected augmentation_parent; no admissible "
            "seed remains to plan augmentation over")

    parents_file = out_dir / f"{category}.extxyz"
    ase_write(str(parents_file), kept, format="extxyz")

    manifest = {
        "schema_version": 1,
        "operation": "post-split-train-augmentation-pool",
        "total_frames": len(kept),
        "categories": [{
            "category": category,
            "sanitized_file": parents_file.name,
            "n_frames": len(kept),
            "sanitized_file_sha256": _sha256_file(parents_file),
        }],
        "source_train_dataset": str(train_extxyz.resolve()),
        "source_train_dataset_sha256": _sha256_file(train_extxyz),
        "protected_augmentation_parents_excluded": sorted(excluded_globals),
        "n_protected_excluded": len(excluded_globals),
    }
    # The manifest MUST sit beside the parents file: load_pool resolves each category's
    # ``sanitized_file`` relative to the MANIFEST's own directory (generic_representation.load_pool
    # -> manifest_dir = manifest_path.parent).
    manifest_file = out_dir / "train_pool_manifest.json"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_body = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_file.write_text(manifest_body)
    # Self-referential provenance SHA (over the body without the field itself), mirroring the
    # sanitized-pool convention load_pool reads (``sanitized_pool_manifest_sha256``).
    manifest["sanitized_pool_manifest_sha256"] = hashlib.sha256(
        manifest_body.encode("utf-8")).hexdigest()
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"manifest_path": str(manifest_file), "parents_path": str(parents_file),
            "n_frames": len(kept), "n_protected_excluded": len(excluded_globals),
            "excluded_globals": sorted(excluded_globals)}


# --------------------------------------------------------------------------------------------
# Autonomous planning (reused producer) + freeze
# --------------------------------------------------------------------------------------------
def plan_train_augmentation(
    controller, *, runtime, agent_specs_dir, exchange_dir, repo_root,
    train_manifest_path, mock_producer_response=None, emitter=None,
):
    """Run the SHARED acquisition producer core, rebound onto the TRAIN-parent pool, to
    autonomously choose the augmentation recipe. Returns a ``ProducerRealizeResult`` (exactly one
    of ``realized`` / ``failure`` set). Binds nothing -- ``freeze_augmentation_plan`` persists the
    result as an AugmentationPlan artifact rather than as the Stage-3 acquisition input."""
    from .acquisition_planner import run_acquisition_producer
    from .default_acquisition_provider import build_augmentation_provider

    provider = build_augmentation_provider(train_manifest_path)
    if not provider.applies(controller):
        from .cli import CAMPAIGN_RESOURCE_BLOCKED, EXIT_BLOCKED_POLICY, CampaignRunResult
        from .acquisition_planner import ProducerRealizeResult
        return ProducerRealizeResult(failure=CampaignRunResult(
            CAMPAIGN_RESOURCE_BLOCKED, EXIT_BLOCKED_POLICY,
            "post-split augmentation planning requires a V2-closure-bound run (a bound "
            "DeploymentScopeContract); none is bound"))
    return run_acquisition_producer(
        controller, runtime=runtime, agent_specs_dir=agent_specs_dir, exchange_dir=exchange_dir,
        repo_root=repo_root, mock_producer_response=mock_producer_response, emitter=emitter,
        provider=provider, producer_role="data-curator", task_id_suffix="augmentation-plan",
        action_label="augmentation_plan_proposal")


def freeze_augmentation_plan(controller, produced, *, train_pool, emitter=None) -> Path:
    """Persist the autonomously-realized AugmentationPlan as a first-class provenance artifact.

    Records the frozen plan's content SHA + strategy, the TRAIN-parent pool manifest identity, the
    Teacher identity the plan is bound to, the protected-augmentation-parent exclusion, and the
    executable projection (existing-pool selection OR local-perturbation recipe) the executor will
    consume. Returns the frozen plan path."""
    realized = produced.realized
    ctx = produced.ctx
    run_dir = controller.run_dir
    run_id = controller.state["run_id"]
    out = plan_path(run_dir, run_id)
    out.parent.mkdir(parents=True, exist_ok=True)

    executable_projection = realized.existing_pool_projection or realized.legacy_projection
    strategy_kind = ctx.strategy.kind.value

    frozen = {
        "schema_version": 1,
        "artifact_kind": "post_split_train_augmentation_plan",
        "run_id": run_id,
        "plan_content_sha256": realized.plan.content_sha256(),
        "strategy_kind": strategy_kind,
        "augmentation_warranted": strategy_kind != "EXISTING_POOL_SELECTION",
        "coverage_gap_sha256": ctx.coverage.content_sha256(),
        "teacher_identity_sha256": ctx.teacher_identity_sha256,
        "train_pool_manifest_path": train_pool["manifest_path"],
        "train_pool_parents_path": train_pool["parents_path"],
        "train_pool_n_frames": train_pool["n_frames"],
        "protected_augmentation_parents_excluded": train_pool["excluded_globals"],
        "n_protected_excluded": train_pool["n_protected_excluded"],
        "required_param_keys": list(ctx.required_param_keys),
        "param_bounds": {k: list(v) for k, v in (ctx.param_bounds or {}).items()},
        "executable_projection": executable_projection,
    }
    out.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    if emitter is not None:
        emitter.emit("augmentation_plan_frozen",
                     detail={"plan_sha256": frozen["plan_content_sha256"],
                             "strategy_kind": strategy_kind,
                             "augmentation_warranted": frozen["augmentation_warranted"]})
    return out


# --------------------------------------------------------------------------------------------
# Child lineage remap (pure, unit-testable seam of the execute path)
# --------------------------------------------------------------------------------------------
def remap_child_lineage(children, parent_item_id_to_seedpool: dict[str, str]) -> None:
    """Rewrite each generated child's ``parent_structure_id`` from the generator's item-id tag
    (``"{category}#{idx}"``) to the canonical ``seed-pool:<global_index>`` lineage of its TRAIN
    parent, so ``prepare_student_distillation_dataset``'s ``assert_parent_lineage_allowed`` (which
    requires the seed-pool lineage) accepts the merged augmentation frames. Mutates in place; fails
    closed if a child's parent has no known seed-pool lineage."""
    for j, child in enumerate(children):
        item_id = child.info.get("parent_structure_id")
        seedpool = parent_item_id_to_seedpool.get(str(item_id))
        if seedpool is None:
            raise ValueError(
                f"augmented child {j} has parent item-id {item_id!r} with no resolved seed-pool "
                "lineage; refusing to merge an unprovenanced augmentation frame")
        child.info["parent_structure_id"] = seedpool


# --------------------------------------------------------------------------------------------
# Live execution (the costly_teacher_labeling boundary)
# --------------------------------------------------------------------------------------------
def _finalize_no_augmentation(controller, *, frozen_plan, base_dataset, emitter=None) -> dict:
    """Honest EXISTING_POOL_SELECTION outcome: the TRAIN core is descriptor-saturated, so the
    autonomous strategy determined augmentation is NOT warranted. final_train == the original
    labeled TRAIN parents (byte-copied to the new artifact path); no Teacher call is made."""
    run_dir = controller.run_dir
    ft = final_train_path(run_dir)
    ft.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base_dataset, ft)
    fm = {
        "schema_version": 1,
        "artifact_kind": "post_split_train_augmentation_finalized",
        "augmentation_warranted": False,
        "strategy_kind": frozen_plan["strategy_kind"],
        "plan_content_sha256": frozen_plan["plan_content_sha256"],
        "base_train_dataset": str(Path(base_dataset).resolve()),
        "base_train_dataset_sha256": _sha256_file(base_dataset),
        "final_train_path": str(ft.resolve()),
        "final_train_sha256": _sha256_file(ft),
        "n_augmented_children": 0,
        "protected_augmentation_parents_excluded":
            frozen_plan.get("protected_augmentation_parents_excluded", []),
        "rationale": ("autonomous post-split strategy selected EXISTING_POOL_SELECTION: the frozen "
                      "TRAIN-parent descriptor core is saturated, so no local perturbation is "
                      "warranted; final_train is the original labeled TRAIN parents unchanged"),
    }
    finalized_manifest_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    finalized_manifest_path(run_dir).write_text(json.dumps(fm, indent=2, sort_keys=True) + "\n")
    if emitter is not None:
        emitter.emit("augmentation_finalized",
                     detail={"augmentation_warranted": False,
                             "final_train_sha256": fm["final_train_sha256"]})
    return fm


def execute_train_augmentation(
    controller, *, base_dataset, base_label_manifest, teacher_config, reference_yaml,
    emitter=None,
) -> dict:
    """Execute the frozen AugmentationPlan end-to-end and produce ``final_train.extxyz``.

    LIVE, GPU/Teacher path (the ``costly_teacher_labeling`` boundary):
      * EXISTING_POOL_SELECTION (no augmentation warranted) -> final_train == labeled TRAIN parents
        (no Teacher call; ``_finalize_no_augmentation``).
      * LOCAL_PERTURBATION -> generate children from the TRAIN parents with the existing
        ``LocalPerturbationGenerator`` (``augment_atoms`` under the frozen Teacher PES), canonically
        Teacher-label them (``adapters.acquisition.label_with_teacher``), remap child lineage to the
        seed-pool parent identity, then merge labeled parents + labeled children via
        ``workflow.steps.prepare_student_distillation_dataset`` into final_train.

    ``base_dataset`` is the frozen Stage-6 labeled TRAIN parents; ``base_label_manifest`` is the
    Stage-5 teacher_labeling manifest for them (used by the merge to prove same-run, same-Teacher
    binding). Returns the finalized provenance manifest dict."""
    from ase.io import read as ase_read

    run_dir = controller.run_dir
    run_id = controller.state["run_id"]
    fp = plan_path(run_dir, run_id)
    if not fp.is_file():
        raise ValueError(f"no frozen AugmentationPlan at {fp}; run the plan phase first")
    frozen_plan = json.loads(fp.read_text())

    if not frozen_plan.get("augmentation_warranted"):
        return _finalize_no_augmentation(
            controller, frozen_plan=frozen_plan, base_dataset=base_dataset, emitter=emitter)

    # -- LOCAL_PERTURBATION: the costly Teacher-driven generation + labeling + merge ---------------
    from adapters import load_config
    from adapters.acquisition import label_with_teacher
    from framework_v2.acquisition.contracts import AcquisitionStrategyKind
    from framework_v2.acquisition.generators.base import GenerationProtocol, TeacherCalculatorProvider
    from framework_v2.acquisition.generators.local_perturbation import LocalPerturbationGenerator
    from workflow.steps import prepare_student_distillation_dataset

    projection = frozen_plan["executable_projection"]
    parents_path = frozen_plan["train_pool_parents_path"]

    # Map each TRAIN-parent item-id ("{category}#{idx}") -> its canonical seed-pool lineage, so the
    # generated children can carry the merge-required parent_structure_id="seed-pool:<global_index>".
    parent_frames = ase_read(parents_path, index=":")
    if not isinstance(parent_frames, list):
        parent_frames = [parent_frames]
    category = json.loads(Path(frozen_plan["train_pool_manifest_path"]).read_text())[
        "categories"][0]["category"]
    item_id_to_seedpool: dict[str, str] = {}
    for idx, atoms in enumerate(parent_frames):
        g = _frame_source_global_index(atoms, idx)
        item_id_to_seedpool[f"{category}#{idx}"] = f"seed-pool:{g}"

    selected_parent_ids = list(projection.get("selected_parent_structure_ids")
                               or projection.get("parent_ids") or [])
    # The LOCAL_PERTURBATION recipe lives at the TOP LEVEL of the legacy projection
    # (framework_v2.acquisition.plan_assembly.build_legacy_projection): n_per_structure / T_K /
    # beta / sigma_range_A / cell_sigma / seed. ``units`` is not a scientific recipe choice; it is
    # the energy-unit declaration the augment executor supplies as a schema default ("eV") via
    # executors._write_executable_augment_config, so we mirror that mechanical default here rather
    # than fabricating a value. Every generation param is read straight from the frozen plan.
    _missing = [k for k in ("n_per_structure", "T_K", "beta", "sigma_range_A", "cell_sigma", "seed")
                if k not in projection]
    if _missing:
        raise ValueError(
            "frozen LOCAL_PERTURBATION AugmentationPlan projection is missing required recipe "
            f"fields {_missing}; refusing to run augment_atoms with an incomplete recipe")
    recipe = {
        "parents_path": parents_path,
        "n_per_structure": int(projection["n_per_structure"]),
        "T_K": float(projection["T_K"]),
        "beta": float(projection["beta"]),
        "sigma_range_A": [float(projection["sigma_range_A"][0]),
                          float(projection["sigma_range_A"][1])],
        "cell_sigma": (None if projection["cell_sigma"] is None
                       else float(projection["cell_sigma"])),
        "seed": int(projection["seed"]),
        "units": projection.get("units", "eV"),
    }

    protocol = GenerationProtocol(
        protocol_id=f"{run_id}-augmentation-protocol",
        backend_id="local_perturbation.augment_atoms",
        strategy_kind=AcquisitionStrategyKind.LOCAL_PERTURBATION,
        strategy_sha256=str(frozen_plan.get("plan_content_sha256", "")),
        n_requested=len(selected_parent_ids) * int(recipe.get("n_per_structure", 1)),
        target_regime_ids=[], parent_ids=selected_parent_ids, params=recipe)

    class _TeacherCalc(TeacherCalculatorProvider):
        def __init__(self, cfg):
            self._cfg = cfg
        def make_ase_calculator(self):
            from adapters.teacher import load_teacher_with_species_evidence
            calc, _ = load_teacher_with_species_evidence(self._cfg)
            return calc

    teacher_cfg = load_config(teacher_config)
    workdir = str(augmentation_dir(run_dir) / "generation")
    gen = LocalPerturbationGenerator().generate(
        protocol, workdir=workdir, teacher=_TeacherCalc(teacher_cfg))
    if not gen.artifact_ref:
        raise ValueError("augment_atoms produced no children; cannot finalize augmentation")

    children = ase_read(gen.artifact_ref, index=":")
    if not isinstance(children, list):
        children = [children]
    remap_child_lineage(children, item_id_to_seedpool)
    from ase.io import write as ase_write
    children_relineaged = augmentation_dir(run_dir) / "generation" / "children_relineaged.extxyz"
    ase_write(str(children_relineaged), children, format="extxyz")

    labeled_children = augmentation_dir(run_dir) / "labeling" / "children_labeled.extxyz"
    labeled_children.parent.mkdir(parents=True, exist_ok=True)
    aug_label_manifest = augmentation_dir(run_dir) / "labeling" / "children_labels.manifest.json"
    label_with_teacher(teacher_cfg, str(children_relineaged), str(labeled_children),
                       str(aug_label_manifest))

    ft = final_train_path(run_dir)
    ft.parent.mkdir(parents=True, exist_ok=True)
    merge_manifest = augmentation_dir(run_dir) / "final_train.merge.manifest.json"
    protection_audit = augmentation_dir(run_dir) / "final_train.protection_audit.json"
    merge = prepare_student_distillation_dataset(
        base_dataset=base_dataset, augmentation_dataset=str(labeled_children),
        output=str(ft), manifest=str(merge_manifest), protection_audit=str(protection_audit),
        reference_yaml=reference_yaml, base_label_manifest=base_label_manifest,
        augmentation_label_manifest=str(aug_label_manifest), run_dir=str(run_dir))

    fm = {
        "schema_version": 1,
        "artifact_kind": "post_split_train_augmentation_finalized",
        "augmentation_warranted": True,
        "strategy_kind": frozen_plan["strategy_kind"],
        "plan_content_sha256": frozen_plan["plan_content_sha256"],
        "teacher_identity_sha256": frozen_plan.get("teacher_identity_sha256"),
        "base_train_dataset": str(Path(base_dataset).resolve()),
        "base_train_dataset_sha256": _sha256_file(base_dataset),
        "base_label_manifest": str(Path(base_label_manifest).resolve()),
        "augmented_children": str(labeled_children.resolve()),
        "augmentation_label_manifest": str(aug_label_manifest.resolve()),
        "n_augmented_children": len(children),
        "child_lineage_remapped": True,
        "merge_manifest": str(merge_manifest.resolve()),
        "merge_output_sha256": merge.get("output_integrity"),
        "protection_audit": str(protection_audit.resolve()),
        "protected_augmentation_parents_excluded":
            frozen_plan.get("protected_augmentation_parents_excluded", []),
        "final_train_path": str(ft.resolve()),
        "final_train_sha256": _sha256_file(ft),
    }
    finalized_manifest_path(run_dir).write_text(json.dumps(fm, indent=2, sort_keys=True) + "\n")
    if emitter is not None:
        emitter.emit("augmentation_finalized",
                     detail={"augmentation_warranted": True,
                             "n_augmented_children": len(children),
                             "final_train_sha256": fm["final_train_sha256"]})
    return fm
