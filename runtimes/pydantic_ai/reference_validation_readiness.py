"""Deterministic, Teacher-free criterion-evidence computation for a Teacher-vs-DFT
``reference_validation`` of an evidence-bearing (e.g. ``recovered-original-holdout``) reference.

One computation serves two callers (FE-032):

* the pre-costly **preflight** (``cli.run_production_stage`` before executor dispatch): the NON-Teacher
  criteria must all be satisfiable, or the stage fails closed BEFORE any Teacher/GPU inference is
  dispatched -- so a run never spends expensive Teacher compute only to discover afterward that the
  gate's lineage / split / identity evidence can never be surfaced (the ffv4g/ffv4h defect).
* the **gate evidence packet** (``cli.run_production_stage`` after execution): the same record, now
  including the post-execution numeric fidelity metrics, is surfaced as a ``validation_outcome`` so a
  Judge can VERIFY each gate criterion against deterministic evidence rather than infer it.

Generic by identity/provenance contract, never a material/filename or hardcoded frame count: it is
keyed on the consumed reference's ``kind`` (an evidence-bearing kind that declares a
``split_source_manifest`` + ``target_split``) and reports the ACTUAL join/hash results it reads. It
NEVER runs the Teacher and NEVER fabricates lineage -- a criterion that cannot be established
deterministically is reported as a gap, never guessed.
"""
from __future__ import annotations

import json
from pathlib import Path

from workflow.controller import EVIDENCE_STRUCTURE_REFERENCE_KINDS
from workflow.integrity import artifact_digest, sha256_file


def _parameters(proposal) -> dict:
    params = (proposal or {}).get("parameters")
    return params if isinstance(params, dict) else {}


def _load_reference(reference_yaml: str):
    """Return the parsed reference doc iff it is an evidence-bearing kind declaring a
    ``split_source_manifest`` (the contract this readiness check applies to); else None."""
    import yaml
    try:
        doc = yaml.safe_load(Path(reference_yaml).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("kind") not in EVIDENCE_STRUCTURE_REFERENCE_KINDS:
        return None
    if not isinstance(doc.get("split_source_manifest"), str):
        return None
    return doc


def _resolve(base_yaml: str, raw: str, project_dir: str) -> Path:
    candidate = Path(str(raw).format(project_dir=str(project_dir)))
    if not candidate.is_absolute():
        candidate = Path(base_yaml).resolve().parent / candidate
    return candidate.resolve()


def _crosswalk(split_manifest_paths):
    from .bounded_evidence import build_split_crosswalk
    return build_split_crosswalk(list(split_manifest_paths or []))


def compute_reference_validation_evidence(controller, proposal, *, split_manifest_paths,
                                          report_path=None) -> dict | None:
    """Compute the criterion-evidence record for a reference_validation proposal, or ``None`` when
    the proposal is not a Teacher-vs-DFT validation of an evidence-bearing reference (not applicable
    -- the caller then imposes no extra preflight/packet surfacing).

    ``split_manifest_paths``: the authoritative source->split crosswalk manifest path(s) the run
    bound (see ``cli._split_membership_manifest_sources``).
    ``report_path``: the post-execution ``reference_validation.json`` when it exists; when absent
    (pre-execution preflight) the Teacher-dependent numeric metrics are marked pending.

    Returns ``{"applicable": True, "ready": bool, "blocking_gaps": [...], "criteria": {...},
    "teacher_metrics_pending": bool}``. ``ready`` is True iff every NON-Teacher criterion is
    satisfied; ``blocking_gaps`` enumerates the ones that are not (the preflight fails closed on any).
    """
    params = _parameters(proposal)
    reference_yaml = params.get("reference_yaml")
    if not isinstance(reference_yaml, str) or not Path(reference_yaml).exists():
        return None
    doc = _load_reference(reference_yaml)
    if doc is None:
        return None

    project_dir = str(controller.state.get("project_dir") or Path.cwd())
    criteria: dict[str, dict] = {}
    gaps: list[str] = []

    def record(key, *, status, gap=None, **fields):
        criteria[key] = {"status": status, **fields}
        if gap:
            gaps.append(gap)

    bound_sources = {str(Path(r["source"]).resolve()) for r in controller.state.get("inputs", [])}

    # --- (1) population / reference identity -------------------------------------------------
    reference_id = doc.get("reference_id")
    target_split = doc.get("target_split")
    frame_count = doc.get("frame_count")
    record("population_identity", status="RESOLVED", reference_id=reference_id,
           reference_kind=doc.get("kind"), target_split=target_split,
           declared_frame_count=frame_count,
           reference_yaml_sha256=sha256_file(Path(reference_yaml)))

    # --- structures population + hash (#4 structure identity) --------------------------------
    structures = doc.get("structures") or {}
    structures_path = None
    frames = None
    if isinstance(structures, dict) and isinstance(structures.get("path"), str):
        structures_path = _resolve(reference_yaml, structures["path"], project_dir)
    if structures_path is None or not structures_path.exists():
        record("structure_identity", status="MISSING",
               gap=f"structures population is missing or undeclared: {structures_path}")
    else:
        actual_sha = sha256_file(structures_path)
        declared_sha = structures.get("sha256")
        sha_ok = (not declared_sha) or declared_sha == actual_sha
        is_bound = str(structures_path) in bound_sources
        record("structure_identity",
               status="VERIFIED" if (sha_ok and is_bound) else "MISMATCH",
               structures_path=str(structures_path), structures_sha256=actual_sha,
               declared_sha256=declared_sha, hash_matches=sha_ok, bound_input=is_bound,
               gap=(None if (sha_ok and is_bound) else
                    ("structures sha256 mismatch" if not sha_ok
                     else "structures population is not a bound Controller input")))
        try:
            from ase.io import read
            frames = read(str(structures_path), index=":")
        except Exception as exc:  # noqa: BLE001 - deterministic read failure is a real gap
            record("structure_readable", status="UNREADABLE",
                   gap=f"structures unreadable: {type(exc).__name__}: {exc}")

    # --- (#2 lineage join) + (#3 TEST membership) -------------------------------------------
    crosswalk = _crosswalk(split_manifest_paths)
    resolved = crosswalk.get("resolved", {})
    ambiguous = crosswalk.get("ambiguous", set())
    record("split_manifest_binding",
           status="BOUND" if crosswalk.get("sources") else "UNBOUND",
           sources=crosswalk.get("sources", []),
           ambiguous_key_count=len(ambiguous),
           declared_split_manifest_sha256=doc.get("split_source_manifest_sha256"),
           gap=(None if crosswalk.get("sources")
                else "authoritative source->split crosswalk manifest is not bound/surfaced"))
    if frames is not None:
        joined = unjoined = amb = 0
        split_counts: dict[str, int] = {}
        for atoms in frames:
            cat = atoms.info.get("source_category")
            local = atoms.info.get("source_local_index")
            if cat is None or local is None:
                unjoined += 1
                continue
            key = (str(cat), int(local))
            if key in ambiguous:
                amb += 1
            elif key in resolved:
                joined += 1
                split_counts[str(resolved[key])] = split_counts.get(str(resolved[key]), 0) + 1
            else:
                unjoined += 1
        n = len(frames)
        count_ok = (frame_count is None) or int(frame_count) == n
        lineage_ok = joined == n and unjoined == 0 and amb == 0
        record("source_split_lineage_join",
               status="COMPLETE" if lineage_ok else "INCOMPLETE",
               n_frames=n, lineage_joined=joined, lineage_unjoined=unjoined,
               lineage_ambiguous=amb, split_distribution=split_counts,
               gap=(None if lineage_ok else
                    f"lineage join incomplete: joined={joined} unjoined={unjoined} "
                    f"ambiguous={amb} of {n}"))
        record("declared_frame_count_consistency",
               status="CONSISTENT" if count_ok else "INCONSISTENT",
               declared_frame_count=frame_count, actual_frame_count=n,
               gap=(None if count_ok else
                    f"declared frame_count {frame_count} != actual {n}"))
        split_ok = bool(split_counts) and set(split_counts) == {str(target_split)}
        record("test_split_membership",
               status="CONFIRMED" if split_ok else "UNCONFIRMED",
               target_split=target_split, split_distribution=split_counts,
               gap=(None if split_ok else
                    f"not all joined frames resolve to target_split={target_split!r}: "
                    f"{split_counts}"))

        # --- (#5 DFT-label identity / provenance) -------------------------------------------
        energy_keys = ("dft_energy", "dft_free_energy", "energy")
        force_keys = ("dft_forces", "forces")
        with_energy = sum(1 for a in frames if any(k in a.info for k in energy_keys))
        with_forces = sum(1 for a in frames if any(k in a.arrays for k in force_keys))
        labels_ok = with_energy == n and with_forces == n
        record("dft_label_provenance",
               status="PRESENT" if labels_ok else "INCOMPLETE",
               frames_with_dft_energy=with_energy, frames_with_dft_forces=with_forces,
               n_frames=n,
               gap=(None if labels_ok else
                    f"DFT labels incomplete: energy {with_energy}/{n}, forces {with_forces}/{n}"))

    # --- (#6 Teacher checkpoint identity) ----------------------------------------------------
    teacher_config = params.get("teacher_config")
    try:
        from adapters import load_config
        from adapters.teacher import teacher_model_reference
        teacher_cfg = load_config(teacher_config) if teacher_config else None
        model_value = teacher_model_reference(teacher_cfg) if teacher_cfg else None
        model_path = Path(model_value).expanduser().resolve() if model_value else None
        if model_path is not None and model_path.is_file():
            record("teacher_checkpoint_identity", status="RESOLVED",
                   teacher_config=str(Path(teacher_config).resolve()),
                   model_path=str(model_path), model_sha256=sha256_file(model_path))
        else:
            record("teacher_checkpoint_identity", status="MISSING",
                   teacher_config=teacher_config, model_path=str(model_path) if model_path else None,
                   gap="Teacher checkpoint could not be resolved to an existing file")
    except Exception as exc:  # noqa: BLE001
        record("teacher_checkpoint_identity", status="UNRESOLVED",
               gap=f"Teacher checkpoint resolution failed: {type(exc).__name__}: {exc}")

    # --- (#8 fresh-run / no-historical-reuse) ------------------------------------------------
    reuses_historical = bool(params.get("historical_report"))
    record("no_historical_reuse",
           status="FRESH" if not reuses_historical else "REUSES_HISTORICAL",
           historical_report_bound=reuses_historical)

    # --- (#9 protected-reference-use policy resolvable) --------------------------------------
    prohibited = doc.get("prohibited_uses")
    protected_roles = controller.state.get("protected_reference_roles") or []
    policy_ok = isinstance(prohibited, list) or bool(protected_roles)
    record("protected_reference_use_policy",
           status="RESOLVED" if policy_ok else "UNRESOLVED",
           prohibited_uses=prohibited, protected_reference_roles=list(protected_roles),
           gap=(None if policy_ok else "no protected-reference-use policy is resolvable"))

    # --- (#7 prediction identity) + (#10/#11 numeric metrics) : post-execution --------------
    metrics_pending = True
    if report_path and Path(report_path).exists():
        try:
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            report = None
        if isinstance(report, dict):
            metrics_pending = False
            pred = report.get("prediction_artifact") or {}
            record("prediction_artifact_identity", status="RECORDED",
                   path=pred.get("path"), integrity=pred.get("integrity"),
                   n_frames=pred.get("n_frames"), labels=pred.get("labels"))
            metrics = report.get("metrics") or {}
            record("global_fidelity_metrics", status="RECORDED",
                   energy_unit=metrics.get("energy_unit"), force_unit=metrics.get("force_unit"),
                   energy_normalization=metrics.get("energy_normalization"),
                   global_metrics=metrics.get("global"))
            by_config = metrics.get("by_config_type") or {}
            record("grouped_fidelity_metrics", status="RECORDED",
                   group_count=len(by_config), by_config_type=by_config)
            record("protected_reference_use_recorded", status="RECORDED",
                   protected_reference_use=report.get("protected_reference_use"),
                   historical_prediction_policy=report.get("historical_prediction_policy"))
    if metrics_pending:
        record("prediction_artifact_identity", status="PENDING_EXECUTION")
        record("global_fidelity_metrics", status="PENDING_EXECUTION")
        record("grouped_fidelity_metrics", status="PENDING_EXECUTION")

    return {
        "applicable": True,
        "ready": not gaps,
        "blocking_gaps": gaps,
        "criteria": criteria,
        "teacher_metrics_pending": metrics_pending,
    }
