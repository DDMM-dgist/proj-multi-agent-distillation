"""Deterministic, Teacher-free criterion-evidence computation for an EXISTING_POOL_SELECTION
``acquisition`` stage (FE-033).

Mirrors ``reference_validation_readiness`` (FE-032): a deterministic per-criterion evidence record
surfaced into the acquisition gate's bounded-evidence / ``gate_outcomes`` path, so a Judge can VERIFY
each acquisition gate criterion against deterministic evidence rather than infer it from the raw
acquisition manifest -- whose semantic join/mapping/attestation fields the generic JSON summariser
drops (the ffv4k defect, where three Judges REVISEd on parent->pool join = null, domain = unknown,
and attestation values = null even though the raw values existed on disk).

It surfaces exactly the three criteria the acquisition gate turns on:

  A. **parent -> sanitized-pool join** -- every selected parent is resolved against the hash-bound
     sanitized pool (joined / unmatched / foreign / duplicate), binding pool path + the declared
     manifest identity/SHA the plan consumed.
  B. **deployment-domain mapping** -- each selected parent's raw source category is resolved to a
     canonical deployment domain via the run's OWN bound frozen scope-classification evidence (the
     ``label_map`` referenced by the closure-bound ``DeploymentScopeContract`` regions'
     ``membership_evidence``); per-parent domain + aggregate coverage vs the frozen primary domains.
     A parent whose category does not map deterministically is reported AMBIGUOUS and fails closed --
     never guessed to eliminate "unknown".
  C. **selection-control attestations** -- the explicit booleans the manifest records
     (``dft_labels_used_as_selection_scores``, ``performs_teacher_inference``) plus the exclusive
     sanitized-pool source-membership result.

Generic by contract shape, never a material/filename or hardcoded category name: keyed on the
acquisition action and on the acquisition-manifest / pool-manifest / scope-contract SHAPES, and
reports the ACTUAL join / mapping / attestation values it reads. It NEVER runs the Teacher, NEVER
selects or re-selects structures, NEVER changes any AcquisitionPlan field, and NEVER fabricates a
domain mapping -- a criterion that cannot be established deterministically is reported as a gap.
"""
from __future__ import annotations

import json
from pathlib import Path

# The acquisition action whose gate this record serves; kept as a module constant rather than a
# literal so the applicability test reads as a contract, not a magic string.
ACQUISITION_ACTION_TYPE = "acquire_structures"


def _parameters(proposal) -> dict:
    params = (proposal or {}).get("parameters")
    return params if isinstance(params, dict) else {}


def _resolve(project_dir: str, raw: str) -> Path:
    candidate = Path(str(raw))
    if not candidate.is_absolute():
        candidate = Path(project_dir) / candidate
    return candidate


def _looks_like_acquisition_manifest(obj) -> bool:
    """Schema-detect the acquisition manifest (never its filename): a select-existing-pool
    acquisition record binds a pool + the concrete selected source records + the selection-control
    attestation booleans."""
    if not isinstance(obj, dict):
        return False
    if obj.get("stage") != "acquisition":
        return False
    if not isinstance(obj.get("selected_source_records"), list):
        return False
    if not isinstance(obj.get("pool_path"), str):
        return False
    return "dft_labels_used_as_selection_scores" in obj


def _locate_acquisition_manifest(controller, proposal):
    """Return the acquisition manifest path among the stage's declared outputs, by SHAPE, or None
    when it has not been produced yet (pre-execution)."""
    run_dir = Path(controller.run_dir)
    for rel in (proposal or {}).get("expected_outputs", []) or []:
        p = rel if Path(rel).is_absolute() else (run_dir / rel)
        p = Path(p)
        if p.suffix.lower() != ".json" or not p.is_file():
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if _looks_like_acquisition_manifest(obj):
            return p
    return None


def _load_scope_classification_evidence(controller, project_dir):
    """Resolve the run's OWN frozen scope-classification evidence (a ``label_map`` scope contract)
    from the closure-bound ``DeploymentScopeContract`` regions' ``membership_evidence`` references.

    Returns ``(scope_v2, binding)`` where ``scope_v2`` is a validated
    ``framework_v2.scientific_adequacy.DeploymentScopeContractV2`` and ``binding`` records the
    provenance actually used; or ``(None, reason)`` when no such evidence is deterministically
    resolvable (fails closed -- never invents a mapping)."""
    v2_state = controller._v2_state() if hasattr(controller, "_v2_state") else {}
    scope_sha = (v2_state or {}).get("scope_contract_sha256")
    if not scope_sha:
        return None, {"reason": "no V2 DeploymentScopeContract is bound to the run"}
    scope_dict = controller.v2_contract(scope_sha)
    if not isinstance(scope_dict, dict):
        return None, {"reason": f"bound scope contract {scope_sha!r} is not resolvable in run state"}

    evidence_refs: list[str] = []
    for region in scope_dict.get("regions", []) or []:
        if not isinstance(region, dict):
            continue
        for ref in region.get("membership_evidence", []) or []:
            if isinstance(ref, str) and ref not in evidence_refs:
                evidence_refs.append(ref)

    from framework_v2.scientific_adequacy import DeploymentScopeContractV2
    for ref in evidence_refs:
        path = _resolve(project_dir, ref)
        if not path.is_file():
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not (isinstance(obj, dict) and isinstance(obj.get("label_map"), list)
                and obj.get("primary_domains")):
            continue
        try:
            scope_v2 = DeploymentScopeContractV2.model_validate(obj)
        except Exception:  # noqa: BLE001 - a malformed evidence ref is simply not usable
            continue
        return scope_v2, {
            "scope_contract_sha256": scope_sha,
            "scope_contract_id": scope_dict.get("contract_id"),
            "classification_evidence_path": str(path),
        }
    return None, {
        "reason": "no scope-classification (label_map) evidence is deterministically resolvable "
                  "from the bound scope contract regions' membership_evidence",
        "scope_contract_sha256": scope_sha,
        "membership_evidence_refs": evidence_refs,
    }


def _selected_records(manifest) -> list[dict]:
    records = manifest.get("selected_source_records")
    return [r for r in records if isinstance(r, dict)] if isinstance(records, list) else []


def _record_category(rec) -> str | None:
    cats = rec.get("categories")
    if isinstance(cats, list) and cats:
        return str(cats[0])
    sid = rec.get("structure_id")
    if isinstance(sid, str) and "#" in sid:
        return sid.rsplit("#", 1)[0]
    return None


def compute_acquisition_evidence(controller, proposal, *, report_path=None) -> dict | None:
    """Compute the criterion-evidence record for an ``acquire_structures`` proposal, or ``None`` when
    the proposal is not that action (not applicable -- the caller then imposes no extra surfacing).

    ``report_path``: unused for acquisition (kept for a signature symmetric to
    ``reference_validation_readiness``); the acquisition manifest is located among the stage's own
    declared outputs. When the manifest is not yet produced (pre-execution) every criterion is
    marked PENDING_EXECUTION.

    Returns ``{"applicable": True, "ready": bool, "blocking_gaps": [...], "criteria": {...},
    "pending_execution": bool}``. ``ready`` is True iff every deterministic surfacing criterion is
    satisfied; ``blocking_gaps`` enumerates the ones that are not. ``ready`` is INFORMATIONAL only --
    the acquisition Judges adjudicate the gate; this record never re-selects or fails a costly action.
    """
    if (proposal or {}).get("action_type") != ACQUISITION_ACTION_TYPE:
        return None

    project_dir = str(controller.state.get("project_dir") or Path.cwd())
    criteria: dict[str, dict] = {}
    gaps: list[str] = []

    def record(key, *, status, gap=None, **fields):
        criteria[key] = {"status": status, **fields}
        if gap:
            gaps.append(gap)

    manifest_path = _locate_acquisition_manifest(controller, proposal)
    if manifest_path is None:
        record("parent_pool_join", status="PENDING_EXECUTION")
        record("deployment_domain_mapping", status="PENDING_EXECUTION")
        record("selection_control_attestations", status="PENDING_EXECUTION")
        return {"applicable": True, "ready": True, "blocking_gaps": [],
                "criteria": criteria, "pending_execution": True}

    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        record("acquisition_manifest_readable", status="UNREADABLE",
               gap=f"acquisition manifest unreadable: {type(exc).__name__}: {exc}")
        return {"applicable": True, "ready": False, "blocking_gaps": gaps,
                "criteria": criteria, "pending_execution": False}

    records = _selected_records(manifest)

    # --- (A) parent -> sanitized-pool join ---------------------------------------------------
    pool_path_raw = manifest.get("pool_path")
    recorded_pool_sha = manifest.get("pool_manifest_sha256")
    pool_path = _resolve(project_dir, pool_path_raw) if isinstance(pool_path_raw, str) else None
    pool = None
    if pool_path is None or not pool_path.is_file():
        record("parent_pool_join", status="POOL_UNBOUND",
               pool_path=(str(pool_path) if pool_path else None),
               gap="sanitized-pool manifest is not resolvable to a bound file")
    else:
        try:
            pool = json.loads(pool_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            record("parent_pool_join", status="POOL_UNREADABLE", pool_path=str(pool_path),
                   gap=f"sanitized-pool manifest unreadable: {type(exc).__name__}: {exc}")
    if pool is not None:
        declared_pool_sha = pool.get("sanitized_pool_manifest_sha256")
        sha_ok = (recorded_pool_sha is None) or (declared_pool_sha == recorded_pool_sha)
        # Global concatenation order = manifest category order (the order the existing-pool executor
        # reproduces); build cumulative [start, end) ranges to resolve each GLOBAL source index.
        ranges: list[tuple[str, int, int]] = []
        offset = 0
        pool_categories: set[str] = set()
        for cat in pool.get("categories", []) or []:
            if not isinstance(cat, dict):
                continue
            name = cat.get("category")
            n = cat.get("n_frames")
            if not isinstance(name, str) or not isinstance(n, int):
                continue
            pool_categories.add(name)
            ranges.append((name, offset, offset + n))
            offset += n
        total_frames = offset

        def resolve_global(g: int):
            for name, start, end in ranges:
                if start <= g < end:
                    return name, g - start
            return None, None

        joined = unmatched = foreign = 0
        duplicates = 0
        seen: set[tuple] = set()
        per_parent: list[dict] = []
        for rec in records:
            claimed_cat = _record_category(rec)
            g = rec.get("source_index")
            key = (claimed_cat, g)
            is_dup = key in seen
            if is_dup:
                duplicates += 1
            seen.add(key)
            status = "JOINED"
            resolved_cat = None
            if claimed_cat is not None and claimed_cat not in pool_categories:
                status = "FOREIGN"
                foreign += 1
            elif not isinstance(g, int) or not (0 <= g < total_frames):
                status = "UNMATCHED"
                unmatched += 1
            else:
                resolved_cat, _local = resolve_global(g)
                if resolved_cat != claimed_cat:
                    status = "UNMATCHED"
                    unmatched += 1
                else:
                    joined += 1
            per_parent.append({
                "structure_id": rec.get("structure_id"), "claimed_category": claimed_cat,
                "source_global_index": g, "resolved_category": resolved_cat,
                "join_status": status, "duplicate": is_dup})
        n = len(records)
        join_ok = (n > 0 and joined == n and unmatched == 0 and foreign == 0
                   and duplicates == 0 and sha_ok)
        gap = None
        if not join_ok:
            if not sha_ok:
                gap = (f"pool manifest SHA mismatch: manifest recorded {recorded_pool_sha!r} but "
                       f"pool declares {declared_pool_sha!r}")
            else:
                gap = (f"parent->pool join incomplete: joined={joined} unmatched={unmatched} "
                       f"foreign={foreign} duplicate={duplicates} of {n}")
        record("parent_pool_join",
               status="COMPLETE" if join_ok else "INCOMPLETE",
               pool_path=str(pool_path), pool_manifest_sha256_recorded=recorded_pool_sha,
               pool_manifest_sha256_declared=declared_pool_sha, pool_sha_matches=sha_ok,
               pool_total_frames=total_frames, n_selected=n,
               joined=joined, unmatched=unmatched, foreign=foreign, duplicate=duplicates,
               per_parent=per_parent, gap=gap)

    # --- (B) deployment-domain mapping -------------------------------------------------------
    scope_v2, binding = _load_scope_classification_evidence(controller, project_dir)
    if scope_v2 is None:
        record("deployment_domain_mapping", status="UNRESOLVABLE_SCOPE",
               **{k: v for k, v in binding.items()},
               gap=f"deployment-domain mapping unavailable: {binding.get('reason')}")
    else:
        from framework_v2.scientific_adequacy import ClaimRole
        label_to_domain = {m.raw_label: m.canonical_domain for m in scope_v2.label_map}
        per_parent: list[dict] = []
        domain_counts: dict[str, int] = {}
        unresolved = 0
        for rec in records:
            cat = _record_category(rec)
            role = scope_v2.role_of(cat) if cat is not None else ClaimRole.AMBIGUOUS
            domain = label_to_domain.get(cat)
            resolved = (domain is not None and role != ClaimRole.AMBIGUOUS)
            if resolved:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            else:
                unresolved += 1
            per_parent.append({
                "structure_id": rec.get("structure_id"), "raw_category": cat,
                "canonical_domain": domain if resolved else None,
                "claim_role": role.value,
                "resolution": "RESOLVED" if resolved else "AMBIGUOUS"})
        primary_domains = list(scope_v2.primary_domains)
        covered = [d for d in primary_domains if d in domain_counts]
        uncovered = [d for d in primary_domains if d not in domain_counts]
        mapping_ok = unresolved == 0 and bool(records)
        record("deployment_domain_mapping",
               status="RESOLVED" if mapping_ok else "AMBIGUOUS",
               scope_contract_sha256=binding.get("scope_contract_sha256"),
               scope_contract_id=binding.get("scope_contract_id"),
               classification_evidence_path=binding.get("classification_evidence_path"),
               per_parent=per_parent, aggregate_domain_counts=domain_counts,
               frozen_primary_domains=primary_domains,
               covered_primary_domains=covered, uncovered_primary_domains=uncovered,
               unresolved_parents=unresolved,
               gap=(None if mapping_ok else
                    f"{unresolved} selected parent(s) do not map deterministically to a canonical "
                    "deployment domain (fail-closed; not guessed)"))

    # --- (C) selection-control attestations --------------------------------------------------
    dft_used = manifest.get("dft_labels_used_as_selection_scores")
    teacher_inf = manifest.get("performs_teacher_inference")
    exclusion = manifest.get("protected_reference_exclusion_report")
    exclusion_status = exclusion.get("status") if isinstance(exclusion, dict) else None
    join_criterion = criteria.get("parent_pool_join", {})
    exclusive_pool_membership = (
        join_criterion.get("status") == "COMPLETE"
        and join_criterion.get("foreign") == 0
        and join_criterion.get("joined") == join_criterion.get("n_selected"))
    attest_missing = [
        name for name, value in (("dft_labels_used_as_selection_scores", dft_used),
                                 ("performs_teacher_inference", teacher_inf))
        if not isinstance(value, bool)]
    record("selection_control_attestations",
           status="ATTESTED" if not attest_missing else "INCOMPLETE",
           dft_labels_used_as_selection_scores=dft_used,
           performs_teacher_inference=teacher_inf,
           exclusive_sanitized_pool_membership=exclusive_pool_membership,
           protected_reference_exclusion_status=exclusion_status,
           gap=(None if not attest_missing else
                f"selection-control attestation value(s) not surfaced: {attest_missing}"))

    return {
        "applicable": True,
        "ready": not gaps,
        "blocking_gaps": gaps,
        "criteria": criteria,
        "pending_execution": False,
    }
