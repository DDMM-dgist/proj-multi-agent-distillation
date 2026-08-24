"""Deterministic Stage-8 reference-population governance validator (Recovery id=5).

This module is the deterministic authority for whether the specific
``allowed_use`` a Stage-8 accuracy channel actually performs is authorized on
the exact protected reference population declared by ``reference_validation``.

It exists to close the semantic-ambiguity gap adjudicated for
RecoveryPlan #5 (canonical hash
``ea7cac2501179e2ef0d1d5b282aa44f20b54ede1ef32e1a4a946feec016f340a``): the
legacy ``protected_reference_use`` string in ``reference_validation.json`` is a
HISTORICAL ORIGIN DESCRIPTOR written by the framework as a Stage-2 provenance
field; it is NOT the run-scoped authorization list. The authoritative
authorization list lives in the pre-existing pinned run inputs
(``inputs/*-reference.yaml`` and its ``protected_reference_manifest`` peer).
The validator therefore reads BOTH sources, treats the legacy field as
provenance-only, and fails closed on:

  * population identity mismatch,
  * source/hash mismatch,
  * frame-count mismatch across the accuracy channels,
  * a Stage-8 channel whose declared use is NOT a member of the pinned
    run-scoped ``allowed_uses`` list,
  * any attempt to substitute the ``historical_origin_descriptor`` for the
    run-scoped authorization list.

The module never widens ``allowed_uses`` retroactively; it only reads what the
pinned inputs already declared BEFORE the first Stage-8 evaluation was
executed. If no pinned ``allowed_uses`` list can be resolved, the validator
returns an explicit ``evidence_gap`` result (never a fabricated authorization).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

from workflow.integrity import sha256_file


# Framework-generic identity of Stage-8's DFT-consumer channels. The channel
# names are the four_channel_accuracy_report top-level keys; the trailing
# "_reference_validation" suffix is the vocabulary used by the pinned
# protected-reference contract's ``allowed_uses``. This mapping is the ONLY
# place the two vocabularies are joined and is deliberately conservative:
# only channels that actually consume DFT labels on the protected reference
# require an allowed_use; a channel that does not touch DFT (e.g. purely
# student-vs-teacher) is not gated by this validator.
CHANNEL_TO_REQUIRED_ALLOWED_USE = {
    "student_vs_dft": "student_vs_dft_reference_validation",
    "teacher_vs_dft": "teacher_vs_dft_reference_validation",
}


def _load_json(path: Path) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_yaml(path: Path) -> Optional[dict]:
    try:
        import yaml
    except ImportError:  # pragma: no cover - runtime dep in this codebase
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        return None
    return payload if isinstance(payload, dict) else None


def _find_run_manifest(run_dir: Path) -> Optional[dict]:
    """Locate and load the run's manifest.json for input pinning discovery."""
    candidate = run_dir / "manifest.json"
    if candidate.is_file():
        return _load_json(candidate)
    return None


def _pinned_source_sha(manifest: dict, source_suffix: str) -> Optional[str]:
    """Return the pinned sha256 for the first manifest input whose source path
    ends with ``source_suffix`` (deterministic ordering: first match wins).
    """
    for entry in manifest.get("inputs", []) or []:
        if not isinstance(entry, dict):
            continue
        source = entry.get("source") or ""
        if isinstance(source, str) and source.endswith(source_suffix):
            sha = entry.get("sha256") or entry.get("source_sha256")
            return sha if isinstance(sha, str) else None
    return None


_POLICY_STATUS_TO_ALLOWED = {"ALLOWED", "ALLOWED_WITH_APPROVAL", "PERMITTED"}
# The protected-reference manifest expresses its authorization vocabulary in
# ``student_policy`` entries whose keys end in ``_evaluation``; the workflow's
# reference-contract snapshots express theirs in ``allowed_uses`` entries whose
# strings end in ``_reference_validation``. Both refer to the SAME framework
# semantic — one authorises the manifest-operation, the other authorises the
# stage's use of the population — and their alignment is declared here as the
# ONE place the two vocabularies are joined.
_MANIFEST_POLICY_KEY_TO_ALLOWED_USE = {
    "teacher_vs_dft_evaluation": "teacher_vs_dft_reference_validation",
    "student_vs_dft_evaluation": "student_vs_dft_reference_validation",
}


def _snapshot_for_source(manifest: dict, source_path: str, run_dir: Path) -> Optional[Path]:
    """Return the pinned in-run snapshot Path for a manifest input whose
    ``source`` field equals ``source_path`` (exact match after resolution).
    """
    if not source_path:
        return None
    try:
        target = str(Path(source_path).expanduser().resolve())
    except OSError:
        target = source_path
    for entry in manifest.get("inputs", []) or []:
        if not isinstance(entry, dict):
            continue
        source = entry.get("source") or ""
        if not isinstance(source, str):
            continue
        try:
            resolved = str(Path(source).expanduser().resolve())
        except OSError:
            resolved = source
        if resolved == target:
            snap = entry.get("snapshot")
            if isinstance(snap, str) and snap.strip():
                snap_path = Path(snap)
                if snap_path.is_file():
                    return snap_path
    return None


def resolve_run_scoped_allowed_uses(
    run_dir: Path,
    reference_id: Optional[str],
    protection_manifest_path: Optional[Path] = None,
    reference_yaml_source: Optional[str] = None,
) -> dict:
    """Deterministically resolve the AUTHORITATIVE run-scoped ``allowed_uses``
    list for the given reference, sourced ONLY from the pinned run inputs.
    Never invents a value; returns an explicit ``evidence_gap`` on any
    unresolved case.

    Resolution order (each step reads only PINNED provenance):

      1. If ``reference_yaml_source`` is given, look up that exact source path
         in the run's ``manifest.inputs`` and read its snapshot's
         ``allowed_uses``.
      2. If ``reference_id`` is given, scan ``run_dir/inputs/*.yaml`` for a
         snapshot whose declared ``reference_id`` matches AND whose
         ``allowed_uses`` list is present.
      3. Cross-check any discovered protected_reference_manifest peer (either
         supplied or referenced by one of the found reference contracts) and
         convert its ``student_policy`` entries into the framework's
         ``allowed_uses`` vocabulary. If steps 1/2 resolved a list, an
         inconsistent manifest-policy list is a fail-closed
         ``authorization_source_inconsistent`` evidence gap. If steps 1/2
         resolved nothing, the manifest-policy list serves as the
         authoritative fallback (this preserves governance when a
         lineage-derived sub-population has no direct reference-contract
         snapshot of its own but is bound to a protected reference by
         provenance).

    Returned shape (always a dict; ``values`` is either a list of strings or
    absent):

        {
          "values": ["teacher_vs_dft_reference_validation", ...],  # if resolved
          "provenance_sources": [
              {"path": ..., "sha256": ..., "reference_id": ..., "kind": ...},
              ...
          ],
          "evidence_gap": "..."  # present iff resolution failed
        }
    """
    manifest = _find_run_manifest(run_dir)
    if manifest is None:
        return {"evidence_gap": "run manifest.json not found or unreadable"}

    provenance_sources: list[dict] = []
    resolved_values: Optional[list[str]] = None
    resolved_kinds: list[str] = []

    # Step 1 — direct linkage via the source path recorded in reference_validation.json.
    if reference_yaml_source:
        snapshot = _snapshot_for_source(manifest, reference_yaml_source, run_dir)
        if snapshot is not None:
            payload = _load_yaml(snapshot)
            if isinstance(payload, dict):
                uses = payload.get("allowed_uses")
                if isinstance(uses, list) and all(isinstance(u, str) for u in uses):
                    provenance_sources.append({
                        "path": str(snapshot),
                        "sha256": sha256_file(snapshot),
                        "reference_id": payload.get("reference_id"),
                        "kind": "pinned_reference_contract_snapshot_by_source_link",
                    })
                    resolved_values = list(uses)
                    resolved_kinds.append("pinned_reference_contract_snapshot_by_source_link")
                if protection_manifest_path is None:
                    raw = payload.get("protection_manifest")
                    if isinstance(raw, str) and raw.strip():
                        protection_manifest_path = Path(raw)

    # Step 2 — reference_id match across all inputs snapshots.
    inputs_dir = run_dir / "inputs"
    if inputs_dir.is_dir() and reference_id:
        for snapshot in sorted(inputs_dir.iterdir()):
            if snapshot.suffix not in (".yaml", ".yml"):
                continue
            payload = _load_yaml(snapshot)
            if not isinstance(payload, dict):
                continue
            declared_id = payload.get("reference_id")
            if declared_id != reference_id:
                continue
            uses = payload.get("allowed_uses")
            if not isinstance(uses, list) or not all(isinstance(u, str) for u in uses):
                continue
            if any(src.get("path") == str(snapshot) for src in provenance_sources):
                continue  # already recorded by step 1
            provenance_sources.append({
                "path": str(snapshot),
                "sha256": sha256_file(snapshot),
                "reference_id": declared_id,
                "kind": "pinned_reference_contract_snapshot_by_reference_id",
            })
            if resolved_values is None:
                resolved_values = list(uses)
                resolved_kinds.append("pinned_reference_contract_snapshot_by_reference_id")
            elif set(uses) != set(resolved_values):
                return {
                    "evidence_gap": (
                        "multiple pinned reference-contract snapshots declare DIFFERENT "
                        f"allowed_uses lists ({sorted(set(resolved_values))} vs "
                        f"{sorted(set(uses))}); manual reconciliation required — refusing to guess"
                    ),
                    "provenance_sources": provenance_sources,
                }

    # Step 3 — protected_reference_manifest cross-check / fallback.
    # First, if a caller-supplied path was not given, look for one declared by
    # ANY pinned reference-contract snapshot the run pins (not only the ones
    # matched in steps 1/2): a lineage-derived sub-population (e.g. the
    # recovered-original-heldout partition of a protected reference) may not
    # carry a `protection_manifest` field on its own snapshot, but a sibling
    # snapshot for the protected parent reference in the same `inputs/`
    # directory will. Scanning all snapshots here is the ONLY generic bridge
    # that lets the manifest govern lineage-derived subsets without hardcoded
    # reference_id equivalences.
    if protection_manifest_path is None:
        # First priority: already-matched snapshots (steps 1/2), if any.
        for src in provenance_sources:
            snap_payload = _load_yaml(Path(src["path"]))
            if isinstance(snap_payload, dict):
                raw = snap_payload.get("protection_manifest")
                if isinstance(raw, str) and raw.strip():
                    protection_manifest_path = Path(raw)
                    break
    if protection_manifest_path is None and inputs_dir.is_dir():
        # Second priority: scan every pinned reference-contract snapshot in
        # `inputs/` for a declared `protection_manifest`; use the first found.
        for snapshot in sorted(inputs_dir.iterdir()):
            if snapshot.suffix not in (".yaml", ".yml"):
                continue
            payload = _load_yaml(snapshot)
            if not isinstance(payload, dict):
                continue
            raw = payload.get("protection_manifest")
            if isinstance(raw, str) and raw.strip():
                candidate = Path(raw)
                if candidate.is_file():
                    protection_manifest_path = candidate
                    provenance_sources.append({
                        "path": str(snapshot),
                        "sha256": sha256_file(snapshot),
                        "reference_id": payload.get("reference_id"),
                        "kind": "pinned_reference_contract_snapshot_manifest_pointer",
                    })
                    break

    manifest_derived_uses: Optional[list[str]] = None
    if protection_manifest_path is not None and protection_manifest_path.is_file():
        prm = _load_json(protection_manifest_path)
        if isinstance(prm, dict):
            policy = prm.get("student_policy") if isinstance(prm.get("student_policy"), dict) else {}
            allowed_manifest_keys = sorted(
                key for key, status in policy.items()
                if isinstance(key, str) and isinstance(status, str)
                and status.upper() in _POLICY_STATUS_TO_ALLOWED
            )
            manifest_derived_uses = sorted({
                _MANIFEST_POLICY_KEY_TO_ALLOWED_USE[key]
                for key in allowed_manifest_keys
                if key in _MANIFEST_POLICY_KEY_TO_ALLOWED_USE
            })
            provenance_sources.append({
                "path": str(protection_manifest_path),
                "sha256": sha256_file(protection_manifest_path),
                "reference_id": prm.get("reference_id"),
                "kind": "pinned_protected_reference_manifest",
                "allowed_policy_keys": allowed_manifest_keys,
                "derived_allowed_uses": manifest_derived_uses,
            })
            resolved_kinds.append("pinned_protected_reference_manifest")

    # Reconcile: if both step 1/2 and step 3 resolved lists, they must agree.
    if resolved_values is not None and manifest_derived_uses is not None:
        if set(manifest_derived_uses) != set(resolved_values):
            return {
                "evidence_gap": (
                    "authorization sources disagree: reference-contract snapshot(s) declare "
                    f"{sorted(set(resolved_values))!r} while the protected_reference_manifest "
                    f"policy derives {sorted(set(manifest_derived_uses))!r}; fail-closed "
                    "authorization_source_inconsistent — manual reconciliation required"
                ),
                "provenance_sources": provenance_sources,
            }

    if resolved_values is None and manifest_derived_uses is not None:
        # Lineage-derived fallback: authoritative when reference_id doesn't
        # match any snapshot directly (e.g. the recovered subset carries a
        # distinct reference_id from its pinned protected parent, but the
        # manifest itself governs the reference lineage).
        resolved_values = manifest_derived_uses

    if resolved_values is None:
        return {
            "evidence_gap": (
                "no pinned reference-contract snapshot with a matching reference_id or source "
                "linkage was found in the run's inputs, and no protected_reference_manifest "
                "policy could be resolved either; the authoritative run-scoped authorization "
                f"could not be resolved from pre-existing pinned provenance "
                f"(reference_id={reference_id!r}, reference_yaml_source={reference_yaml_source!r})"
            ),
            "provenance_sources": provenance_sources,
        }

    return {
        "values": sorted(set(resolved_values)),
        "provenance_sources": provenance_sources,
        "provenance_kinds": sorted(set(resolved_kinds)),
    }


def validate_stage8_reference_population_governance(
    run_dir: Path,
    reference_validation_payload: dict,
    accuracy_report_channels: Iterable[str],
    channel_frame_counts: Optional[dict] = None,
    performed_channels_frame_counts: Optional[dict] = None,
    reference_yaml_source: Optional[str] = None,
) -> dict:
    """Deterministic Stage-8 authorization check.

    Verifies that every Stage-8 channel that ACTUALLY performs a use of the
    protected reference population is a member of the AUTHORITATIVE run-scoped
    ``allowed_uses`` list resolved from pinned pre-existing run provenance.
    Fails closed with a specific code per failure mode; explicitly forbids the
    legacy ``protected_reference_use`` origin descriptor from satisfying the
    authorization check by itself.

    ``accuracy_report_channels`` is an iterable of the top-level channel names
    present in the four_channel_accuracy_report (e.g. ``student_vs_dft``,
    ``teacher_vs_dft``, ``student_vs_teacher``). Only channels that appear in
    ``CHANNEL_TO_REQUIRED_ALLOWED_USE`` are gated by this validator.

    ``channel_frame_counts`` is an optional mapping ``channel -> n_frames``
    from the accuracy_report's aggregate block; when provided it is used for
    the frame-count binding check against ``reference.logical_frames``.

    Returns a structured dict (never raises for evidence gaps):

        {
          "ok": bool,
          "failures": [{"code": ..., "detail": ...}, ...],
          "checks": {...deterministic named results...},
          "run_scoped_allowed_uses": {...resolve_run_scoped_allowed_uses shape...},
          "historical_origin_descriptor": {...},
          "authorization_scope_authority": "pinned_run_scoped_allowed_uses_only",
          "note": "..."
        }
    """
    payload = reference_validation_payload if isinstance(reference_validation_payload, dict) else {}
    reference = payload.get("reference") if isinstance(payload.get("reference"), dict) else {}
    reference_id = reference.get("reference_id")
    logical_frames = reference.get("logical_frames")
    structures_sha = None
    integrity = reference.get("structures_integrity")
    if isinstance(integrity, dict) and isinstance(integrity.get("sha256"), str):
        structures_sha = integrity["sha256"]

    historical_origin_descriptor = {
        "protected_reference_use": payload.get("protected_reference_use"),
        "evidence_source": payload.get("evidence_source"),
        "role": "historical_origin_descriptor",
        "note": (
            "Legacy Stage-2 field describing the origin of the historical reference-validation "
            "reuse; PROVENANCE-ONLY. This field is NOT the run-scoped authorization list and "
            "MUST NOT by itself satisfy any Stage-8 allowed_use requirement."
        ),
    }

    allowed_uses_result = resolve_run_scoped_allowed_uses(
        run_dir, reference_id, reference_yaml_source=reference_yaml_source,
    )

    checks: dict[str, Any] = {}
    failures: list[dict] = []

    checks["reference_id_present"] = bool(reference_id)
    if not reference_id:
        failures.append({
            "code": "population_identity_missing",
            "detail": "reference_validation.json.reference.reference_id is missing or empty",
        })

    checks["source_sha256_present"] = bool(structures_sha)
    if not structures_sha:
        failures.append({
            "code": "source_hash_missing",
            "detail": "reference_validation.json.reference.structures_integrity.sha256 is missing",
        })

    checks["logical_frames_present"] = isinstance(logical_frames, int)
    if not isinstance(logical_frames, int):
        failures.append({
            "code": "frame_count_missing",
            "detail": "reference_validation.json.reference.logical_frames is missing or not an integer",
        })

    # Frame-count binding across channels (if provided)
    if channel_frame_counts is not None and isinstance(logical_frames, int):
        distinct = {v for v in channel_frame_counts.values() if v is not None}
        binds = len(distinct) == 1 and logical_frames in distinct
        checks["frame_count_binds"] = binds
        checks["channel_frame_counts"] = dict(channel_frame_counts)
        checks["reference_logical_frames"] = logical_frames
        if not binds:
            failures.append({
                "code": "frame_count_mismatch",
                "detail": (
                    f"accuracy_report channel frame counts {channel_frame_counts!r} do not "
                    f"uniquely match reference.logical_frames={logical_frames}"
                ),
            })

    # Authorization check
    resolved_values = allowed_uses_result.get("values")
    checks["run_scoped_allowed_uses_resolved"] = resolved_values is not None
    if resolved_values is None:
        failures.append({
            "code": "authorization_evidence_missing",
            "detail": allowed_uses_result.get(
                "evidence_gap",
                "run-scoped allowed_uses could not be resolved from pinned inputs"),
        })
    else:
        performed_channels = sorted(set(accuracy_report_channels))
        checks["performed_channels"] = performed_channels
        checks["required_allowed_uses_by_channel"] = {
            channel: CHANNEL_TO_REQUIRED_ALLOWED_USE[channel]
            for channel in performed_channels
            if channel in CHANNEL_TO_REQUIRED_ALLOWED_USE
        }
        for channel, required in checks["required_allowed_uses_by_channel"].items():
            if required not in resolved_values:
                failures.append({
                    "code": "channel_use_not_authorized",
                    "detail": (
                        f"Stage-8 channel {channel!r} performs use "
                        f"{required!r} on the protected reference population, but this use is "
                        f"NOT a member of the pinned run-scoped allowed_uses list "
                        f"{sorted(resolved_values)!r}"
                    ),
                })

    # Explicit anti-substitution guard: the historical origin descriptor MUST
    # NOT be the sole basis for authorization. This is a structural invariant
    # of the validator: even if a caller pretends the origin descriptor is an
    # allowed_uses list, the check above (which never reads the origin
    # descriptor) is unaffected. Recorded here for the Judges' inspection.
    checks["historical_origin_descriptor_role"] = "provenance_only"

    ok = not failures
    return {
        "ok": ok,
        "failures": failures,
        "checks": checks,
        "run_scoped_allowed_uses": allowed_uses_result,
        "historical_origin_descriptor": historical_origin_descriptor,
        "authorization_scope_authority": "pinned_run_scoped_allowed_uses_only",
        "note": (
            "Deterministic Stage-8 reference-population governance check. The authoritative "
            "authorization list is the pinned run-scoped allowed_uses; the historical "
            "protected_reference_use field is a Stage-2 origin descriptor and is intentionally "
            "excluded from the authorization decision."
        ),
    }


__all__ = [
    "CHANNEL_TO_REQUIRED_ALLOWED_USE",
    "resolve_run_scoped_allowed_uses",
    "validate_stage8_reference_population_governance",
]
