"""Deterministic protection checks for held-out reference structures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from ase.io import read

from workflow.integrity import sha256_file


EXPECTED_REFERENCE_CLASS = "ORIGINAL_TEACHER_TEST"


def _structure_fingerprint(atoms) -> str:
    """Canonical geometry fingerprint for protected-reference leakage checks.

    Coordinates and cell vectors are rounded to 8 decimal places so harmless
    extxyz serialization differences do not bypass the protection check.
    """
    payload = {
        "numbers": np.asarray(
            atoms.numbers, dtype=np.int32
        ).tolist(),
        "positions": np.round(
            np.asarray(atoms.positions, dtype=np.float64), 8
        ).tolist(),
        "cell": np.round(
            np.asarray(atoms.cell.array, dtype=np.float64), 8
        ).tolist(),
        "pbc": np.asarray(
            atoms.pbc, dtype=bool
        ).tolist(),
    }

    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def load_protected_indices(path):
    path = Path(path).resolve()

    values = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()

        if not raw:
            continue

        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"invalid protected source index at {path}:{lineno}: {raw!r}"
            ) from exc

        values.append(value)

    if len(values) != len(set(values)):
        raise ValueError("protected source-index file contains duplicates")

    return set(values)


def validate_reference_config(reference_yaml):
    """Validate a Teacher-vs-DFT reference.yaml, dispatched by its declared ``kind``.

    Generic entry point: every caller (``_exec_validate_teacher_reference``,
    ``validate_reference_validation_report``) calls this one function regardless of which
    reference population backs it. Each ``kind`` has its own validator function, registered in
    ``_REFERENCE_KIND_VALIDATORS`` below; a new reference population class (e.g. a recovered
    original held-out test partition, as opposed to a physically-recovered historical artifact)
    registers a new validator here rather than growing new branches inside one function.
    """
    reference_yaml = Path(reference_yaml).resolve()
    cfg = yaml.safe_load(reference_yaml.read_text(encoding="utf-8")) or {}
    kind = cfg.get("kind")
    validator = _REFERENCE_KIND_VALIDATORS.get(kind)
    if validator is None:
        raise ValueError(
            f"reference.kind {kind!r} is not a recognized Teacher-vs-DFT reference kind "
            f"(known kinds: {sorted(_REFERENCE_KIND_VALIDATORS)})"
        )
    return validator(reference_yaml, cfg)


def _validate_protected_existing_dft_reference(reference_yaml, cfg):
    """Validate the frozen R2 reference.yaml and protected-reference package
    (``kind: protected-existing-dft``) -- a physically-recovered historical artifact whose own
    original-selection provenance may be unresolved (see
    local_inputs/sio2_fresh/protected_reference/protected_reference_manifest.json's
    ``reference_class``/``evaluation_role`` for its current, honest scientific-role
    description)."""

    if cfg.get("reference_class") != EXPECTED_REFERENCE_CLASS:
        raise ValueError(
            f"reference_class must be {EXPECTED_REFERENCE_CLASS!r}"
        )

    if cfg.get("status") != "AVAILABLE_AND_PROTECTED":
        raise ValueError("protected reference is not marked AVAILABLE_AND_PROTECTED")

    structures = cfg.get("structures")
    if not isinstance(structures, dict):
        raise ValueError("reference.structures must be a mapping")

    ref_path = Path(structures["path"]).resolve()

    if not ref_path.is_file():
        raise FileNotFoundError(ref_path)

    expected_sha = structures.get("sha256")
    observed_sha = sha256_file(ref_path)

    if expected_sha != observed_sha:
        raise RuntimeError(
            "protected DFT reference SHA-256 mismatch: "
            f"{observed_sha} != {expected_sha}"
        )

    frames = read(str(ref_path), index=":")

    expected_logical = int(cfg["logical_test_frames"])

    if len(frames) != expected_logical:
        raise RuntimeError(
            f"protected logical-frame mismatch: {len(frames)} != {expected_logical}"
        )

    if int(structures.get("logical_frames", -1)) != expected_logical:
        raise ValueError(
            "structures.logical_frames does not match logical_test_frames"
        )

    manifest_path = Path(cfg["protection_manifest"]).resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text())

    mapping = manifest.get("mapping", {})

    checks = {
        "logical_test_frames":
            expected_logical,

        "matched_logical_frames":
            expected_logical,

        "unmatched_logical_frames":
            0,

        "protected_source_rows":
            int(cfg["protected_source_rows"]),

        "conflicting_label_duplicates":
            0,
    }

    for field, expected in checks.items():
        observed = mapping.get(field)

        if observed != expected:
            raise ValueError(
                f"protected-reference manifest mismatch for {field}: "
                f"{observed!r} != {expected!r}"
            )

    indices_path = Path(cfg["protected_source_rows_file"]).resolve()

    protected_indices = load_protected_indices(indices_path)

    if len(protected_indices) != int(cfg["protected_source_rows"]):
        raise ValueError(
            "protected source-index count does not match reference.yaml"
        )

    duplicate = cfg.get("duplicate_equivalent", {})

    duplicate_rows = set(
        int(x) for x in duplicate.get("source_global_indices", [])
    )

    if duplicate_rows != {760, 761}:
        raise ValueError(
            "expected duplicate-equivalent protected rows {760, 761}"
        )

    if not duplicate_rows.issubset(protected_indices):
        raise ValueError(
            "duplicate-equivalent source rows are not both protected"
        )

    if duplicate.get("label_conflict") is not False:
        raise ValueError(
            "duplicate-equivalent held-out structure must have label_conflict=false"
        )

    prohibited = set(cfg.get("prohibited_uses", []))

    required_prohibitions = {
        "student_training",
        "student_validation_tuning",
        "acquisition_seed",
        "augmentation_parent",
        "recovery_training",
    }

    missing = required_prohibitions - prohibited

    if missing:
        raise ValueError(
            "protected reference is missing prohibited uses: "
            + ", ".join(sorted(missing))
        )

    fingerprints = [_structure_fingerprint(a) for a in frames]

    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError(
            "protected logical reference itself contains duplicate geometries"
        )

    return {
        "reference_id": cfg["reference_id"],
        "logical_frames": len(frames),
        "protected_source_rows": len(protected_indices),
        "protected_source_indices": protected_indices,
        "reference_fingerprints": set(fingerprints),
        "reference_path": ref_path,
    }


RECOVERED_HOLDOUT_REFERENCE_CLASS = "RECOVERED_ORIGINAL_HELDOUT_TEST_PARTITION"

RECOVERED_HOLDOUT_REQUIRED_PROHIBITIONS = {
    "student_training",
    "student_validation_tuning",
    "acquisition_seed",
    "augmentation_parent",
    "recovery_training",
}


def _validate_recovered_holdout_reference_config(reference_yaml, cfg):
    """Validate a ``kind: recovered-original-holdout`` reference.yaml: an algorithmically
    RECONSTRUCTED partition of a Teacher's own original training pool (e.g. the genuine
    train/validation/test split membership reproduced from a recovered seed/fractions/order --
    see configs/provenance/teacher_training_split_manifest.json for the reference demonstration
    of this class), as opposed to ``protected-existing-dft``'s physically-recovered historical
    artifact.

    Never hardcodes a frame count, split name, or campaign identity: ``target_split``,
    ``frame_count``, and the split-source manifest are all read from ``cfg`` and cross-checked
    against each other and against the structures file itself. Every frame in the structures
    file must (a) join the declared split-source manifest via ``(source_category,
    source_local_index)`` to EXACTLY ``target_split`` -- ambiguous, unjoined, or
    wrong-partition frames fail closed rather than being silently included -- and (b) already
    carry finite ``dft_energy``/``dft_forces`` labels, since this reference class must never
    require a fresh DFT calculation.
    """
    from runtimes.pydantic_ai.bounded_evidence import (
        _is_split_membership_manifest, build_split_crosswalk,
    )

    if cfg.get("reference_class") != RECOVERED_HOLDOUT_REFERENCE_CLASS:
        raise ValueError(f"reference_class must be {RECOVERED_HOLDOUT_REFERENCE_CLASS!r}")

    if cfg.get("status") != "AVAILABLE_AND_VERIFIED":
        raise ValueError("recovered-holdout reference is not marked AVAILABLE_AND_VERIFIED")

    target_split = cfg.get("target_split")
    if not isinstance(target_split, str) or not target_split.strip():
        raise ValueError("recovered-holdout reference must declare a non-empty target_split")

    structures = cfg.get("structures")
    if not isinstance(structures, dict):
        raise ValueError("reference.structures must be a mapping")
    ref_path = Path(structures["path"]).resolve()
    if not ref_path.is_file():
        raise FileNotFoundError(ref_path)
    expected_sha = structures.get("sha256")
    observed_sha = sha256_file(ref_path)
    if expected_sha != observed_sha:
        raise RuntimeError(
            f"recovered-holdout structures SHA-256 mismatch: {observed_sha} != {expected_sha}"
        )

    manifest_path = Path(cfg["split_source_manifest"]).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    observed_manifest_sha = sha256_file(manifest_path)
    if observed_manifest_sha != cfg.get("split_source_manifest_sha256"):
        raise RuntimeError(
            "split_source_manifest SHA-256 mismatch: "
            f"{observed_manifest_sha} != {cfg.get('split_source_manifest_sha256')}"
        )
    manifest_payload = json.loads(manifest_path.read_text())
    if not _is_split_membership_manifest(manifest_payload):
        raise ValueError("split_source_manifest does not match the split-membership shape")

    frames = read(str(ref_path), index=":")
    expected_count = int(cfg["frame_count"])
    if len(frames) != expected_count:
        raise RuntimeError(f"recovered-holdout frame-count mismatch: {len(frames)} != {expected_count}")
    if int(structures.get("logical_frames", -1)) != expected_count:
        raise ValueError("structures.logical_frames does not match frame_count")

    crosswalk = build_split_crosswalk([manifest_path])
    resolved = crosswalk["resolved"]
    ambiguous = crosswalk["ambiguous"]
    resolved_keys = set()
    for index, atoms in enumerate(frames):
        cat = atoms.info.get("source_category")
        local_index = atoms.info.get("source_local_index")
        if cat is None or local_index is None:
            raise ValueError(f"frame {index} is missing source_category/source_local_index")
        key = (str(cat), int(local_index))
        if key in ambiguous:
            raise ValueError(f"frame {index} ({key!r}) is ambiguous in the split-source manifest")
        if key not in resolved:
            raise ValueError(f"frame {index} ({key!r}) does not join the split-source manifest")
        if resolved[key] != target_split:
            raise ValueError(
                f"frame {index} ({key!r}) belongs to split {resolved[key]!r}, not "
                f"target_split {target_split!r} -- this reference must contain ONLY "
                "target_split members"
            )
        if key in resolved_keys:
            raise ValueError(f"duplicate source key {key!r} in recovered-holdout structures")
        resolved_keys.add(key)
        for label_key in ("dft_energy",):
            value = atoms.info.get(label_key)
            if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
                raise ValueError(f"frame {index} is missing a finite {label_key}")
        forces = atoms.arrays.get("dft_forces")
        if forces is None or not np.all(np.isfinite(np.asarray(forces, dtype=float))):
            raise ValueError(f"frame {index} is missing finite dft_forces")

    prohibited = set(cfg.get("prohibited_uses", []))
    missing = RECOVERED_HOLDOUT_REQUIRED_PROHIBITIONS - prohibited
    if missing:
        raise ValueError(
            "recovered-holdout reference is missing prohibited uses: " + ", ".join(sorted(missing))
        )

    fingerprints = [_structure_fingerprint(a) for a in frames]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("recovered-holdout reference contains duplicate geometries")

    return {
        "reference_id": cfg["reference_id"],
        "logical_frames": len(frames),
        "protected_source_rows": len(resolved_keys),
        "protected_source_indices": resolved_keys,
        "reference_fingerprints": set(fingerprints),
        "reference_path": ref_path,
    }


# --------------------------------------------------------------------------------------
# PROTECTION REFERENCE vs EVALUATION REFERENCE (framework-level separation)
# --------------------------------------------------------------------------------------
# Two distinct concepts share the reference.yaml surface but need different information and
# obey different access-control invariants:
#
#   PROTECTION REFERENCE  -- used by acquisition / teacher_labeling / dataset_split /
#     training (early stages 1-7) purely to verify that a Student-side population does NOT
#     overlap a protected population. It needs ONLY structure identity: source-pool indices
#     and/or geometry (species/positions/cell/pbc) fingerprints. It MUST NOT carry, and this
#     module MUST NOT read, DFT/Teacher energy/force/stress truth -- exposing those at an
#     early stage would leak protected labels into the blind Student route.
#
#   EVALUATION REFERENCE  -- used ONLY at Stage 8 (evaluation) / Stage 9 (uncertainty) to
#     compute actual DFT fidelity and uncertainty calibration. It legitimately carries and
#     reads finite DFT/Teacher labels, under the frozen access policy that restricts label
#     access to those late stages.
#
# `protected-existing-dft` and `recovered-original-holdout` are EVALUATION references (they
# materialize a DFT-labeled structures file). `protected-structure-identity` is a
# PROTECTION-ONLY reference (identity + geometry, no labels). The classification is made
# first-class below so callers can require the correct capability rather than assuming every
# reference kind carries labels.
PROTECTION_ONLY_STRUCTURE_IDENTITY_REFERENCE_CLASS = "PROTECTION_ONLY_STRUCTURE_IDENTITY"

# Info/array keys that would constitute DFT/Teacher *truth* (as opposed to structure
# identity). A protection-only reference carrying any of these is rejected fail-closed so a
# geometry-only protection manifest can never indirectly expose label truth to early stages.
_FORBIDDEN_LABEL_INFO_KEYS = frozenset({
    "dft_energy", "dft_free_energy", "dft_stress", "dft_virial",
    "teacher_energy", "teacher_free_energy", "teacher_stress", "teacher_virial",
    "energy", "free_energy", "stress", "virial",
})
_FORBIDDEN_LABEL_ARRAY_KEYS = frozenset({
    "dft_forces", "teacher_forces", "forces", "stress", "stresses",
})
_FORBIDDEN_CALC_RESULT_KEYS = frozenset({
    "energy", "free_energy", "forces", "stress",
})


def _assert_frame_carries_no_label_truth(atoms, index):
    """Fail closed if a protection-only reference frame carries DFT/Teacher label truth.

    Only key *presence* is inspected; label values are never read. This enforces the
    access-control invariant that a protection manifest exposes structure identity only.
    """
    present_info = _FORBIDDEN_LABEL_INFO_KEYS & set(atoms.info)
    if present_info:
        raise ValueError(
            f"protection-only reference frame {index} carries forbidden label field(s) "
            f"{sorted(present_info)} -- a protection reference must expose structure identity "
            "only, never DFT/Teacher energy/force/stress truth"
        )
    present_arrays = _FORBIDDEN_LABEL_ARRAY_KEYS & set(atoms.arrays)
    if present_arrays:
        raise ValueError(
            f"protection-only reference frame {index} carries forbidden label array(s) "
            f"{sorted(present_arrays)} -- a protection reference must expose structure identity "
            "only, never DFT/Teacher energy/force/stress truth"
        )
    calc = getattr(atoms, "calc", None)
    if calc is not None:
        results = getattr(calc, "results", None) or {}
        present_calc = _FORBIDDEN_CALC_RESULT_KEYS & set(results)
        if present_calc:
            raise ValueError(
                f"protection-only reference frame {index} carries attached calculator results "
                f"{sorted(present_calc)} -- a protection reference must expose structure identity "
                "only, never DFT/Teacher energy/force/stress truth"
            )


def _validate_protection_only_structure_identity_reference(reference_yaml, cfg):
    """Validate a ``kind: protected-structure-identity`` reference.yaml: a PROTECTION-ONLY
    reference that expresses a protected population by STRUCTURE IDENTITY (source-pool indices
    + geometry fingerprints) and carries NO DFT/Teacher labels.

    This is the generic, material-agnostic representation the early-stage disjointness checks
    (acquisition / teacher_labeling / dataset_split / training) actually need. It never requires
    a DFT label to exist, never reads energy/force/stress, and is explicitly separated from the
    DFT-labeled EVALUATION references (``protected-existing-dft`` / ``recovered-original-holdout``)
    consumed only at Stages 8/9.

    Nothing about a particular material, campaign, frame count, or split name is hardcoded:
    ``protected_source_indices_file`` (newline-separated integer source-pool rows) and a
    geometry-only ``structures`` file are both read from ``cfg`` and cross-checked against their
    declared sha256 + counts, and every frame is asserted to carry NO label truth.
    """
    if cfg.get("reference_class") != PROTECTION_ONLY_STRUCTURE_IDENTITY_REFERENCE_CLASS:
        raise ValueError(
            f"reference_class must be {PROTECTION_ONLY_STRUCTURE_IDENTITY_REFERENCE_CLASS!r}"
        )
    if cfg.get("status") != "IDENTITY_AVAILABLE_AND_PROTECTED":
        raise ValueError(
            "protection-only reference is not marked IDENTITY_AVAILABLE_AND_PROTECTED"
        )
    if not cfg.get("reference_id"):
        raise ValueError("protection-only reference must declare a non-empty reference_id")

    indices_path = Path(cfg["protected_source_indices_file"]).resolve()
    if not indices_path.is_file():
        raise FileNotFoundError(indices_path)
    observed_indices_sha = sha256_file(indices_path)
    if observed_indices_sha != cfg.get("protected_source_indices_sha256"):
        raise RuntimeError(
            "protected_source_indices_file SHA-256 mismatch: "
            f"{observed_indices_sha} != {cfg.get('protected_source_indices_sha256')}"
        )
    protected_indices = load_protected_indices(indices_path)
    if len(protected_indices) != int(cfg["protected_source_rows"]):
        raise ValueError(
            "protected source-index count does not match protected_source_rows"
        )

    structures = cfg.get("structures")
    if not isinstance(structures, dict):
        raise ValueError("reference.structures must be a mapping")
    ref_path = Path(structures["path"]).resolve()
    if not ref_path.is_file():
        raise FileNotFoundError(ref_path)
    observed_sha = sha256_file(ref_path)
    if structures.get("sha256") != observed_sha:
        raise RuntimeError(
            f"protection-only structures SHA-256 mismatch: {observed_sha} != {structures.get('sha256')}"
        )

    frames = read(str(ref_path), index=":")
    expected_count = int(structures.get("logical_frames", -1))
    if len(frames) != expected_count:
        raise ValueError("structures.logical_frames does not match the structures file")

    for index, atoms in enumerate(frames):
        _assert_frame_carries_no_label_truth(atoms, index)

    prohibited = set(cfg.get("prohibited_uses", []))
    missing = RECOVERED_HOLDOUT_REQUIRED_PROHIBITIONS - prohibited
    if missing:
        raise ValueError(
            "protection-only reference is missing prohibited uses: " + ", ".join(sorted(missing))
        )

    fingerprints = [_structure_fingerprint(a) for a in frames]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError(
            "protection-only reference itself contains duplicate geometries"
        )

    return {
        "reference_id": cfg["reference_id"],
        "logical_frames": len(frames),
        "protected_source_rows": len(protected_indices),
        "protected_source_indices": protected_indices,
        "reference_fingerprints": set(fingerprints),
        "reference_path": ref_path,
    }


_REFERENCE_KIND_VALIDATORS = {
    "protected-existing-dft": _validate_protected_existing_dft_reference,
    "recovered-original-holdout": _validate_recovered_holdout_reference_config,
    "protected-structure-identity": _validate_protection_only_structure_identity_reference,
}

# Capability classification: which reference kinds carry DFT/Teacher label truth (EVALUATION
# references, Stages 8/9 only) versus identity-only PROTECTION references (safe for Stages 1-7).
_PROTECTION_ONLY_REFERENCE_KINDS = frozenset({"protected-structure-identity"})
_EVALUATION_REFERENCE_KINDS = frozenset({"protected-existing-dft", "recovered-original-holdout"})


def reference_kind_provides_dft_labels(kind):
    """True iff ``kind`` is a DFT-labeled EVALUATION reference (usable at Stages 8/9)."""
    if kind in _EVALUATION_REFERENCE_KINDS:
        return True
    if kind in _PROTECTION_ONLY_REFERENCE_KINDS:
        return False
    raise ValueError(
        f"reference.kind {kind!r} is not a recognized reference kind "
        f"(known kinds: {sorted(_REFERENCE_KIND_VALIDATORS)})"
    )


def reference_kind_of(reference_yaml):
    reference_yaml = Path(reference_yaml).resolve()
    cfg = yaml.safe_load(reference_yaml.read_text(encoding="utf-8")) or {}
    return cfg.get("kind")


def assert_reference_is_protection_capable(reference_yaml):
    """Every recognized kind can back an early-stage disjointness (protection) check, because
    all validators return protected_source_indices + reference_fingerprints. Returns the
    validated protection dict. This is the accessor early stages (1-7) should use."""
    return validate_reference_config(reference_yaml)


def assert_reference_provides_dft_labels(reference_yaml):
    """Guard for the EVALUATION path (Stages 8/9): fail closed if the run-bound reference is a
    protection-only identity reference that carries no DFT labels. Early stages must never route
    through here."""
    kind = reference_kind_of(reference_yaml)
    if not reference_kind_provides_dft_labels(kind):
        raise ValueError(
            f"reference.kind {kind!r} is a protection-only structure-identity reference and "
            "carries no DFT/Teacher labels; it cannot be used as a Stage-8/9 evaluation reference"
        )
    return validate_reference_config(reference_yaml)


def resolve_protected_population(reference_yaml):
    """THE ONE canonical resolution of a run-bound protected reference's protected source population.

    Both the autonomous acquisition PLANNER (which must EXCLUDE these rows from the eligible pool
    BEFORE descriptor/FPS selection and marginal-novelty sizing) and the acquisition EXECUTOR guard
    (which independently re-checks the selected rows via ``assert_source_indices_allowed`` AFTER
    selection) resolve the protected population through THIS function -- so the two enforcement paths
    can never diverge from two independent hand-rolled interpretations of ``reference.yaml``. That
    shared-object guarantee is exactly what the ffv4o Stage-3 defect violated: the planner fabricated
    a PASS exclusion report with ``protected_excluded_count=0`` while the executor's own resolution
    carried 1143 protected rows and fail-closed on the leaked overlap.

    Returns a normalized dict:
      - ``reference_id``             the run-bound reference identity (matched by the executor guard)
      - ``reference_path``           the resolved on-disk reference config path
      - ``protected_source_indices`` sorted unique non-negative seed-pool global rows to protect
      - ``protected_source_rows``    authoritative protected row count (fail-closed: may exceed the
                                     number of unique protected geometries when one geometry appears
                                     at more than one seed-pool row)

    Fails closed (propagates ``validate_reference_config``'s ``ValueError``) on any ambiguous or
    incomplete lineage -- it NEVER silently returns an empty protected set for a malformed reference."""
    protection = validate_reference_config(reference_yaml)
    indices = sorted({int(x) for x in protection["protected_source_indices"]})
    return {
        "reference_id": protection["reference_id"],
        "reference_path": protection.get("reference_path"),
        "protected_source_indices": indices,
        "protected_source_rows": int(protection.get("protected_source_rows", len(indices))),
    }


def assert_source_indices_allowed(selected_indices, protected_indices):
    """Fail if an acquisition plan selects any protected source-pool row."""

    selected = {int(x) for x in selected_indices}
    protected = {int(x) for x in protected_indices}

    overlap = sorted(selected & protected)

    if overlap:
        preview = ", ".join(map(str, overlap[:20]))
        raise ValueError(
            "protected reference leakage through source indices: "
            f"{len(overlap)} overlap(s): {preview}"
        )

    return True


def assert_dataset_geometry_disjoint(dataset_path, reference_fingerprints):
    """Fail if a Student-side dataset contains a protected reference geometry."""

    dataset_path = Path(dataset_path).resolve()
    frames = read(str(dataset_path), index=":")

    overlaps = []

    for index, atoms in enumerate(frames):
        if _structure_fingerprint(atoms) in reference_fingerprints:
            overlaps.append(index)

    if overlaps:
        preview = ", ".join(map(str, overlaps[:20]))
        raise ValueError(
            "protected reference geometry found in Student-side dataset: "
            f"{len(overlaps)} frame(s): {preview}"
        )

    return True


def assert_parent_lineage_allowed(dataset_path, protected_indices):
    """Reject descendants whose top-level parent is a protected seed-pool row.

    Required workflow lineage format:
        parent_structure_id = "seed-pool:<global_index>"
    """

    dataset_path = Path(dataset_path).resolve()
    frames = read(str(dataset_path), index=":")

    protected = {int(x) for x in protected_indices}
    overlaps = []

    for index, atoms in enumerate(frames):

        parent = atoms.info.get("parent_structure_id")

        if parent is None:
            raise ValueError(
                f"frame {index} is missing parent_structure_id"
            )

        parent = str(parent)

        if not parent.startswith("seed-pool:"):
            raise ValueError(
                f"frame {index} has unsupported parent_structure_id: {parent!r}"
            )

        try:
            source_index = int(parent.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(
                f"frame {index} has malformed parent_structure_id: {parent!r}"
            ) from exc

        if source_index in protected:
            overlaps.append((index, source_index))

    if overlaps:
        preview = ", ".join(
            f"frame={frame}/source={source}"
            for frame, source in overlaps[:20]
        )

        raise ValueError(
            "protected reference descendant detected: "
            f"{len(overlaps)} frame(s): {preview}"
        )

    return True



def _artifact_contains(path, submitted_artifacts):
    """Return True when path itself, or a parent directory, was submitted."""
    path = Path(path).resolve()

    for raw in submitted_artifacts or []:
        root = Path(raw).resolve()

        if path == root:
            return True

        if root.is_dir() and path.is_relative_to(root):
            return True

    return False


def validate_protection_audit_report(
    manifest_path,
    reference_yaml,
    submitted_artifacts=None,
    allowed_evidence=None,
    enforce_required_pass=False,
):
    """Controller contract for protected-reference exclusion.

    The report does not decide PASS itself. This function recomputes all
    protection checks from the referenced datasets and raises on leakage.
    """

    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text())

    if payload.get("schema_version") != 1:
        raise ValueError(
            "protection audit requires schema_version=1"
        )

    stage = payload.get("stage")

    if stage not in {
        "acquisition",
        "dataset_split",
        "teacher_labeling",
    }:
        raise ValueError(
            f"unsupported protection-audit stage: {stage!r}"
        )

    protection = validate_reference_config(reference_yaml)

    selected = payload.get("selected_source_indices", [])

    if not isinstance(selected, list):
        raise ValueError(
            "selected_source_indices must be a list"
        )

    if any(
        isinstance(x, bool) or not isinstance(x, int)
        for x in selected
    ):
        raise ValueError(
            "selected_source_indices must contain integers only"
        )

    assert_source_indices_allowed(
        selected,
        protection["protected_source_indices"],
    )

    datasets = payload.get("datasets")

    if not isinstance(datasets, list) or not datasets:
        raise ValueError(
            "protection audit requires at least one dataset"
        )

    submitted = list(submitted_artifacts or [])

    roles = set()
    checked = []

    for item in datasets:
        if not isinstance(item, dict):
            raise ValueError(
                "each protection-audit dataset must be an object"
            )

        role = item.get("role")
        raw_path = item.get("path")

        if (
            not isinstance(role, str)
            or not role.strip()
            or not isinstance(raw_path, str)
            or not raw_path.strip()
        ):
            raise ValueError(
                "each protection-audit dataset requires role and path"
            )

        if role in roles:
            raise ValueError(
                f"duplicate protection-audit dataset role: {role}"
            )

        roles.add(role)

        path = Path(raw_path).expanduser()

        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        else:
            path = path.resolve()

        if not path.is_file():
            raise FileNotFoundError(path)

        if submitted and not _artifact_contains(
            path,
            submitted,
        ):
            raise ValueError(
                f"protection-audit dataset was not submitted "
                f"as a stage artifact: {path}"
            )

        assert_dataset_geometry_disjoint(
            path,
            protection["reference_fingerprints"],
        )

        assert_parent_lineage_allowed(
            path,
            protection["protected_source_indices"],
        )

        checked.append(
            {
                "role": role,
                "path": str(path),
            }
        )

    return {
        "schema_version": 1,
        "stage": stage,
        "status": "PASS",
        "reference_id": protection["reference_id"],
        "logical_reference_frames":
            protection["logical_frames"],
        "protected_source_rows":
            protection["protected_source_rows"],
        "selected_source_indices_checked":
            len(selected),
        "datasets_checked": checked,
    }


def write_protection_audit(
    stage,
    reference_yaml,
    dataset_specs,
    output,
    selected_source_indices=None,
):
    """Create an audit artifact only after deterministic checks pass."""

    output = Path(output).resolve()

    datasets = []

    for spec in dataset_specs:
        if "=" not in spec:
            raise ValueError(
                "dataset spec must be ROLE=PATH"
            )

        role, raw_path = spec.split("=", 1)

        path = Path(raw_path).expanduser().resolve()

        datasets.append(
            {
                "role": role,
                "path": str(path),
            }
        )

    payload = {
        "schema_version": 1,
        "stage": stage,
        "selected_source_indices": [
            int(x)
            for x in (selected_source_indices or [])
        ],
        "datasets": datasets,
    }

    # Write temporarily so the same contract implementation performs
    # the scientific validation.
    output.parent.mkdir(parents=True, exist_ok=True)

    tmp = output.with_suffix(
        output.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n"
    )

    try:
        validate_protection_audit_report(
            tmp,
            reference_yaml=reference_yaml,
        )
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    tmp.replace(output)

    return output
