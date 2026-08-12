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
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
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
    """Validate the frozen R2 reference.yaml and protected-reference package."""

    reference_yaml = Path(reference_yaml).resolve()
    cfg = yaml.safe_load(reference_yaml.read_text()) or {}

    if cfg.get("kind") != "protected-existing-dft":
        raise ValueError("reference.kind must be 'protected-existing-dft'")

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
