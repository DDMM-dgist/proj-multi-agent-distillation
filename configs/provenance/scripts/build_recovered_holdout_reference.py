"""One-off provenance script: materialize the ``recovered-original-holdout`` Teacher-vs-DFT
reference for the SiO2 campaign (R26 forensic fix -- see configs/provenance/PROVENANCE.md and
validation/protected_reference.py's ``_validate_recovered_holdout_reference_config``).

The 24 per-category seed_pool_11424/<category>/<category>.xyz files (verified, in
configs/provenance/scripts/parse_and_split.py, to be frame-for-frame identical to the
teacher_training_split_manifest.json global reconstruction) do not themselves carry
``source_category``/``source_local_index`` in ``atoms.info`` -- this script's only job is to
annotate each frame with that join key (0-based, file order, matching the manifest's own
per-category local-index convention) and concatenate them into one combined source dataset, then
call the generic, reusable ``workflow.steps.build_split_membership_population`` producer to
extract exactly the manifest's ``target_split`` partition. This script is a one-off data-prep /
provenance-reconstruction step (like parse_and_split.py before it), not production dispatch
logic -- literal category names and paths here are fine per the run's no-hardcode rule, which
applies to runtime/dispatch code, not to provenance scripts that build config/data inputs.

Run once: python configs/provenance/scripts/build_recovered_holdout_reference.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ase.io import read, write

REPO_ROOT = Path(__file__).resolve().parents[3]
SEEDPOOL = REPO_ROOT / "local_inputs/sio2_fresh/seed_pool_11424"
SPLIT_MANIFEST = REPO_ROOT / "configs/provenance/teacher_training_split_manifest.json"
OUT_DIR = REPO_ROOT / "local_inputs/sio2_fresh/recovered_original_holdout"

COMBINED_SOURCE_PATH = OUT_DIR / "annotated_source_pool_11424.xyz"
STRUCTURES_PATH = OUT_DIR / "recovered_original_holdout_test.xyz"
STRUCTURES_MANIFEST_PATH = OUT_DIR / "recovered_original_holdout_test_manifest.json"
REFERENCE_YAML_PATH = OUT_DIR / "reference.yaml"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def annotate_and_combine() -> Path:
    """Concatenate every seed_pool_11424 category file, in category-name-sorted order (the same
    deterministic order parse_and_split.py used to verify category<->manifest correspondence),
    stamping ``source_category``/``source_local_index`` onto every frame in file order."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = []
    for catdir in sorted(SEEDPOOL.iterdir()):
        if not catdir.is_dir():
            continue
        category = catdir.name
        xyz_path = catdir / f"{category}.xyz"
        if not xyz_path.is_file():
            continue
        frames = read(str(xyz_path), index=":")
        for local_index, atoms in enumerate(frames):
            atoms.info["source_category"] = category
            atoms.info["source_local_index"] = local_index
            combined.append(atoms)
    write(str(COMBINED_SOURCE_PATH), combined)
    return COMBINED_SOURCE_PATH


def build_reference() -> None:
    from workflow.steps import build_split_membership_population
    from validation.protected_reference import (
        RECOVERED_HOLDOUT_REQUIRED_PROHIBITIONS, RECOVERED_HOLDOUT_REFERENCE_CLASS,
        validate_reference_config,
    )

    combined_path = annotate_and_combine()

    manifest_payload = json.loads(SPLIT_MANIFEST.read_text())
    split_roles = manifest_payload.get("split_roles", {})
    heldout_splits = [split for split, role in split_roles.items() if role == "heldout_evaluation"]
    if len(heldout_splits) != 1:
        raise ValueError(
            f"expected exactly one split_roles entry with role 'heldout_evaluation', found "
            f"{heldout_splits!r} in {SPLIT_MANIFEST}"
        )
    target_split = heldout_splits[0]

    result = build_split_membership_population(
        source_dataset=str(combined_path),
        split_source_manifest=str(SPLIT_MANIFEST),
        target_split=target_split,
        output_path=str(STRUCTURES_PATH),
        manifest_path=str(STRUCTURES_MANIFEST_PATH),
    )

    reference_id = "recovered-original-heldout-" + target_split
    reference_yaml = {
        "kind": "recovered-original-holdout",
        "reference_id": reference_id,
        "reference_class": RECOVERED_HOLDOUT_REFERENCE_CLASS,
        "status": "AVAILABLE_AND_VERIFIED",
        "target_split": target_split,
        "split_source_manifest": str(SPLIT_MANIFEST.resolve()),
        "split_source_manifest_sha256": _sha256_file(SPLIT_MANIFEST),
        "frame_count": result["frame_count"],
        "structures": {
            "path": result["structures"]["path"],
            "logical_frames": result["structures"]["logical_frames"],
            "sha256": result["structures"]["sha256"],
        },
        "prohibited_uses": sorted(RECOVERED_HOLDOUT_REQUIRED_PROHIBITIONS),
        "note": (
            "Algorithmically reconstructed original Teacher held-out partition (see "
            "configs/provenance/teacher_training_split_manifest.json), derived from the same "
            "24-category seed pool as the historical protected reference but joined via the "
            "recovered split-generation manifest rather than a physically-recovered historical "
            "artifact. This is the ORIGINAL_HELDOUT_FIDELITY component's execution binding; it "
            "must never be substituted with the separate historical protected-existing-dft "
            "reference (local_inputs/sio2_fresh/protected_reference/), which remains valid only "
            "for its own protected-reference role."
        ),
    }

    import yaml
    REFERENCE_YAML_PATH.write_text(yaml.safe_dump(reference_yaml, sort_keys=False))

    validate_reference_config(str(REFERENCE_YAML_PATH))
    print(f"wrote and validated {REFERENCE_YAML_PATH}")
    print(f"frame_count={result['frame_count']} target_split={target_split!r}")


if __name__ == "__main__":
    build_reference()
