"""Concrete stage commands used by the persistent run controller."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from ase.io import read, write

from adapters import load_config
from adapters.md_backend import render_input as render_md_input, run as run_md_backend
from adapters.student import load_student, predict_student, train_student
from validation.four_channel_audit import channel
from workflow.integrity import artifact_digest, sha256_file, verify_artifact


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def split_dataset(dataset, output_dir, manifest, seed=2026, validation_fraction=0.1,
                  test_fraction=0.1, grouping_key="parent_structure_id",
                  allow_unique_parent_fallback=False):
    """Create leakage-resistant splits by keeping related structures together."""
    if validation_fraction < 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("split fractions require test > 0 and validation + test < 1")
    frames = read(dataset, index=":")
    if len(frames) < 3:
        raise ValueError("at least three structures are required for train/validation/test splitting")
    groups = {}
    for index, atoms in enumerate(frames):
        structure_id = str(atoms.info.get("structure_id", f"frame-{index:08d}"))
        if grouping_key not in atoms.info and not allow_unique_parent_fallback:
            raise ValueError(f"frame {index} is missing required lineage key {grouping_key!r}")
        group_id = str(atoms.info.get(grouping_key, structure_id))
        atoms.info.setdefault("structure_id", structure_id)
        groups.setdefault(group_id, []).append(atoms)
    if len(groups) < 3:
        raise ValueError("at least three independent structure groups are required to prevent leakage")
    group_ids = sorted(groups)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(group_ids)
    n_test = max(1, round(len(group_ids) * test_fraction))
    n_validation = max(1, round(len(group_ids) * validation_fraction))
    if n_test + n_validation >= len(group_ids):
        n_validation = 1
        n_test = 1
    split_groups = {
        "test": group_ids[:n_test],
        "validation": group_ids[n_test:n_test + n_validation],
        "train": group_ids[n_test + n_validation:],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, ids in split_groups.items():
        split_frames = [atoms for group_id in ids for atoms in groups[group_id]]
        path = output_dir / f"{name}.extxyz"
        write(path, split_frames)
        outputs[name] = {"path": str(path.resolve()), "sha256": sha256_file(path),
                         "n_frames": len(split_frames), "group_ids": sorted(ids)}
    result = {"schema_version": 1, "source": str(Path(dataset).resolve()),
              "source_sha256": sha256_file(dataset), "seed": int(seed),
              "grouping_key": grouping_key, "validation_fraction": validation_fraction,
              "test_fraction": test_fraction, "splits": outputs,
              "overlap_checks": {"train_validation": 0, "train_test": 0,
                                  "validation_test": 0}}
    _write_json(manifest, result)
    return result


def _structure_fingerprint(atoms):
    """Exact geometry fingerprint for duplicate control; no tolerance is implied."""
    payload = {
        "numbers": atoms.numbers.tolist(), "positions": atoms.positions.tolist(),
        "cell": atoms.cell.array.tolist(), "pbc": atoms.pbc.tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def merge_datasets(sources, output, manifest, grouping_key="parent_structure_id",
                   duplicate_policy="error"):
    """Merge acquisition outputs while preserving lineage and recording schemas.

    This intentionally does not align labels or mix reference conventions.
    Such transformations require an explicit, reviewed project-specific stage.
    """
    if duplicate_policy not in {"error", "keep-first", "keep"}:
        raise ValueError("duplicate_policy must be error, keep-first, or keep")
    merged, seen, duplicate_count, source_records = [], {}, 0, []
    for source_index, raw_path in enumerate(sources):
        path = Path(raw_path).resolve()
        frames = read(path, index=":")
        schemas = set()
        accepted = 0
        for frame_index, atoms in enumerate(frames):
            if grouping_key not in atoms.info:
                raise ValueError(
                    f"{path} frame {frame_index} is missing lineage key {grouping_key!r}"
                )
            fingerprint = _structure_fingerprint(atoms)
            if fingerprint in seen:
                duplicate_count += 1
                if duplicate_policy == "error":
                    raise ValueError(
                        f"exact duplicate structure in {path} frame {frame_index}; "
                        f"first seen at {seen[fingerprint]}"
                    )
                if duplicate_policy == "keep-first":
                    continue
            else:
                seen[fingerprint] = f"source {source_index} frame {frame_index}"
            original_id = atoms.info.get("structure_id", f"frame-{frame_index:08d}")
            atoms.info["source_structure_id"] = str(original_id)
            atoms.info["structure_id"] = (
                f"source-{source_index:03d}:frame-{frame_index:08d}:{original_id}"
            )
            atoms.info["acquisition_source"] = str(path)
            schemas.add((tuple(sorted(atoms.info)), tuple(sorted(atoms.arrays))))
            merged.append(atoms)
            accepted += 1
        source_records.append({"path": str(path), "integrity": artifact_digest(path),
                               "n_frames": len(frames), "n_accepted": accepted,
                               "schemas": [{"info": list(info), "arrays": list(arrays)}
                                           for info, arrays in sorted(schemas)]})
    if not merged:
        raise ValueError("dataset merge produced no structures")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, merged)
    result = {"schema_version": 1, "operation": "merge-acquisitions",
              "grouping_key": grouping_key, "duplicate_policy": duplicate_policy,
              "n_frames": len(merged), "n_exact_duplicates": duplicate_count,
              "sources": source_records, "output": str(output.resolve()),
              "output_integrity": artifact_digest(output)}
    _write_json(manifest, result)
    return result


def train_committee(student_config, dataset, output_dir, manifest):
    cfg = load_config(student_config)
    output_dir = Path(output_dir)
    models = []
    for seed in range(1, int(cfg.get("committee", {}).get("n_seeds", 4)) + 1):
        artifact = train_student(cfg, dataset, output_dir / f"seed-{seed}", seed)
        models.append({"kind": artifact.kind, "seed": seed, "path": str(artifact.path),
                       "integrity": artifact_digest(artifact.path),
                       "metadata": artifact.metadata})
    result = {"schema_version": 1, "student_config": str(Path(student_config).resolve()),
              "student_config_integrity": artifact_digest(student_config),
              "dataset": str(Path(dataset).resolve()),
              "dataset_integrity": artifact_digest(dataset), "models": models}
    _write_json(manifest, result)
    return result


def evaluate_committee(student_config, committee_manifest, frames_path, labeled_output, report,
                       required_channels=None):
    cfg = load_config(student_config)
    committee = json.loads(Path(committee_manifest).read_text())
    for model in committee["models"]:
        verify_artifact(model["path"], model.get("integrity", {}))
    frames = read(frames_path, index=":")
    for model in committee["models"]:
        prediction = predict_student(cfg, load_student(cfg, model["path"]), frames)
        if len(prediction.energies) != len(frames):
            raise RuntimeError(
                f"committee seed {model['seed']} returned {len(prediction.energies)} predictions "
                f"for {len(frames)} held-out frames"
            )
        for index, (atoms, forces) in enumerate(zip(frames, prediction.forces)):
            if np.asarray(forces).shape != (len(atoms), 3):
                raise RuntimeError(
                    f"committee seed {model['seed']} returned invalid force shape at frame "
                    f"{index}: {np.asarray(forces).shape}"
                )
        key = f"{int(model['seed']):02d}"
        for atoms, energy, forces in zip(frames, prediction.energies, prediction.forces):
            atoms.info[f"student_energy_seed{key}"] = float(energy)
            atoms.arrays[f"student_forces_seed{key}"] = np.asarray(forces)
    write(labeled_output, frames)
    results = {}
    required_channels = set(required_channels or [])
    for label, ref, pred in (("teacher_vs_dft", "dft", "teacher"),
                             ("student_vs_teacher", "teacher", "student"),
                             ("student_vs_dft", "dft", "student")):
        results[label] = channel(frames, ref, pred, per_config_type=True,
                                 require_complete=label in required_channels)
    missing = [name for name in required_channels if results.get(name) is None]
    if missing:
        raise RuntimeError("required evaluation channels have missing labels: " + ", ".join(missing))
    _write_json(report, results)
    return results


def run_md(md_config, student_config, checkpoint, template_name, context_yaml, input_path, run_dir,
           manifest, committee_manifest=None, selected_seed=None, evidence_paths=None):
    md_cfg, student_cfg = load_config(md_config), load_config(student_config)
    context = yaml.safe_load(Path(context_yaml).read_text())
    render_md_input(md_cfg, student_cfg, checkpoint, template_name, context, input_path)
    run_md_backend(md_cfg, input_path, run_dir, mpi_ranks=int(context.get("MPI_RANKS", 1)))
    checkpoint = Path(checkpoint).resolve()
    evidence = [{"role": "input", "path": str(Path(input_path).resolve()),
                 "integrity": artifact_digest(input_path)}]
    for item in evidence_paths or []:
        if "=" not in item:
            raise ValueError("MD evidence must use ROLE=PATH")
        role, raw_path = item.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not role or not path.exists():
            raise ValueError(f"invalid MD evidence: {item}")
        evidence.append({"role": role, "path": str(path),
                         "integrity": artifact_digest(path)})
    result = {"schema_version": 1, "input": str(Path(input_path).resolve()),
              "run_dir": str(Path(run_dir).resolve()), "checkpoint": str(checkpoint),
              "checkpoint_integrity": artifact_digest(checkpoint), "evidence": evidence}
    if committee_manifest is not None:
        result["committee_manifest"] = str(Path(committee_manifest).resolve())
    if selected_seed is not None:
        result["selected_seed"] = int(selected_seed)
    _write_json(manifest, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    train = sub.add_parser("train-committee")
    train.add_argument("student_config"); train.add_argument("dataset")
    train.add_argument("output_dir"); train.add_argument("manifest")
    evaluate = sub.add_parser("evaluate-committee")
    evaluate.add_argument("student_config"); evaluate.add_argument("committee_manifest")
    evaluate.add_argument("frames"); evaluate.add_argument("labeled_output"); evaluate.add_argument("report")
    evaluate.add_argument("--require-channel", action="append", default=[])
    md = sub.add_parser("run-md")
    md.add_argument("md_config"); md.add_argument("student_config"); md.add_argument("checkpoint")
    md.add_argument("template"); md.add_argument("context_yaml"); md.add_argument("input_path")
    md.add_argument("run_dir"); md.add_argument("manifest")
    md.add_argument("--committee-manifest")
    md.add_argument("--selected-seed", type=int)
    md.add_argument("--evidence", action="append", default=[], metavar="ROLE=PATH")
    split = sub.add_parser("split-dataset")
    split.add_argument("dataset"); split.add_argument("output_dir"); split.add_argument("manifest")
    split.add_argument("--seed", type=int, default=2026)
    split.add_argument("--validation-fraction", type=float, default=0.1)
    split.add_argument("--test-fraction", type=float, default=0.1)
    split.add_argument("--grouping-key", default="parent_structure_id")
    split.add_argument("--allow-unique-parent-fallback", action="store_true")
    merge = sub.add_parser("merge-datasets")
    merge.add_argument("output"); merge.add_argument("manifest")
    merge.add_argument("--source", action="append", required=True)
    merge.add_argument("--grouping-key", default="parent_structure_id")
    merge.add_argument("--duplicate-policy", choices=["error", "keep-first", "keep"],
                       default="error")
    args = p.parse_args()
    if args.action == "split-dataset":
        split_dataset(args.dataset, args.output_dir, args.manifest, args.seed,
                      args.validation_fraction, args.test_fraction, args.grouping_key,
                      args.allow_unique_parent_fallback)
    elif args.action == "merge-datasets":
        merge_datasets(args.source, args.output, args.manifest, args.grouping_key,
                       args.duplicate_policy)
    elif args.action == "train-committee":
        train_committee(args.student_config, args.dataset, args.output_dir, args.manifest)
    elif args.action == "evaluate-committee":
        evaluate_committee(args.student_config, args.committee_manifest, args.frames,
                           args.labeled_output, args.report, args.require_channel)
    elif args.action == "run-md":
        run_md(args.md_config, args.student_config, args.checkpoint, args.template,
               args.context_yaml, args.input_path, args.run_dir, args.manifest,
               args.committee_manifest, args.selected_seed, args.evidence)


if __name__ == "__main__":
    main()
