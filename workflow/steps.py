"""Concrete stage commands used by the persistent run controller."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from ase.io import read, write

from adapters import load_config
from adapters.md_backend import render_lammps_input, run as run_md_backend
from adapters.student import load_student, predict_student, train_student
from validation.four_channel_audit import channel


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def train_committee(student_config, dataset, output_dir, manifest):
    cfg = load_config(student_config)
    output_dir = Path(output_dir)
    models = []
    for seed in range(1, int(cfg.get("committee", {}).get("n_seeds", 4)) + 1):
        artifact = train_student(cfg, dataset, output_dir / f"seed-{seed}", seed)
        models.append({"kind": artifact.kind, "seed": seed, "path": str(artifact.path)})
    result = {"schema_version": 1, "student_config": str(Path(student_config).resolve()),
              "dataset": str(Path(dataset).resolve()), "models": models}
    _write_json(manifest, result)
    return result


def evaluate_committee(student_config, committee_manifest, frames_path, labeled_output, report):
    cfg = load_config(student_config)
    committee = json.loads(Path(committee_manifest).read_text())
    frames = read(frames_path, index=":")
    for model in committee["models"]:
        prediction = predict_student(cfg, load_student(cfg, model["path"]), frames)
        key = f"{int(model['seed']):02d}"
        for atoms, energy, forces in zip(frames, prediction.energies, prediction.forces):
            atoms.info[f"student_energy_seed{key}"] = float(energy)
            atoms.arrays[f"student_forces_seed{key}"] = np.asarray(forces)
    write(labeled_output, frames)
    results = {}
    for label, ref, pred in (("teacher_vs_dft", "dft", "teacher"),
                             ("student_vs_teacher", "teacher", "student"),
                             ("student_vs_dft", "dft", "student")):
        results[label] = channel(frames, ref, pred, per_config_type=True)
    _write_json(report, results)
    return results


def run_md(md_config, student_config, checkpoint, template_name, context_yaml, input_path, run_dir, manifest):
    md_cfg, student_cfg = load_config(md_config), load_config(student_config)
    context = yaml.safe_load(Path(context_yaml).read_text())
    render_lammps_input(md_cfg, student_cfg, checkpoint, template_name, context, input_path)
    run_md_backend(md_cfg, input_path, run_dir, mpi_ranks=int(context.get("MPI_RANKS", 1)))
    result = {"schema_version": 1, "input": str(Path(input_path).resolve()),
              "run_dir": str(Path(run_dir).resolve()), "checkpoint": str(Path(checkpoint).resolve())}
    _write_json(manifest, result)
    return result


def capture_validation(command, report):
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    payload = {"schema_version": 1, "command": command, "stdout": result.stdout,
               "stderr": result.stderr, "returncode": result.returncode}
    _write_json(report, payload)
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    train = sub.add_parser("train-committee")
    train.add_argument("student_config"); train.add_argument("dataset")
    train.add_argument("output_dir"); train.add_argument("manifest")
    evaluate = sub.add_parser("evaluate-committee")
    evaluate.add_argument("student_config"); evaluate.add_argument("committee_manifest")
    evaluate.add_argument("frames"); evaluate.add_argument("labeled_output"); evaluate.add_argument("report")
    md = sub.add_parser("run-md")
    md.add_argument("md_config"); md.add_argument("student_config"); md.add_argument("checkpoint")
    md.add_argument("template"); md.add_argument("context_yaml"); md.add_argument("input_path")
    md.add_argument("run_dir"); md.add_argument("manifest")
    validate = sub.add_parser("capture-validation")
    validate.add_argument("report"); validate.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.action == "train-committee":
        train_committee(args.student_config, args.dataset, args.output_dir, args.manifest)
    elif args.action == "evaluate-committee":
        evaluate_committee(args.student_config, args.committee_manifest, args.frames,
                           args.labeled_output, args.report)
    elif args.action == "run-md":
        run_md(args.md_config, args.student_config, args.checkpoint, args.template,
               args.context_yaml, args.input_path, args.run_dir, args.manifest)
    else:
        if not args.command:
            p.error("capture-validation requires a command after --")
        capture_validation(args.command, args.report)


if __name__ == "__main__":
    main()
