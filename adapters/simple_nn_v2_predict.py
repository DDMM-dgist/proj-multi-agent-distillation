"""Batch prediction shim for the SIMPLE-NN v2 test-mode interface.

The command consumes Teacher-labeled extxyz structures, evaluates one exported
SIMPLE-NN potential, and writes the same structures with ``student_*`` fields.
It intentionally runs inside an isolated work directory because SIMPLE-NN v2
uses fixed filenames such as ``total_list``, ``potential_saved`` and
``test_result``.
"""
from __future__ import annotations

import argparse
import copy
import shutil
import tempfile
from pathlib import Path

import numpy as np
import yaml

from adapters.simple_nn_v2_wrapper import (
    _dataset_to_extxyz_with_ref_labels,
    _write_structure_list,
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--structures", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-stress", action="store_true")
    return parser.parse_args(argv)


def _load_test_result(path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch versions do not expose weights_only
        return torch.load(path, map_location="cpu")


def _run_simple_nn(input_path, cwd):
    import os
    from simple_nn.simple_nn import run

    previous = Path.cwd()
    try:
        os.chdir(cwd)
        run(str(input_path))
    finally:
        os.chdir(previous)


def _copy_training_state(training_dir, work_dir, checkpoint, training_input):
    shutil.copy2(checkpoint, work_dir / "potential_saved")
    nn = training_input.get("neural_network", {})
    required = []
    if nn.get("use_scale", True):
        required.append("scale_factor")
    if nn.get("use_pca", True):
        required.append("pca")
    for name in required:
        source = training_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"SIMPLE-NN prediction requires training artifact: {source}")
        shutil.copy2(source, work_dir / name)


def _generation_input(training_input, structure_list, include_stress):
    payload = copy.deepcopy(training_input)
    payload["generate_features"] = True
    payload["preprocess"] = False
    payload["train_model"] = False
    data = dict(payload.get("data") or {})
    data.update(struct_list=str(structure_list.resolve()), absolute_path=True,
                save_directory="./data", save_list="./total_list",
                refdata_format="extxyz", read_force=True,
                read_stress=bool(include_stress))
    payload["data"] = data
    nn = dict(payload.get("neural_network") or {})
    nn.update(train=False, test=False, use_force=True,
              use_stress=bool(include_stress))
    payload["neural_network"] = nn
    return payload


def _evaluation_input(training_input, test_list, include_stress):
    payload = copy.deepcopy(training_input)
    payload["generate_features"] = False
    payload["preprocess"] = False
    payload["train_model"] = True
    nn = dict(payload.get("neural_network") or {})
    nn.update(train=False, test=True, use_force=True,
              use_stress=bool(include_stress),
              **{"continue": "weights", "test_list": str(test_list.resolve())})
    payload["neural_network"] = nn
    return payload


def _materialize_test_list(total_list, work_dir, expected_n_frames):
    """Convert SIMPLE-NN's tagged ``total_list`` (lines of ``TAG:PATH``) into a
    plain-path ``test_list`` that SIMPLE-NN's test-mode ``FilelistDataset``
    consumes directly.

    Reuses SIMPLE-NN's own regex-based tag parser
    (``simple_nn.utils.features._make_str_data_list``) so we do not
    re-implement its file-format rules. Preprocess is intentionally NOT
    invoked here: running it would overwrite the training-time
    ``scale_factor`` and ``pca`` with values fit on the held-out frames
    (``simple_nn/features/preprocessing.py:43-54``).
    """
    from simple_nn.utils.features import _make_str_data_list

    grouped = _make_str_data_list(str(total_list))
    test_paths = [path for group in grouped for path in group]
    if len(test_paths) != expected_n_frames:
        raise RuntimeError(
            f"SIMPLE-NN test_list has {len(test_paths)} entries; "
            f"expected {expected_n_frames} (one per input frame)"
        )
    for path in test_paths:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = work_dir / path
        if not candidate.is_file():
            raise RuntimeError(f"SIMPLE-NN feature file is missing: {candidate}")

    test_list = work_dir / "test_list"
    test_list.write_text("\n".join(test_paths) + "\n")
    return test_list


def predict(checkpoint, structures, output, include_stress=False):
    checkpoint = Path(checkpoint).resolve()
    structures = Path(structures).resolve()
    output = Path(output).resolve()
    if not checkpoint.is_file() or not structures.is_file():
        raise FileNotFoundError("SIMPLE-NN prediction checkpoint/structures are missing")
    training_dir = checkpoint.parent
    training_input_path = training_dir / "simple_nn_input.yaml"
    if not training_input_path.is_file():
        raise FileNotFoundError(
            f"SIMPLE-NN prediction requires the resolved training input: {training_input_path}"
        )
    training_input = yaml.safe_load(training_input_path.read_text()) or {}
    with tempfile.TemporaryDirectory(
            prefix="simple-nn-evaluation-", dir=Path.cwd()) as tmp:
        return _predict_in_workdir(
            checkpoint, structures, output, training_input,
            Path(tmp), include_stress
        )


def _predict_in_workdir(checkpoint, structures, output, training_input,
                        work_dir, include_stress=False):
    from ase.io import read, write

    training_dir = checkpoint.parent

    labeled = work_dir / "labeled_structures.extxyz"
    n_frames, _ = _dataset_to_extxyz_with_ref_labels(
        structures, labeled, require_stress=include_stress
    )
    structure_list = work_dir / "structure_list"
    _write_structure_list(structure_list, labeled, "student-heldout-evaluation")

    generation_path = work_dir / "generate_input.yaml"
    generation_path.write_text(yaml.safe_dump(
        _generation_input(training_input, structure_list, include_stress),
        sort_keys=False,
    ))
    _run_simple_nn(generation_path, work_dir)
    total_list = work_dir / "total_list"
    if not total_list.is_file():
        raise RuntimeError("SIMPLE-NN feature generation produced no total_list")
    test_list = _materialize_test_list(total_list, work_dir, n_frames)

    _copy_training_state(training_dir, work_dir, checkpoint, training_input)
    evaluation_path = work_dir / "evaluate_input.yaml"
    evaluation_path.write_text(yaml.safe_dump(
        _evaluation_input(training_input, test_list, include_stress),
        sort_keys=False,
    ))
    _run_simple_nn(evaluation_path, work_dir)
    result_path = work_dir / "test_result"
    if not result_path.is_file():
        raise RuntimeError("SIMPLE-NN test mode produced no test_result")
    result = _load_test_result(result_path)

    # DFT_* are the reference labels read by SIMPLE-NN. Rechecking them here
    # binds every prediction to its input frame instead of trusting ordering
    # merely because the result counts match.
    required = {"N", "NN_E", "NN_F", "DFT_E", "DFT_F"}
    missing = required - set(result)
    if missing:
        raise ValueError("SIMPLE-NN test_result is missing: " + ", ".join(sorted(missing)))
    frames = read(structures, index=":")
    frame_keys = ("N", "NN_E", "NN_F", "DFT_E", "DFT_F")
    if len(frames) != n_frames or any(len(result[key]) != n_frames for key in frame_keys):
        raise ValueError("SIMPLE-NN test_result frame count does not match the input")
    if include_stress and ("NN_S" not in result or len(result["NN_S"]) != n_frames):
        raise ValueError("SIMPLE-NN test_result has no complete stress predictions")

    for index, atoms in enumerate(frames):
        if int(result["N"][index]) != len(atoms):
            raise ValueError(f"SIMPLE-NN test_result atom count mismatch at frame {index}")
        energy = float(result["NN_E"][index])
        forces = np.asarray(result["NN_F"][index], dtype=float)
        if not np.isfinite(energy) or forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
            raise ValueError(f"SIMPLE-NN predictions are invalid at frame {index}")
        reference_energy = float(result["DFT_E"][index])
        reference_forces = np.asarray(result["DFT_F"][index], dtype=float)
        teacher_energy = float(atoms.info["teacher_energy"])
        teacher_forces = np.asarray(atoms.arrays["teacher_forces"], dtype=float)
        if (not np.isclose(reference_energy, teacher_energy, atol=1e-10, rtol=1e-10) or
                reference_forces.shape != teacher_forces.shape or
                not np.allclose(reference_forces, teacher_forces,
                                atol=1e-10, rtol=1e-10)):
            raise ValueError(
                f"SIMPLE-NN test_result reference labels do not match input frame {index}"
            )
        atoms.info["student_energy"] = energy
        atoms.arrays["student_forces"] = forces
        if include_stress:
            stress = np.asarray(result["NN_S"][index], dtype=float)
            if stress.shape != (6,) or not np.all(np.isfinite(stress)):
                raise ValueError(f"SIMPLE-NN stress prediction is invalid at frame {index}")
            atoms.info["student_stress"] = stress.tolist()
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, frames, format="extxyz")
    return output


def main(argv=None):
    args = _parse_args(argv)
    predict(args.checkpoint, args.structures, args.output, args.include_stress)


if __name__ == "__main__":
    main()
