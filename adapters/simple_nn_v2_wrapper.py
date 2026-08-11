"""CLI shim between adapters.student._train_simple_nn and installed SIMPLE-NN v2.

The internal `_train_simple_nn` helper invokes
    python -m <runner.module> \\
        --config <rendered_input.yaml> \\
        --descriptor-param Si=<params_Si> \\
        --descriptor-param O=<params_O> \\
        --dataset <path/to/train.extxyz> \\
        --out <output_dir> \\
        --seed <int> \\
        --epochs <int> \\
        --precision <double|single> \\
        --batch-size <int>

Installed SIMPLE-NN v2 exposes only `simple_nn.simple_nn.run(input_yaml)` which
consumes a single YAML file and reads structures via `ase.io.read()` through a
`structure_list` file. This module translates the CLI form above into the
SIMPLE-NN v2 API and runs one training seed.

Design constraints:
- Never fabricates training data. The extxyz dataset passed via --dataset must
  already contain teacher_energy (info) and teacher_forces (arrays). SIMPLE-NN
  reads energy/forces from ASE's standard slots (info['REF_energy'] or
  atoms.calc), so this wrapper attaches a lightweight ASE Calculator that
  returns the pre-computed teacher labels — SIMPLE-NN then treats those labels
  as its reference E/F.
- All artifacts land in `--out`; no writes elsewhere.
- Ends with either potential_saved_bestmodel present (success) or a raised
  exception (failure). Never a silent partial state.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import yaml


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run one SIMPLE-NN v2 training seed via the adapter CLI"
    )
    p.add_argument("--config", required=True,
                   help="rendered SIMPLE-NN input.yaml from adapters.student._render_simple_nn_config")
    p.add_argument("--descriptor-param", action="append", default=[],
                   help="ELEMENT=path/to/params_ELEMENT, may be repeated")
    p.add_argument("--dataset", required=True,
                   help="ASE-readable dataset (extxyz) with teacher labels")
    p.add_argument("--out", required=True, help="output directory for this seed")
    p.add_argument("--seed", type=int, required=True,
                   help="deterministic seed for this committee member")
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--precision", choices=("double", "single"), required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--use-stress", action="store_true")
    p.add_argument("--stress-loss-weight", type=float, default=0.0)
    p.add_argument("--struct-weight-policy", default="none",
                   choices=("none", "c_size_normalized_bounded"),
                   help="per-structure SIMPLE-NN struct_weight: none=uniform 1.0; "
                        "c_size_normalized_bounded=clip((1/N)/geomean(1/N),1/sqrt8,sqrt8) [training only]")
    return p.parse_args(argv)


def _dataset_to_extxyz_with_ref_labels(source_path: Path, target_path: Path,
                                       require_stress: bool = False):
    """Rewrite the dataset so SIMPLE-NN reads teacher_energy/teacher_forces as
    reference labels. SIMPLE-NN v2 uses ase.io.read() and pulls energies/forces
    from a calculator's SinglePointCalculator results dict, so we attach one.
    """
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import read, write
    import numpy as np

    frames = read(str(source_path), index=":")
    if not frames:
        raise RuntimeError(f"dataset is empty: {source_path}")
    materialized = []
    stress_count = 0
    for index, atoms in enumerate(frames):
        if "teacher_energy" not in atoms.info:
            raise ValueError(f"frame {index} is missing teacher_energy")
        if "teacher_forces" not in atoms.arrays:
            raise ValueError(f"frame {index} is missing teacher_forces")
        energy = float(atoms.info["teacher_energy"])
        forces = np.asarray(atoms.arrays["teacher_forces"], dtype=float)
        if not math.isfinite(energy):
            raise ValueError(f"frame {index} teacher_energy is not finite")
        if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
            raise ValueError(f"frame {index} teacher_forces are invalid")
        stress = None
        if "teacher_stress" in atoms.info:
            raw = np.asarray(atoms.info["teacher_stress"], dtype=float)
            # SinglePointCalculator expects 6-vector Voigt stress
            if raw.shape == (3, 3):
                stress = np.array([raw[0, 0], raw[1, 1], raw[2, 2],
                                   raw[1, 2], raw[0, 2], raw[0, 1]])
            elif raw.shape == (6,):
                stress = raw
            else:
                raise ValueError(f"frame {index} teacher_stress has invalid shape {raw.shape}")
            if not np.all(np.isfinite(stress)):
                raise ValueError(f"frame {index} teacher_stress is not finite")
            stress_count += 1
        cleaned = atoms.copy()
        results = {"energy": energy, "forces": forces}
        if stress is not None:
            results["stress"] = stress
        cleaned.calc = SinglePointCalculator(cleaned, **results)
        materialized.append(cleaned)
    if require_stress and stress_count != len(materialized):
        raise ValueError(
            f"stress training requires teacher_stress on every frame: "
            f"{stress_count}/{len(materialized)} present"
        )
    write(str(target_path), materialized, format="extxyz")
    return len(materialized), stress_count == len(materialized)


def _write_structure_list(struct_list_path: Path, dataset_path: Path, tag: str, weights=None):
    """Write a SIMPLE-NN structure_list file.

    weights=None -> single tag over all frames (SIMPLE-NN default weight 1.0).
    weights=[w0,w1,...] -> ONE weighted tag PER FRAME ([tag-i : w_i] + '<dataset> i') so a
    per-structure struct_weight (e.g. C_SIZE_NORMALIZED_BOUNDED) actually reaches SIMPLE-NN's
    structure_weights mechanism instead of silently defaulting to 1.0.
    """
    d = dataset_path.resolve()
    if not weights:
        struct_list_path.write_text(f"[{tag}]\n{d} :\n")
        return
    # NOTE: use a SLICE 'i:i+1' (not bare 'i') so SIMPLE-NN's ase.io.read returns a LIST of one
    # Atoms; a bare integer index yields a single Atoms which SIMPLE-NN iterates into Atom objects
    # (AttributeError: 'Atom' has no 'cell').
    lines = [f"[{tag}-{i:06d} : {float(w):.6f}]\n{d} {i}:{i+1}\n" for i, w in enumerate(weights)]
    struct_list_path.write_text("".join(lines))


def _build_input_yaml(rendered: dict, args, descriptor_params: dict,
                      structure_list_path: Path,
                      valid_rate: float = 0.1) -> dict:
    """Merge the pre-rendered adapter template with SIMPLE-NN v2's required
    schema and the CLI seed/epoch/batch overrides.
    """
    payload = dict(rendered) if isinstance(rendered, dict) else {}

    # Explicit generate_features + preprocess + train_model — always all three
    # so the seed produces both feature files AND a trained checkpoint from the
    # extxyz dataset in one call.
    payload["generate_features"] = True
    payload["preprocess"] = True
    payload["train_model"] = True
    payload["random_seed"] = int(args.seed)

    payload["params"] = {element: str(Path(path).resolve())
                          for element, path in descriptor_params.items()}

    data = dict(payload.get("data") or {})
    data.setdefault("type", "symmetry_function")
    data["refdata_format"] = "extxyz"
    data["compress_outcar"] = False
    data["struct_list"] = str(structure_list_path.resolve())
    data["absolute_path"] = True
    data["read_force"] = True
    # read_stress: True only when the teacher pipeline stored stress; the
    # adapter's rendered use_stress flag is authoritative for the training-side
    # decision.
    nn_rendered = dict(payload.get("neural_network") or {})
    use_stress = bool(args.use_stress)
    data["read_stress"] = use_stress
    payload["data"] = data

    nn = dict(payload.get("neural_network") or {})
    nn["train"] = True
    nn["test"] = False
    nn["use_force"] = True
    nn["use_stress"] = use_stress
    nn["batch_size"] = int(args.batch_size)
    nn["total_epoch"] = int(args.epochs)
    nn["double_precision"] = (args.precision == "double")
    nn["stress_loss_weight"] = float(args.stress_loss_weight)
    # Ensure GPU/CPU is honored automatically. use_gpu default is True; if no
    # CUDA is available, SIMPLE-NN falls back to CPU.
    nn.setdefault("use_gpu", True)
    payload["neural_network"] = nn

    preprocessing = dict(payload.get("preprocessing") or {})
    preprocessing.setdefault("valid_rate", valid_rate)
    preprocessing.setdefault("shuffle", True)
    payload["preprocessing"] = preprocessing

    return payload


def main(argv=None):
    args = _parse_args(argv)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the adapter-rendered SIMPLE-NN input template.
    rendered_input_path = Path(args.config).resolve()
    if not rendered_input_path.is_file():
        raise FileNotFoundError(f"--config not found: {rendered_input_path}")
    rendered = yaml.safe_load(rendered_input_path.read_text()) or {}

    # 2. Parse descriptor-param overrides.
    descriptor_params = {}
    for spec in args.descriptor_param:
        if "=" not in spec:
            raise ValueError(f"--descriptor-param must be ELEMENT=path, got: {spec}")
        element, path = spec.split("=", 1)
        element = element.strip()
        path = path.strip()
        if not element or not path:
            raise ValueError(f"invalid --descriptor-param entry: {spec}")
        if element in descriptor_params:
            raise ValueError(f"duplicate --descriptor-param element: {element}")
        descriptor_params[element] = path
    if not descriptor_params:
        raise ValueError("at least one --descriptor-param is required")
    for element, path in descriptor_params.items():
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"descriptor param file for {element} is missing: {path}"
            )

    # 3. Materialize a labeled extxyz copy inside out_dir so ase.io.read sees
    #    reference labels through a SinglePointCalculator.
    labeled_dataset = out_dir / "labeled_dataset.extxyz"
    n_frames, has_stress = _dataset_to_extxyz_with_ref_labels(
        Path(args.dataset).resolve(), labeled_dataset, require_stress=args.use_stress
    )
    if n_frames == 0:
        raise RuntimeError("no frames were materialized for SIMPLE-NN training")

    # 4. Write structure_list pointing at the labeled dataset. Apply the per-structure
    #    struct_weight policy so it actually reaches SIMPLE-NN (not a silent 1.0 fallback).
    struct_list_path = out_dir / "structure_list"
    weights = None
    if args.struct_weight_policy == "c_size_normalized_bounded":
        import math
        from ase.io import read as _read
        _lab = _read(str(labeled_dataset), index=":")
        natoms = [len(a) for a in _lab]
        geo = math.exp(sum(math.log(1.0 / n) for n in natoms) / len(natoms))
        cap = math.sqrt(8.0)
        weights = [min(max((1.0 / n) / geo, 1.0 / cap), cap) for n in natoms]
        (out_dir / "struct_weights.json").write_text(json.dumps(
            {"policy": "c_size_normalized_bounded", "n": len(weights),
             "min": min(weights), "max": max(weights), "ratio": max(weights) / min(weights),
             "weights": [round(w, 6) for w in weights]}, indent=2) + "\n")
    _write_structure_list(struct_list_path, labeled_dataset,
                           tag=f"student-train-seed-{args.seed}", weights=weights)

    # 5. Compose the final SIMPLE-NN input.yaml with CLI-derived overrides.
    payload = _build_input_yaml(rendered, args, descriptor_params,
                                 struct_list_path)
    if has_stress and payload["neural_network"].get("use_stress"):
        # Only enable read_stress when both the dataset carries stress AND the
        # adapter told us to train on stress. Otherwise stay off — SIMPLE-NN
        # will raise if it expects stress and none is present.
        payload["data"]["read_stress"] = True
    else:
        payload["data"]["read_stress"] = False
        payload["neural_network"]["use_stress"] = False

    resolved_input_path = out_dir / "simple_nn_input.yaml"
    resolved_input_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    # 6. Run SIMPLE-NN v2 from inside out_dir so its side-effect artifacts
    #    (LOG, potential_saved_bestmodel, data/, total_list, ...) land there.
    previous_cwd = Path.cwd()
    try:
        os.chdir(out_dir)
        from simple_nn.simple_nn import run
        run(str(resolved_input_path))
    finally:
        os.chdir(previous_cwd)

    checkpoint = out_dir / "potential_saved_bestmodel"
    if not checkpoint.exists():
        raise RuntimeError(
            "SIMPLE-NN run finished without potential_saved_bestmodel; "
            "do not substitute an arbitrary epoch checkpoint"
        )

    print(f"[simple_nn_v2_wrapper] seed={args.seed} -> {checkpoint}")


if __name__ == "__main__":
    main()
