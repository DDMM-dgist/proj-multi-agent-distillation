"""Stage D-2 C3 trusted executor — ONE teacher-model single-point (E/F) on one structure.

DESIGN/PREPARATION. Defines the trusted executor interface + all deterministic guards and artifact/
computation-validity recording. It is MODEL-AGNOSTIC: the actual Allegro/nequip forward pass is an
INJECTED callable ``forward_fn`` supplied only at approved execution time (real inference). Preparation
and tests never run the real model — ``forward_fn`` defaults to None (raises), and unit tests inject a
tiny synthetic forward_fn on a parsed structure. One structure, one forward pass, one GPU; NO training,
NO MD, NO DFT, NO scheduler, NO network.

Validity layers (kept separate, per the C3 contract):
  A. ARTIFACT/COMPUTATION VALIDITY (always authoritative): sha matches, parsed, atom count preserved,
     outputs finite, force shape N x 3, energy/E-per-atom/max|F| finite, hashes recorded, source+model
     unchanged, writes only under the fresh run dir.
  B. PHYSICAL VALIDITY (reused FROZEN SiO2 DFT-scale ranges, documented provenance): E_per_atom in
     [-11,-8] eV/atom and max|F| <= 50 eV/A — applicable because the Allegro teacher predicts DFT-scale
     SiO2 energies/forces (DFT-labeled AL cells sit at -9.4..-9.9 eV/atom, inside the range). Invalidating.
  C. SEMANTIC interpretation -> optional advisory Judge (deterministic_authoritative=false); never binds A/B.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Frozen SiO2 DFT-scale physical-validity bounds, REUSED from the Stage D-1 DFT-label criteria
# (examples/stage_d1_replay/criteria/d1-dft-*.json). Provenance documented in examples/stage_d2_c3/CONTRACT.md.
E_PER_ATOM_RANGE = (-11.0, -8.0)     # eV/atom
MAX_FORCE_BOUND = 50.0               # eV/Angstrom


class ExecutorGuardError(RuntimeError):
    """A safety guard refused the action (approval, overwrite, path, sha, shape, finite)."""


@dataclass
class TeacherSinglePointResult:
    status: str                       # "OK" | "STOP_GUARD"
    validity: dict = field(default_factory=dict)   # axis-A + axis-B fields
    artifact: dict = field(default_factory=dict)   # the E/F artifact
    reason: str = ""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_lammps_data(path: str):
    """Minimal LAMMPS 'atomic'-style data parser: returns (positions[N][3], types[N], box_L, n_atoms).
    Positions sorted by atom id. Pure Python; no ASE dependency for parsing/validation."""
    lines = Path(path).read_text().splitlines()
    n_atoms = None
    lo = hi = None
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.endswith("atoms") and len(s.split()) == 2:
            n_atoms = int(s.split()[0])
        if "xlo xhi" in s:
            lo, hi = float(s.split()[0]), float(s.split()[1])
        if s == "Atoms" or s.startswith("Atoms "):
            i += 1
            break
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    rows = []
    while i < len(lines):
        s = lines[i].strip()
        if not s or not s[0].isdigit():
            break
        p = s.split()
        rows.append((int(p[0]), int(p[1]), float(p[2]), float(p[3]), float(p[4])))
        i += 1
    rows.sort(key=lambda r: r[0])
    positions = [(r[2], r[3], r[4]) for r in rows]
    types = [r[1] for r in rows]
    box_L = (hi - lo) if (lo is not None and hi is not None) else None
    return positions, types, (n_atoms if n_atoms is not None else len(rows)), box_L


def require_approval(approval: Optional[dict]) -> None:
    if not approval or approval.get("approved") is not True or not str(approval.get("approver", "")).strip():
        raise ExecutorGuardError("execution requires explicit human approval (approval.approved=true)")


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def run_teacher_single_point(*, proposal: dict, run_dir: str, approval: Optional[dict],
                             forward_fn: Callable = None, clock: Callable[[], float]) -> TeacherSinglePointResult:
    """Trusted executor. Guards -> parse structure -> ONE injected teacher forward pass -> E/F artifact
    + validity. ``forward_fn(positions, types, box_L, type_symbol_map) -> (energy_eV, forces_eV_A[N][3])``
    is injected ONLY at approved execution (the real Allegro/nequip model). Prep/tests never run the
    real model. Never mutates the source/model or anything outside run_dir."""
    require_approval(approval)
    if forward_fn is None:
        raise ExecutorGuardError("no forward_fn injected — real teacher inference is a separate approved step")
    rd = Path(run_dir)
    if rd.exists():
        raise ExecutorGuardError(f"run directory already exists (no overwrite / idempotency): {rd}")
    params = proposal.get("parameters", {})
    src = params["source_structure"]
    model = params["teacher_model"]
    allow = params.get("read_allow_prefixes", [])

    def _allowed(p):
        return any(str(Path(p).resolve()).startswith(str(Path(a).resolve())) for a in allow)
    if not (_allowed(src) and _allowed(model)):
        raise ExecutorGuardError("source or model outside the allow-list")
    src_sha = sha256_file(src)
    model_sha = sha256_file(model)
    declared_src = proposal.get("input_artifact_hashes", {}).get(src) or params.get("source_sha256")
    declared_model = params.get("model_sha256")
    if declared_src and src_sha != declared_src:
        raise ExecutorGuardError("source structure sha256 mismatch")
    if declared_model and model_sha != declared_model:
        raise ExecutorGuardError("teacher model sha256 mismatch")

    t0 = clock()
    positions, types, n_atoms, box_L = parse_lammps_data(src)
    parsed_ok = (len(positions) == n_atoms and n_atoms > 0)
    expected_n = int(params.get("expected_n_atoms", n_atoms))
    type_symbol_map = params.get("type_symbol_map", {"1": "O", "2": "Si"})

    energy, forces = forward_fn(positions, types, box_L, type_symbol_map)   # the ONE forward pass
    runtime_s = clock() - t0

    forces = [list(map(float, f)) for f in (forces or [])]
    force_shape_ok = (len(forces) == n_atoms and all(len(f) == 3 for f in forces))
    forces_finite = force_shape_ok and all(_finite(c) for f in forces for c in f)
    energy_finite = _finite(energy)
    e_per_atom = (energy / n_atoms) if (energy_finite and n_atoms) else float("nan")
    e_per_atom_finite = _finite(e_per_atom)
    max_force = max((math.sqrt(sum(c * c for c in f)) for f in forces), default=float("nan")) if forces_finite else float("nan")
    max_force_finite = _finite(max_force)

    rd.mkdir(parents=True, exist_ok=False)
    import json
    forces_path = rd / "forces.csv"
    forces_path.write_text("id,fx,fy,fz\n" +
                           "\n".join(f"{i+1},{f[0]},{f[1]},{f[2]}" for i, f in enumerate(forces)) + "\n")
    artifact = {
        "source_structure": src, "source_sha256": src_sha,
        "teacher_model": model, "model_sha256": model_sha,
        "n_atoms": n_atoms, "composition": {type_symbol_map.get(str(t), str(t)): types.count(t) for t in sorted(set(types))},
        "predicted_total_energy_eV": energy if energy_finite else None,
        "energy_per_atom_eV": e_per_atom if e_per_atom_finite else None,
        "forces_artifact": "forces.csv", "max_force_eV_A": max_force if max_force_finite else None,
        "units": {"energy": "eV", "energy_per_atom": "eV/atom", "force": "eV/Angstrom", "length": "Angstrom"},
        "inference_metadata": {"cutoff_A": params.get("cutoff_A", 5.0), "type_symbol_map": type_symbol_map,
                               "one_forward_pass": True, "device": params.get("device", "cuda:selected"),
                               "runtime_s": runtime_s},
    }
    (rd / "teacher_ef.json").write_text(json.dumps(artifact, indent=2) + "\n")
    validity = {
        # A. artifact / computation validity
        "input_sha256_matches": bool(not declared_src or src_sha == declared_src),
        "model_sha256_matches": bool(not declared_model or model_sha == declared_model),
        "structure_parsed": parsed_ok, "atom_count_preserved": n_atoms == expected_n,
        "energy_finite": energy_finite, "energy_per_atom_finite": e_per_atom_finite,
        "forces_finite": forces_finite, "force_shape_is_Nx3": force_shape_ok,
        "max_force_finite": max_force_finite,
        "artifact_hashes_recorded": True,
        "source_model_unchanged": (sha256_file(src) == src_sha and sha256_file(model) == model_sha),
        "writes_under_run_dir_only": True,
        "ef_sha256": sha256_file(str(rd / "teacher_ef.json")),
        "forces_sha256": sha256_file(str(forces_path)),
        # B. physical validity (reused frozen SiO2 DFT-scale ranges; values for the criterion evaluator)
        "E_per_atom_eV": e_per_atom if e_per_atom_finite else None,
        "max_force_eV_A": max_force if max_force_finite else None,
        "runtime_s": runtime_s,
    }
    return TeacherSinglePointResult("OK", validity=validity, artifact=artifact)
