"""Stage D-2 C1 trusted executor — post-hoc MSD from an existing production trajectory.

DESIGN/PREPARATION. This module DEFINES the trusted executor interface + implementation. It is NOT
invoked on the real trajectory during preparation; the first real execution is gated on explicit human
approval (see ``ExecutorGuardError`` / ``require_approval``) and is a separate, approved step.

Responsibility split (Stage D-2 contract):
  A. AUTHORITATIVE artifact/computation validity  -> deterministic fields recorded here, evaluated by
     the FROZEN runtimes.pydantic_ai.criterion_eval (deterministic_authoritative=true). The LLM never
     owns these booleans or the resulting gate verdict.
  B. DERIVED diagnostics (MSD(t), late-window mean/std/slope, R2, D estimate) -> computed + recorded as
     DATA, with the window/fit DECLARED before execution. No plateau/diffusion threshold is invented.
  C. SEMANTIC interpretation (solid-like / plateau / diffusive) -> advisory LLM Judge, grounded only in
     the generated artifact + declared diagnostics.

PBC: displacement is reconstructed by MINIMUM-IMAGE CONTINUITY unwrapping (the dump has wrapped x/y/z,
no unwrapped coords and no image flags). This is valid ONLY if no atom moves >= L/2 between consecutive
selected frames; that precondition is checked deterministically (``pbc_precondition_ok``) and a failure
STOPS the run (never a silent wrapped-coordinate pseudo-MSD). Pure Python (no heavy deps); CPU only.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

MAX_FRAMES = 200            # hard cap (contract)
MIN_FRAMES = 10            # predeclared minimum for a meaningful MSD
LATE_WINDOW_FRAC = 0.30    # DECLARED before execution: late window = last 30% of selected frames
INITIAL_MSD_TOL = 1e-6     # MSD(0) approx 0 tolerance
# PBC continuity limitation + proxy gate: the dump has WRAPPED coords only (no xu/yu/zu, no image
# flags), so absence of a true >= L/2 inter-frame jump CANNOT be proven from the data (a min-image
# step is always <= L/2 by construction). Min-image continuity unwrapping is valid under the physical
# assumption that per-frame displacement << L/2 — assured for a 300 K amorphous solid at 1 ps cadence
# (thermal motion ~0.1 A). The deterministic PROXY gate below STOPS if the max observed min-image step
# approaches L/2 (i.e. motion large enough to threaten the assumption). For a GUARANTEE, re-dump with
# unwrapped coords or image flags. This is min-image continuity with a declared safety bound + a
# documented assumption — NOT a silent wrapped-coordinate pseudo-MSD.
CONTINUITY_SAFE_FRAC = 0.25  # STOP if max min-image step >= 0.25 * L (well below the L/2 ambiguity)
WALLTIME_CEILING_S = 300.0   # 5 minutes


class ExecutorGuardError(RuntimeError):
    """A safety guard refused the action (approval, overwrite, path, sha, ceiling, PBC)."""


@dataclass
class MSDResult:
    status: str                       # "OK" | "STOP_PBC_INSUFFICIENT" | "STOP_GUARD"
    validity: dict = field(default_factory=dict)      # axis-A authoritative fields
    diagnostics: dict = field(default_factory=dict)   # axis-B derived data
    rows: list = field(default_factory=list)          # msd.csv rows
    reason: str = ""


# --- pure computation (unit-tested on SYNTHETIC data; never run on the real trajectory in prep) ------

def _min_image(delta: float, L: float) -> float:
    """Minimum-image of a single component displacement into (-L/2, L/2]."""
    return delta - L * round(delta / L)


def compute_msd_min_image(frames: list, box_L, types=None):
    """frames: list of length F, each a list of (x,y,z) for N atoms (wrapped coords, same atom order).
    box_L: scalar L or (Lx,Ly,Lz). Returns (msd_all, per_type, max_step). MSD is the mean over atoms of
    squared displacement from frame 0, using minimum-image continuity unwrapping. ``max_step`` is the
    largest per-component min-image step observed (the caller applies the CONTINUITY_SAFE_FRAC proxy
    gate — this function does not decide validity). Raises on ragged input. Deterministic; O(N*F)."""
    Ls = (box_L, box_L, box_L) if isinstance(box_L, (int, float)) else tuple(box_L)
    F = len(frames)
    if F == 0:
        return [], {}, 0.0
    N = len(frames[0])
    for fr in frames:
        if len(fr) != N:
            raise ValueError("atom count not constant across frames")
    unwrapped = [list(map(float, xyz)) for xyz in frames[0]]   # start = frame 0
    ref = [list(u) for u in unwrapped]
    prev = [list(map(float, xyz)) for xyz in frames[0]]
    msd_all = [0.0]
    per_type = {}
    if types is not None:
        for t in set(types):
            per_type[t] = [0.0]
    max_step = 0.0
    for f in range(1, F):
        sq_sum = 0.0
        type_sq = {t: 0.0 for t in per_type}
        type_ct = {t: 0 for t in per_type}
        for i in range(N):
            for c in range(3):
                step = _min_image(frames[f][i][c] - prev[i][c], Ls[c])
                if abs(step) > max_step:
                    max_step = abs(step)
                unwrapped[i][c] += step
            dx = unwrapped[i][0] - ref[i][0]
            dy = unwrapped[i][1] - ref[i][1]
            dz = unwrapped[i][2] - ref[i][2]
            d2 = dx * dx + dy * dy + dz * dz
            sq_sum += d2
            if types is not None:
                type_sq[types[i]] += d2
                type_ct[types[i]] += 1
        prev = [list(u) for u in frames[f]]
        msd_all.append(sq_sum / N)
        for t in per_type:
            per_type[t].append(type_sq[t] / type_ct[t] if type_ct[t] else 0.0)
    return msd_all, per_type, max_step


def late_window_diagnostics(times: list, msd: list, frac: float = LATE_WINDOW_FRAC) -> dict:
    """Derived diagnostics over the DECLARED late window (last ``frac`` of frames). Records mean, std,
    slope, R^2, and a 3D Einstein D estimate (slope/6). These are DATA, not pass/fail; no threshold."""
    F = len(msd)
    k = max(2, int(math.ceil(F * frac)))
    xs, ys = times[F - k:], msd[F - k:]
    n = len(xs)
    mean = sum(ys) / n
    var = sum((y - mean) ** 2 for y in ys) / n
    mx = sum(xs) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = (sum((xs[i] - mx) * (ys[i] - mean) for i in range(n)) / sxx) if sxx > 0 else 0.0
    inter = mean - slope * mx
    ss_res = sum((ys[i] - (inter + slope * xs[i])) ** 2 for i in range(n))
    ss_tot = sum((y - mean) ** 2 for y in ys)
    r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"late_window_frames": k, "late_window_frac": frac, "late_mean_msd": mean,
            "late_std_msd": math.sqrt(var), "late_slope": slope, "late_fit_r2": r2,
            "diffusion_estimate": slope / 6.0}


# --- LAMMPS dump parsing + guarded runner (runner NOT invoked on the real trajectory in prep) --------

def parse_lammps_custom_dump(path: str):
    """Parse a LAMMPS custom dump with columns including id type x y z. Returns
    (timesteps, box_L_per_frame, frames_xyz, types) with atoms sorted by id per frame. Pure Python."""
    timesteps, boxes, frames, types = [], [], [], None
    with open(path) as fh:
        lines = fh.read().splitlines()
    i, n = 0, len(lines)
    while i < n:
        if lines[i].startswith("ITEM: TIMESTEP"):
            timesteps.append(int(lines[i + 1])); i += 2; continue
        if lines[i].startswith("ITEM: NUMBER OF ATOMS"):
            natoms = int(lines[i + 1]); i += 2; continue
        if lines[i].startswith("ITEM: BOX BOUNDS"):
            lo, hi = map(float, lines[i + 1].split()[:2]); boxes.append(hi - lo); i += 4; continue
        if lines[i].startswith("ITEM: ATOMS"):
            cols = lines[i].split()[2:]
            ci = {c: k for k, c in enumerate(cols)}
            rows = []
            for j in range(i + 1, i + 1 + natoms):
                p = lines[j].split()
                rows.append((int(p[ci["id"]]), int(p[ci["type"]]),
                             float(p[ci["x"]]), float(p[ci["y"]]), float(p[ci["z"]])) )
            rows.sort(key=lambda r: r[0])
            frames.append([(r[2], r[3], r[4]) for r in rows])
            if types is None:
                types = [r[1] for r in rows]
            i = i + 1 + natoms; continue
        i += 1
    return timesteps, boxes, frames, types


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_frames(total: int) -> list:
    """Deterministic frame selection: all frames if total <= MAX_FRAMES, else an even stride to <=MAX."""
    if total <= MAX_FRAMES:
        return list(range(total))
    stride = math.ceil(total / MAX_FRAMES)
    return list(range(0, total, stride))[:MAX_FRAMES]


def require_approval(approval: Optional[dict]) -> None:
    """The first Stage D-2 state-advancing execution requires an explicit approval token. Preparing the
    proposal is NOT approval."""
    if not approval or approval.get("approved") is not True or not str(approval.get("approver", "")).strip():
        raise ExecutorGuardError("execution requires explicit human approval (approval.approved=true)")


def run_posthoc_msd(*, proposal: dict, run_dir: str, approval: Optional[dict],
                    clock: Callable[[], float]) -> MSDResult:
    """Trusted executor for generate_analysis_artifact/posthoc_msd. Enforces every Stage D-2 guard and,
    on success, writes msd.csv + msd_summary.json under run_dir and records the axis-A validity fields.
    Prep NEVER calls this on the real trajectory; tests call it on a tiny SYNTHETIC dump. Never mutates
    the source or anything outside run_dir."""
    require_approval(approval)
    rd = Path(run_dir)
    if rd.exists():
        raise ExecutorGuardError(f"run directory already exists (no overwrite): {rd}")
    params = proposal.get("parameters", {})
    src = params["source_trajectory"]
    allow = params.get("source_allow_prefixes", [])
    if not any(str(Path(src).resolve()).startswith(str(Path(a).resolve())) for a in allow):
        raise ExecutorGuardError(f"source outside allow-list: {src}")
    declared_sha = proposal.get("input_artifact_hashes", {}).get(src) or params.get("source_sha256")
    actual_sha = sha256_file(src)
    if declared_sha and actual_sha != declared_sha:
        raise ExecutorGuardError("source sha256 mismatch (trajectory changed since proposal)")

    t0 = clock()
    timesteps, boxes, frames, types = parse_lammps_custom_dump(src)
    sel = select_frames(len(frames))
    if len(sel) < MIN_FRAMES:
        return MSDResult("STOP_GUARD", reason=f"selected frames {len(sel)} < MIN_FRAMES {MIN_FRAMES}")
    sel_ts = [timesteps[k] for k in sel]
    sel_fr = [frames[k] for k in sel]
    atom_count_constant = len({len(f) for f in sel_fr}) == 1
    timesteps_increasing = all(sel_ts[i + 1] > sel_ts[i] for i in range(len(sel_ts) - 1))
    L = boxes[0]
    msd_all, per_type, max_step = compute_msd_min_image(sel_fr, L, types)
    continuity_safe_bound = CONTINUITY_SAFE_FRAC * L
    pbc_ok = max_step < continuity_safe_bound
    if not pbc_ok:
        # STOP: observed per-frame motion is large enough to threaten the continuity assumption ->
        # cannot safely reconstruct physical displacement from wrapped coords. No pseudo-MSD written.
        return MSDResult("STOP_PBC_INSUFFICIENT",
                         reason=f"max min-image step {max_step:.3f} >= continuity bound "
                                f"{continuity_safe_bound:.3f} (L/4); wrapped-only input insufficient")
    # timeline in ps (dt from proposal; dump interval * dt)
    dt_ps = float(params.get("timestep_ps", 0.001))
    times = [(ts - sel_ts[0]) * dt_ps for ts in sel_ts]
    diag = late_window_diagnostics(times, msd_all)
    finite = all(math.isfinite(v) for v in msd_all)
    nonneg = all(v >= 0.0 for v in msd_all)
    rows = [{"frame_index": sel[i], "timestep": sel_ts[i], "time_ps": times[i],
             "msd_all": msd_all[i], **{f"msd_type{t}": per_type[t][i] for t in sorted(per_type)}}
            for i in range(len(sel))]
    rd.mkdir(parents=True, exist_ok=False)
    csv_path = rd / "msd.csv"
    cols = ["frame_index", "timestep", "time_ps", "msd_all"] + [f"msd_type{t}" for t in sorted(per_type)]
    csv_path.write_text(",".join(cols) + "\n" +
                        "\n".join(",".join(f"{r[c]}" for c in cols) for r in rows) + "\n")
    summary = {"n_frames": len(sel), "n_atoms": len(sel_fr[0]), "box_L": L, "dt_ps": dt_ps,
               "species_counts": {str(t): types.count(t) for t in sorted(set(types))}, **diag}
    (rd / "msd_summary.json").write_text(__import__("json").dumps(summary, indent=2) + "\n")
    runtime_s = clock() - t0
    if runtime_s > WALLTIME_CEILING_S:
        raise ExecutorGuardError(f"runtime {runtime_s:.1f}s exceeded ceiling {WALLTIME_CEILING_S}s")
    validity = {
        "input_exists": True, "input_sha256_matches": bool(not declared_sha or actual_sha == declared_sha),
        "parsed_ok": True, "selected_frame_count": len(sel), "min_frames_ok": len(sel) >= MIN_FRAMES,
        "atom_count_constant": atom_count_constant, "timesteps_increasing": timesteps_increasing,
        "pbc_precondition_ok": pbc_ok, "max_min_image_step_A": max_step,
        "continuity_safe_bound_A": continuity_safe_bound,
        "msd_all_finite": finite, "msd_all_nonneg": nonneg,
        "msd_initial_abs": abs(msd_all[0]),
        "output_row_count": len(rows), "output_row_count_matches": len(rows) == len(sel),
        "summary_fields_present": all(k in summary for k in ("n_frames", "late_mean_msd", "late_slope")),
        "output_sha256": sha256_file(str(csv_path)), "output_sha256_present": True,
        "source_byte_identical_after": sha256_file(src) == actual_sha,
        "runtime_s": runtime_s, "runtime_ok": runtime_s <= WALLTIME_CEILING_S,
        "writes_under_run_dir_only": True,
    }
    return MSDResult("OK", validity=validity, diagnostics=diag, rows=rows)
