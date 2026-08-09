"""Stage D-2 C1 (post-hoc MSD) network-free preparation tests. Exercises the trusted executor + the
authoritative validity gate on SYNTHETIC tiny dumps in a temp dir — NEVER the real trajectory, NO
model, NO scheduler. Covers the 10 required contracts: approval, path allow-list, no-overwrite, frame
cap, wall-time/CPU, PBC precondition, source-SHA, output schema, deterministic criterion evaluation,
and no Stage-D1 mutation.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    import pydantic  # noqa: F401
    from runtimes.pydantic_ai import stage_d2_executor as EX
    from runtimes.pydantic_ai.criterion_eval import derive_severity, evaluate_criteria
    _HAS = True
except ImportError:  # pragma: no cover
    _HAS = False


def _write_dump(path, frames, types, L, ts=None):
    ts = ts or [1000 * i for i in range(len(frames))]
    N = len(types)
    out = []
    for f, fr in enumerate(frames):
        out += ["ITEM: TIMESTEP", str(ts[f]), "ITEM: NUMBER OF ATOMS", str(N),
                "ITEM: BOX BOUNDS pp pp pp", f"0 {L}", f"0 {L}", f"0 {L}",
                "ITEM: ATOMS id type x y z fx fy fz"]
        for i in range(N):
            x, y, z = fr[i]
            out.append(f"{i+1} {types[i]} {x} {y} {z} 0 0 0")
    Path(path).write_text("\n".join(out) + "\n")


def _solid_frames(nframes=12, L=10.0):
    import math
    fr = []
    for f in range(nframes):
        fr.append([(1.0 + 0.02 * math.sin(f), 1.0, 1.0), (6.0, 6.0 + 0.02 * math.cos(f), 6.0)])
    return fr, [1, 2]


def _proposal(src, run_dir, sha=None):
    return {"parameters": {"source_trajectory": src, "source_allow_prefixes": [str(Path(src).parent)],
                           "source_sha256": sha, "timestep_ps": 0.001, "run_dir": run_dir},
            "input_artifact_hashes": {src: sha} if sha else {}}


def _clock():
    t = {"n": 0.0}

    def c():
        t["n"] += 0.001
        return t["n"]
    return c


APPROVED = {"approved": True, "approver": "tester"}


@unittest.skipUnless(_HAS, "pydantic/executor not importable")
class StageD2PosthocMSDTests(unittest.TestCase):
    def _sha(self, p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    # 10. execution impossible without explicit approval
    def test_requires_explicit_approval(self):
        with tempfile.TemporaryDirectory() as d:
            src = f"{d}/traj.dump"; fr, ty = _solid_frames(); _write_dump(src, fr, ty, 10.0)
            for appr in (None, {}, {"approved": False}, {"approved": True}):  # last: no approver
                with self.assertRaises(EX.ExecutorGuardError):
                    EX.run_posthoc_msd(proposal=_proposal(src, f"{d}/run", self._sha(src)),
                                       run_dir=f"{d}/run", approval=appr, clock=_clock())

    # 1. path allow-list
    def test_source_must_be_in_allow_list(self):
        with tempfile.TemporaryDirectory() as d:
            src = f"{d}/traj.dump"; fr, ty = _solid_frames(); _write_dump(src, fr, ty, 10.0)
            prop = _proposal(src, f"{d}/run", self._sha(src))
            prop["parameters"]["source_allow_prefixes"] = ["/nonexistent/other"]
            with self.assertRaisesRegex(EX.ExecutorGuardError, "allow-list"):
                EX.run_posthoc_msd(proposal=prop, run_dir=f"{d}/run", approval=APPROVED, clock=_clock())

    # 2. no overwrite
    def test_refuses_existing_run_dir(self):
        with tempfile.TemporaryDirectory() as d:
            src = f"{d}/traj.dump"; fr, ty = _solid_frames(); _write_dump(src, fr, ty, 10.0)
            Path(f"{d}/run").mkdir()
            with self.assertRaisesRegex(EX.ExecutorGuardError, "no overwrite"):
                EX.run_posthoc_msd(proposal=_proposal(src, f"{d}/run", self._sha(src)),
                                   run_dir=f"{d}/run", approval=APPROVED, clock=_clock())

    # 6. source SHA verification
    def test_source_sha_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as d:
            src = f"{d}/traj.dump"; fr, ty = _solid_frames(); _write_dump(src, fr, ty, 10.0)
            with self.assertRaisesRegex(EX.ExecutorGuardError, "sha256 mismatch"):
                EX.run_posthoc_msd(proposal=_proposal(src, f"{d}/run", "deadbeef" * 8),
                                   run_dir=f"{d}/run", approval=APPROVED, clock=_clock())

    # 3. frame cap
    def test_frame_cap(self):
        self.assertEqual(len(EX.select_frames(151)), 151)
        self.assertLessEqual(len(EX.select_frames(1000)), EX.MAX_FRAMES)
        self.assertLessEqual(len(EX.select_frames(201)), EX.MAX_FRAMES)

    # 5. PBC method precondition: solid-like passes; a large jump STOPS (no pseudo-MSD written)
    def test_pbc_precondition(self):
        with tempfile.TemporaryDirectory() as d:
            src = f"{d}/ok.dump"; fr, ty = _solid_frames(14, 10.0); _write_dump(src, fr, ty, 10.0)
            r = EX.run_posthoc_msd(proposal=_proposal(src, f"{d}/run_ok", self._sha(src)),
                                   run_dir=f"{d}/run_ok", approval=APPROVED, clock=_clock())
            self.assertEqual(r.status, "OK"); self.assertTrue(r.validity["pbc_precondition_ok"])
            # a jump of ~0.4*L between two frames (> 0.25*L bound) -> STOP, no output written
            jump = [[(0.0, 0.0, 0.0)] for _ in range(12)]
            for f in range(12):
                jump[f] = [(0.0 if f % 2 == 0 else 4.0, 0.0, 0.0)]  # oscillate 0<->4 in L=10 (0.4L)
            src2 = f"{d}/jump.dump"; _write_dump(src2, jump, [1], 10.0)
            r2 = EX.run_posthoc_msd(proposal=_proposal(src2, f"{d}/run_j", self._sha(src2)),
                                    run_dir=f"{d}/run_j", approval=APPROVED, clock=_clock())
            self.assertEqual(r2.status, "STOP_PBC_INSUFFICIENT")
            self.assertFalse(Path(f"{d}/run_j").exists())    # no run dir / no pseudo-MSD

    # 7. output schema  +  9. writes only under run dir (no Stage-D1 / source mutation)
    def test_output_schema_and_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            src = f"{d}/traj.dump"; fr, ty = _solid_frames(15, 10.0); _write_dump(src, fr, ty, 10.0)
            before = self._sha(src)
            rd = f"{d}/run"
            r = EX.run_posthoc_msd(proposal=_proposal(src, rd, before), run_dir=rd,
                                   approval=APPROVED, clock=_clock())
            self.assertEqual(r.status, "OK")
            cols = Path(f"{rd}/msd.csv").read_text().splitlines()[0].split(",")
            self.assertEqual(cols[:4], ["frame_index", "timestep", "time_ps", "msd_all"])
            summ = json.loads(Path(f"{rd}/msd_summary.json").read_text())
            for k in ("n_frames", "n_atoms", "box_L", "late_mean_msd", "late_slope", "diffusion_estimate"):
                self.assertIn(k, summ)
            # source untouched; all writes under run dir
            self.assertEqual(self._sha(src), before)
            self.assertTrue(r.validity["source_byte_identical_after"])
            self.assertTrue(r.validity["writes_under_run_dir_only"])
            self.assertTrue(all(Path(p).resolve().is_relative_to(Path(rd).resolve())
                                for p in (f"{rd}/msd.csv", f"{rd}/msd_summary.json")))

    # 4. wall-time / CPU contract
    def test_walltime_ceiling(self):
        with tempfile.TemporaryDirectory() as d:
            src = f"{d}/traj.dump"; fr, ty = _solid_frames(12, 10.0); _write_dump(src, fr, ty, 10.0)
            slow = iter([0.0, EX.WALLTIME_CEILING_S + 1.0])  # start, end -> exceeds ceiling
            with self.assertRaisesRegex(EX.ExecutorGuardError, "ceiling"):
                EX.run_posthoc_msd(proposal=_proposal(src, f"{d}/run", self._sha(src)),
                                   run_dir=f"{d}/run", approval=APPROVED, clock=lambda: next(slow))

    # 8. deterministic criterion evaluation via the FROZEN criterion_eval on the recorded validity
    def test_validity_gate_via_frozen_criterion_eval(self):
        spec = json.loads((ROOT / "examples/stage_d2/criteria/posthoc_msd_validity.json").read_text())
        with tempfile.TemporaryDirectory() as d:
            src = f"{d}/traj.dump"; fr, ty = _solid_frames(15, 10.0); _write_dump(src, fr, ty, 10.0)
            r = EX.run_posthoc_msd(proposal=_proposal(src, f"{d}/run", self._sha(src)),
                                   run_dir=f"{d}/run", approval=APPROVED, clock=_clock())
            self.assertEqual(derive_severity(evaluate_criteria(r.validity, spec)), "PASS")
            self.assertEqual(derive_severity(evaluate_criteria(dict(r.validity, pbc_precondition_ok=False), spec)), "FAIL")
            self.assertEqual(derive_severity(evaluate_criteria(dict(r.validity, summary_fields_present=False), spec)), "REVISE")

    def test_proposal_validates_against_frozen_schema(self):
        from runtimes.pydantic_ai.actions import AnalystActionProposal
        prop = json.loads((ROOT / "examples/stage_d2/action_proposal.json").read_text())
        AnalystActionProposal(**prop)                       # raises on drift
        self.assertEqual(prop["action_type"], "summarize_md_stability")
        self.assertEqual(prop["parameters"]["subtype"], "posthoc_msd")
        # judge interpretation gate is advisory (semantic), not authoritative
        task = json.loads((ROOT / "examples/stage_d2/judge_interpretation_task.json").read_text())
        self.assertIs(task["context"]["deterministic_authoritative"], False)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
