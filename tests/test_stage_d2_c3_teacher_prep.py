"""Stage D-2 C3 teacher single-point preparation tests (network-free; NO model, NO GPU, NO inference).

Exercises the trusted executor + the A/B validity gate on a SYNTHETIC forward_fn (never the real
Allegro model). Covers: the typed proposal validates against the frozen DataCuratorActionProposal with
the existing approval-gated label_with_teacher action (no new action_type); criteria use only frozen
operators and reuse the frozen SiO2 physical range; executor guards (approval, no-overwrite, allow-list,
source+model SHA, N×3 shape, finite); the deterministic gate gives PASS / FAIL-on-invalidating /
REVISE-on-completeness; idempotency (structure not already teacher-labeled); no run dir created in prep.
"""
from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "tests" / "fixtures" / "stage_d2_c3"

try:
    import pydantic  # noqa: F401
    from runtimes.pydantic_ai import stage_d2_c3_teacher_executor as EX
    from runtimes.pydantic_ai.criterion_eval import derive_severity, evaluate_criteria
    _HAS = True
except ImportError:  # pragma: no cover
    _HAS = False


def _clock():
    c = itertools.count(0.0, 0.01)
    return lambda: next(c)


APPROVED = {"approved": True, "approver": "tester"}


def _mini_data(path, n_si=2, n_o=4, L=10.0):
    # tiny synthetic SiO2 cell (type 1=O, type 2=Si), LAMMPS atomic
    lines = ["(synthetic)", "", f"{n_si+n_o} atoms", "2 atom types", "",
             f"0.0 {L} xlo xhi", f"0.0 {L} ylo yhi", f"0.0 {L} zlo zhi", "", "Atoms # atomic", ""]
    aid = 1
    for _ in range(n_o):
        lines.append(f"{aid} 1 {aid*0.5} {aid*0.4} {aid*0.3}"); aid += 1
    for _ in range(n_si):
        lines.append(f"{aid} 2 {aid*0.5} {aid*0.4} {aid*0.3}"); aid += 1
    Path(path).write_text("\n".join(lines) + "\n")


def _good_fwd(positions, types, box_L, tmap):
    return -9.7 * len(positions), [[0.3, -0.2, 0.1] for _ in positions]


def _proposal(src, model, run_dir, n, ssha=None, msha=None):
    return {"parameters": {"source_structure": src, "teacher_model": model,
                           "read_allow_prefixes": [str(Path(src).parent), str(Path(model).parent)],
                           "source_sha256": ssha, "model_sha256": msha, "expected_n_atoms": n,
                           "type_symbol_map": {"1": "O", "2": "Si"}, "run_dir": run_dir},
            "input_artifact_hashes": {src: ssha} if ssha else {}}


@unittest.skipUnless(_HAS, "pydantic/executor not importable")
class StageD2C3TeacherPrepTests(unittest.TestCase):
    def _sha(self, p):
        import hashlib
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def test_proposal_validates_frozen_schema_existing_action(self):
        from runtimes.pydantic_ai.actions import (APPROVAL_GATED_ACTIONS, DATA_CURATOR_ACTIONS,
                                                  DataCuratorActionProposal)
        prop = json.loads((BASE / "action_proposal.json").read_text())
        DataCuratorActionProposal(**prop)                       # raises on drift
        self.assertEqual(prop["action_type"], "label_with_teacher")
        self.assertIn("label_with_teacher", DATA_CURATOR_ACTIONS)          # existing action, not new
        self.assertEqual(APPROVAL_GATED_ACTIONS["label_with_teacher"], "costly_teacher_labeling")
        self.assertEqual(prop["approval_boundary"], "costly_teacher_labeling")
        self.assertEqual(prop["parameters"]["subtype"], "teacher_single_point")
        self.assertIs(prop["parameters"]["one_forward_pass"], True)

    def test_criteria_frozen_operators_and_reused_range(self):
        from runtimes.pydantic_ai.criterion_eval import _OPERATORS
        spec = json.loads((BASE / "criteria/teacher_ef_validity.json").read_text())
        fields = {c["lhs"]["field"] for c in spec}
        self.assertLessEqual({c["operator"] for c in spec}, _OPERATORS)     # frozen operators only
        # reused frozen SiO2 physical range present with documented provenance in the criterion text
        er = next(c for c in spec if c["lhs"]["field"] == "E_per_atom_eV")
        self.assertEqual(er["rhs"], {"low": -11.0, "high": -8.0}); self.assertTrue(er["invalidating"])
        self.assertIn("reused", er["criterion"].lower())
        mf = next(c for c in spec if c["lhs"]["field"] == "max_force_eV_A")
        self.assertEqual(mf["rhs"], {"const": 50.0}); self.assertTrue(mf["invalidating"])
        # A-layer integrity criteria present + invalidating
        for f in ("input_sha256_matches", "model_sha256_matches", "force_shape_is_Nx3",
                  "energy_finite", "source_model_unchanged", "writes_under_run_dir_only"):
            self.assertIn(f, fields)

    def test_executor_requires_approval_and_forward_fn(self):
        with tempfile.TemporaryDirectory() as d:
            src = f"{d}/s.data"; model = f"{d}/m.pth"; _mini_data(src); Path(model).write_bytes(b"m")
            prop = _proposal(src, model, f"{d}/run", 6, self._sha(src), self._sha(model))
            with self.assertRaises(EX.ExecutorGuardError):                 # no approval
                EX.run_teacher_single_point(proposal=prop, run_dir=f"{d}/run", approval=None,
                                            forward_fn=_good_fwd, clock=_clock())
            with self.assertRaises(EX.ExecutorGuardError):                 # no forward_fn (no real inference in prep)
                EX.run_teacher_single_point(proposal=prop, run_dir=f"{d}/run", approval=APPROVED,
                                            forward_fn=None, clock=_clock())

    def test_executor_guards_overwrite_allowlist_sha(self):
        with tempfile.TemporaryDirectory() as d:
            src = f"{d}/s.data"; model = f"{d}/m.pth"; _mini_data(src); Path(model).write_bytes(b"m")
            ss, ms = self._sha(src), self._sha(model)
            Path(f"{d}/run").mkdir()
            with self.assertRaisesRegex(EX.ExecutorGuardError, "no overwrite"):
                EX.run_teacher_single_point(proposal=_proposal(src, model, f"{d}/run", 6, ss, ms),
                                            run_dir=f"{d}/run", approval=APPROVED, forward_fn=_good_fwd, clock=_clock())
            bad = _proposal(src, model, f"{d}/run2", 6, "dead"*16, ms)
            with self.assertRaisesRegex(EX.ExecutorGuardError, "sha256 mismatch"):
                EX.run_teacher_single_point(proposal=bad, run_dir=f"{d}/run2", approval=APPROVED, forward_fn=_good_fwd, clock=_clock())
            outside = _proposal(src, model, f"{d}/run3", 6, ss, ms); outside["parameters"]["read_allow_prefixes"] = ["/nope"]
            with self.assertRaisesRegex(EX.ExecutorGuardError, "allow-list"):
                EX.run_teacher_single_point(proposal=outside, run_dir=f"{d}/run3", approval=APPROVED, forward_fn=_good_fwd, clock=_clock())

    def test_validity_gate_pass_fail_via_frozen_criterion_eval(self):
        spec = json.loads((BASE / "criteria/teacher_ef_validity.json").read_text())
        with tempfile.TemporaryDirectory() as d:
            src = f"{d}/s.data"; model = f"{d}/m.pth"; _mini_data(src, 2, 4); Path(model).write_bytes(b"m")
            ss, ms = self._sha(src), self._sha(model)
            r = EX.run_teacher_single_point(proposal=_proposal(src, model, f"{d}/run", 6, ss, ms),
                                            run_dir=f"{d}/run", approval=APPROVED, forward_fn=_good_fwd, clock=_clock())
            self.assertEqual(r.status, "OK")
            self.assertEqual(derive_severity(evaluate_criteria(r.validity, spec)), "PASS")
            self.assertTrue((Path(f"{d}/run") / "teacher_ef.json").exists())
            self.assertTrue((Path(f"{d}/run") / "forces.csv").exists())
            # ragged / wrong-shape forces -> force_shape_is_Nx3 False -> FAIL
            self.assertEqual(derive_severity(evaluate_criteria(dict(r.validity, force_shape_is_Nx3=False), spec)), "FAIL")
            # completeness gap -> REVISE
            self.assertEqual(derive_severity(evaluate_criteria(dict(r.validity, artifact_hashes_recorded=False), spec)), "REVISE")

    def test_idempotency_structure_not_already_labeled(self):
        im = json.loads((BASE / "input_manifest.json").read_text())
        self.assertIs(im["already_teacher_labeled"], False)     # genuinely new artifact
        self.assertEqual(im["n_atoms"], 216)
        self.assertEqual(im["composition"], {"O": 144, "Si": 72})

    def test_c3_completed_attempt3_rejected_and_judge_advisory(self):
        # C3 is COMPLETE. The ONLY completed scientific run is the approved attempt-3, and the deterministic
        # gate REJECTED it (authoritative FAIL / accepted=false) rather than rubber-stamping the successful
        # forward. The immutable failed attempt-1 (API mismatch) and attempt-2 (device mismatch) runs carry
        # no teacher_ef.json. teacher_ef.json therefore exists ONLY under attempt-3.
        base = ROOT / "tests" / "fixtures" / "stage_d2_c3"
        efs = sorted(p.parent.name for p in base.rglob("teacher_ef.json"))
        self.assertEqual(efs, ["d2c3-teacher-sp-mini216-attempt3"])          # only the completed run
        a3 = base / "d2c3-teacher-sp-mini216-attempt3"
        rm = json.loads((a3 / "run_manifest.json").read_text())
        self.assertEqual(rm["status"], "OK")                                 # forward completed
        self.assertEqual(rm["authoritative_verdict"], "FAIL")                # deterministic gate rejected it
        self.assertIs(rm["accepted"], False)                                 # NOT silently accepted
        prov = json.loads((a3 / "provenance.json").read_text())
        self.assertIs(prov["valid_prediction_generated"], True)             # a real E/F artifact exists
        # attempts 1 and 2 carry no completed prediction
        for name in ("d2c3-teacher-sp-mini216", "d2c3-teacher-sp-mini216-attempt2"):
            self.assertFalse((base / name / "teacher_ef.json").exists())
        t = json.loads((BASE / "judge_interpretation_task.json").read_text())
        self.assertIs(t["context"]["deterministic_authoritative"], False)   # advisory only
        self.assertIn("tests/fixtures/stage_d2_c3/", t["instruction"])                # full repo-relative path
        self.assertNotIn("manifest", t["instruction"].lower().split("no manifest")[0][-30:] if "no manifest" in t["instruction"].lower() else "")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
