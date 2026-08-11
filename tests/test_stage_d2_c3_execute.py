"""Stage D-2 C3 execution-wrapper tests — EXTERNAL-approval flow + failure atomicity (network-free; NO
real mini216 forward; synthetic adapter).

Proves the committed one-shot wrapper enforces: external approval required (approved=false / wrong
action/subtype / wrong SHA / wrong limits / authorizes_subsequent=true all refuse); run dir must not
pre-exist; the fresh run dir is created + the approval snapshotted ONLY after validation; the external
approval is never modified and its SHA is recorded; exactly one forward; no arbitrary CLI/agent
forward_fn; source+model immutable; failures BEFORE vs AFTER the forward are distinguishable; no auto
retry; and no scheduler/training/MD/DFT path. Synthetic model output only.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C3 = ROOT / "examples" / "stage_d2_c3"
sys.path.insert(0, str(ROOT / "work"))

RES = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation"
MINI216 = f"{RES}/teacher_diag/nve_drift/mini216_nvt_fixed.data"

try:
    import pydantic  # noqa: F401
    import stage_d2_c3_execute as W
    _HAS = True
except ImportError:  # pragma: no cover
    _HAS = False
_HAS_STRUCT = Path(MINI216).is_file()


class _FakeAdapter:
    """Mirrors the real adapter's forward-phase contract: build_forward_fn resets forward_invoked/
    forward_completed; forward_fn sets forward_invoked=True immediately BEFORE the (synthetic) model call
    and forward_completed=True after a successful return. ``raise_before_model`` fails in the PRE_FORWARD
    input-build stage (invoked stays False); ``raise_in_model`` fails AFTER invocation begins (invoked
    True, completed False) — the attempt-2 DURING_FORWARD device-mismatch class."""

    def __init__(self, calls, raise_before_model=False, raise_in_model=False):
        self.type_names = ["O", "Si"]; self.r_max = 5.0; self.model_dtype = "float32"; self._model = object()
        self._calls = calls
        self._raise_before = raise_before_model
        self._raise_in = raise_in_model
        self.forward_invoked = False
        self.forward_completed = False

    def load(self, device="cpu"):
        return {"python": "3.10", "torch": "x", "nequip": "y", "allegro": "?",
                "type_names": self.type_names, "r_max": self.r_max, "model_dtype": self.model_dtype}

    def build_forward_fn(self):
        self.forward_invoked = False
        self.forward_completed = False

        def fwd(positions, lammps_types, box_L, tmap):
            if self._raise_before:                      # PRE_FORWARD (input build) failure
                raise RuntimeError("synthetic input-build failure")
            self.forward_invoked = True                 # FORWARD_STARTED
            if self._raise_in:                          # DURING_FORWARD (in-model device mismatch)
                raise RuntimeError("synthetic in-model device mismatch (cuda:1 vs cpu)")
            self._calls.append(len(positions))
            self.forward_completed = True               # FORWARD_COMPLETED
            return -9.7 * len(positions), [[0.3, -0.2, 0.1] for _ in positions]
        return fwd


def _approval(**over):
    a = json.loads((C3 / "approvals" / "d2c3-teacher-sp-mini216.approval.json").read_text())
    a.update(approved=True, approver="tester")
    a.update(over)
    return a


@unittest.skipUnless(_HAS and _HAS_STRUCT, "wrapper import / mini216 required")
class StageD2C3ExecuteTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.factory = lambda mp, sha, allow: _FakeAdapter(self.calls)

    def _write_approval(self, d, approval):
        Path(d).mkdir(parents=True, exist_ok=True)
        p = Path(d) / "external_approval.json"; p.write_text(json.dumps(approval, indent=2) + "\n"); return p

    def _run(self, d, approval=None, **over):
        ap = self._write_approval(d, approval if approval is not None else _approval())
        # torch/nequip are absent in this test env, so inject a PASSING env_check by default; the
        # env-failure path is tested explicitly below with a failing env_check.
        kw = dict(approval_path=str(ap), device="cpu", run_dir=f"{d}/run",
                  adapter_factory=self.factory, env_check=lambda **k: (True, {"stub": "ok"}))
        kw.update(over)
        return W.execute(**kw)

    def test_env_preflight_fails_closed_before_run_dir(self):
        with tempfile.TemporaryDirectory() as d:
            # simulate the launch-attempt-1 class: missing pydantic -> env preflight not ok
            fail = lambda **k: (False, {"pydantic": "FAIL:ModuleNotFoundError"})  # noqa: E731
            with self.assertRaisesRegex(W.ExecutionRefused, "PRE_EXECUTION_IMPORT"):
                self._run(d, env_check=fail)
            self.assertFalse(Path(f"{d}/run").exists())     # no run dir created
            self.assertEqual(len(self.calls), 0)            # no forward call
            self.assertFalse((Path(f"{d}/run") / "teacher_ef.json").exists())

    def test_real_env_preflight_does_no_forward_and_fail_closes(self):
        # the committed preflight is import/load-only and returns a boolean; it never runs a forward
        import stage_d2_c3_env_preflight as P
        ok, checks = P.check_env(device="cpu")
        self.assertIn("pydantic", checks)
        self.assertIsInstance(ok, bool)                     # torch/nequip may be absent here -> ok False
        src = (ROOT / "work" / "stage_d2_c3_env_preflight.py").read_text()
        self.assertNotIn("run_teacher_single_point(", src)  # preflight invokes no forward
        self.assertNotIn("build_forward_fn(", src)

    def test_no_cli_forward_function_accepted(self):
        src = (ROOT / "work" / "stage_d2_c3_execute.py").read_text()
        added = {ln.split('add_argument("')[1].split('"')[0] for ln in src.splitlines() if 'add_argument("' in ln}
        self.assertEqual(added, {"--device", "--expect-head", "--approval", "--attempt"})
        self.assertNotIn("--forward", added)
        self.assertNotIn("eval(", src); self.assertNotIn("exec(", src)
        self.assertIn("adapter_factory", src)

    def test_happy_path_snapshot_and_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            ext = self._write_approval(d, _approval())
            ext_sha = hashlib.sha256(ext.read_bytes()).hexdigest()
            rep = W.execute(approval_path=str(ext), device="cpu", run_dir=f"{d}/run",
                            adapter_factory=self.factory, env_check=lambda **k: (True, {"stub": "ok"}))
            self.assertEqual(rep["authoritative_verdict"], "PASS")
            self.assertEqual(rep["n_atoms"], 216); self.assertEqual(rep["composition"], {"O": 144, "Si": 72})
            self.assertEqual(len(self.calls), 1)                       # exactly one forward
            rd = Path(f"{d}/run")
            for f in ("approval.json", "input_manifest.json", "model_manifest.json", "teacher_ef.json",
                      "forces.csv", "criterion_results.json", "provenance.json", "run_manifest.json"):
                self.assertTrue((rd / f).exists(), f)
            # approval snapshotted into run dir; external approval unchanged; sha recorded
            self.assertEqual(json.loads((rd / "approval.json").read_text())["approved"], True)
            self.assertEqual(hashlib.sha256(ext.read_bytes()).hexdigest(), ext_sha)   # external unchanged
            prov = json.loads((rd / "provenance.json").read_text())
            self.assertEqual(prov["source_approval"]["external_approval_sha256"], ext_sha)
            self.assertEqual(prov["source_approval"]["external_approval_path"], str(ext))
            self.assertEqual(prov["semantic_judge"], "NOT run (separate later approval)")
            ef = json.loads((rd / "teacher_ef.json").read_text())
            for k in ("source_sha256", "model_sha256", "n_atoms", "composition", "predicted_total_energy_eV",
                      "energy_per_atom_eV", "max_force_eV_A", "force_array_shape", "model_dtype",
                      "model_device", "model_type_names", "cutoff_A", "inference_wall_time_s", "software_versions"):
                self.assertIn(k, ef, k)
            self.assertEqual(ef["force_array_shape"], [216, 3])
            rows = (rd / "forces.csv").read_text().splitlines()
            self.assertEqual(rows[0], "id,fx,fy,fz"); self.assertEqual(len(rows), 217); self.assertTrue(rows[1].startswith("1,"))

    def test_approval_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            for bad in (_approval(approved=False),
                        _approval(approver=""),
                        _approval(action="run_dft"),
                        _approval(subtype="something_else"),
                        _approval(structure_sha256="0" * 64),
                        _approval(teacher_sha256="0" * 64),
                        _approval(authorizes_subsequent_actions=True)):
                with self.assertRaises(W.ExecutionRefused):
                    self._run(d + "/x", approval=bad, run_dir=f"{d}/x/run")
            # limit violations
            base = _approval()
            for limover in ({"structures": 2}, {"forward_passes": 2}, {"gpus": 2},
                            {"scientific_inference_wall_time_s_max": 120}, {"no_scheduler": False}):
                bad = _approval(); bad["limits"] = dict(base["limits"], **limover)
                with self.assertRaises(W.ExecutionRefused):
                    self._run(d + "/y", approval=bad, run_dir=f"{d}/y/run")

    def test_fresh_dir_only_after_validation(self):
        with tempfile.TemporaryDirectory() as d:
            # invalid approval -> run dir must NOT be created
            with self.assertRaises(W.ExecutionRefused):
                self._run(d, approval=_approval(approved=False), run_dir=f"{d}/run")
            self.assertFalse(Path(f"{d}/run").exists())
            # pre-existing run dir -> refuse (fresh-run guard not weakened)
            Path(f"{d}/run2").mkdir()
            with self.assertRaises(W.ExecutionRefused):
                self._run(d, run_dir=f"{d}/run2")

    def test_expect_head_guard(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(W.ExecutionRefused):
                self._run(d, expect_head="0" * 40)

    def test_source_model_byte_identical_after(self):
        before = hashlib.sha256(Path(MINI216).read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            self._run(d)
        self.assertEqual(hashlib.sha256(Path(MINI216).read_bytes()).hexdigest(), before)

    def test_failure_before_forward_distinguishable(self):
        # PRE_FORWARD: the model was never invoked (input-build stage fails) -> BEFORE_FORWARD.
        with tempfile.TemporaryDirectory() as d:
            self.factory = lambda mp, sha, allow: _FakeAdapter(self.calls, raise_before_model=True)
            with self.assertRaises(W.ExecutionRefused):
                self._run(d, run_dir=f"{d}/run")
            rm = json.loads((Path(f"{d}/run") / "run_manifest.json").read_text())
            self.assertEqual(rm["status"], "EXECUTION_FAILED_BEFORE_FORWARD")
            self.assertFalse(rm["model_forward_invoked"]); self.assertFalse(rm["model_forward_completed"])
            self.assertFalse(rm["valid_prediction_generated"])
            self.assertIsNone(rm["scientific_verdict"]); self.assertFalse(rm["automatic_retry"])
            self.assertFalse((Path(f"{d}/run") / "teacher_ef.json").exists())

    def test_failure_during_forward_classified(self):
        # DURING_FORWARD (attempt-2's class): the model forward was INVOKED but did not complete; no
        # teacher_ef.json is written, yet it must NOT be mislabeled BEFORE_FORWARD.
        with tempfile.TemporaryDirectory() as d:
            self.factory = lambda mp, sha, allow: _FakeAdapter(self.calls, raise_in_model=True)
            with self.assertRaisesRegex(W.ExecutionRefused, "during forward"):
                self._run(d, run_dir=f"{d}/run")
            rm = json.loads((Path(f"{d}/run") / "run_manifest.json").read_text())
            self.assertEqual(rm["status"], "EXECUTION_FAILED_DURING_FORWARD")
            self.assertTrue(rm["model_forward_invoked"])          # forward WAS invoked
            self.assertFalse(rm["model_forward_completed"])       # but did not complete
            self.assertFalse(rm["valid_prediction_generated"])    # no valid prediction
            self.assertFalse(rm["forward_pass_completed"])
            self.assertIsNone(rm["scientific_verdict"]); self.assertFalse(rm["automatic_retry"])
            self.assertFalse((Path(f"{d}/run") / "teacher_ef.json").exists())
            self.assertEqual(len(self.calls), 0)                  # no successful forward recorded

    def test_classify_failure_state_machine(self):
        # pure state-machine mapping
        self.assertEqual(W.classify_failure(forward_invoked=False, forward_completed=False, artifact_exists=False),
                         "EXECUTION_FAILED_BEFORE_FORWARD")
        self.assertEqual(W.classify_failure(forward_invoked=True, forward_completed=False, artifact_exists=False),
                         "EXECUTION_FAILED_DURING_FORWARD")
        self.assertEqual(W.classify_failure(forward_invoked=True, forward_completed=True, artifact_exists=False),
                         "EXECUTION_FAILED_AFTER_FORWARD")
        self.assertEqual(W.classify_failure(forward_invoked=True, forward_completed=False, artifact_exists=True),
                         "EXECUTION_FAILED_AFTER_FORWARD")

    def test_failure_after_forward_marker(self):
        # unit: the failure recorder records the explicit completed counter for AFTER-forward failures
        with tempfile.TemporaryDirectory() as d:
            rd = Path(d) / "run"; rd.mkdir(); (rd / "teacher_ef.json").write_text("{}")
            W._write_failure(rd, "HEAD", {"external_approval_path": "x", "external_approval_sha256": "y"},
                             "EXECUTION_FAILED_AFTER_FORWARD", "post-forward error",
                             model_forward_invoked=True, model_forward_completed=True,
                             valid_prediction_generated=False)
            rm = json.loads((rd / "run_manifest.json").read_text())
            self.assertEqual(rm["status"], "EXECUTION_FAILED_AFTER_FORWARD")
            self.assertTrue(rm["model_forward_invoked"]); self.assertTrue(rm["model_forward_completed"])
            self.assertTrue(rm["forward_pass_completed"]); self.assertFalse(rm["automatic_retry"])

    def test_no_side_job_paths(self):
        src = (ROOT / "work" / "stage_d2_c3_execute.py").read_text().lower()
        for banned in ("sbatch", "qsub", "srun", "nequip-train", "lammps", "run_md", "run_dft"):
            self.assertNotIn(banned, src)

    def test_attempt_identity_and_immutable_attempts_1_2(self):
        # deterministic attempt identity
        self.assertEqual(W.run_id_for(1), "d2c3-teacher-sp-mini216")
        self.assertEqual(W.run_id_for(2), "d2c3-teacher-sp-mini216-attempt2")
        self.assertEqual(W.run_id_for(3), "d2c3-teacher-sp-mini216-attempt3")
        with tempfile.TemporaryDirectory() as d:
            # attempts 1 and 2 (the immutable failed identities) are refused for scientific execution
            for att in (1, 2):
                with self.assertRaisesRegex(W.ExecutionRefused, "immutable failed run"):
                    self._run(d, attempt=att, run_dir=f"{d}/run")
            # targeting either immutable run directory is refused
            for name in ("d2c3-teacher-sp-mini216", "d2c3-teacher-sp-mini216-attempt2"):
                with self.assertRaisesRegex(W.ExecutionRefused, "immutable failed attempt"):
                    self._run(d, run_dir=str(ROOT / "runs" / "stage_d2_c3" / name))
        # committed failed attempt-1 run: immutable evidence, BEFORE_FORWARD, no teacher_ef.json
        a1 = ROOT / "runs" / "stage_d2_c3" / "d2c3-teacher-sp-mini216"
        if a1.is_dir():
            self.assertEqual(json.loads((a1 / "run_manifest.json").read_text())["status"],
                             "EXECUTION_FAILED_BEFORE_FORWARD")
            self.assertFalse((a1 / "teacher_ef.json").exists())

    def test_attempt2_immutable_raw_and_additive_correction(self):
        a2 = ROOT / "runs" / "stage_d2_c3" / "d2c3-teacher-sp-mini216-attempt2"
        if not a2.is_dir():
            self.skipTest("attempt-2 run not present")
        # raw provenance preserved byte-identically with the ORIGINAL coarse classification
        raw = json.loads((a2 / "run_manifest.json").read_text())
        self.assertEqual(raw["status"], "EXECUTION_FAILED_BEFORE_FORWARD")   # the coarse original, untouched
        self.assertIn("cuda:1 and cpu", raw["reason"])                       # model WAS entered (device mismatch)
        self.assertFalse((a2 / "teacher_ef.json").exists())
        self.assertFalse((a2 / "criterion_results.json").exists())
        # the correction is ADDITIVE (separate file), not a rewrite of the raw record
        corr = json.loads((a2 / "CORRECTED_INTERPRETATION.json").read_text())["CORRECTED_CLASSIFICATION"]
        self.assertEqual(corr["failure_class"], "EXECUTION_FAILED_DURING_FORWARD")
        self.assertEqual(corr["ROOT_CAUSE"], "MODEL_INPUT_OR_BUFFER_DEVICE_MISMATCH_CPU_VS_CUDA1")
        self.assertEqual(corr["model_forward_invocation_count"], 1)
        self.assertEqual(corr["valid_teacher_prediction_count"], 0)
        self.assertIs(corr["model_forward_completed"], False)

    def _assert_prepared_approval(self, path, attempt):
        ap = json.loads(Path(path).read_text())
        self.assertIs(ap["approved"], False)               # prepared, not active
        self.assertEqual(ap["attempt"], attempt)
        self.assertEqual(ap["action"], "label_with_teacher")
        self.assertEqual(ap["subtype"], "teacher_single_point")
        self.assertIs(ap["authorizes_subsequent_actions"], False)
        return ap

    def test_attempt2_external_approval_prepared_not_active(self):
        ap = self._assert_prepared_approval(
            ROOT / "examples/stage_d2_c3/approvals/d2c3-teacher-sp-mini216-attempt2.approval.json", 2)
        self.assertIn("d2c3-teacher-sp-mini216", ap["supersedes"]["attempt1_run"])

    def test_attempt3_external_approval_prepared_and_references_1_2(self):
        ap = self._assert_prepared_approval(
            ROOT / "examples/stage_d2_c3/approvals/d2c3-teacher-sp-mini216-attempt3.approval.json", 3)
        sup = ap["supersedes"]
        self.assertIn("NEQUIP_0_16_1_ATOMICDATADICT_API_MISMATCH", sup["attempt1_root_cause"])
        self.assertIn("MODEL_INPUT_OR_BUFFER_DEVICE_MISMATCH_CPU_VS_CUDA1", sup["attempt2_root_cause"])
        self.assertEqual(sup["attempt2_model_forward_invocation_count"], 1)
        self.assertEqual(sup["attempt2_valid_teacher_prediction_count"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
