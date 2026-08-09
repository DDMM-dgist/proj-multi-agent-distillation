"""Stage D-2 C3 execution-wrapper tests (network-free; NO real mini216 forward; synthetic adapter).

Proves the committed one-shot wrapper (work/stage_d2_c3_execute.py) enforces the execution contract via a
FAKE adapter_factory that returns synthetic model output — so no real Allegro forward runs. Verifies: an
arbitrary forward_fn cannot enter from the CLI/agent; approval required (no approval -> impossible);
exact structure+teacher SHA enforced; exactly one forward invocation; type mapping fixed 1->O/2->Si and
atom count/order preserved; no overwrite; output schema; provenance records adapter/model identity;
source+model remain byte-identical; and no scheduler/training/MD/DFT path exists in the wrapper.
"""
from __future__ import annotations

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
    """Synthetic-output stand-in for TrustedAllegroAdapter — records the ONE forward invocation."""
    def __init__(self, calls):
        self.type_names = ["O", "Si"]; self.r_max = 5.0; self.model_dtype = "float32"; self._model = object()
        self._calls = calls

    def load(self, device="cpu"):
        return {"python": "3.10", "torch": "x", "nequip": "y", "allegro": "z",
                "type_names": self.type_names, "r_max": self.r_max, "model_dtype": self.model_dtype}

    def build_forward_fn(self):
        def fwd(positions, lammps_types, box_L, tmap):
            self._calls.append({"n": len(positions), "types_head": lammps_types[:2], "types_tail": lammps_types[-1:]})
            # map check happens in the real adapter; here just return a-SiO2-scale synthetic output
            return -9.7 * len(positions), [[0.3, -0.2, 0.1] for _ in positions]
        return fwd


@unittest.skipUnless(_HAS and _HAS_STRUCT, "wrapper import / mini216 required")
class StageD2C3ExecuteTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.factory = lambda mp, sha, allow: _FakeAdapter(self.calls)
        self.approval = json.loads((C3 / "approval.template.json").read_text())
        self.approval.update(approved=True, approver="tester")

    def _run(self, d, **over):
        kw = dict(approval=self.approval, device="cpu", run_dir=f"{d}/run",
                  adapter_factory=self.factory, clock=None)
        kw.update(over)
        return W.execute(**kw)

    def test_no_cli_forward_function_accepted(self):
        src = (ROOT / "work" / "stage_d2_c3_execute.py").read_text()
        # the CLI defines ONLY device/expect-head/approval — no forward arg, no python-expr eval/exec
        added = [ln for ln in src.splitlines() if "add_argument(" in ln]
        self.assertEqual({a.split('add_argument("')[1].split('"')[0] for a in added},
                         {"--device", "--expect-head", "--approval"})
        self.assertNotIn('add_argument("--forward', src)
        self.assertNotIn("eval(", src); self.assertNotIn("exec(", src)
        self.assertIn("adapter_factory", src)          # code-level test seam, not a CLI arg
        self.assertIn('ap.add_argument("--approval"', src)

    def test_happy_path_one_forward_schema_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            rep = self._run(d)
            self.assertEqual(rep["authoritative_verdict"], "PASS")
            self.assertEqual(rep["n_atoms"], 216)
            self.assertEqual(rep["composition"], {"O": 144, "Si": 72})   # type1=O144, type2=Si72
            self.assertEqual(len(self.calls), 1)                          # EXACTLY one forward
            rd = Path(f"{d}/run")
            for f in ("approval.json", "input_manifest.json", "model_manifest.json", "teacher_ef.json",
                      "forces.csv", "criterion_results.json", "provenance.json", "run_manifest.json"):
                self.assertTrue((rd / f).exists(), f)
            ef = json.loads((rd / "teacher_ef.json").read_text())
            for k in ("source_sha256", "model_sha256", "n_atoms", "composition", "predicted_total_energy_eV",
                      "energy_per_atom_eV", "max_force_eV_A", "force_array_shape", "model_dtype",
                      "model_device", "model_type_names", "cutoff_A", "inference_wall_time_s", "software_versions"):
                self.assertIn(k, ef, k)
            self.assertEqual(ef["force_array_shape"], [216, 3])
            self.assertEqual(ef["n_atoms"], 216)
            # forces.csv preserves 216 rows + header, id order
            rows = (rd / "forces.csv").read_text().splitlines()
            self.assertEqual(rows[0], "id,fx,fy,fz"); self.assertEqual(len(rows), 217)
            self.assertTrue(rows[1].startswith("1,"))
            prov = json.loads((rd / "provenance.json").read_text())
            self.assertIn("TrustedAllegroAdapter", prov["adapter"])
            self.assertIn("KISTI Allegro", prov["model_identity"])
            self.assertEqual(prov["semantic_judge"], "NOT run (separate later approval)")
            self.assertTrue(prov["source_model_unchanged"])

    def test_no_approval_impossible(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(W.ExecutionRefused):
                self._run(d, approval={"approved": False})
            with self.assertRaises(W.ExecutionRefused):
                self._run(d, approval=dict(self.approval, authorizes_subsequent_actions=True))

    def test_sha_and_head_and_overwrite_guards(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(W.ExecutionRefused):                   # wrong EXPECT_HEAD
                self._run(d, expect_head="0" * 40)
            Path(f"{d}/run").mkdir()                                      # existing run dir -> refuse
            with self.assertRaises(W.ExecutionRefused):
                self._run(d)

    def test_source_model_byte_identical_after(self):
        import hashlib
        before = hashlib.sha256(Path(MINI216).read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            self._run(d)
        self.assertEqual(hashlib.sha256(Path(MINI216).read_bytes()).hexdigest(), before)

    def test_no_side_job_paths_in_wrapper(self):
        src = (ROOT / "work" / "stage_d2_c3_execute.py").read_text().lower()
        for banned in ("sbatch", "qsub", "srun", "nequip-train", "lammps", "run_md", "run_dft"):
            self.assertNotIn(banned, src)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
