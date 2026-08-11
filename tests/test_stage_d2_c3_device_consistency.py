"""Stage D-2 C3 device-consistency preflight tests (network-free; NO GPU; NO model forward).

The attempt-2 scientific execution invoked the deployed model and died inside edge normalization with
``Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cpu``. These
tests prove the new device-consistency preflight (a) is a pure no-forward inspection that would have
detected that mismatch, and (b) its detection logic flags a CPU-model+CUDA-input and a CUDA-model+
CPU-input as inconsistent while passing fully-consistent placement. The pure logic is exercised without
torch; the real inspection runs on the GPU host.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "harness"))

try:
    from runtimes.pydantic_ai.stage_d2_c3_teacher_adapter import device_consistency_report
    _HAS = True
except ImportError:  # pragma: no cover
    _HAS = False


@unittest.skipUnless(_HAS, "adapter import failed")
class StageD2C3DeviceConsistencyTests(unittest.TestCase):
    def test_preflight_source_makes_no_model_call(self):
        src = (ROOT / "tests" / "harness" / "stage_d2_c3_device_consistency_preflight.py").read_text()
        # inspects devices, never invokes the model or any forward path
        self.assertNotIn("run_teacher_single_point(", src)
        self.assertNotIn("build_forward_fn(", src)
        self.assertNotIn("_model(", src)                    # no forward invocation anywhere
        self.assertIn("model_device_report", src)
        self.assertIn("input_device_report", src)
        self.assertIn("device_consistency_report", src)
        self.assertIn('"model_forward_called": False', src)

    def test_detects_attempt2_style_mismatch(self):
        # model on cuda:1, an input tensor left on cpu -> inconsistent (attempt-2's failure)
        r = device_consistency_report(["cuda:1"], ["cuda:1", "cpu"], "cuda:1")
        self.assertFalse(r["ok"]); self.assertTrue(r["mixed"]); self.assertEqual(r["input_offenders"], ["cpu"])

    def test_detects_model_buffer_left_on_cpu(self):
        # a required model buffer on cpu while inputs are on cuda:1 -> inconsistent
        r = device_consistency_report(["cuda:1", "cpu"], ["cuda:1"], "cuda:1")
        self.assertFalse(r["ok"]); self.assertEqual(r["model_offenders"], ["cpu"])

    def test_consistent_placement_passes(self):
        self.assertTrue(device_consistency_report(["cuda:1"], ["cuda:1"], "cuda:1")["ok"])
        self.assertTrue(device_consistency_report(["cpu"], ["cpu"], "cpu")["ok"])

    def test_preflight_runs_on_cpu_if_torch_present(self):
        try:
            import torch  # noqa: F401
            import nequip  # noqa: F401
        except ImportError:
            self.skipTest("torch/nequip absent (real device-consistency verified in the allegro env)")
        if not Path("/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/"
                    "research-sio2-allegro-simplenn-distillation/gpu_finetune_handoff/models/"
                    "teacher_current_compiled.nequip.pth").is_file():
            self.skipTest("teacher model not present")
        import stage_d2_c3_device_consistency_preflight as P
        ok, checks = P.device_consistency_preflight(device="cpu")   # NO forward; cpu -> consistent
        self.assertTrue(ok)
        self.assertIs(checks["model_forward_called"], False)
        self.assertFalse(checks["mixed_cpu_cuda"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
