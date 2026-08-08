"""Stage D-1 runner resource-policy config: the co-scheduled 7B profile is parameterized with the
Stage-C-validated defaults (util 0.36 / MIN_FREE 18000), the model + frozen runtime invariants are
unchanged, and the safety behaviour (selected GPU only, fail-closed, no auto-switch, PGID-only
cleanup, no broad pkill) is preserved. Text-level checks on the committed runner; network-free.
"""
from __future__ import annotations

import unittest
from pathlib import Path

RUNNER = (Path(__file__).resolve().parent.parent / "work" / "stage_d1_shadow_replay.sh").read_text()


class StageD1RunnerConfigTests(unittest.TestCase):
    def test_resource_profile_defaults_are_the_validated_coscheduled_values(self):
        self.assertIn('STAGE_D1_GPU_MEM_UTIL="${STAGE_D1_GPU_MEM_UTIL:-0.36}"', RUNNER)
        self.assertIn('MIN_FREE_MIB="${STAGE_D1_MIN_FREE_MIB:-18000}"', RUNNER)
        self.assertIn('STAGE_D1_CUDA_DEVICE="${STAGE_D1_CUDA_DEVICE:-1}"', RUNNER)
        # the old conservative defaults must be gone
        self.assertNotIn(":-0.50}", RUNNER)
        self.assertNotIn(":-26000}", RUNNER)

    def test_launch_uses_the_resource_variables(self):
        self.assertIn('--gpu-memory-utilization "$STAGE_D1_GPU_MEM_UTIL"', RUNNER)
        self.assertIn('CUDA_VISIBLE_DEVICES="$DEV"', RUNNER)
        self.assertIn('[ "${FREE:-0}" -ge "$MIN_FREE_MIB" ]', RUNNER)   # fail-closed gate

    def test_model_and_frozen_invariants_unchanged(self):
        self.assertIn('STAGE_D1_MODEL_PATH="${STAGE_D1_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"', RUNNER)
        self.assertIn('STAGE_D1_SERVED_MODEL_NAME="${STAGE_D1_SERVED_MODEL_NAME:-qwen2.5-7b-instruct}"', RUNNER)
        for flag in ("--dtype bfloat16", "--max-model-len 8192", "--max-num-seqs 1",
                     "--enforce-eager", "--enable-auto-tool-choice", "--tool-call-parser hermes"):
            self.assertIn(flag, RUNNER, flag)

    def test_safety_behaviour_preserved(self):
        self.assertNotIn("pkill", RUNNER)                       # PGID-only cleanup, no broad pkill
        self.assertIn("no GPU switch", RUNNER)                  # gate never auto-switches
        self.assertIn("--mode shadow", RUNNER)                  # shadow only


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
