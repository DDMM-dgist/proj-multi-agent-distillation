"""Stage C runner model/GPU parameterization: the ONLY independent variable is the model. With no
env overrides the defaults reproduce the exact frozen 3B configuration, and every frozen benchmark
invariant (vLLM flags, tool parser, no hard-coded model on the launch line) is preserved. Text-level
checks on the committed runner; network-free; no GPU.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

RUNNER = (Path(__file__).resolve().parent.parent / "tests" / "harness" / "stage_c_golden_shadow.sh").read_text()


class StageCRunnerConfigTests(unittest.TestCase):
    def test_config_is_parameterized_with_3b_defaults(self):
        self.assertIn('STAGE_C_MODEL_PATH="${STAGE_C_MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"', RUNNER)
        self.assertIn('STAGE_C_SERVED_MODEL_NAME="${STAGE_C_SERVED_MODEL_NAME:-qwen2.5-3b-instruct}"', RUNNER)
        self.assertIn('STAGE_C_GPU_MEM_UTIL="${STAGE_C_GPU_MEM_UTIL:-0.18}"', RUNNER)
        self.assertIn('MIN_FREE_MIB="${STAGE_C_MIN_FREE_MIB:-12000}"', RUNNER)
        self.assertIn('STAGE_C_CUDA_DEVICE="${STAGE_C_CUDA_DEVICE:-1}"', RUNNER)

    def test_launch_uses_variables_not_a_hardcoded_model(self):
        self.assertIn('vllm serve "$STAGE_C_MODEL_PATH" --served-model-name "$STAGE_C_SERVED_MODEL_NAME"', RUNNER)
        self.assertIn('--gpu-memory-utilization "$STAGE_C_GPU_MEM_UTIL"', RUNNER)
        self.assertIn('CUDA_VISIBLE_DEVICES="$DEV"', RUNNER)
        self.assertIn('PYDANTIC_AI_MODEL="$STAGE_C_SERVED_MODEL_NAME"', RUNNER)
        # the launch line must NOT hard-code the 3B repo id (only the default assignment may name it)
        self.assertNotIn("vllm serve Qwen/Qwen2.5-3B-Instruct", RUNNER)
        self.assertNotIn("PYDANTIC_AI_MODEL=qwen2.5-3b-instruct ", RUNNER)

    def test_frozen_invariants_preserved(self):
        for flag in ("--dtype bfloat16", "--max-model-len 8192", "--max-num-seqs 1",
                     "--enforce-eager", "--enable-auto-tool-choice", "--tool-call-parser hermes"):
            self.assertIn(flag, RUNNER, flag)
        # single sequential loop, no retries, PGID-only cleanup, no broad pkill
        self.assertIn("for tid in \"${IDS[@]}\"; do run_one \"$tid\"; done", RUNNER)
        self.assertNotIn("pkill", RUNNER)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
