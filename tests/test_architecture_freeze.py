"""ARCHITECTURE FREEZE guard — v2 (deterministic-verdict-ownership refactor).

The v1 freeze (development-pass revision 87c51d3) was consumed by the Stage D-1 holdout, which
exposed an LLM_VERDICT_REGENERATION_CONSISTENCY_FAILURE. The architecture was then deliberately
revised (the deterministic policy now OWNS the authoritative verdict; the LLM owns only
interpretation) and RE-FROZEN here at the post-refactor revision. This test pins the SHA-256 of
every frozen runtime/semantic file so any later edit to one — the deterministic criterion evaluator
+ result schema + authoritative/advisory + verdict-ownership binding, the Judge prompt, canonical
validation/acceptance, the duplicate-read guard, request_limit, authorization policy, controller
logic, and role schemas — fails the suite. The NEXT holdout package may only ADD
fixtures/specs/runner/evaluator-glue/tests; it may not change anything below. Frozen model is a
constant, not a file.

If a frozen file legitimately must change, that is (by definition) a NEW architecture revision and a
NEW freeze — it must not be done silently while a holdout is in flight.
"""
from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FREEZE_REVISION = "deterministic-verdict-ownership-refactor (v2; post-consumed-holdout)"
FROZEN_MODEL = "qwen2.5-7b-instruct"

FROZEN = {
    "runtimes/pydantic_ai/criterion_eval.py":
        "65a1c4fd5560660eef4825b4f4aa0687868cda7e4e6e863c19ec2b8923d12b96",
    "agents/judge.md":
        "5e902597944c4e700c4d82b77e7462a6971781cbd1b0614dc33015a48a3682f9",
    "orchestration/exchange.py":
        "f29362891c18c728dc9de0a8c3ee51590c9be6be53f5b811bdfd861d0e82a8ae",
    "runtimes/pydantic_ai/tool_registry.py":
        "3d398a718da1c9e89d03585acdc9fafcfeb2d4767569ffac6027edcd13c1e467",
    "runtimes/pydantic_ai/models.py":
        "962a77cb4c553b3b1f96c73c9b62769d8f403c1a70ebf321e9d3352e20a72cff",
    "runtimes/pydantic_ai/pydantic_ai_runtime.py":
        "4cf343d6e3b16c66560e6ee47ddd4d6b36d1eb941ce59bf95f65bcc077a87507",
    "runtimes/pydantic_ai/role_outputs.py":
        "929695fad270279f839e8d7cd7ca441516fc2fd6c6717ec0359e5a52050972eb",
    "runtimes/pydantic_ai/production_router.py":
        "cc390924d8ccc02e11963e52c2a3fff70ea86fab4a6f7cdc30bdcd8b1d9d44de",
    "runtimes/pydantic_ai/driver.py":
        "571636918a2827ceded12e9ee3b0cad7f23ab73887d61ed0cc2b6d5727986719",
    "runtimes/pydantic_ai/actions.py":
        "2efc14e661056581b54e6a93be8ab9bca9b670b009f9e97091b1dad9aee1d78f",
    "runtimes/pydantic_ai/controller_bridge.py":
        "5b23ee61b4bb399fe4c5b17f545fd1386806e74e5d419f0c8469765895289385",
    "orchestration/specs.py":
        "12058f13b1a498a34b5aabf0dcff5186c72f1754b429ff682737a56180db1010",
    "workflow/controller.py":
        "cf6875f5c188e312e57525cfd43daa2dd6504c98a41437298197065967863244",
    "orchestration/schema/agent_result.schema.json":
        "a38afea9c06c21e647376efd835dec32a16b2f247583a090560cb1843e0eda31",
    "orchestration/schema/agent_spec.schema.json":
        "8b59189f55f72a3c5853093ec88c3284d353475a322c50ca55403bbf5282151b",
    "orchestration/schema/agent_task.schema.json":
        "60d3d49c33c85107830c237cbcc6db23b9c30225990cad7c6f152337f57ce0a5",
    "orchestration/schema/judge_vote.schema.json":
        "682ade03213da8483d2089ed21f34081be612b01c4df615f1ae6facbc4ea18df",
    "agent_specs/analyst.yaml":
        "7fac8bc650f2b06d689327a206cf0802bc0e1cf9351dbd4f44e5f315d2820306",
    "agent_specs/data-curator.yaml":
        "7500edebd058b82be6bb6ca048c5f9cb7136f3440cf12e3161f7b450342ff774",
    "agent_specs/judge.yaml":
        "94727231c06c51daf5f400867454a273c4600fc96f7ab10b7b5df11e52f8fd5d",
    "agent_specs/literature.yaml":
        "7709db297c330083abe4a396a7374e5f9fd4bfdaba62950f72094f24993f9368",
    "agent_specs/ml-trainer.yaml":
        "17ecf5a3244f8e0e798d4f27aa6995d55fe31d5a14741c6f06e6f033e3e9b8ad",
    "agent_specs/orchestrator.yaml":
        "37bb4da36e5075f7f94cb0008a44d07ef7c600db8da48a3b247aaf8b815c3950",
    "agent_specs/simulation.yaml":
        "aab7a7842f1aefe175ca718c830469d6fa42b1ffae3e9cc1c599936f22fdf7f3",
}


class ArchitectureFreezeTests(unittest.TestCase):
    def test_frozen_files_unchanged(self):
        drift = []
        for rel, want in FROZEN.items():
            p = ROOT / rel
            self.assertTrue(p.is_file(), f"frozen file missing: {rel}")
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            if got != want:
                drift.append(f"{rel}: {got} != {want}")
        self.assertEqual(drift, [], "frozen architecture files changed:\n" + "\n".join(drift))

    def test_request_limit_default_is_six(self):
        # request_limit=6 is part of the freeze; assert the default constant is intact.
        from runtimes.pydantic_ai.models import RuntimeContext
        self.assertEqual(RuntimeContext.model_fields["request_limit"].default, 6)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
