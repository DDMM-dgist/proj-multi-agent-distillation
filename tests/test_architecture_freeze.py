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
FREEZE_REVISION = (
    "approval-boundary reconciliation (v9; costly_teacher_labeling added to "
    "agent_specs/simulation.yaml to match SIMULATION_ACTIONS gaining "
    "build_teacher_baseline/validate_teacher_reference, and evaluate_heldout_fidelity "
    "added to APPROVAL_GATED_ACTIONS in actions.py to close a gap where its HPC binding "
    "was READY_HPC_APPROVAL_GATED with a real executor but was never actually gated by "
    "dispatch.authorize_and_execute's approval-boundary check; no role/action-set/contract/"
    "Judge change)"
)
FROZEN_MODEL = "qwen2.5-7b-instruct"

FROZEN = {
    "runtimes/pydantic_ai/criterion_eval.py":
        "65a1c4fd5560660eef4825b4f4aa0687868cda7e4e6e863c19ec2b8923d12b96",
    "agents/judge.md":
        "cc32f81efdbf825067f2688eb78a2f41982f9183ef78a33badb31906dabc8aa8",
    "orchestration/exchange.py":
        "f29362891c18c728dc9de0a8c3ee51590c9be6be53f5b811bdfd861d0e82a8ae",
    "runtimes/pydantic_ai/tool_registry.py":
        "3d398a718da1c9e89d03585acdc9fafcfeb2d4767569ffac6027edcd13c1e467",
    "runtimes/pydantic_ai/models.py":
        "b6e5efbb6ccc89c9be17d39e8b0255b8b97178292a6dc0619ada506e71fbfd1a",
    "runtimes/pydantic_ai/pydantic_ai_runtime.py":
        "3d2105d5f15824fbd26a599996181e09f2bb555c331aff514df14a4122d2bcb9",
    "runtimes/pydantic_ai/role_outputs.py":
        "929695fad270279f839e8d7cd7ca441516fc2fd6c6717ec0359e5a52050972eb",
    "runtimes/pydantic_ai/production_router.py":
        "cc390924d8ccc02e11963e52c2a3fff70ea86fab4a6f7cdc30bdcd8b1d9d44de",
    "runtimes/pydantic_ai/driver.py":
        "571636918a2827ceded12e9ee3b0cad7f23ab73887d61ed0cc2b6d5727986719",
    "runtimes/pydantic_ai/actions.py":
        "c63f8d42bf208f87c2b7d220264e27556fd3ce79ac5d48c13becb0557c66c141",
    "runtimes/pydantic_ai/controller_bridge.py":
        "3eb11b9075bd25d5b45f09fc9d0b7c0c65f032c1293bd5e38f754fa26b752100",
    "orchestration/specs.py":
        "4b6dc829fe2b6b594cc87e8a62bd944ea9df181cd7f420ae3732c861ce8e43cb",
    "workflow/controller.py":
        "ec81c8db63ea64874d677077c32a46d6bfc8fc6f920e294249e79695004999b6",
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
        "6b9fd7de3a5248b50622a3c33950411e3864c6b9ef1d28b17f7a27b19d7544e9",
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
        # pydantic is an optional runtime dep; on the core-only install skip (like every other
        # pydantic-ai test) instead of erroring — the assertion still runs in the pydantic-ai jobs.
        try:
            from runtimes.pydantic_ai.models import RuntimeContext
        except ModuleNotFoundError:
            self.skipTest("pydantic (optional runtime dep) not installed")
        self.assertEqual(RuntimeContext.model_fields["request_limit"].default, 6)


    def test_production_wiring_keeps_frozen_architecture_dimensions(self):
        from runtimes.pydantic_ai.actions import ROLE_ALLOWED_ACTIONS
        from workflow.controller import RunController
        roles = {"orchestrator", "literature", "data-curator", "ml-trainer", "simulation", "analyst", "judge"}
        import json
        import tempfile
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "workflow.yaml"
            cfg.write_text(yaml.safe_dump({"run_id": "freeze-dimensions", "stages": [{
                "name": "s", "command": None,
                "contract": {"kind": "validation_manifest", "manifest": "m.json", "validator": "validation.report.validate_validation_report"},
            }]}))
            c = RunController.initialize(cfg, root / "run")
        self.assertEqual(set(ROLE_ALLOWED_ACTIONS), {"data-curator", "ml-trainer", "simulation", "analyst"})
        self.assertIn("build_teacher_baseline", ROLE_ALLOWED_ACTIONS["simulation"])
        self.assertIn("validate_teacher_reference", ROLE_ALLOWED_ACTIONS["simulation"])
        self.assertIn("acquire_structures", ROLE_ALLOWED_ACTIONS["data-curator"])
        self.assertNotIn("acquire_structures", ROLE_ALLOWED_ACTIONS["simulation"])
        self.assertNotIn("validate_teacher_reference", ROLE_ALLOWED_ACTIONS["data-curator"])
        self.assertNotIn("validate_teacher_reference", ROLE_ALLOWED_ACTIONS["ml-trainer"])
        self.assertEqual(len(roles), 7)
        self.assertEqual(c.stage("s")["contract"]["kind"], "validation_manifest")

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
