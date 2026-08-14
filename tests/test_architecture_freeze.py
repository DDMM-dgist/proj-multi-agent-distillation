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

v10 (validation-target-lock) deliberately re-froze workflow/controller.py: it is a real, disclosed
architecture change (see FREEZE_REVISION below for exactly what changed and, just as importantly,
what did NOT), not a workaround to avoid touching frozen code.

v11 (validation-target-lock, automatic establishment) closed a real wiring gap a runtime-route
audit found in v10: contract establishment was still an optional, manual CLI step, so nothing
actually stopped a real run from executing Teacher-side scientific stages before the target
contract existed. v11 deliberately re-froze workflow/controller.py again for the same reason as
v10 — see FREEZE_REVISION below for exactly what changed and what deliberately did not.
"""
from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FREEZE_REVISION = (
    "validation-target-lock (v10; schema_version bumped 7->8 and workflow/controller.py "
    "gained a write-once validation_contract: a new top-level state key, None until "
    "establish_validation_contract() freezes it from the Teacher applicability domain, "
    "validation scope, and dataset split policy, each hash-bound at establishment; "
    "identical re-establishment is an idempotent no-op, differing content hard-fails and "
    "requires a new run; stages may declare produces_student_results, and BOTH run_stage() "
    "and complete_external_stage() (via the shared "
    "_require_validation_contract_for_student_stage helper) refuse to run/complete one "
    "until the contract exists — the guard was applied to both execution paths because "
    "real workflows complete Student-producing stages externally (agent/executor "
    "dispatch), not via run_stage's subprocess path; completing one via either path sets "
    "a permanent student_stage_ever_completed provenance marker on the contract record "
    "(via the shared _mark_student_stage_completed helper). "
    "propose_recovery/start_iteration/verify_recovery_execution were deliberately left "
    "UNCHANGED — no recovery method writes validation_contract, so recovery may re-run any "
    "stage, including a contract-consuming one like teacher_baseline or "
    "reference_validation under the unchanged frozen contract, but can never mutate, "
    "re-establish, or replace it; no role/action-set/Judge change); "
    "v11 (automatic establishment) closes the wiring gap a runtime-route audit found in v10: "
    "RunController.initialize() now accepts an optional top-level validation_contract_sources "
    "mapping ({distillation_scope, validation_profile, dataset_policy} paths); when a workflow "
    "declares it, initialize() snapshots the exact content of those three files into the new "
    "run's own inputs/contract_sources/ area (via shutil.copy2, inside the same temporary "
    "init directory used for every other run-bound artifact) BEFORE building anything from "
    "them, then builds the contract's components from those run-local snapshots — never from "
    "the still-mutable external paths — via the single shared "
    "workflow.contracts.build_validation_contract_components, and establishes the record via "
    "the single shared _build_validation_contract_record (module-level in workflow/controller.py) "
    "that both RunController.establish_validation_contract and initialize() now call — there is "
    "exactly one construction path, never two independently-mutable representations; the "
    "resulting record is written to state['validation_contract'] AND to a validation_contract.json "
    "file inside the same temporary directory, so both are byte-identical from the same write. "
    "Initialization remains atomic: source resolution/snapshotting, domain-equality verification "
    "(distillation_scope's deployment_domain is authoritative; validation_profile's copy must "
    "match exactly or this hard-fails), and contract establishment all happen inside initialize()'s "
    "existing temporary-directory transaction, so any failure (a missing source file, a domain "
    "mismatch) is caught by the existing except-Exception/rmtree(temporary)/raise handler and "
    "leaves no run directory behind at all — no new rollback mechanism was introduced. Existing/"
    "historical workflows that do not declare validation_contract_sources are entirely unaffected: "
    "validation_contract stays None exactly as before, so v10 behavior is preserved byte-for-byte "
    "for them (this is why R11, which has no validation_contract_sources key, was never touched). "
    "The manual establish-validation-contract CLI helper "
    "(workflow.steps.establish_validation_contract_from_configs) still exists as a compatibility/"
    "manual tool and now itself calls the same shared "
    "workflow.contracts.build_validation_contract_components instead of duplicating the "
    "domain-equality/component-building logic inline — a pure refactor with no behavior change, "
    "verified by the pre-existing test suite. No role/action-set/Judge/schema_version change; "
    "schema_version remains 8)"
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
        "5832123d9fb1ce235cb705bb551bd56598e3a9655a9dbecc2c8dc0bec89ab851",
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
