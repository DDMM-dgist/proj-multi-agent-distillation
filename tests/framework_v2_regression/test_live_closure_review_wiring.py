"""Framework V2 closure — the LIVE judge gate is wired to the CanonicalReviewPacket
/ JudgeReview path (Sections H & J), enforced by the real RunController.

The other V2 regression tests exercise ``validate_judge_review`` /
``_enforce_v2_review`` standalone. This module proves the production wiring: with
a frozen StageReviewSpec bound to a stage (``bind-closure``), the exact bridge
``runtimes.pydantic_ai.closure_review`` that ``run_three_judge_gate`` uses turns
the real per-lens Judge votes into a ``v2_review`` bundle that the Controller
deterministically validates before any PASS. It checks the positive path (a
unanimous, well-formed committee PASSes) and two teeth:

  * a structurally invalid review (packet SHA tampered) is INVALID_JUDGE_OUTPUT
    and refuses PASS — it never becomes a scientific vote (Section J);
  * a non-unanimous review set refuses PASS even if the legacy aggregate said PASS
    (defense in depth over the closure objects).

No expensive compute: the stage command is a one-line writer and all Judges are
constructed in-process (this is a wiring test, not the real vLLM campaign).
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from workflow.controller import RunController
from framework_v2.contracts import (
    DeploymentScopeContract, ScopeCategory, ScopeRegion)
from framework_v2.review_spec import default_stage_review_specs
from runtimes.pydantic_ai import closure_review as closure

STAGE = "training"  # a canonical stage value with a default StageReviewSpec
GATE_CRITERION = "student committee is trained and internally consistent"


def _scope_contract():
    return DeploymentScopeContract(
        contract_id="closure-wiring-scope",
        objective="closure wiring test deployment scope",
        regions=[ScopeRegion(
            region_id="primary", category=ScopeCategory.PRIMARY_DEPLOYMENT,
            membership_rule="descriptor in the primary deployment set")])


class LiveClosureReviewWiringTests(unittest.TestCase):
    def _bound_controller(self, root):
        cfg = root / "workflow.yaml"
        cfg.write_text(yaml.safe_dump({"run_id": "closure-wiring", "stages": [
            {"name": STAGE,
             "command": [sys.executable, "-c",
                         "from pathlib import Path; "
                         "Path('artifacts/model.txt').write_text('trained')"],
             "outputs": ["artifacts/model.txt"],
             "gate": {"criteria": [GATE_CRITERION]}},
        ]}))
        c = RunController.initialize(cfg, root / "run")
        c.run_stage(STAGE)
        c = RunController(c.run_dir)
        c.bind_v2_scope_contract(_scope_contract().model_dump(mode="json"))
        spec = default_stage_review_specs()[STAGE]
        c.bind_v2_stage_review_spec(STAGE, spec.model_dump(mode="json"))
        return c, spec

    def _legacy_votes(self, c, spec, *, verdicts=None):
        """Build the three legacy votes exactly as the live judges would: all
        three mutually-blind lenses answer the SAME shared free-text gate
        criteria (the historical legacy model). The frozen StageReviewSpec's
        per-lens partition is carried separately in the v2_review bundle."""
        lenses = c.stage(STAGE)["gate_review_lenses"]
        criteria = c.stage(STAGE)["gate_criteria"]
        verdicts = verdicts or {}
        votes = []
        for i, lens in enumerate(lenses, 1):
            lens_id = lens["id"]
            verdict = verdicts.get(lens_id, "PASS")
            ok = verdict == "PASS"
            votes.append({
                "judge_id": f"judge-{i}", "review_lens": lens_id,
                "verdict": verdict,
                "criteria_checked": [
                    {"criterion": q, "value_read": "verified", "ok": ok}
                    for q in criteria],
                "rationale": "committee reviewed",
                "required_fix": "" if ok else "address the unmet criterion"})
        return votes

    def _bundle(self, c, spec, votes):
        artifacts = {a["path"]: a["sha256"] for a in c.stage_artifacts(STAGE)}
        criteria = c.stage(STAGE)["gate_criteria"]
        lenses = c.stage(STAGE)["gate_review_lenses"]
        facts = closure.deterministic_facts_for_stage(STAGE, artifacts, [])
        packet = closure.compile_review_packet(
            controller=c, stage_name=STAGE, spec=spec, facts=facts,
            decision_sha256="closure-wiring-decision")
        reviews = [
            closure.judge_vote_to_review(
                v, v["review_lens"], spec, packet,
                run_id=c.state["run_id"], stage=STAGE, judge_index=i)
            for i, v in enumerate(votes, 1)]
        decision = "FAIL" if any(v["verdict"] == "FAIL" for v in votes) else (
            "PASS" if all(v["verdict"] == "PASS" for v in votes) else "REVISE")
        bundle = {"stage": STAGE, "criteria": criteria, "review_lenses": lenses,
                  "artifact_sha256": artifacts, "decision": decision, "votes": votes,
                  "v2_review": closure.assemble_v2_review(packet, reviews)}
        return bundle, packet, reviews

    def _record(self, c, bundle):
        path = c.run_dir / "gates" / f"{STAGE}.votes.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bundle))
        c.record_gate(STAGE, votes_path=path)

    # --- positive ----------------------------------------------------------
    def test_unanimous_closure_pass_is_enforced_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, spec = self._bound_controller(Path(tmp))
            bundle, packet, _ = self._bundle(c, spec, self._legacy_votes(c, spec))
            self._record(c, bundle)
            c = RunController(c.run_dir)
            self.assertEqual(c.stage(STAGE)["gate"], "PASS")
            gate = [e for e in c.state["events"] if e.get("type") == "gate"][-1]
            fw = gate.get("framework_v2") or {}
            self.assertEqual(fw.get("v2_packet_sha256"), packet.packet_sha256())
            self.assertEqual(fw.get("v2_review_spec_sha256"), spec.content_sha256())

    # --- teeth: invalid review refuses PASS --------------------------------
    def test_tampered_packet_sha_is_invalid_and_refuses_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, spec = self._bound_controller(Path(tmp))
            bundle, _, _ = self._bundle(c, spec, self._legacy_votes(c, spec))
            # Simulate a review compiled against a different/edited packet.
            bundle["v2_review"]["reviews"][0]["packet_sha256"] = "0" * 64
            with self.assertRaises(RuntimeError) as ctx:
                self._record(c, bundle)
            self.assertIn("INVALID_JUDGE_OUTPUT", str(ctx.exception))
            self.assertNotEqual(RunController(c.run_dir).stage(STAGE)["gate"], "PASS")

    def test_non_unanimous_review_refuses_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, spec = self._bound_controller(Path(tmp))
            bundle, _, _ = self._bundle(c, spec, self._legacy_votes(c, spec))
            # Legacy aggregate still says PASS, but one typed review dissents.
            bundle["v2_review"]["reviews"][1]["verdict"] = "REVISE"
            bundle["v2_review"]["reviews"][1]["required_fix"] = "revisit convergence"
            with self.assertRaises(RuntimeError) as ctx:
                self._record(c, bundle)
            self.assertIn("unanimous", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
