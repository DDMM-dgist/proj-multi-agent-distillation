import json
import tempfile
import unittest
from pathlib import Path

import yaml

from workflow.controller import RunController
from workflow.review_lenses import DEFAULT_REVIEW_LENSES, normalize_review_lenses


class JudgeReviewLensTests(unittest.TestCase):
    CRITERIA = ["artifact is complete", "reported threshold is satisfied"]

    def make_completed_gate(self, root, review_lenses=None):
        root.mkdir(parents=True, exist_ok=True)
        gate = {"criteria": self.CRITERIA}
        if review_lenses is not None:
            gate["review_lenses"] = review_lenses
        config = root / "workflow.yaml"
        config.write_text(yaml.safe_dump({
            "run_id": "lens-test",
            "stages": [{"name": "validation", "command": None,
                        "outputs": ["artifacts/report.json"], "gate": gate}],
        }))
        controller = RunController.initialize(config, root / "run")
        artifact = controller.run_dir / "artifacts/report.json"
        artifact.write_text('{"status": "ok"}\n')
        controller.complete_external_stage("validation", [artifact])
        return controller

    @staticmethod
    def passing_bundle(controller):
        context = controller.gate_context("validation")
        votes = []
        for index, lens in enumerate(context["review_lenses"], 1):
            votes.append({
                "judge_id": f"judge-{index}",
                "review_lens": lens["id"],
                "verdict": "PASS",
                "criteria_checked": [
                    {"criterion": criterion, "value_read": "verified", "ok": True}
                    for criterion in context["criteria"]
                ],
                "rationale": "All common criteria and the assigned lens were checked.",
                "required_fix": "",
            })
        return {"stage": "validation", "criteria": context["criteria"],
                "review_lenses": context["review_lenses"],
                "artifact_sha256": context["artifact_sha256"],
                "decision": "PASS", "votes": votes}

    @staticmethod
    def write_bundle(controller, payload, name="votes.json"):
        path = controller.run_dir / "gates" / name
        path.write_text(json.dumps(payload))
        return path

    def test_default_lenses_are_three_complementary_records(self):
        lenses = normalize_review_lenses()
        self.assertEqual(lenses, list(DEFAULT_REVIEW_LENSES))
        self.assertEqual(len({lens["id"] for lens in lenses}), 3)

    def test_lens_contract_rejects_wrong_count_duplicate_and_incomplete(self):
        with self.assertRaisesRegex(ValueError, "exactly three"):
            normalize_review_lenses(list(DEFAULT_REVIEW_LENSES[:2]))
        duplicated = [dict(item) for item in DEFAULT_REVIEW_LENSES]
        duplicated[2]["id"] = duplicated[0]["id"]
        with self.assertRaisesRegex(ValueError, "must be unique"):
            normalize_review_lenses(duplicated)
        incomplete = [dict(item) for item in DEFAULT_REVIEW_LENSES]
        incomplete[0].pop("focus")
        with self.assertRaisesRegex(ValueError, "exactly id, title, and focus"):
            normalize_review_lenses(incomplete)
        non_string = [dict(item) for item in DEFAULT_REVIEW_LENSES]
        non_string[0]["focus"] = None
        with self.assertRaisesRegex(ValueError, "must be strings"):
            normalize_review_lenses(non_string)

    def test_controller_binds_default_lenses_and_accepts_one_vote_per_lens(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = self.make_completed_gate(Path(tmp))
            # A freshly initialized run targets the current schema version (v7 adds only
            # additive operational metadata; gate/recovery semantics are unchanged).
            from workflow.controller import SCHEMA_VERSION
            self.assertEqual(controller.state["schema_version"], SCHEMA_VERSION)
            bundle = self.passing_bundle(controller)
            controller.record_gate(
                "validation", votes_path=self.write_bundle(controller, bundle)
            )
            self.assertEqual(controller.stage("validation")["gate"], "PASS")

    def test_run_can_bind_exactly_three_custom_domain_lenses(self):
        custom = [
            {"id": "data_audit", "title": "Data audit", "focus": "Audit data."},
            {"id": "physics_audit", "title": "Physics audit", "focus": "Audit physics."},
            {"id": "ops_audit", "title": "Operations audit", "focus": "Audit operations."},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            controller = self.make_completed_gate(Path(tmp), custom)
            self.assertEqual(controller.gate_context("validation")["review_lenses"], custom)

    def test_bundle_cannot_change_run_bound_lens_definitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = self.make_completed_gate(Path(tmp))
            bundle = self.passing_bundle(controller)
            bundle["review_lenses"][0]["focus"] = "softened after seeing results"
            with self.assertRaisesRegex(ValueError, "run-bound review lenses"):
                controller.record_gate(
                    "validation", votes_path=self.write_bundle(controller, bundle)
                )

    def test_duplicate_missing_unknown_or_reordered_vote_lenses_are_rejected(self):
        mutations = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label in ("duplicate", "missing", "unknown", "reordered"):
                controller = self.make_completed_gate(root / label)
                bundle = self.passing_bundle(controller)
                if label == "duplicate":
                    bundle["votes"][1]["review_lens"] = bundle["votes"][0]["review_lens"]
                elif label == "missing":
                    bundle["votes"][1].pop("review_lens")
                elif label == "unknown":
                    bundle["votes"][1]["review_lens"] = "unknown_lens"
                else:
                    bundle["votes"][0], bundle["votes"][1] = (
                        bundle["votes"][1], bundle["votes"][0]
                    )
                mutations.append((controller, bundle, label))
            for controller, bundle, label in mutations:
                with self.subTest(label=label), self.assertRaises(ValueError):
                    controller.record_gate(
                        "validation",
                        votes_path=self.write_bundle(controller, bundle, f"{label}.json"),
                    )

    def test_lens_does_not_relax_common_criteria(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = self.make_completed_gate(Path(tmp))
            bundle = self.passing_bundle(controller)
            bundle["votes"][2]["criteria_checked"][1]["ok"] = False
            with self.assertRaisesRegex(ValueError, "Judge PASS requires every"):
                controller.record_gate(
                    "validation", votes_path=self.write_bundle(controller, bundle)
                )


if __name__ == "__main__":
    unittest.main()
