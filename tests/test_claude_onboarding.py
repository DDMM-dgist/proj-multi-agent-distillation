import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


class ClaudeOnboardingTests(unittest.TestCase):
    def test_fresh_clone_has_default_director_and_specialists(self):
        settings = json.loads((ROOT / ".claude/settings.json").read_text())
        self.assertEqual(settings["agent"], "director")
        expected = {"director", "literature", "data-curator", "ml-trainer", "simulation", "analyst", "judge"}
        found = {p.stem for p in (ROOT / ".claude/agents").glob("*.md")}
        self.assertTrue(expected <= found)

    def test_start_status_resume_skills_are_packaged(self):
        for name in ("distill-start", "distill-status", "distill-resume"):
            path = ROOT / ".claude/skills" / name / "SKILL.md"
            self.assertTrue(path.is_file(), path)
            self.assertIn(f"name: {name}", path.read_text())

    def test_packaged_readme_starts_from_clone_and_claude(self):
        readme = (ROOT / "README.md").read_text()
        self.assertIn("git clone", readme)
        self.assertIn("/distill-start", readme)
        self.assertIn("/distill-resume", readme)

    def test_start_skill_only_requires_packaged_guidance(self):
        skill = (ROOT / ".claude/skills/distill-start/SKILL.md").read_text()
        self.assertIn("`README.md`", skill)
        self.assertNotIn("MANUAL_KO.md", skill)

    def test_generic_templates_are_parseable_and_case_neutral(self):
        template_dir = ROOT / "configs/templates"
        expected = {"teacher.yaml", "student.yaml", "acquisition.yaml",
                    "md_backend.yaml", "reference.yaml",
                    "validation_profile.yaml", "workflow.yaml"}
        self.assertTrue(expected <= {path.name for path in template_dir.glob("*.yaml")})
        text = ""
        for path in template_dir.glob("*.yaml"):
            yaml.safe_load(path.read_text())
            text += path.read_text()
        for case_token in ("MACE", "GRACE", "Allegro", "SIMPLE-NN", "SiO2", "SiO₂"):
            self.assertNotIn(case_token, text)
        workflow = yaml.safe_load((template_dir / "workflow.yaml").read_text())
        self.assertTrue(all("criteria" in stage.get("gate", {})
                            for stage in workflow["stages"]))

    def test_canonical_agent_prompts_do_not_select_case_models(self):
        text = "\n".join(path.read_text() for path in (ROOT / "agents").glob("*.md"))
        for case_token in ("MACE-MH-1", "GRACE/FS", "Allegro", "SIMPLE-NN", "SiO2", "SiO₂"):
            self.assertNotIn(case_token, text)

    def test_optional_gate_workflow_carries_hashes_and_failed_judge_slots(self):
        workflow = (ROOT / "gates/gate_vote.workflow.js").read_text()
        self.assertIn("artifact_sha256: artifactSha256", workflow)
        self.assertIn("invocationFailureVote", workflow)
        self.assertIn("criteria.map", workflow)

    def test_release_build_outputs_are_ignored_and_agents_are_registered_only(self):
        ignored = (ROOT / ".gitignore").read_text()
        for pattern in ("build/", "dist/", "*.egg-info/"):
            self.assertIn(pattern, ignored)
        self.assertFalse((ROOT / "agents/github-manager.md").exists())
        judge = (ROOT / "agents/judge.md").read_text()
        self.assertNotIn("genuinely independent", judge)
        self.assertIn("separate-context", judge)


if __name__ == "__main__":
    unittest.main()
