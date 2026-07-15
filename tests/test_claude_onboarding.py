import json
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
