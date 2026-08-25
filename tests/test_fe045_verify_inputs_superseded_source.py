"""FE-045 -- ``RunController.verify_inputs`` must not re-verify the mutable SOURCE bytes of a
*superseded* input record.

Background (the live ffv4t-eng1 blocker): a superseded ``acquisition_plan`` input (043) and the
fresh superseding plan (044) legitimately share ONE mutable source path
(``acquisition/plans/<run_id>.acquisition_plan.json`` -- the canonical Stage-3 planner writes a fixed
filename). Binding the superseding plan overwrites that shared source with the new bytes, so 043's
recorded historical bytes no longer live at the source path -- they survive only in 043's immutable
content-addressed snapshot. Pre-FE-045 ``verify_inputs`` iterated ALL inputs (including superseded)
and re-checked ``verify_artifact(source, source_integrity)`` for every one, so the superseded 043
source re-check raised ``declared workflow input changed after initialization`` and blocked the
acquisition re-gate immediately after the corrective reacquisition executed.

FE-045 gates the SOURCE re-check behind ``if not record.get("superseded")`` -- mirroring the existing
``active_inputs()`` active/superseded semantics -- while leaving the per-input SNAPSHOT integrity
check mandatory and fail-closed for EVERY input (superseded or active). These tests drive the REAL
``RunController.verify_inputs`` against genuine bound/superseded input records (built through the real
``bind_new_input`` / ``supersede_bound_acquisition_plan`` lifecycle), plus a fixture derived from the
exact ffv4t-eng1 043/044 input shape.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from workflow.controller import RunController
from workflow.integrity import sha256_file

ROOT = Path(__file__).resolve().parents[1]
EItNG1_RUN = ROOT / "runs" / "sio2-sox-allegro-simplenn-ffv4t-eng1"

# The fixed shared filename the canonical Stage-3 planner writes (source path both the superseded and
# the superseding acquisition_plan inputs point at).
PLAN_BASENAME = "acquisition_plan.json"


def _plan_bytes(*, seed: int) -> str:
    """A distinct, valid-looking existing-pool AcquisitionPlan JSON body (bytes differ by seed so a
    superseding plan overwrites the shared source with genuinely different content)."""
    return json.dumps({
        "schema_version": 1,
        "pool_path": "artifacts/pool.extxyz",
        "selected_source_global_indices": [seed, seed + 1, seed + 2],
        "selected_parent_structure_ids": [f"p-{seed}-0", f"p-{seed}-1"],
        "labeling_population_sizing": {"n_target": 11},
        "seed": seed,
    }, indent=2) + "\n"


class _Fe045Fixture(unittest.TestCase):
    GATE_CRITERION = "artifact is complete and internally consistent"

    def _controller(self, root):
        cfg_dict = {"run_id": "fe045", "stages": [{
            "name": "acquisition", "command": None, "outputs": ["artifacts/result.txt"],
            "gate": {"criteria": [self.GATE_CRITERION]},
        }]}
        cfg = root / "workflow.yaml"
        cfg.write_text(yaml.safe_dump(cfg_dict))
        return RunController.initialize(cfg, root / "run")

    def _bind_plan(self, controller, *, seed: int) -> Path:
        """Write the shared-path acquisition_plan source and bind it as a copied input (creates a
        content-addressed snapshot). Returns the shared source path."""
        plans_dir = controller.run_dir / "acquisition" / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        source = plans_dir / f"fe045.{PLAN_BASENAME}"
        source.write_text(_plan_bytes(seed=seed))
        controller.bind_new_input(source, copy=True)
        return source


class Fe045VerifyInputsTests(_Fe045Fixture):
    # 1 -- the exact live shape: a superseded plan whose shared source was overwritten by the
    #      superseding plan, with an intact frozen snapshot, verifies WITHOUT error.
    def test_1_superseded_overwritten_source_intact_snapshot_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._controller(Path(tmp))
            source = self._bind_plan(c, seed=100)               # 043 (will be superseded)
            v1_sha = sha256_file(source)
            c.supersede_bound_acquisition_plan(reason="coverage-gap reacquisition")
            source.write_text(_plan_bytes(seed=200))            # superseding plan overwrites shared src
            c.bind_new_input(source, copy=True)                 # 044 (active)
            v2_sha = sha256_file(source)
            self.assertNotEqual(v1_sha, v2_sha)                 # source genuinely changed

            superseded = [r for r in c.state["inputs"] if r.get("superseded")]
            self.assertEqual(len(superseded), 1)
            # Superseded record's recorded bytes no longer live at the (now-overwritten) shared source.
            self.assertEqual(superseded[0]["source_sha256"], v1_sha)
            self.assertEqual(sha256_file(Path(superseded[0]["source"])), v2_sha)
            # But its immutable snapshot still holds the historical bytes.
            self.assertEqual(sha256_file(Path(superseded[0]["snapshot"])), v1_sha)

            c.verify_inputs()  # FE-045: must NOT raise

    # 2 -- an ACTIVE input whose source bytes change still fails closed (source verification is only
    #      skipped for superseded records, never for active ones).
    def test_2_active_source_change_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._controller(Path(tmp))
            source = self._bind_plan(c, seed=300)               # active, never superseded
            source.write_text(_plan_bytes(seed=999))            # tamper the active source
            with self.assertRaises(RuntimeError) as ctx:
                c.verify_inputs()
            self.assertIn("declared workflow input changed after initialization", str(ctx.exception))

    # 3 -- a SUPERSEDED input whose immutable snapshot is corrupted still fails closed (the snapshot
    #      integrity check remains mandatory for EVERY input; FE-045 only relaxes the SOURCE re-check).
    def test_3_superseded_snapshot_corruption_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._controller(Path(tmp))
            source = self._bind_plan(c, seed=400)
            c.supersede_bound_acquisition_plan(reason="coverage-gap reacquisition")
            source.write_text(_plan_bytes(seed=500))
            c.bind_new_input(source, copy=True)
            superseded = next(r for r in c.state["inputs"] if r.get("superseded"))
            Path(superseded["snapshot"]).write_text("corrupted-frozen-provenance")
            with self.assertRaises(RuntimeError) as ctx:
                c.verify_inputs()
            self.assertIn("run input snapshot integrity check failed", str(ctx.exception))

    # 4 -- the ACTIVE superseding plan is still fully source-verified (not skipped): intact -> passes,
    #      and tampering ONLY the active superseding source -> fails closed.
    def test_4_active_superseding_plan_still_source_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._controller(Path(tmp))
            source = self._bind_plan(c, seed=600)
            c.supersede_bound_acquisition_plan(reason="coverage-gap reacquisition")
            source.write_text(_plan_bytes(seed=700))
            c.bind_new_input(source, copy=True)                 # active superseding plan
            c.verify_inputs()                                   # intact active source -> passes

            source.write_text(_plan_bytes(seed=701))            # tamper ONLY the active source
            with self.assertRaises(RuntimeError) as ctx:
                c.verify_inputs()
            self.assertIn("declared workflow input changed after initialization", str(ctx.exception))

    # 5 -- exact ffv4t-eng1 043/044 fixture: the real recorded input records (superseded 043 sharing
    #      the plans/ source path now holding active 044's bytes) verify WITHOUT stopping on 043.
    def test_5_ffv4t_eng1_fixture_no_longer_stops_on_input_043(self):
        if not EItNG1_RUN.exists():
            self.skipTest("ffv4t-eng1 diagnostic run dir not present")
        real = json.loads((EItNG1_RUN / "manifest.json").read_text())
        real_inputs = real["inputs"]

        # Sanity: the live condition still holds on disk (superseded 043 source == active 044 source,
        # and that shared source now holds 044's bytes, not 043's recorded bytes).
        rec043 = next(r for r in real_inputs
                      if (r.get("snapshot") or "").endswith("043-" + EItNG1_RUN.name
                                                            + ".acquisition_plan.json"))
        rec044 = next(r for r in real_inputs
                      if (r.get("snapshot") or "").endswith("044-" + EItNG1_RUN.name
                                                            + ".acquisition_plan.json"))
        self.assertTrue(rec043.get("superseded"))
        self.assertFalse(rec044.get("superseded"))
        self.assertEqual(rec043["source"], rec044["source"])            # shared source path
        self.assertEqual(sha256_file(Path(rec043["source"])), rec044["source_sha256"])
        self.assertNotEqual(sha256_file(Path(rec043["source"])), rec043["source_sha256"])

        # Build a fixture controller that reuses the EXACT recorded input records (read-only against the
        # real, immutable ffv4t-eng1 files) and isolates the input-verification path from the run's
        # code_revision guard (this is a derived fixture, not a resume of the immutable run).
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            state = {
                "run_id": "fe045-eng1-fixture",
                "project_dir": real["project_dir"],
                "code_revision": {"available": False},
                "inputs": real_inputs,
                "events": [],
            }
            (run_dir / "manifest.json").write_text(json.dumps(state))
            c = RunController(run_dir)
            c.verify_inputs()  # FE-045: must NOT raise on the superseded 043 shared source


if __name__ == "__main__":
    unittest.main()
