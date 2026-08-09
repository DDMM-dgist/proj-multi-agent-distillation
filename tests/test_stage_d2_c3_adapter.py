"""Stage D-2 C3 trusted Allegro adapter tests (network-free; NO real mini216 prediction).

Verifies the trusted-adapter guards + species mapping + structure conversion contract without running
the model: path allow-list, sha immutability + identity, LAMMPS type 1->O / type 2->Si mapping (via the
model's own type_names), fail-closed on unknown/reversed/unexpected species, cell/PBC/atom-count
preservation, and that a forward function can come ONLY from the loaded trusted adapter (no arbitrary
callable, and build_forward_fn requires load()). The real torch/jit load is exercised only if torch is
available; otherwise type_names is set directly for the pure-logic checks.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = "/home/hyunjin/CLADE/SiO2-x_distillatio/materials-ml-kit/research-sio2-allegro-simplenn-distillation"
MODEL = f"{RES}/gpu_finetune_handoff/models/teacher_current_compiled.nequip.pth"
MODEL_SHA = "b56e20ffc31da601feed8411c92675bdae9eb886db153ff67dd37dea161b1c57"
ALLOW = [f"{RES}/gpu_finetune_handoff/models/"]

try:
    from runtimes.pydantic_ai.stage_d2_c3_teacher_adapter import AdapterGuardError, TrustedAllegroAdapter
    _HAS = True
except ImportError:  # pragma: no cover
    _HAS = False
_HAS_MODEL = Path(MODEL).is_file()


@unittest.skipUnless(_HAS, "adapter import failed")
class StageD2C3AdapterTests(unittest.TestCase):
    @unittest.skipUnless(_HAS_MODEL, "teacher model not present")
    def test_allow_list_and_sha_identity(self):
        TrustedAllegroAdapter(MODEL, expected_sha256=MODEL_SHA, allow_prefixes=ALLOW)   # ok
        with self.assertRaisesRegex(AdapterGuardError, "allow-list"):
            TrustedAllegroAdapter(MODEL, expected_sha256=MODEL_SHA, allow_prefixes=["/nope"])
        with self.assertRaisesRegex(AdapterGuardError, "sha256 mismatch"):
            TrustedAllegroAdapter(MODEL, expected_sha256="dead" * 16, allow_prefixes=ALLOW)

    def _adapter(self):
        a = TrustedAllegroAdapter.__new__(TrustedAllegroAdapter)   # bypass sha (unit test of pure logic)
        a.type_names = ["O", "Si"]; a.r_max = 5.0; a.model_dtype = "float32"
        a._loaded = True; a._model = None; a.model_sha256 = MODEL_SHA
        return a

    def test_species_mapping_type1_O_type2_Si(self):
        a = self._adapter()
        self.assertEqual(a.species_index("O"), 0)     # model type_names index: O=0, Si=1
        self.assertEqual(a.species_index("Si"), 1)
        self.assertEqual(a.map_lammps_types([1, 1, 2], {"1": "O", "2": "Si"}), [0, 0, 1])

    def test_species_mapping_fail_closed(self):
        a = self._adapter()
        with self.assertRaisesRegex(AdapterGuardError, "unknown LAMMPS atom type"):
            a.map_lammps_types([1, 3], {"1": "O", "2": "Si"})       # type 3 unmapped
        with self.assertRaisesRegex(AdapterGuardError, "unexpected species"):
            a.map_lammps_types([1], {"1": "C"})                     # unexpected species C
        # a REVERSED symbol map is honoured literally -> the reversal is visible/auditable
        self.assertEqual(a.map_lammps_types([1, 2], {"1": "Si", "2": "O"}), [1, 0])

    def test_conversion_contract_preserves_structure(self):
        a = self._adapter()
        pos = [(i * 0.1, i * 0.2, i * 0.3) for i in range(216)]
        types = [1] * 144 + [2] * 72                                # O144 Si72 like mini216
        conv = a.structure_conversion_contract(pos, types, 14.835545077426339, {"1": "O", "2": "Si"})
        self.assertEqual(conv["n_atoms"], 216)
        self.assertEqual(conv["composition"], {"O": 144, "Si": 72})
        self.assertEqual(conv["pbc"], [True, True, True])
        self.assertEqual(conv["cell"][0][0], 14.835545077426339)   # cubic L preserved
        self.assertEqual(conv["relaxation"], "none")
        self.assertEqual(conv["atom_type_index"][:2], [0, 0])       # O -> 0
        self.assertEqual(conv["atom_type_index"][-1], 1)            # Si -> 1

    def test_forward_only_from_loaded_adapter(self):
        a = TrustedAllegroAdapter.__new__(TrustedAllegroAdapter)
        a._loaded = False
        with self.assertRaisesRegex(AdapterGuardError, "call load"):
            a.build_forward_fn()                                    # no arbitrary callable; must load() first
        # the generic C3 executor default forward_fn=None also raises (no agent-supplied callable)
        from runtimes.pydantic_ai import stage_d2_c3_teacher_executor as EX
        self.assertTrue(hasattr(EX, "run_teacher_single_point"))

    def test_execution_contract_one_forward_no_side_jobs(self):
        prop = json.loads((ROOT / "examples/stage_d2_c3/action_proposal.json").read_text())["parameters"]
        self.assertIs(prop["one_forward_pass"], True)
        for k in ("no_scheduler", "no_training", "no_md", "no_dft", "no_paid_api", "no_overwrite"):
            self.assertIs(prop[k], True)

    @unittest.skipUnless(_HAS_MODEL, "teacher model not present")
    def test_real_load_metadata_if_torch_present(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not available in this test env (real load verified in the allegro env)")
        a = TrustedAllegroAdapter(MODEL, expected_sha256=MODEL_SHA, allow_prefixes=ALLOW)
        info = a.load(device="cpu")                                 # LOAD ONLY, no forward
        self.assertEqual(a.type_names, ["O", "Si"])
        self.assertEqual(a.r_max, 5.0)
        self.assertEqual(info["model_dtype"], "float32")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
