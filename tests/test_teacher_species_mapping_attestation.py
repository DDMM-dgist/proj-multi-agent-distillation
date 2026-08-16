"""Regression tests for the R20/R21 forensic-audit fix to
adapters.teacher.load_teacher_with_species_evidence: the runtime species/type
mapping must come from the ACTUAL constructed calculator's own state, cross-
checked against every other available mapping source, never merely from the
declared config kwargs or a legacy exception-handling fallback that real
NequIP 0.15 never triggers.

Network-free: mocks stand in for real nequip/torch objects using only the same
duck-typed `lookup_table` shape adapters.teacher introspects (a lookup indexed
by ASE atomic number, sentinel -1 = species not mapped) -- no nequip or torch
import is required to run these tests.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from ase.data import atomic_numbers


class _MockLookupTransform:
    """Mirrors NequIP's ChemicalSpeciesToAtomTypeMapper shape (a `lookup_table`
    indexed by ASE atomic number) using a plain Python list -- no torch needed."""

    def __init__(self, chemical_symbols):
        table = [-1] * max(atomic_numbers.values())
        for index, symbol in enumerate(chemical_symbols):
            table[atomic_numbers[symbol]] = index
        self.lookup_table = table


class _MockNequIPStyleCalculator:
    """Mirrors the shape adapters.teacher targets for real NequIP/Allegro: a
    `module`/`class`/`constructor="from_compiled_model"` calculator whose
    __init__ stores `self.transforms` as a plain list containing the type
    mapper -- exactly what NequIPCalculator.__init__ does."""

    def __init__(self, compile_path, chemical_symbols=None, **kwargs):
        self.compile_path = compile_path
        self.transforms = [_MockLookupTransform(chemical_symbols or [])]

    @classmethod
    def from_compiled_model(cls, compile_path, chemical_symbols=None, **kwargs):
        return cls(compile_path, chemical_symbols=chemical_symbols, **kwargs)


class _MockNequIPStyleCalculatorReordered(_MockNequIPStyleCalculator):
    """Simulates a calculator that binds a DIFFERENT species order at runtime
    than what was declared (e.g. silently substituting compiled type_names) --
    the exact disagreement class Scope B's cross-check must catch."""

    @classmethod
    def from_compiled_model(cls, compile_path, chemical_symbols=None, **kwargs):
        return cls(compile_path, chemical_symbols=list(reversed(chemical_symbols or [])), **kwargs)


class _MockNequIPStyleCalculatorNoTransforms:
    """A constructed calculator exposing no introspectable species/type
    mapper at all -- used to prove a declared convention with no resolvable
    runtime mapping still fails closed (via species_mapping_is_attested),
    without a source conflict being involved."""

    def __init__(self, compile_path, chemical_symbols=None, **kwargs):
        self.compile_path = compile_path
        self.transforms = []

    @classmethod
    def from_compiled_model(cls, compile_path, chemical_symbols=None, **kwargs):
        return cls(compile_path, chemical_symbols=chemical_symbols, **kwargs)


class _MockLegacyCalculator:
    """Simulates an older calculator API that rejects `chemical_symbols` and
    instead requires `chemical_species_to_atom_type_map`, raising a ValueError
    naming it -- the exact shape adapters.teacher's identity-mapping fallback
    branch catches. Exposes no `transforms`, matching that legacy API's shape."""

    def __init__(self, compile_path, chemical_symbols=None,
                chemical_species_to_atom_type_map=None, **kwargs):
        if chemical_species_to_atom_type_map is None:
            raise ValueError(
                "this calculator version requires chemical_species_to_atom_type_map, "
                "not chemical_symbols")
        self.compile_path = compile_path
        self.chemical_species_to_atom_type_map = chemical_species_to_atom_type_map

    @classmethod
    def from_compiled_model(cls, compile_path, chemical_symbols=None, **kwargs):
        return cls(compile_path, chemical_symbols=chemical_symbols, **kwargs)


def _cfg(calc_class_name, chemical_symbols):
    return {
        "kind": "mock-nequip",
        "model": "/tmp/mock_checkpoint.nequip.pth",
        "calculator": {
            "module": __name__,
            "class": calc_class_name,
            "constructor": "from_compiled_model",
            "model_arg": "__positional__",
            "kwargs": {"chemical_symbols": list(chemical_symbols)},
        },
    }


class RuntimeMappingExtractionTests(unittest.TestCase):
    """Scope D #1/#2: the runtime mapping is read off the actual constructed
    calculator's transforms, generically, for arbitrary species."""

    def test_nequip_style_lookup_table_resolves_correctly(self):
        from adapters.teacher import load_teacher_with_species_evidence
        _calc, evidence = load_teacher_with_species_evidence(
            _cfg("_MockNequIPStyleCalculator", ["O", "Si"]))
        self.assertEqual(evidence["runtime_chemical_species_to_atom_type_map"], {"O": 0, "Si": 1})
        self.assertIn("lookup_table", evidence["runtime_mapping_source"])

    def test_arbitrary_chemical_species_handled_generically(self):
        # Deliberately NOT O/Si: proves nothing in adapters.teacher hardcodes a species list.
        from adapters.teacher import load_teacher_with_species_evidence
        _calc, evidence = load_teacher_with_species_evidence(
            _cfg("_MockNequIPStyleCalculator", ["Fe", "Cu", "Zn"]))
        self.assertEqual(evidence["runtime_chemical_species_to_atom_type_map"],
                         {"Fe": 0, "Cu": 1, "Zn": 2})


class CrossSourceAttestationTests(unittest.TestCase):
    """Scope D #3/#4/#5/#6: consistent sources attest; any disagreement, or an
    unresolvable runtime mapping for a calculator that declares a typing
    convention, fails closed."""

    def test_declared_and_runtime_mapping_agreement_passes(self):
        from adapters.teacher import load_teacher_with_species_evidence, species_mapping_is_attested
        _calc, evidence = load_teacher_with_species_evidence(
            _cfg("_MockNequIPStyleCalculator", ["O", "Si"]))
        self.assertTrue(species_mapping_is_attested(evidence))
        self.assertEqual(evidence["declared_chemical_symbols"], ["O", "Si"])
        self.assertEqual(evidence["runtime_chemical_species_to_atom_type_map"], {"O": 0, "Si": 1})

    def test_runtime_config_conflict_fails_closed(self):
        from adapters.teacher import load_teacher_with_species_evidence, SpeciesMappingConflictError
        with self.assertRaises(SpeciesMappingConflictError):
            load_teacher_with_species_evidence(
                _cfg("_MockNequIPStyleCalculatorReordered", ["O", "Si"]))

    def test_compiled_metadata_runtime_disagreement_fails_closed(self):
        from adapters.teacher import load_teacher_with_species_evidence, SpeciesMappingConflictError
        # declared and runtime agree ({"O": 0, "Si": 1}); the compiled-model metadata source
        # (best-effort, patched here to avoid depending on a real .nequip.pth/torch/nequip)
        # disagrees -- must still fail closed rather than accepting the majority.
        with patch("adapters.teacher._compiled_model_type_names",
                  return_value=({"O": 1, "Si": 0}, "compiled_model[/tmp/mock_checkpoint.nequip.pth].type_names")):
            with self.assertRaises(SpeciesMappingConflictError):
                load_teacher_with_species_evidence(_cfg("_MockNequIPStyleCalculator", ["O", "Si"]))

    def test_compiled_metadata_agreement_does_not_block_attestation(self):
        from adapters.teacher import load_teacher_with_species_evidence, species_mapping_is_attested
        with patch("adapters.teacher._compiled_model_type_names",
                  return_value=({"O": 0, "Si": 1}, "compiled_model[/tmp/mock_checkpoint.nequip.pth].type_names")):
            _calc, evidence = load_teacher_with_species_evidence(
                _cfg("_MockNequIPStyleCalculator", ["O", "Si"]))
        self.assertTrue(species_mapping_is_attested(evidence))
        self.assertEqual(evidence["compiled_model_type_names_map"], {"O": 0, "Si": 1})

    def test_missing_runtime_mapping_for_typed_calculator_fails_closed(self):
        from adapters.teacher import load_teacher_with_species_evidence, species_mapping_is_attested
        _calc, evidence = load_teacher_with_species_evidence(
            _cfg("_MockNequIPStyleCalculatorNoTransforms", ["O", "Si"]))
        self.assertIsNone(evidence["runtime_chemical_species_to_atom_type_map"])
        self.assertFalse(species_mapping_is_attested(evidence))


class LegacyFallbackPathTests(unittest.TestCase):
    """Scope D #7: the pre-existing identity-mapping fallback (for calculator
    APIs that reject `chemical_symbols` and require
    `chemical_species_to_atom_type_map` instead) remains functional and
    unaffected by the new cross-source attestation."""

    def test_legacy_value_error_fallback_still_constructs_and_is_attested(self):
        from adapters.teacher import load_teacher_with_species_evidence, species_mapping_is_attested
        calc, evidence = load_teacher_with_species_evidence(
            _cfg("_MockLegacyCalculator", ["O", "Si"]))
        self.assertEqual(calc.chemical_species_to_atom_type_map, {"O": "O", "Si": "Si"})
        self.assertTrue(evidence["fallback_applied"])
        self.assertEqual(evidence["runtime_chemical_species_to_atom_type_map"], {"O": "O", "Si": "Si"})
        self.assertTrue(species_mapping_is_attested(evidence))


class FailFastOrderingTests(unittest.TestCase):
    """Scope D #8: an unattested species mapping must be discovered
    immediately after calculator construction, in _exec_build_teacher_baseline,
    strictly before the expensive per-frame Teacher inference
    (adapters.acquisition.label_with_teacher) or Teacher-MD sanity checks run."""

    def test_unattested_mapping_blocks_before_label_with_teacher_is_called(self):
        from runtimes.pydantic_ai.executors import _exec_build_teacher_baseline
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_cfg_path = root / "teacher.yaml"
            teacher_cfg_path.write_text(yaml.safe_dump(_cfg("_MockNequIPStyleCalculatorNoTransforms",
                                                            ["O", "Si"])))
            proposal = {"parameters": {
                "structures_path": str(root / "structures.extxyz"),
                "teacher_config": str(teacher_cfg_path),
                "report_path": str(root / "report.json"),
            }}
            with patch("adapters.acquisition.label_with_teacher") as mock_label:
                with self.assertRaisesRegex(ValueError, "species_mapping is not attested"):
                    _exec_build_teacher_baseline(proposal)
            mock_label.assert_not_called()

    def test_attested_mapping_proceeds_to_label_with_teacher(self):
        from runtimes.pydantic_ai.executors import _exec_build_teacher_baseline
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_cfg_path = root / "teacher.yaml"
            teacher_cfg_path.write_text(yaml.safe_dump(_cfg("_MockNequIPStyleCalculator", ["O", "Si"])))
            proposal = {"parameters": {
                "structures_path": str(root / "structures.extxyz"),
                "teacher_config": str(teacher_cfg_path),
                "report_path": str(root / "report.json"),
            }}
            with patch("adapters.acquisition.label_with_teacher") as mock_label:
                mock_label.side_effect = RuntimeError("stop: reached the expensive-inference stage")
                with self.assertRaisesRegex(RuntimeError, "stop: reached the expensive-inference stage"):
                    _exec_build_teacher_baseline(proposal)
            mock_label.assert_called_once()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
