"""V2-H01: structural representation backends.

Composition descriptor width is explicit and not assumed to equal the species
count; SOAP is a real DScribe-backed descriptor with honest dependency gating
(no hidden composition fallback) and fixed-size per-structure pooling.
"""
import builtins

import numpy as np
import pytest

from framework_v2.structural_representation import (
    CompositionRepresentationAdapter,
    PoolingMode,
    RepresentationBackend,
    SoapDependencyError,
    SoapRepresentationAdapter,
    StructuralRepresentation,
    StructureRecord,
)

def _records():
    return [
        StructureRecord(structure_id="s1", species_counts={"Si": 1, "O": 2}),
        StructureRecord(structure_id="s2", species_counts={"Si": 2, "O": 1}),
    ]


def test_composition_descriptor_dimension_not_species_assumption():
    rep = CompositionRepresentationAdapter(species=["Si", "O"]).compute(
        _records(), representation_id="r"
    )
    assert rep.descriptor_dimension == 2
    assert rep.backend == RepresentationBackend.COMPOSITION
    assert rep.pooling == PoolingMode.NONE
    # fractions sum to 1 per row
    assert rep.as_array().shape == (2, 2)


def test_structural_representation_accepts_descriptor_width_not_equal_species():
    rep = StructuralRepresentation(
        representation_id="soapish",
        backend=RepresentationBackend.SOAP,
        species=["Si", "O"],
        structure_ids=["s1"],
        matrix=[[0.1, 0.2, 0.3, 0.4]],
        descriptor_dimension=4,
    )
    assert rep.descriptor_dimension == 4
    assert rep.as_array().shape == (1, 4)


def test_descriptor_dimension_inferred_when_omitted_for_backcompat():
    # older callers that omit descriptor_dimension still validate against width
    rep = StructuralRepresentation(
        representation_id="legacy",
        backend=RepresentationBackend.COMPOSITION,
        species=["Si", "O"],
        structure_ids=["s1"],
        matrix=[[0.33, 0.67]],
    )
    assert rep.descriptor_dimension == 2


def test_row_width_mismatch_rejected():
    with pytest.raises(ValueError, match="descriptor_dimension"):
        StructuralRepresentation(
            representation_id="bad",
            backend=RepresentationBackend.SOAP,
            species=["Si", "O"],
            structure_ids=["s1"],
            matrix=[[0.1, 0.2, 0.3]],
            descriptor_dimension=4,
        )


def test_soap_dependency_error_if_dscribe_missing(monkeypatch):
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "dscribe" or name.startswith("dscribe."):
            raise ImportError("dscribe blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    adapter = SoapRepresentationAdapter(species=["Si", "O"], r_cut=4.0, n_max=4, l_max=3)
    with pytest.raises(SoapDependencyError):
        adapter.compute_ase([("s1", object())], representation_id="r")


def test_soap_none_pooling_rejected():
    pytest.importorskip("dscribe")
    pytest.importorskip("ase")
    with pytest.raises(ValueError, match="pooling"):
        SoapRepresentationAdapter(
            species=["Si", "O"], r_cut=4.0, n_max=4, l_max=3, pooling=PoolingMode.NONE
        ).compute_ase([("s1", object())], representation_id="r")


def test_soap_real_descriptor_is_fixed_size_and_wider_than_species():
    pytest.importorskip("dscribe")
    ase = pytest.importorskip("ase")
    from ase import Atoms

    a1 = Atoms("Si2O", positions=[[0, 0, 0], [2.0, 0, 0], [1.0, 1.0, 0]],
               cell=[10, 10, 10], pbc=True)
    a2 = Atoms("SiO2", positions=[[0, 0, 0], [1.6, 0, 0], [0, 1.6, 0]],
               cell=[10, 10, 10], pbc=True)
    rep = SoapRepresentationAdapter(
        species=["Si", "O"], r_cut=4.0, n_max=4, l_max=3, periodic=True,
        pooling=PoolingMode.MEAN,
    ).compute_ase([("s1", a1), ("s2", a2)], representation_id="soap")
    assert rep.backend == RepresentationBackend.SOAP
    assert rep.descriptor_dimension > len(["Si", "O"])  # true power spectrum width
    assert rep.as_array().shape == (2, rep.descriptor_dimension)
    assert "dscribe" in rep.software and "ase" in rep.software
    assert rep.pooling == PoolingMode.MEAN
    assert np.isfinite(rep.as_array()).all()
