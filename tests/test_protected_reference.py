from pathlib import Path

import json
import os
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from validation.protected_reference import (
    assert_dataset_geometry_disjoint,
    assert_parent_lineage_allowed,
    assert_source_indices_allowed,
    _structure_fingerprint,
    validate_protection_audit_report,
)


ROOT = Path(__file__).resolve().parent.parent


def _atoms(x=0.0, parent="seed-pool:10"):
    a = Atoms(
        "SiO2",
        positions=[
            [x, 0.0, 0.0],
            [1.6 + x, 0.0, 0.0],
            [-1.6 + x, 0.0, 0.0],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    a.info["parent_structure_id"] = parent
    return a


def test_source_index_protection_passes():
    assert assert_source_indices_allowed(
        [1, 2, 3],
        {10, 11, 12},
    )


def test_source_index_protection_rejects_overlap():
    with pytest.raises(ValueError, match="protected reference leakage"):
        assert_source_indices_allowed(
            [1, 11, 20],
            {10, 11, 12},
        )


def test_geometry_protection_rejects_exact_reference(tmp_path):
    reference = _atoms(0.0)

    dataset = tmp_path / "dataset.extxyz"
    write(dataset, [reference])

    with pytest.raises(ValueError, match="protected reference geometry"):
        assert_dataset_geometry_disjoint(
            dataset,
            {_structure_fingerprint(reference)},
        )


def test_geometry_protection_accepts_different_structure(tmp_path):
    reference = _atoms(0.0)
    candidate = _atoms(0.2)

    dataset = tmp_path / "dataset.extxyz"
    write(dataset, [candidate])

    assert assert_dataset_geometry_disjoint(
        dataset,
        {_structure_fingerprint(reference)},
    )


def test_parent_lineage_rejects_protected_seed(tmp_path):
    dataset = tmp_path / "dataset.extxyz"

    write(
        dataset,
        [_atoms(parent="seed-pool:761")],
    )

    with pytest.raises(ValueError, match="protected reference descendant"):
        assert_parent_lineage_allowed(
            dataset,
            {760, 761},
        )


def test_parent_lineage_accepts_eligible_seed(tmp_path):
    dataset = tmp_path / "dataset.extxyz"

    write(
        dataset,
        [_atoms(parent="seed-pool:900")],
    )

    assert assert_parent_lineage_allowed(
        dataset,
        {760, 761},
    )


def test_parent_lineage_requires_explicit_lineage(tmp_path):
    a = _atoms()
    del a.info["parent_structure_id"]

    dataset = tmp_path / "dataset.extxyz"
    write(dataset, [a])

    with pytest.raises(ValueError, match="missing parent_structure_id"):
        assert_parent_lineage_allowed(
            dataset,
            {760, 761},
        )



def test_load_protected_indices_reads_non_ascii_content_regardless_of_locale(tmp_path):
    # A hand-annotated protected-indices file containing a non-ASCII character (e.g. an
    # em dash) must be read correctly -- and fail on its own domain validation, not a
    # UnicodeDecodeError -- even in a subprocess whose ambient locale resolves to ASCII.
    path = tmp_path / "protected.txt"
    path.write_bytes("760\n761\n# note — non-ascii annotation\n".encode("utf-8"))
    env = dict(os.environ)
    env.update({"LC_ALL": "C", "LANG": "C", "PYTHONCOERCECLOCALE": "0", "PYTHONUTF8": "0"})
    result = subprocess.run(
        [sys.executable, "-c",
         "from validation.protected_reference import load_protected_indices; import sys; "
         "load_protected_indices(sys.argv[1])", str(path)],
        cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "UnicodeDecodeError" not in result.stderr
    assert "invalid protected source index" in result.stderr


def test_protection_audit_contract_passes(tmp_path):
    dataset = tmp_path / "eligible.extxyz"

    write(
        dataset,
        [_atoms(parent="seed-pool:900")],
    )

    audit = tmp_path / "audit.json"

    audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "acquisition",
                "selected_source_indices": [900],
                "datasets": [
                    {
                        "role": "acquired",
                        "path": str(dataset),
                    }
                ],
            }
        )
    )

    fake_reference = {
        "reference_id": "test-reference",
        "logical_frames": 1,
        "protected_source_rows": 2,
        "protected_source_indices": {760, 761},
        "reference_fingerprints": set(),
    }

    with patch(
        "validation.protected_reference.validate_reference_config",
        return_value=fake_reference,
    ):
        result = validate_protection_audit_report(
            audit,
            reference_yaml=tmp_path / "reference.yaml",
            submitted_artifacts=[audit, dataset],
        )

    assert result["status"] == "PASS"
    assert result["selected_source_indices_checked"] == 1


def test_protection_audit_contract_rejects_protected_parent(tmp_path):
    dataset = tmp_path / "leaked.extxyz"

    write(
        dataset,
        [_atoms(parent="seed-pool:760")],
    )

    audit = tmp_path / "audit.json"

    audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "acquisition",
                "selected_source_indices": [],
                "datasets": [
                    {
                        "role": "acquired",
                        "path": str(dataset),
                    }
                ],
            }
        )
    )

    fake_reference = {
        "reference_id": "test-reference",
        "logical_frames": 1,
        "protected_source_rows": 2,
        "protected_source_indices": {760, 761},
        "reference_fingerprints": set(),
    }

    with patch(
        "validation.protected_reference.validate_reference_config",
        return_value=fake_reference,
    ):
        with pytest.raises(
            ValueError,
            match="protected reference descendant",
        ):
            validate_protection_audit_report(
                audit,
                reference_yaml=tmp_path / "reference.yaml",
                submitted_artifacts=[audit, dataset],
            )
