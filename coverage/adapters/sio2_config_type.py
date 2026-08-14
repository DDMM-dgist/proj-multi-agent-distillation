"""SiO2-x campaign example: config_type-based slice-membership adapter.

This is the ONLY place in this repository's generic coverage architecture where
`atoms.info["config_type"]` is read. Generic coverage code
(coverage.reference_pool, coverage.nn_distance, coverage.aggregate,
coverage.report) has no notion of `config_type` at all -- it only accepts
caller-supplied `slice_membership` / `query_slice_labels` sequences of
free-form label strings, keyed by whatever `structure_id`s the caller chose.
This module is the SiO2-x campaign's own concrete choice: one slice per
distinct `config_type` value, single-membership (each structure belongs to
exactly the one slice named after its own `config_type`).

A different campaign (different material system, different slice semantics,
possibly multi-membership slices) needs a different adapter module, not a
change to any generic coverage.* module.
"""
from __future__ import annotations

from typing import Sequence


def config_type_of(atoms) -> str:
    """This campaign's `config_type` convention: `atoms.info["config_type"]`,
    falling back to `atoms.info["source"]`, then the literal string "unknown"
    -- matching the now-superseded coverage.soap_descriptors._config_type
    helper this replaces.
    """
    return str(atoms.info.get("config_type", atoms.info.get("source", "unknown")))


def config_type_slice_membership(structures: Sequence) -> list:
    """Return one single-element slice-membership tuple per structure, suitable
    for `coverage.reference_pool.build_reference_pool`'s `slice_membership` argument.
    """
    return [(config_type_of(atoms),) for atoms in structures]


def config_type_slice_labels_by_id(structures: Sequence, structure_ids: Sequence) -> dict:
    """Return a structure_id -> (config_type,) mapping, suitable for
    `coverage.nn_distance.compute_environment_distances`'s `query_slice_labels` argument.
    """
    if len(structures) != len(structure_ids):
        raise ValueError("structures and structure_ids must have the same length")
    membership = config_type_slice_membership(structures)
    return dict(zip(structure_ids, membership))
