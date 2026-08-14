"""Descriptive summary statistics over generic directed nearest-neighbor distance evidence.

SUMMARY_STATS is a fixed, comprehensive set of descriptive statistics -- not a
policy choice, unchanged from the previous per-category implementation.
Computing mean/percentiles/max together is pure reporting; it does not "bless"
any single one of them (e.g. max, or p95) as the eventual pass/fail decision
metric. Choosing which statistic (and which view -- global vs. any particular
reference/query slice) to act on, and at what threshold, is a later,
human-approved policy step and is deliberately out of scope here.

Every grouping function here operates on generic `EnvironmentDistanceRecord`s
(see coverage.nn_distance) and their free-form `query_slice_labels`/
`slice_distances` fields -- never on `config_type` or any other fixed metadata
field. Records with no eligible reference match (`matched=False`, distance is
`None`) are excluded from the numeric summary but their count -- and, as of
this refactor, their fraction of the total -- are always reported via
`n_unmatched`/`unmatched_fraction`, so an unmatched central species, an
unmatched periodicity, or any other opaque compatibility_key with no eligible
reference environment (see coverage.nn_distance) is visible, quantitative
evidence rather than being silently dropped or reported as a bare count with
no denominator.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

SUMMARY_STATS = ("mean", "p50", "p75", "p90", "p95", "p99", "max")


def summarize(values) -> dict:
    """Compute the fixed SUMMARY_STATS set over `values`. Raises on empty input."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("summarize requires at least one value")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _summarize_with_unmatched(values, n_unmatched: int) -> dict:
    result = summarize(values) if values else {"n": 0}
    total = result["n"] + n_unmatched
    result["n_unmatched"] = n_unmatched
    result["unmatched_fraction"] = (n_unmatched / total) if total else 0.0
    return result


def overall_global_summary(records: list) -> dict:
    """Global (mandatory-index) distance summary across all query environments."""
    matched = [r.global_distance for r in records if r.global_matched]
    unmatched = sum(1 for r in records if not r.global_matched)
    return _summarize_with_unmatched(matched, unmatched)


def structure_global_summaries(records: list) -> dict:
    """Per-query-structure global-distance summary: query_structure_id -> summarize(...)."""
    by_structure = defaultdict(list)
    unmatched_by_structure = defaultdict(int)
    for record in records:
        if record.global_matched:
            by_structure[record.query_structure_id].append(record.global_distance)
        else:
            unmatched_by_structure[record.query_structure_id] += 1
    all_structure_ids = set(by_structure) | set(unmatched_by_structure)
    return {
        sid: _summarize_with_unmatched(by_structure.get(sid, []), unmatched_by_structure.get(sid, 0))
        for sid in all_structure_ids
    }


def query_slice_resolved_summaries(records: list) -> dict:
    """Global-distance summary grouped by each QUERY-side slice label.

    A record contributes to every label in its (possibly multi-membership)
    `query_slice_labels` -- this is deliberate: overlapping query slices are
    not required to partition the query population.
    """
    by_label = defaultdict(list)
    unmatched_by_label = defaultdict(int)
    for record in records:
        for label in record.query_slice_labels:
            if record.global_matched:
                by_label[label].append(record.global_distance)
            else:
                unmatched_by_label[label] += 1
    all_labels = set(by_label) | set(unmatched_by_label)
    return {
        label: _summarize_with_unmatched(by_label.get(label, []), unmatched_by_label.get(label, 0))
        for label in all_labels
    }


def reference_slice_resolved_summaries(records: list) -> dict:
    """Summary of distances to each individual REFERENCE slice, over all query environments.

    This is the raw material for slice/domain-balanced reweighting: a
    per-slice summary computed once here can be recombined into any custom
    weighting across reference slices later, without re-querying any index.
    """
    by_slice = defaultdict(list)
    unmatched_by_slice = defaultdict(int)
    for record in records:
        for slice_name, distance in record.slice_distances.items():
            if record.slice_matched[slice_name]:
                by_slice[slice_name].append(distance)
            else:
                unmatched_by_slice[slice_name] += 1
    all_slices = set(by_slice) | set(unmatched_by_slice)
    return {
        slice_name: _summarize_with_unmatched(
            by_slice.get(slice_name, []), unmatched_by_slice.get(slice_name, 0)
        )
        for slice_name in all_slices
    }
