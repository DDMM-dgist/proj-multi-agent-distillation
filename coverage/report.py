"""Assembles JSON-serializable directed structural coverage evidence payloads.

This module computes EVIDENCE only -- descriptive statistics and provenance. It
does not choose a pass/fail threshold, an acquisition count, or a
parent-selection policy (see coverage/__init__.py).

There is exactly ONE generic evidence-builder, `build_directed_coverage_evidence`,
rather than one hardcoded function per fixed channel name -- a campaign wires it
up once per directed coverage question it cares about, e.g.:

* `direction="teacher_support"`, `query_population="candidate_population"`,
  reference_pool built with `population_role="teacher_train_partition"` --
  are candidate/Student-distillation environments supported by the Teacher's
  actual training distribution?
* `direction="deployment_coverage"`, `query_population="deployment_target_population"`,
  reference_pool built with `population_role="student_training_dataset"" --
  does the Student's own dataset cover the frozen deployment-target population?
* any additional direction a campaign defines, e.g. a second Teacher-side
  direction (relevant Teacher-train/deployment-supported environments ->
  Student dataset) -- kept distinct from candidate->Teacher support rather
  than collapsed into one "Teacher training coverage" channel.

When `reference_pool.population_role == "teacher_train_partition"`, this module
still enforces the validation/test exclusion rule as a defense-in-depth check
(the primary enforcement point for schema-level validation is
validation.data_coverage) -- `excluded_partitions` must positively record that
exclusion, never leave it implicit.

`protected_reference_pointer` returns a pointer-only reference to the
independent DFT validation channel (validation.reference_validation) -- it
never recomputes or merges that channel's evidence into this one.
"""
from __future__ import annotations

from coverage import aggregate
from coverage.reference_pool import ReferencePool

_TEACHER_TRAIN_PARTITION_ROLE = "teacher_train_partition"
_REQUIRED_TEACHER_TRAIN_EXCLUSIONS = ("validation", "test")


def _provenance(reference_pool: ReferencePool) -> dict:
    return {
        "representation_provenance": reference_pool.representation_provenance,
        "representation_hash": reference_pool.representation_hash,
        "search_backend_provenance": reference_pool.search_backend_provenance,
        "reference_manifest_sha256": reference_pool.reference_manifest_sha256,
        "reference_slice_counts": reference_pool.slice_counts(),
        "reference_total_atoms": reference_pool.total_atoms,
        "reference_total_frames": reference_pool.total_frames,
    }


def build_directed_coverage_evidence(
    direction: str,
    query_population: str,
    reference_pool: ReferencePool,
    records: list,
    excluded_partitions: tuple = (),
) -> dict:
    """Build one directed coverage-evidence payload.

    `records` must be the `EnvironmentDistanceRecord`s (see coverage.nn_distance)
    produced by querying `reference_pool` for exactly this `direction`/
    `query_population` pair -- every record's own `direction`/`query_population`/
    `reference_population` is cross-checked against the arguments here so a
    caller cannot accidentally assemble evidence for the wrong directed question.
    """
    if not records:
        raise ValueError("cannot build directed coverage evidence from zero query environment records")
    mismatched = [
        r for r in records
        if r.direction != direction
        or r.query_population != query_population
        or r.reference_population != reference_pool.population_role
    ]
    if mismatched:
        raise ValueError(
            "one or more records do not match the requested "
            f"direction={direction!r}/query_population={query_population!r}/"
            f"reference_population={reference_pool.population_role!r} -- refusing to assemble "
            "evidence that could silently mix directed coverage questions"
        )
    if reference_pool.population_role == _TEACHER_TRAIN_PARTITION_ROLE:
        missing = set(_REQUIRED_TEACHER_TRAIN_EXCLUSIONS) - set(excluded_partitions)
        if missing:
            raise ValueError(
                f"reference_pool.population_role={_TEACHER_TRAIN_PARTITION_ROLE!r} requires "
                f"excluded_partitions to include {_REQUIRED_TEACHER_TRAIN_EXCLUSIONS}, "
                f"missing {sorted(missing)}"
            )

    return {
        "direction": direction,
        "query_population": query_population,
        "reference_population": reference_pool.population_role,
        "provenance": _provenance(reference_pool),
        "excluded_partitions": list(excluded_partitions),
        "n_query_environments": len(records),
        "n_query_structures": len({record.query_structure_id for record in records}),
        "overall_global_summary": aggregate.overall_global_summary(records),
        "structure_global_summaries": aggregate.structure_global_summaries(records),
        "query_slice_resolved_summaries": aggregate.query_slice_resolved_summaries(records),
        "reference_slice_resolved_summaries": aggregate.reference_slice_resolved_summaries(records),
    }


def protected_reference_pointer(
    protected_reference_report_path: str, *, report_sha256: str,
    logical_frames: int, protected_source_rows: int,
) -> dict:
    """Return a pointer-only reference to the independent DFT validation channel.

    Does not recompute, re-read, or merge validation.reference_validation's evidence,
    and never hardcodes a campaign's frame counts -- ``logical_frames``,
    ``protected_source_rows``, and ``report_sha256`` must be the real, hash-verified
    values the caller already obtained from that channel's own report (see
    validation.reference_validation.validate_reference_validation_report), so this
    pointer can never silently drift from that channel's own hash-bound evidence.
    """
    from validation.data_coverage import PROTECTED_REFERENCE_POINTER_ROLE

    return {
        "channel": "protected_reference_status",
        "role": PROTECTED_REFERENCE_POINTER_ROLE,
        "report_path": protected_reference_report_path,
        "report_sha256": report_sha256,
        "required_logical_frames": int(logical_frames),
        "required_protected_source_rows": int(protected_source_rows),
    }
