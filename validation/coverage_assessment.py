"""Typed data-coverage assessability contract (FE-038).

Stage 4 must answer a coverage question with three honest outcomes, never conflating them:

    COVERAGE_SUFFICIENT   every declared coverage dimension has an evaluable criterion AND meets it
    COVERAGE_INSUFFICIENT at least one declared dimension has an evaluable criterion it FAILS
    NOT_ASSESSABLE        at least one declared dimension has no evaluable criterion (and none FAIL)

The two invariants this module exists to enforce:

  * The framework MUST NOT claim FAIL / insufficiency solely because a criterion is absent. A
    dimension with ``criterion_provenance == "absent"`` can only ever be ``NOT_ASSESSABLE``.
  * The framework MUST NOT claim PASS / sufficiency merely because no threshold exists. A ``PASS``
    dimension must name a real criterion (provenance != "absent") and carry the observed support the
    criterion was evaluated against.

It hard-codes NO SiO2-x category name, structure-class quota, minimum frame count, or percentage.
Criteria arrive by provenance precedence (see ``CRITERION_PROVENANCES``); when none is available the
dimension is reported ``NOT_ASSESSABLE`` rather than being filled with an invented constant.
"""

ASSESSMENT_STATUSES = {"COVERAGE_SUFFICIENT", "COVERAGE_INSUFFICIENT", "NOT_ASSESSABLE"}
DIMENSION_STATUSES = {"PASS", "FAIL", "NOT_ASSESSABLE"}
# Precedence order (most authoritative first). "absent" is the fail-closed terminal state: it is
# not a source of a criterion, it is the explicit record that no legitimate criterion was found.
CRITERION_PROVENANCES = (
    "frozen_deployment_domain",   # a frozen coverage_requirement carried by the locked domain
    "frozen_coverage_policy",     # a separately frozen, run-bound coverage policy artifact
    "autonomously_derived",       # an objective-conditioned criterion derived WITH explicit provenance
    "absent",                     # genuinely underdetermined -> NOT_ASSESSABLE, never invented
)
LINEAGE_RESULTS = {"PASS", "FAIL"}
PROTECTION_RESULTS = {"PASS", "FAIL"}


def _nonneg_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def make_dimension(*, dimension_id, declared_target, metric, criterion_provenance,
                   observed_support, criterion=None, reason):
    """Build ONE typed per-dimension assessment record, choosing PASS/FAIL/NOT_ASSESSABLE by the two
    invariants above -- never by prose. ``criterion`` is the concrete evaluable threshold (e.g. a
    minimum frame count) when ``criterion_provenance != 'absent'``; ``met`` decides PASS vs FAIL.
    """
    if criterion_provenance not in CRITERION_PROVENANCES:
        raise ValueError(f"unknown criterion_provenance: {criterion_provenance!r}")
    if criterion_provenance == "absent":
        status = "NOT_ASSESSABLE"
        met = None
    else:
        if criterion is None:
            raise ValueError(
                f"dimension {dimension_id!r} declares provenance {criterion_provenance!r} but carries "
                "no concrete criterion -- a non-absent provenance must supply an evaluable threshold")
        met = bool(criterion.get("met"))
        status = "PASS" if met else "FAIL"
    return {
        "dimension_id": dimension_id,
        "declared_target": declared_target,
        "metric": metric,
        "criterion_provenance": criterion_provenance,
        "criterion": criterion,
        "observed_support": observed_support,
        "assessment_status": status,
        "met": met,
        "reason": reason,
    }


def aggregate_assessment_status(dimensions):
    """Fold typed per-dimension statuses into the report-level tri-state.

    INSUFFICIENT if ANY dimension FAILs (a real, evaluable criterion is unmet). Otherwise SUFFICIENT
    only if EVERY declared dimension PASSes (an evaluable criterion, met). Otherwise NOT_ASSESSABLE
    (at least one dimension had no evaluable criterion, and nothing FAILed). Empty dimension list is
    NOT_ASSESSABLE, never SUFFICIENT.
    """
    statuses = [d.get("assessment_status") for d in dimensions]
    if any(s == "FAIL" for s in statuses):
        return "COVERAGE_INSUFFICIENT"
    if statuses and all(s == "PASS" for s in statuses):
        return "COVERAGE_SUFFICIENT"
    return "NOT_ASSESSABLE"


def build_coverage_assessment(*, teacher_training_data_access, teacher_access_limitations,
                              dimensions, acquisition_lineage, protected_reference_exclusion):
    """Assemble the typed ``coverage_assessment`` block; report-level status is DERIVED from the
    per-dimension records, never independently asserted."""
    return {
        "schema_version": 1,
        "assessment_status": aggregate_assessment_status(dimensions),
        "teacher_training_data_access": {
            "mode": teacher_training_data_access,
            "limitations": list(teacher_access_limitations or []),
        },
        "dimensions": list(dimensions),
        "acquisition_lineage": acquisition_lineage,
        "protected_reference_exclusion": protected_reference_exclusion,
    }


def validate_coverage_assessment(assessment):
    """Deterministically re-check a ``coverage_assessment`` block. Fails closed on any inconsistency,
    and specifically enforces the two never-fabricate invariants. Returns the block on success."""
    if not isinstance(assessment, dict):
        raise ValueError("coverage_assessment must be an object")
    status = assessment.get("assessment_status")
    if status not in ASSESSMENT_STATUSES:
        raise ValueError(f"coverage_assessment.assessment_status must be one of {sorted(ASSESSMENT_STATUSES)}")

    access = assessment.get("teacher_training_data_access")
    if (not isinstance(access, dict) or not isinstance(access.get("mode"), str)
            or not access["mode"].strip()
            or not isinstance(access.get("limitations"), list)
            or any(not isinstance(x, str) for x in access["limitations"])):
        raise ValueError(
            "coverage_assessment.teacher_training_data_access requires a non-empty mode and a "
            "list-of-strings limitations (geometry-only / labels-available / unavailable access must "
            "be stated explicitly with its limitations)")

    dimensions = assessment.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError("coverage_assessment.dimensions must be a non-empty list")
    seen = set()
    for dim in dimensions:
        if not isinstance(dim, dict):
            raise ValueError("each coverage dimension must be an object")
        for field in ("dimension_id", "metric", "reason"):
            if not isinstance(dim.get(field), str) or not dim[field].strip():
                raise ValueError(f"coverage dimension requires a non-empty {field}")
        dim_id = dim["dimension_id"]
        if dim_id in seen:
            raise ValueError(f"duplicate coverage dimension_id: {dim_id!r}")
        seen.add(dim_id)
        if "declared_target" not in dim:
            raise ValueError(f"coverage dimension {dim_id!r} requires declared_target")
        if "observed_support" not in dim:
            raise ValueError(f"coverage dimension {dim_id!r} requires observed_support")
        provenance = dim.get("criterion_provenance")
        if provenance not in CRITERION_PROVENANCES:
            raise ValueError(
                f"coverage dimension {dim_id!r} criterion_provenance must be one of "
                f"{list(CRITERION_PROVENANCES)}")
        dim_status = dim.get("assessment_status")
        if dim_status not in DIMENSION_STATUSES:
            raise ValueError(
                f"coverage dimension {dim_id!r} assessment_status must be one of {sorted(DIMENSION_STATUSES)}")
        # Invariant 1: absent criterion can NEVER be PASS or FAIL.
        if provenance == "absent" and dim_status != "NOT_ASSESSABLE":
            raise ValueError(
                f"coverage dimension {dim_id!r} has criterion_provenance='absent' but "
                f"assessment_status={dim_status!r}; a missing criterion must be NOT_ASSESSABLE, never "
                "a FAIL/insufficiency or a PASS")
        # Invariant 2: PASS/FAIL require a real, evaluable criterion AND observed support.
        if dim_status in ("PASS", "FAIL"):
            if provenance == "absent":
                raise ValueError(
                    f"coverage dimension {dim_id!r} is {dim_status} without an evaluable criterion")
            criterion = dim.get("criterion")
            if not isinstance(criterion, dict) or "met" not in criterion:
                raise ValueError(
                    f"coverage dimension {dim_id!r} is {dim_status} but carries no concrete "
                    "criterion with a 'met' result -- PASS/FAIL must name the threshold evaluated")
            if bool(criterion["met"]) != (dim_status == "PASS"):
                raise ValueError(
                    f"coverage dimension {dim_id!r} assessment_status={dim_status!r} is inconsistent "
                    f"with criterion.met={criterion['met']!r}")
        if dim_status == "NOT_ASSESSABLE" and provenance != "absent":
            # A dimension may only be NOT_ASSESSABLE when no criterion exists; a present criterion
            # must resolve to PASS or FAIL, so we do not silently drop an evaluable result.
            raise ValueError(
                f"coverage dimension {dim_id!r} is NOT_ASSESSABLE but declares criterion_provenance "
                f"{provenance!r}; an evaluable criterion must resolve to PASS or FAIL")

    recomputed = aggregate_assessment_status(dimensions)
    if recomputed != status:
        raise ValueError(
            f"coverage_assessment.assessment_status={status!r} disagrees with the per-dimension "
            f"aggregation {recomputed!r}")

    lineage = assessment.get("acquisition_lineage")
    if not isinstance(lineage, dict):
        raise ValueError("coverage_assessment.acquisition_lineage must be an object")
    for field in ("acquisition_manifest_path", "acquisition_manifest_sha256",
                  "expected_identity", "observed_identity"):
        if not isinstance(lineage.get(field), str) or not lineage[field].strip():
            raise ValueError(f"acquisition_lineage requires a non-empty {field}")
    if lineage.get("equality_result") not in LINEAGE_RESULTS:
        raise ValueError(f"acquisition_lineage.equality_result must be one of {sorted(LINEAGE_RESULTS)}")
    if lineage["equality_result"] == "PASS" and lineage["expected_identity"] != lineage["observed_identity"]:
        raise ValueError(
            "acquisition_lineage.equality_result=PASS but expected_identity != observed_identity")
    if lineage["equality_result"] == "FAIL" and lineage["expected_identity"] == lineage["observed_identity"]:
        raise ValueError(
            "acquisition_lineage.equality_result=FAIL but expected_identity == observed_identity")

    prot = assessment.get("protected_reference_exclusion")
    if not isinstance(prot, dict):
        raise ValueError("coverage_assessment.protected_reference_exclusion must be an object")
    if not isinstance(prot.get("reference_id"), str) or not prot["reference_id"].strip():
        raise ValueError("protected_reference_exclusion requires a non-empty reference_id")
    for field in ("protected_candidate_count", "protected_excluded_count",
                  "eligible_population_after_exclusion", "post_selection_overlap_count"):
        if not _nonneg_int(prot.get(field)):
            raise ValueError(f"protected_reference_exclusion.{field} must be a non-negative integer")
    if prot.get("result") not in PROTECTION_RESULTS:
        raise ValueError(f"protected_reference_exclusion.result must be one of {sorted(PROTECTION_RESULTS)}")
    # A protected row leaking into the selection is a hard FAIL; PASS requires zero overlap.
    if (prot["result"] == "PASS") != (prot["post_selection_overlap_count"] == 0):
        raise ValueError(
            "protected_reference_exclusion.result must be PASS iff post_selection_overlap_count == 0")
    return assessment
