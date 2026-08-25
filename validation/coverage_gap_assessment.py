"""Gap-based Stage-4 structural coverage assessment (FE-039).

Stage 3 proposes an INITIAL acquisition population with the generic, material-independent
FPS / marginal-novelty sizing heuristic (``generic_sizing_v1``); the size it returns is an
autonomous *starting proposal*, never an accepted adequacy result. Stage 4 is the INDEPENDENT
configuration-space adequacy gate: does the ACQUIRED population structurally SUPPORT each
declared deployment structure class of the FROZEN scope?

The only support signal used here is DECLARED-CLASS OCCUPANCY, resolved through the frozen,
human-authored ``config_type -> canonical structure-class`` ``label_map`` carried by the locked
deployment scope (criterion provenance ``frozen_deployment_domain``). A declared structure class
with ZERO acquired representatives is UNSUPPORTED. That is a definitional presence/absence fact
-- you cannot claim to cover a region you sampled zero times -- NOT an invented minimum-N,
per-class quota, percentage, or descriptor distance threshold. When no frozen ``label_map`` is
bound, per-class support is honestly ``NOT_ASSESSABLE`` (never a fabricated PASS, never a false
insufficiency).

This module hard-codes NO SiO2-x class name, NO frame count, and NO material-specific constant.
It reasons only over the frozen label_map's own entries and the observed per-``config_type``
acquired counts. The ``label_map`` distinguishes real declared classes from the frozen scope's
own non-class sentinels (``OUT_OF_SCOPE_DIAGNOSTIC``) and its fail-closed ``AMBIGUOUS`` entries;
this module fails those closed rather than silently crediting them to any declared class.
"""
from __future__ import annotations

# Canonical-domain sentinels the frozen scope uses for raw labels that are NOT one of the
# declared deployment structure classes. Neither can ever count as structural support.
OUT_OF_SCOPE_DOMAIN = "OUT_OF_SCOPE_DIAGNOSTIC"
AMBIGUOUS_DOMAIN = "AMBIGUOUS"
_NON_CLASS_DOMAINS = frozenset({OUT_OF_SCOPE_DOMAIN, AMBIGUOUS_DOMAIN})

# The definitional presence bound: a declared class must have at least one acquired representative
# to be structurally supported at the occupancy dimension. This is presence/absence, NOT a quota.
MIN_STRUCTURAL_OCCUPANCY = 1


def build_label_index(label_map):
    """Index a frozen ``label_map`` (list of ``{raw_label, canonical_domain, claim_role, ...}``)
    as ``raw_label -> entry``. Fails closed on a malformed entry or a duplicated ``raw_label`` that
    disagrees on ``canonical_domain`` (an internally inconsistent frozen map must never be silently
    disambiguated)."""
    index: dict[str, dict] = {}
    for entry in label_map or []:
        if not isinstance(entry, dict):
            raise ValueError("each label_map entry must be an object")
        raw = entry.get("raw_label")
        dom = entry.get("canonical_domain")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("label_map entry requires a non-empty raw_label")
        if not isinstance(dom, str) or not dom.strip():
            raise ValueError(f"label_map entry {raw!r} requires a non-empty canonical_domain")
        prior = index.get(raw)
        if prior is not None and prior.get("canonical_domain") != dom:
            raise ValueError(
                f"label_map maps raw_label {raw!r} to two canonical_domains "
                f"({prior.get('canonical_domain')!r} and {dom!r}); the frozen map is inconsistent")
        index[raw] = entry
    return index


def resolve_config_type_domain(config_type, label_index):
    """Resolve one acquired ``config_type`` to a declared canonical structure class, or to a
    fail-closed reason. Returns ``(canonical_domain, claim_role)`` for a real declared class; returns
    ``(None, reason)`` where ``reason`` is one of ``"unmapped"``, ``"out_of_scope_diagnostic"``,
    ``"ambiguous"`` -- none of which credits structural support to any declared class."""
    entry = label_index.get(config_type)
    if entry is None:
        return None, "unmapped"
    dom = entry.get("canonical_domain")
    if dom == OUT_OF_SCOPE_DOMAIN:
        return None, "out_of_scope_diagnostic"
    if dom == AMBIGUOUS_DOMAIN:
        return None, "ambiguous"
    return dom, entry.get("claim_role", "primary_claim")


def compute_structure_class_occupancy(acquired_config_type_counts, declared_structure_classes,
                                      label_map):
    """Fold observed per-``config_type`` acquired counts into per-declared-class occupancy via the
    frozen label_map. Frames whose ``config_type`` is unmapped / out-of-scope / ambiguous, or maps to
    a domain that is not a declared class, are recorded separately (never credited to a class)."""
    index = build_label_index(label_map)
    declared = list(declared_structure_classes or [])
    declared_set = set(declared)
    occupancy = {c: 0 for c in declared}
    unmapped: dict[str, int] = {}
    out_of_scope: dict[str, int] = {}
    ambiguous: dict[str, int] = {}
    mapped_outside_declared: dict[str, int] = {}
    for config_type, count in (acquired_config_type_counts or {}).items():
        n = int(count)
        domain, reason = resolve_config_type_domain(config_type, index)
        if domain is None:
            if reason == "ambiguous":
                ambiguous[config_type] = n
            elif reason == "out_of_scope_diagnostic":
                out_of_scope[config_type] = n
            else:
                unmapped[config_type] = n
            continue
        if domain in declared_set:
            occupancy[domain] += n
        else:
            mapped_outside_declared[config_type] = n
    return {
        "occupancy": occupancy,
        "unmapped": unmapped,
        "out_of_scope": out_of_scope,
        "ambiguous": ambiguous,
        "mapped_outside_declared_classes": mapped_outside_declared,
    }


def build_structure_class_dimensions(declared_structure_classes, acquired_config_type_counts,
                                     label_map):
    """Build one FE-038 typed coverage dimension per declared deployment structure class.

    With a frozen ``label_map``: each class gets a ``frozen_deployment_domain`` presence criterion --
    PASS iff at least one acquired frame maps to it, FAIL (zero-occupancy => UNSUPPORTED) otherwise.
    Without a frozen ``label_map``: every class is ``NOT_ASSESSABLE`` (criterion absent), exactly as
    before -- no fabricated PASS, no false insufficiency."""
    from validation.coverage_assessment import make_dimension

    declared = list(declared_structure_classes or [])
    if not label_map:
        return [
            make_dimension(
                dimension_id=f"structure_class:{c}",
                declared_target={"structure_class": c},
                metric="declared_structure_class_occupancy",
                criterion_provenance="absent",
                observed_support={"note": ("no frozen config_type->structure_class label_map is "
                                           "bound; per-class structural support is not assessable"),
                                  "config_type_counts": dict(acquired_config_type_counts or {})},
                reason=("declared deployment structure class carries no frozen or derivable "
                        "occupancy criterion; structural support is not assessable"))
            for c in declared
        ]

    result = compute_structure_class_occupancy(acquired_config_type_counts, declared, label_map)
    occupancy = result["occupancy"]
    dimensions = []
    for c in declared:
        n = int(occupancy.get(c, 0))
        met = n >= MIN_STRUCTURAL_OCCUPANCY
        criterion = {
            "kind": "structural_presence",
            "min_occupancy": MIN_STRUCTURAL_OCCUPANCY,
            "observed_occupancy": n,
            "met": met,
        }
        dimensions.append(make_dimension(
            dimension_id=f"structure_class:{c}",
            declared_target={"structure_class": c},
            metric="declared_structure_class_occupancy_via_frozen_label_map",
            criterion_provenance="frozen_deployment_domain",
            criterion=criterion,
            observed_support={
                "acquired_occupancy": n,
                "config_type_counts": dict(acquired_config_type_counts or {}),
                "unmapped_config_types": result["unmapped"],
                "out_of_scope_config_types": result["out_of_scope"],
                "ambiguous_config_types": result["ambiguous"],
            },
            reason=(f"{n} acquired frame(s) map to this declared class via the frozen label_map"
                    if met else
                    "no acquired frame maps to this declared deployment structure class; "
                    "zero-occupancy => structurally UNSUPPORTED (definitional presence, not a quota)")))
    return dimensions


def unsupported_structure_classes(dimensions):
    """Extract the declared structure classes whose occupancy dimension FAILed (UNSUPPORTED).

    Reads the same typed FE-038 dimension records the coverage verdict aggregated, so the recovery
    RootCause names exactly the classes the gate found unsupported -- never a re-derived guess."""
    out = []
    for dim in dimensions or []:
        target = dim.get("declared_target") or {}
        sc = target.get("structure_class")
        if sc is not None and dim.get("assessment_status") == "FAIL":
            out.append(sc)
    return out


def derive_reacquisition_targets(unsupported_classes, label_map, pool_config_type_counts, *,
                                 already_eligible_source_categories=(), primary_claim_only=True):
    """Deterministically derive which source ``config_type`` families a targeted REACQUISITION must
    widen its eligibility to, in order to remediate the unsupported declared classes.

    For each unsupported class, the target families are the pool ``config_type``s the FROZEN label_map
    maps to that class (optionally restricted to ``primary_claim`` evidence) that still have available
    candidates in the pool. A class with NO such family is ``unremediable`` from the current pool --
    a genuine scientific boundary surfaced (never fabricated away). ``widened_eligible_source_categories``
    are the NEW families (not already eligible) the reacquisition adds; when it is non-empty the
    reacquisition necessarily supersedes the prior plan with a genuinely changed eligibility (so it
    cannot dispatch a byte-identical acquisition). ``materializable`` is True iff at least one new
    target family exists to acquire from.

    This function invents NO frame count and selects NO frame identity: the additional population size
    and the additional frame identities remain the autonomous job of the acquisition sizing/selection
    machinery over the widened, protected-reference-excluded, already-acquired-excluded pool."""
    index = build_label_index(label_map)
    pool_counts = {str(k): int(v) for k, v in (pool_config_type_counts or {}).items()}
    already = set(already_eligible_source_categories or ())

    targets: dict[str, list[str]] = {}
    unremediable: list[str] = []
    widened: set[str] = set()
    for c in unsupported_classes or []:
        families = sorted(
            raw for raw, entry in index.items()
            if entry.get("canonical_domain") == c
            and (not primary_claim_only or entry.get("claim_role") == "primary_claim")
            and pool_counts.get(raw, 0) > 0)
        if not families:
            unremediable.append(c)
            continue
        targets[c] = families
        widened.update(f for f in families if f not in already)

    return {
        "target_config_types_by_class": targets,
        "unremediable_classes": unremediable,
        "widened_eligible_source_categories": sorted(widened),
        "materializable": bool(widened),
    }


__all__ = [
    "OUT_OF_SCOPE_DOMAIN",
    "AMBIGUOUS_DOMAIN",
    "MIN_STRUCTURAL_OCCUPANCY",
    "build_label_index",
    "resolve_config_type_domain",
    "compute_structure_class_occupancy",
    "build_structure_class_dimensions",
    "unsupported_structure_classes",
    "derive_reacquisition_targets",
]
