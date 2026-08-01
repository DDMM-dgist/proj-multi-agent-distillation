"""Run-bound, architecture-neutral review lenses for Judge committees."""
import re
from collections.abc import Mapping


DEFAULT_REVIEW_LENSES = (
    {
        "id": "evidence_provenance",
        "title": "Evidence and provenance",
        "focus": (
            "Audit artifact identity, hashes, lineage, split leakage, checkpoint binding, "
            "configuration provenance, and completeness of the evidence chain."
        ),
    },
    {
        "id": "scientific_validity",
        "title": "Scientific validity",
        "focus": (
            "Audit units, metrics, thresholds, Teacher applicability, reference anchors, "
            "physical plausibility, and whether the scientific interpretation follows."
        ),
    },
    {
        "id": "reproducibility_deployment",
        "title": "Reproducibility and deployment risk",
        "focus": (
            "Audit executable settings, reproducibility, uncertainty or OOD interpretation, "
            "deployment stability, operational limitations, and whether claims stay in scope."
        ),
    },
)

_LENS_ID = re.compile(r"^[a-z][a-z0-9_-]*$")


def normalize_review_lenses(value=None):
    """Return exactly three unique review-lens records.

    A run may replace the default names and focus text, but not the three-slot
    committee contract. This keeps the controller decision rule and audit
    bundle stable while allowing domain-specific review perspectives.
    """
    raw = DEFAULT_REVIEW_LENSES if value is None else value
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError("gate review_lenses must contain exactly three mappings")
    lenses = []
    for index, item in enumerate(raw, 1):
        if not isinstance(item, Mapping):
            raise ValueError(f"gate review_lenses item {index} must be a mapping")
        if set(item) != {"id", "title", "focus"}:
            raise ValueError(
                f"gate review_lenses item {index} requires exactly id, title, and focus"
            )
        if any(not isinstance(item[key], str) for key in ("id", "title", "focus")):
            raise ValueError("gate review lens id, title, and focus must be strings")
        lens = {key: item[key].strip() for key in ("id", "title", "focus")}
        if not _LENS_ID.fullmatch(lens["id"]):
            raise ValueError(
                f"gate review lens id must use lowercase letters, digits, '_' or '-': {lens['id']!r}"
            )
        if not lens["title"] or not lens["focus"]:
            raise ValueError("gate review lens title and focus must be non-empty")
        lenses.append(lens)
    ids = [lens["id"] for lens in lenses]
    if len(set(ids)) != 3:
        raise ValueError("gate review lens ids must be unique")
    return lenses
