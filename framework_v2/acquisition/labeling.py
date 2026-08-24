"""Framework V2 -- CanonicalTeacherLabeling request (provenance separation).

The selected candidates must be (re-)labeled canonically under the frozen
Teacher before any of them can become training data. This is the ONLY
provenance training labels may originate from. Any exploration PES recorded
during candidate generation (augment-atoms relaxation energies, MD-trajectory
energies) is discarded for labeling purposes -- it is never conflated with a
training label (Section K).

This module produces the ``CanonicalLabelingRequest`` that instructs the
downstream teacher_labeling stage; the actual labeling is executed by the
canonical Teacher-labeling stage under the existing frozen-Teacher execution
path, not here.
"""
from __future__ import annotations

from framework_v2.acquisition.contracts import (
    CandidateSelectionResult,
    CanonicalLabelingRequest,
)


def build_labeling_request(
    *,
    request_id: str,
    selection_result: CandidateSelectionResult,
    teacher_identity_sha256: str,
) -> CanonicalLabelingRequest:
    """Emit the canonical relabeling instruction for the selected candidates.

    ``relabel_from_scratch`` is fixed True by the contract validator: labels
    are always produced canonically under the frozen Teacher, never reused
    from exploration dynamics."""
    return CanonicalLabelingRequest(
        request_id=request_id,
        selection_result_sha256=selection_result.content_sha256(),
        teacher_identity_sha256=teacher_identity_sha256,
        candidate_ids=list(selection_result.selected_candidate_ids),
        relabel_from_scratch=True,
    )
