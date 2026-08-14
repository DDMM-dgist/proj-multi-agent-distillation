"""Structural coverage evidence infrastructure (Priority #2), representation-
and campaign-agnostic.

Computes descriptive evidence about how well one directed query population of
local atomic environments is covered by one directed reference population --
e.g. "are candidate/Student-distillation environments supported by the
Teacher's actual training distribution?" (`direction="teacher_support"`), or
"does the Student's own dataset cover the frozen deployment-target
population?" (`direction="deployment_coverage"`). Every piece of evidence is
tagged with an explicit `direction`, `query_population`, and
`reference_population` (see coverage.nn_distance, coverage.report) so two
directed coverage questions can never be silently collapsed into one another.

This package computes EVIDENCE only: raw per-environment nearest-neighbor
distances (global plus per-reference-slice), descriptive statistics
(mean/percentiles/max -- see coverage.aggregate.SUMMARY_STATS), and full
method provenance. It does not choose a pass/fail threshold, an acquisition
count, or a parent-selection policy -- those are later, separately
human-approved decisions.

Local-environment coverage is split into three independent, generic
interfaces, each fully pluggable:

* `coverage.representation.CoverageRepresentation` -- descriptor/vector
  computation and identity only (`coverage.soap_representation.
  SoapCoverageRepresentation`, SOAP via the optional `dscribe` package, see
  pyproject.toml's `[structural-coverage]` extra, is ONE implementation);
* `coverage.distance_policy.DistancePolicy` -- normalization and comparison
  semantics only (`coverage.soap_distance_policy.SoapDistancePolicy` is ONE
  implementation);
* `coverage.search_backend.SearchBackend` -- index build/query mechanics only,
  declaring exact vs. approximate (`coverage.exact_kdtree_backend.
  ExactKDTreeBackend`, always exact, is ONE implementation).

This split is what allows, with no change to coverage.reference_pool,
coverage.nn_distance, coverage.aggregate, coverage.report, or
DataCoverageReport semantics: SOAP + exact cKDTree; SOAP + a future validated
approximate backend; a Teacher-latent representation + exact backend; or a
Teacher-latent representation + a future approximate backend.
Representation-optional-dependency errors (e.g. `dscribe` not installed) must
surface as ModuleNotFoundError at call time, not import time, matching this
repo's existing optional-dependency convention (see runtimes/pydantic_ai for
the same pattern).

Reference pools (coverage.reference_pool) store each environment's
representation vector exactly once (canonical storage); slice/domain
membership references environment positions, never duplicated vectors. A
search backend's own index structure is a derived/cache artifact, not the
source of truth.

Generic modules in this package never read `config_type` or any other fixed,
campaign-specific metadata field; slice/domain membership is always supplied
externally as free-form labels (see coverage.reference_pool,
coverage.nn_distance) by a campaign-specific adapter (see coverage.adapters).

Teacher latent-space coverage remains a documented, deferred alternative (see
configs/provenance/PROVENANCE.md and the Priority #2 design discussion) --
nothing in this package runs Teacher inference.
"""
