# Structural coverage (Priority #2) -- production cost/storage estimate (revised)

This is a durable, version-controlled estimate produced BEFORE any full-scale
directed-coverage computation has been run, revised after the generic,
directed, representation-agnostic architecture correction (see
`coverage/representation.py`, `coverage/reference_pool.py`,
`coverage/nn_distance.py`). It does not run, schedule, or authorize the
full-scale computation itself -- that remains a separate, later step. It is
grounded in numbers measured directly in this repo/session, not Big-O
guesswork, and reuses the same underlying measurements as the prior estimate
(the measurements themselves did not change; what changed is which
populations are compared, in which role, and how many index passes each
directed question now requires).

## Generic per-direction cost model

Nothing in this estimate assumes SOAP is the only representation, or that any
specific material system's counts are architectural facts. The reusable,
representation-agnostic cost model is:

```
per_direction_query_cost ~= n_query_environments * n_reference_environments * per_pair_rate
```

where `per_pair_rate` is an empirically measured, representation- and
hardware-specific constant (below, measured for SOAP D=144 features on this
machine). A different representation (different feature dimensionality,
different distance backend) requires re-measuring `per_pair_rate`, not a
change to this formula.

**Real, generalization-relevant addition**: `coverage.reference_pool` now
always builds a *mandatory global index* over the full reference population,
**plus** one additional sub-index per distinct slice label that exists in
`slice_membership` -- and `coverage.nn_distance` queries the global index
**and every existing slice index** for every query environment (see
`coverage/reference_pool.py`, `coverage/nn_distance.py` module docstrings).
This is different from, and more expensive than, the previous
category-tree-only design:

* If slices **partition** the reference population with no overlap (as the
  current SiO2-x campaign's per-`config_type` adapter does -- see
  `coverage/adapters/sio2_config_type.py`), total reference atoms scanned per
  query environment = (1x reference population, via the global index) + (1x
  reference population, split across the partitioning slice indices) = **2x**
  what the old category-tree-only design scanned per query environment.
* If a campaign's slice adapter produces **overlapping, multi-membership**
  slices (explicitly allowed by the new architecture -- see
  `coverage/representation.py`), the slice-index term can exceed 1x reference
  population, scaling roughly with the average number of slices a reference
  environment belongs to. This is a real, disclosed cost/architecture
  trade-off of supporting multi-membership slices, not an oversight: a
  campaign that wants the old "exactly 1x" query cost should keep its slice
  adapter partitioning (as the SiO2-x example does), not overlapping.

## Real inputs used

**Measured benchmark** (this machine, `scipy.spatial.cKDTree`, D=144, float64,
`query(..., k=1, workers=-1)` -- i.e. already using all available CPU cores):

| Operation | Scale | Measured wall time |
|---|---|---|
| Tree build | N=250,000 points | 0.704 s |
| Batch query | 50,000 queries vs. N_ref=250,000 | 290.863 s -> 5,817.26 us/query |

This benchmark is unchanged by the directional/architecture correction --
`per_pair_rate` is a property of the representation and search backend, not of
which direction or population is being queried.

**Measured SOAP descriptor cost** (unchanged from the prior estimate,
benchmark-only parameters, NOT proposed as production defaults): ~0.012
ms/atom, scaling linearly in total atom count.

**Illustrative example counts**, presented ONLY as one campaign's numbers, not
as architectural facts (see `seed_pool_11424` and
`configs/provenance/teacher_training_split_manifest.json`):

| Population | Frames | Atoms | Role in this example |
|---|---|---|---|
| Teacher-train partition | 9,140 | 1,009,444 | reference for `teacher_support` |
| Candidate/Student-distillation population | ~2,206 (unverified) | ~243k-332k (density-based estimate, unverified) | query for `teacher_support` |
| Frozen deployment-target population | 2,134 | 321,256 | query for `deployment_coverage` |
| Current Student/distillation dataset | **not established in this repo** | **not established in this repo** | reference for `deployment_coverage` |

The last row is a genuine, disclosed gap, not an oversight: `deployment_coverage`'s
reference population is now the Student's own accumulated training dataset (a
cumulative, per-round-growing population), which is a *different* quantity
from "candidates awaiting evaluation in one round" -- the ~243k-332k figure
above must NOT be silently reused as a stand-in for it. Until a campaign
records its actual current Student dataset size, `deployment_coverage`'s cost
below is given as "same order of magnitude as the candidate estimate," an
explicitly flagged placeholder, not a verified number.

## Directed compute cost estimate

Applying the generic model with the doubled (partitioning-slice) query-cost
factor derived above:

| Direction | Query population (count) | Reference population (count) | Estimated wall time (with mandatory global + partitioning slice indices) |
|---|---|---|---|
| `teacher_support` | candidate (~243k-332k, unverified) | Teacher-train partition (1,009,444) | ~190-260 min (2x the prior single-pass ~95-130 min) |
| `deployment_coverage` | deployment-target (321,256) | current Student dataset (**unverified size** -- using the candidate-count order of magnitude as an explicitly flagged placeholder) | ~60-82 min (2x the prior single-pass ~30-41 min) |
| **Combined (both directions, one pass each)** | | | **~250-340 min (~4.2-5.7 hours)**, roughly double the prior ~2-3 hour estimate |

**The corrected architecture does alter the estimate**: the total combined
wall time roughly **doubles** relative to the prior estimate, not because the
two directions individually got more expensive per population pair compared,
but because `coverage.reference_pool`/`coverage.nn_distance` now always
additionally build and query a mandatory global index on top of whatever
slice indices exist (see above) -- this is the real, load-bearing
architectural cost of always guaranteeing a correct "distance to nearest
reference environment overall" even when slices are allowed to be
multi-membership and non-partitioning.

**Optional second Teacher-side direction** (relevant Teacher-train/
deployment-supported environments -> Student dataset, kept distinct from
`teacher_support` per the corrected design -- see `coverage/report.py`'s
module docstring): if a campaign computes this as a third directed pass, its
cost is bounded above by (a subset of the 1,009,444-atom Teacher-train
population) x (current Student dataset size), i.e. no larger than the
`teacher_support` estimate above, and likely substantially smaller in
practice since "relevant/deployment-supported" is expected to be a filtered
subset, not the full Teacher-train partition. This is presented as an
upper-bound-only estimate: a campaign choosing to compute this direction
should re-measure with its actual subset size before relying on this bound.

SOAP descriptor computation itself remains comparatively negligible (~20-25 s
for the full illustrative-example atom counts across both directions) --
unchanged from the prior estimate, since descriptor cost is independent of how
many index passes the NN-query step performs.

All of the above already reflects `workers=-1` (full local CPU parallelism);
it is a single-machine, no-GPU estimate.

## Storage estimate (revised)

Because each reference pool now stores a mandatory global index PLUS one
sub-index per existing slice, and `scipy.spatial.cKDTree` retains its own copy
of the input vector array, reference-pool storage also roughly **doubles**
relative to the prior estimate when slices partition the reference population
(as the SiO2-x per-`config_type` adapter's slices do):

| Artifact | Prior estimate | Revised estimate (global + partitioning slices) | Persisted? |
|---|---|---|---|
| Teacher-train reference pool (1,009,444 x 144 float64) | ~1.08 GiB | ~2.16 GiB | only if cached across runs |
| Current-Student-dataset reference pool (size TBD, illustrative ~243k-332k x 144 float64) | ~280-380 MiB (candidate-pool figure, different role) | ~560 MiB-760 MiB (as a reference pool, global + slices) | only if cached across runs |
| Per-environment directed distance evidence (global + all existing reference slices, both directions, float64) | ~93-128 MiB | ~150-210 MiB (extra column for the now-always-present global distance) | yes -- durable raw evidence artifact |
| Frame-level / slice-resolved summary JSON | KB-scale | KB-scale, unchanged | yes |

If a campaign's slice adapter produces overlapping, multi-membership slices,
storage scales further with the average slice-membership count per reference
environment, same as the query-cost scaling described above.

**Peak RAM**, holding both reference pools and one query batch in memory
simultaneously: still on the order of **4-6 GiB** (revised up from 2-3 GiB to
reflect the doubled per-pool storage) -- still no specialized hardware
required.

Cached reference pools remain invalidated only when a representation's
`representation_hash()` (covering descriptor parameters, distance/normalization
policy, and library version -- see `coverage/representation.py`,
`coverage/soap_distance_policy.py`) or the reference manifest's sha256
changes, both already recorded in `coverage.report`'s provenance block, so
cache-invalidation remains a direct hash comparison, not a new mechanism.

## Known limitation, disclosed rather than resolved

Exact k-d trees remain not an efficient search structure at D=144 -- the
measured rate is consistent with brute-force-like scanning, not logarithmic
pruning. This is unchanged by the directional/architecture correction, since
it is a property of the representation's feature dimensionality and the exact
search backend, not of which populations are compared or in which role.

This estimate still does not propose approximate nearest-neighbor methods
(e.g. dimensionality reduction, FAISS/ANN) as a fix. Approximate NN remains a
documented, **not-yet-implemented** future option: a campaign MAY propose it
as an alternative `CoverageRepresentation`/index backend, but it must never
silently substitute an approximate result for exact NN in place of the
current exact-cKDTree behavior without an explicit, separately-approved
accuracy-validation policy (e.g. a documented bound on approximate-vs-exact
distance error, verified on a held-out sample) -- that policy does not exist
yet and is out of scope for this Priority #2 evidence-computation step.

## Framework interface refactor (Representation / DistancePolicy / SearchBackend split) -- cost impact

After this estimate was written, `coverage.representation.CoverageRepresentation`
was split into three independent interfaces: `CoverageRepresentation`
(vector/identity construction only), `coverage.distance_policy.DistancePolicy`
(normalization/metric semantics only), and `coverage.search_backend.SearchBackend`
(index build/query mechanics only, with `coverage.exact_kdtree_backend.
ExactKDTreeBackend` as the promoted, now-standalone cKDTree implementation).
This is a pure interface/module-boundary change, not a change to the
underlying algorithm:

- **Time estimate: unchanged.** The same number of search indices is built
  and queried per level (global, plus one per existing slice) -- what was
  previously "one cKDTree per central species, decided inside
  `SoapCoverageRepresentation`" is now "one `ExactKDTreeBackend` index per
  opaque `compatibility_key`, decided generically in `coverage.reference_pool`"
  -- for the SiO2-x campaign's actual policy settings, this produces the
  identical per-species tree cardinality as before, so the directed-compute
  estimates above are unaffected by this refactor.
- **Storage estimate: unchanged in aggregate, clarified in composition.**
  `coverage.reference_pool.ReferencePool` now stores each environment's
  (distance-policy-normalized) vector exactly ONCE, in a canonical batch;
  slice membership references index positions, never a duplicated vector
  array. This removes one redundant Python-level copy layer that the prior
  `_filter_batch`-based implementation created per slice, but does NOT
  change the dominant storage cost, which is each search backend's own
  index structure (e.g. a cKDTree's internal copy of the vectors it was
  built from) -- one such derived/cache artifact still exists per
  (level, compatibility_key) combination, exactly as before. A campaign that
  persists only the canonical vectors to disk (rebuilding backend indices on
  load) can keep durable raw-evidence storage at the un-doubled, single-copy
  figure; a campaign that also persists backend index caches keeps the
  doubled figure documented above, now explicitly labeled as a derived/cache
  artifact rather than primary evidence.
- **Unmatched-evidence reporting: no cost change, richer evidence.** Every
  directed-coverage summary now also reports `unmatched_fraction` alongside
  the existing `n_unmatched` count (see coverage.aggregate) -- computed from
  data already produced by the same query pass, not an additional pass.
- **cKDTree is now fully backend-only.** `coverage.soap_representation` no
  longer imports `scipy` at all; the only remaining `scipy.spatial.cKDTree`
  import in this package is inside `coverage.exact_kdtree_backend`. This is
  what licenses swapping in a future approximate backend (see the
  not-yet-implemented-ANN caveat above) without touching
  `coverage.soap_representation`, `coverage.reference_pool`,
  `coverage.nn_distance`, `coverage.aggregate`, or `coverage.report`.

## What this estimate does NOT do

- Does not run the full-scale computation for any direction.
- Does not select a pass/fail threshold, acquisition count, or descriptor
  parameter set for production use.
- Does not confirm the "~2,206 Student candidates" figure, nor the current
  Student/distillation dataset's real size (the `deployment_coverage`
  direction's reference population) -- both remain unverified in this repo
  and are used only as order-of-magnitude planning inputs, explicitly flagged
  where used.
- Does not assume SOAP is the only representation this estimate could apply
  to -- `per_pair_rate` and descriptor cost must be re-measured for any other
  `CoverageRepresentation` implementation.
- Does not propose or silently adopt approximate nearest-neighbor search as a
  production substitute for exact NN.
