"""Deterministic governed access-partition system for a protected DFT reference.

A protected DFT EVALUATION population (see :mod:`validation.protected_reference` --
``protected-existing-dft`` / ``recovered-original-holdout``) is consumed by more than one
late stage:

  * Stage 8 (evaluation) reports Student final accuracy on a *held-out* slice.
  * Stage 9 (uncertainty) FITS a conformal/calibration quantile on one slice and then
    VALIDATES empirical coverage on a *different, disjoint* slice.

Re-using the same frames for calibration-fit and calibration-eval (or for both final
evaluation and calibration) is a leakage that silently inflates coverage/accuracy claims.
This module is the deterministic authority that partitions the protected population into
mutually-disjoint governed ROLES, exactly once, from PRE-RESULT information only
(structure identity, provenance category, geometry), never from Student error, model
uncertainty, or DFT label *values*. The plan is hash-bound, replayable, and fail-closed.

Design invariants (all enforced by :func:`validate_access_partition_contract`):

  * DETERMINISTIC / REPLAYABLE -- the same population + role set always yields the same
    assignment; the validator re-derives it and rejects any drift.
  * DISJOINT -- roles never share a frame.
  * COVERING -- every population frame lands in exactly one role.
  * STRATIFIED -- frames are stratified by their own structural category
    (``config_type`` / ``source_category``), so every role samples the category mix
    (rare-category preservation via carried round-robin), material-independently.
  * AUTONOMOUS SIZING -- no human frame count, percentage, or quota. Per-role size is an
    emergent ~n/R stratified round-robin split; the only structural input is the number of
    roles R declared by the reference. A support floor tied to the population's own number
    of strata is enforced fail-closed.
  * INPUT-ONLY FEATURES -- assignment reads only category / provenance index / geometry
    fingerprint; it never reads DFT energy/force/stress values or any model output.

Nothing about a specific material, campaign, frame count, category name, or split ratio is
hardcoded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

from validation.protected_reference import _structure_fingerprint
from workflow.integrity import sha256_file


# Canonical governed role vocabulary. These strings are the same tokens the pinned
# reference contract declares in ``allowed_uses`` (see inputs/*-reference.yaml), so the
# partition contract and the reference-authorization contract speak one vocabulary.
ROLE_STUDENT_FINAL_EVALUATION = "protected_stage8_evaluation"
ROLE_CALIBRATION_FIT = "uncertainty_calibration_fit"
ROLE_CALIBRATION_EVAL = "uncertainty_calibration_eval"

CANONICAL_PARTITION_ROLES = (
    ROLE_STUDENT_FINAL_EVALUATION,
    ROLE_CALIBRATION_FIT,
    ROLE_CALIBRATION_EVAL,
)

# Fail-closed stage -> allowed-role access policy. A stage may read ONLY the frames of the
# role(s) listed here; the three roles are MUST_BE_DISJOINT, so honoring this policy makes
# calibration-fit, calibration-eval, and final evaluation populations provably isolated.
STAGE_ROLE_ACCESS_POLICY = {
    "evaluation": frozenset({ROLE_STUDENT_FINAL_EVALUATION}),
    "uncertainty": frozenset({ROLE_CALIBRATION_FIT, ROLE_CALIBRATION_EVAL}),
}

# Ordered candidate info keys used to resolve a frame's structural stratum. These are
# generic extxyz/ASE provenance field names, NOT material category values.
DEFAULT_STRATIFY_KEYS = ("config_type", "source_category")

CONTRACT_KIND = "protected_reference_access_partition"
SCHEMA_VERSION = 1


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256_of(obj) -> str:
    return hashlib.sha256(_canonical_json(obj).encode()).hexdigest()


class FrameDescriptor:
    """Pre-result identity of one population frame used for partition assignment.

    Only fields available BEFORE any Student/DFT result are carried: the structural
    ``stratum`` (category), a deterministic ``order_key`` (provenance index when present,
    else the fingerprint), the geometry ``fingerprint``, and an optional human-readable
    ``source_key`` = (source_category, source_local_index).
    """

    __slots__ = ("stratum", "order_key", "fingerprint", "source_key")

    def __init__(self, stratum, order_key, fingerprint, source_key=None):
        self.stratum = str(stratum)
        self.order_key = order_key
        self.fingerprint = str(fingerprint)
        self.source_key = source_key


def _resolve_stratum(info: dict, stratify_keys: Sequence[str]):
    for key in stratify_keys:
        value = info.get(key)
        if value is not None and str(value).strip():
            return key, str(value)
    return None, None


def load_population_descriptors(
    structures_path,
    stratify_keys: Sequence[str] = DEFAULT_STRATIFY_KEYS,
) -> list[FrameDescriptor]:
    """Read a DFT-labeled protected structures file into partition descriptors.

    Reads ONLY structure identity: species/positions/cell/pbc geometry (via the shared
    protected-reference fingerprint), the structural category, and the provenance index.
    DFT energy/force/stress *values* are never read here -- assignment must not depend on
    label truth. Fails closed if any frame lacks a resolvable stratum.
    """
    from ase.io import read

    structures_path = Path(structures_path).resolve()
    frames = read(str(structures_path), index=":")
    descriptors: list[FrameDescriptor] = []
    resolved_key: Optional[str] = None
    for index, atoms in enumerate(frames):
        key, stratum = _resolve_stratum(atoms.info, stratify_keys)
        if stratum is None:
            raise ValueError(
                f"frame {index} lacks a resolvable structural stratum among "
                f"{list(stratify_keys)}; cannot stratify the governed partition"
            )
        if resolved_key is None:
            resolved_key = key
        elif key != resolved_key:
            raise ValueError(
                f"frame {index} resolves its stratum from {key!r} but earlier frames used "
                f"{resolved_key!r}; the population must use one consistent stratify field"
            )
        fingerprint = _structure_fingerprint(atoms)
        local_index = atoms.info.get("source_local_index")
        category = atoms.info.get("source_category")
        source_key = None
        if category is not None and local_index is not None:
            source_key = [str(category), int(local_index)]
        # Deterministic ordering: provenance local index first when available, else the
        # geometry fingerprint. Fingerprint is always the final tiebreaker.
        order_key = int(local_index) if isinstance(local_index, (int, float)) else fingerprint
        descriptors.append(FrameDescriptor(stratum, order_key, fingerprint, source_key))
    return descriptors, (resolved_key or (stratify_keys[0] if stratify_keys else "stratum"))


def derive_minimum_support_per_role(n_strata: int) -> int:
    """Autonomous per-role support floor tied to the population's own complexity.

    A partition smaller than the number of structural categories cannot, even in
    principle, represent every category, so calibration/evaluation on it would silently
    drop categories. The floor is therefore the number of strata -- derived from the data,
    never a human-chosen N. Bounded below by 1 so a single-category population is legal.
    """
    return max(1, int(n_strata))


def plan_stratified_partition(
    descriptors: Sequence[FrameDescriptor],
    roles: Sequence[str],
) -> dict:
    """Deterministically assign descriptors to roles via carried stratified round-robin.

    Within each stratum (sorted by category key), frames are ordered by (order_key,
    fingerprint) and dealt to the roles round-robin. The starting role rotates across
    strata by the running count so remainder frames do not systematically favor one role;
    this yields near-equal ~n/R role sizes AND spreads each category across the roles as
    evenly as its member count allows (rare-category preservation).

    Returns ``{role: [descriptor_index, ...]}`` with disjoint, covering index lists.
    """
    roles = list(roles)
    if len(roles) < 2:
        raise ValueError("a governed partition requires at least two roles")
    if len(set(roles)) != len(roles):
        raise ValueError(f"partition roles must be unique: {roles!r}")

    by_stratum: dict[str, list[tuple]] = {}
    for idx, d in enumerate(descriptors):
        # (order_key, fingerprint, idx) -- order_key may be int or str; keep per-stratum
        # lists type-homogeneous by sorting on (str(order_key), fingerprint).
        by_stratum.setdefault(d.stratum, []).append((str(d.order_key), d.fingerprint, idx))

    assignment: dict[str, list[int]] = {r: [] for r in roles}
    offset = 0
    r = len(roles)
    for stratum in sorted(by_stratum):
        members = sorted(by_stratum[stratum])
        for i, (_ok, _fp, idx) in enumerate(members):
            assignment[roles[(offset + i) % r]].append(idx)
        offset = (offset + len(members)) % r
    for role in assignment:
        assignment[role].sort()
    return assignment


def build_access_partition_contract(
    structures_path,
    reference_id: str,
    reference_kind: str,
    roles: Sequence[str] = CANONICAL_PARTITION_ROLES,
    stratify_keys: Sequence[str] = DEFAULT_STRATIFY_KEYS,
    min_support_per_role: Optional[int] = None,
) -> dict:
    """Produce the canonical governed access-partition contract for a protected population.

    Fails closed if the population is too small to yield disjoint, category-representative
    roles at the autonomous support floor.
    """
    structures_path = Path(structures_path).resolve()
    descriptors, resolved_stratify_key = load_population_descriptors(structures_path, stratify_keys)
    n_frames = len(descriptors)
    if n_frames == 0:
        raise ValueError("protected population is empty; cannot build a governed partition")

    strata_counts: dict[str, int] = {}
    for d in descriptors:
        strata_counts[d.stratum] = strata_counts.get(d.stratum, 0) + 1
    n_strata = len(strata_counts)

    floor = derive_minimum_support_per_role(n_strata) if min_support_per_role is None \
        else int(min_support_per_role)
    required = floor * len(roles)
    if n_frames < required:
        raise ValueError(
            "POPULATION_TOO_SMALL_FOR_GOVERNED_PARTITION: "
            f"{n_frames} frames < {len(roles)} roles x support floor {floor} = {required} "
            f"(support floor derived from {n_strata} structural strata); refusing to build a "
            "partition that cannot represent every category in each role"
        )

    assignment = plan_stratified_partition(descriptors, roles)

    partitions: dict[str, dict] = {}
    all_indices: list[int] = []
    for role in roles:
        idxs = assignment[role]
        all_indices.extend(idxs)
        if len(idxs) < floor:
            raise ValueError(
                f"role {role!r} received {len(idxs)} frames, below the autonomous support "
                f"floor {floor}; population too small or too skewed for governed partitioning"
            )
        fingerprints = sorted(descriptors[i].fingerprint for i in idxs)
        source_keys = [descriptors[i].source_key for i in idxs
                       if descriptors[i].source_key is not None]
        partitions[role] = {
            "n_frames": len(idxs),
            "frame_fingerprints_sha256": _sha256_of(fingerprints),
            "frame_fingerprints": fingerprints,
            "source_keys": sorted(source_keys) if source_keys else [],
            # split-conformal finite-sample coverage resolution supported by this slice
            "coverage_resolution": 1.0 / (len(idxs) + 1),
        }

    # Disjointness + covering (defensive; the algorithm guarantees both).
    if len(all_indices) != n_frames or len(set(all_indices)) != n_frames:
        raise ValueError("internal partition error: assignment is not a disjoint cover")

    assignment_binding = {
        role: partitions[role]["frame_fingerprints"] for role in roles
    }
    partition_assignment_sha256 = _sha256_of(assignment_binding)

    contract = {
        "schema_version": SCHEMA_VERSION,
        "contract_kind": CONTRACT_KIND,
        "reference_id": reference_id,
        "reference_kind": reference_kind,
        "population": {
            "structures_path": str(structures_path),
            "structures_sha256": sha256_file(structures_path),
            "n_frames": n_frames,
        },
        "stratify_by": resolved_stratify_key,
        "strata": dict(sorted(strata_counts.items())),
        "roles": list(roles),
        "sizing": {
            "policy": "deterministic_stratified_carried_round_robin_equal_resolution",
            "minimum_support_per_role": floor,
            "minimum_support_rationale": (
                "floor = number of structural strata; a role smaller than the category count "
                "cannot represent every category. Derived from the population, not a human N."
            ),
            "achieved": {role: partitions[role]["n_frames"] for role in roles},
        },
        "assignment_algorithm": {
            "name": "stratified_carried_round_robin",
            "frame_order_key": "(source_local_index_or_fingerprint, geometry_fingerprint)",
            "role_rotation": "per-stratum start offset carried by running assigned count",
        },
        "input_only_features_attestation": (
            "partition assignment used only pre-result structure identity (geometry "
            "fingerprint), structural category, and provenance index; it did NOT read "
            "Student error, model uncertainty, or DFT label values"
        ),
        "partitions": partitions,
        "disjointness": {
            "pairwise_disjoint": True,
            "union_covers_population": True,
            "n_assigned": n_frames,
        },
        "partition_assignment_sha256": partition_assignment_sha256,
    }
    return contract


def write_access_partition_contract(
    structures_path,
    reference_id: str,
    reference_kind: str,
    output,
    roles: Sequence[str] = CANONICAL_PARTITION_ROLES,
    stratify_keys: Sequence[str] = DEFAULT_STRATIFY_KEYS,
    min_support_per_role: Optional[int] = None,
) -> Path:
    """Build, self-validate, and atomically write the governed partition contract."""
    output = Path(output).resolve()
    contract = build_access_partition_contract(
        structures_path, reference_id, reference_kind,
        roles=roles, stratify_keys=stratify_keys, min_support_per_role=min_support_per_role,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    try:
        validate_access_partition_contract(tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(output)
    return output


def validate_access_partition_contract(
    contract_path,
    expected_reference_id: Optional[str] = None,
    expected_structures_sha256: Optional[str] = None,
) -> dict:
    """Re-derive the partition from the pinned population and fail closed on any drift.

    This is the controller-side authority. It does NOT trust the artifact's stored index
    lists; it recomputes the deterministic assignment from the structures file and its
    declared roles, then verifies disjointness, covering, the support floor, the per-role
    fingerprint hashes, and the top-level assignment hash. Optionally binds identity and
    source-hash expectations from the run's reference contract.
    """
    contract_path = Path(contract_path).resolve()
    contract = json.loads(contract_path.read_text())

    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"access-partition contract requires schema_version={SCHEMA_VERSION}")
    if contract.get("contract_kind") != CONTRACT_KIND:
        raise ValueError(f"contract_kind must be {CONTRACT_KIND!r}")

    population = contract.get("population")
    if not isinstance(population, dict):
        raise ValueError("access-partition contract requires a population block")
    structures_path = Path(population.get("structures_path", "")).expanduser()
    if not structures_path.is_absolute():
        structures_path = (contract_path.parent / structures_path).resolve()
    if not structures_path.is_file():
        raise FileNotFoundError(structures_path)
    observed_sha = sha256_file(structures_path)
    if observed_sha != population.get("structures_sha256"):
        raise RuntimeError(
            "access-partition population SHA-256 mismatch: "
            f"{observed_sha} != {population.get('structures_sha256')}"
        )
    if expected_structures_sha256 is not None and observed_sha != expected_structures_sha256:
        raise RuntimeError(
            "access-partition population does not match the run-bound reference structures "
            f"sha256 ({observed_sha} != {expected_structures_sha256})"
        )

    reference_id = contract.get("reference_id")
    if expected_reference_id is not None and reference_id != expected_reference_id:
        raise ValueError(
            f"access-partition reference_id {reference_id!r} != expected {expected_reference_id!r}"
        )

    roles = contract.get("roles")
    if not isinstance(roles, list) or len(roles) < 2 or len(set(roles)) != len(roles):
        raise ValueError("access-partition contract requires >=2 unique roles")

    stratify_keys = (contract.get("stratify_by"),) if contract.get("stratify_by") else DEFAULT_STRATIFY_KEYS
    descriptors, _ = load_population_descriptors(structures_path, stratify_keys)
    n_frames = len(descriptors)
    if n_frames != population.get("n_frames"):
        raise ValueError(
            f"population.n_frames {population.get('n_frames')!r} != actual {n_frames}"
        )

    strata_counts: dict[str, int] = {}
    for d in descriptors:
        strata_counts[d.stratum] = strata_counts.get(d.stratum, 0) + 1

    stored_floor = (contract.get("sizing") or {}).get("minimum_support_per_role")
    floor = derive_minimum_support_per_role(len(strata_counts)) \
        if stored_floor is None else int(stored_floor)

    replay = plan_stratified_partition(descriptors, roles)

    partitions = contract.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != set(roles):
        raise ValueError("partitions block must have exactly one entry per declared role")

    all_indices: list[int] = []
    for role in roles:
        idxs = replay[role]
        all_indices.extend(idxs)
        if len(idxs) < floor:
            raise ValueError(
                f"replayed role {role!r} has {len(idxs)} frames < support floor {floor}"
            )
        recomputed = sorted(descriptors[i].fingerprint for i in idxs)
        stored = partitions[role]
        if stored.get("n_frames") != len(idxs):
            raise ValueError(
                f"role {role!r} n_frames {stored.get('n_frames')!r} != replayed {len(idxs)}"
            )
        if stored.get("frame_fingerprints") != recomputed:
            raise ValueError(
                f"role {role!r} frame_fingerprints do not match the deterministic replay; "
                "the partition artifact drifted from its population"
            )
        if stored.get("frame_fingerprints_sha256") != _sha256_of(recomputed):
            raise ValueError(f"role {role!r} frame_fingerprints_sha256 mismatch")

    if len(all_indices) != n_frames or len(set(all_indices)) != n_frames:
        raise ValueError("replayed assignment is not a disjoint cover of the population")

    # Cross-role disjointness of the stored fingerprint sets (belt and suspenders).
    seen: set[str] = set()
    for role in roles:
        fps = set(partitions[role]["frame_fingerprints"])
        overlap = seen & fps
        if overlap:
            raise ValueError(
                f"roles are not disjoint: {len(overlap)} shared frame fingerprint(s) at {role!r}"
            )
        seen |= fps

    assignment_binding = {role: sorted(descriptors[i].fingerprint for i in replay[role])
                          for role in roles}
    recomputed_top = _sha256_of(assignment_binding)
    if contract.get("partition_assignment_sha256") != recomputed_top:
        raise ValueError(
            "partition_assignment_sha256 does not match the deterministic replay hash"
        )

    return contract


def materialize_partition_slice(
    contract: dict,
    role: str,
    source_population_path,
    out_path,
    require_committee_seeds: Optional[Iterable[int]] = None,
):
    """Write the frames of one governed role out of a source population, by fingerprint.

    ``source_population_path`` must CONTAIN every frame of ``role`` (matched by the same
    rounded geometry fingerprint the contract is bound to); all info/arrays -- including any
    embedded ``student_forces_seed<NN>`` and DFT labels -- are preserved. Fails closed if any
    role fingerprint is absent, if the source has duplicate geometries, or (when
    ``require_committee_seeds`` is given) if a selected frame lacks a required committee-force
    array. The written slice is exactly the role's frames, nothing more.
    """
    from ase.io import read, write

    role_fps = resolve_partition_fingerprints(contract, role)
    frames = read(str(Path(source_population_path).resolve()), index=":")
    selected = []
    seen: set = set()
    for atoms in frames:
        fp = _structure_fingerprint(atoms)
        if fp in role_fps:
            if fp in seen:
                raise ValueError(
                    f"source population has a duplicate geometry for role {role!r}; cannot "
                    "materialize a well-defined slice"
                )
            selected.append(atoms)
            seen.add(fp)
    missing = role_fps - seen
    if missing:
        raise ValueError(
            f"materialize_partition_slice: {len(missing)} of {len(role_fps)} role {role!r} "
            f"fingerprints were not found in the source population {source_population_path}"
        )
    if require_committee_seeds is not None:
        for i, atoms in enumerate(selected):
            for seed in require_committee_seeds:
                key = f"student_forces_seed{int(seed):02d}"
                if key not in atoms.arrays:
                    raise ValueError(
                        f"slice frame {i} for role {role!r} is missing committee forces {key!r}; "
                        "the source population must carry embedded per-seed student forces"
                    )
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_path), selected)
    return {
        "path": str(out_path),
        "role": role,
        "n_frames": len(selected),
        "frame_fingerprints_sha256": _sha256_of(sorted(seen)),
    }


def enforce_and_materialize(
    contract_path,
    stage: str,
    role: str,
    source_population_path,
    out_path,
    require_committee_seeds: Optional[Iterable[int]] = None,
    expected_reference_id: Optional[str] = None,
    expected_structures_sha256: Optional[str] = None,
):
    """Validate the contract, fail-closed access-check ``stage``->``role``, then slice.

    This is the single entry point late stages should use: it guarantees a stage can only
    ever materialize a role it is authorized to read, from a replay-verified contract.
    """
    contract = validate_access_partition_contract(
        contract_path,
        expected_reference_id=expected_reference_id,
        expected_structures_sha256=expected_structures_sha256,
    )
    assert_stage_partition_access(contract, stage, role)
    result = materialize_partition_slice(
        contract, role, source_population_path, out_path,
        require_committee_seeds=require_committee_seeds,
    )
    stored = (contract.get("partitions") or {}).get(role, {})
    if stored.get("frame_fingerprints_sha256") != result["frame_fingerprints_sha256"]:
        raise ValueError(
            f"materialized role {role!r} fingerprint hash does not match the contract; "
            "the source population is not the governed reference population"
        )
    result["contract"] = contract
    return result


def resolve_partition_fingerprints(contract: dict, role: str) -> set:
    """Return the geometry-fingerprint set for one governed role (for isolation checks)."""
    partitions = contract.get("partitions") or {}
    entry = partitions.get(role)
    if not isinstance(entry, dict):
        raise ValueError(f"partition contract has no role {role!r}")
    return set(entry.get("frame_fingerprints") or [])


def assert_stage_partition_access(contract: dict, stage: str, role: str) -> str:
    """Fail closed unless ``stage`` is authorized to read ``role`` and the role exists.

    This is the access-enforcement gate the late stages call before touching any protected
    slice. Stage->role authorizations are fixed by :data:`STAGE_ROLE_ACCESS_POLICY`; a
    stage may never read a role outside its allowed set, guaranteeing that calibration-fit,
    calibration-eval, and final evaluation stay isolated.
    """
    allowed = STAGE_ROLE_ACCESS_POLICY.get(stage)
    if allowed is None:
        raise ValueError(
            f"stage {stage!r} has no governed protected-reference access policy; "
            f"known stages: {sorted(STAGE_ROLE_ACCESS_POLICY)}"
        )
    if role not in allowed:
        raise ValueError(
            f"stage {stage!r} is NOT authorized to access partition role {role!r} "
            f"(authorized roles for this stage: {sorted(allowed)})"
        )
    if role not in (contract.get("roles") or []):
        raise ValueError(f"partition contract does not define role {role!r}")
    return role


__all__ = [
    "CANONICAL_PARTITION_ROLES",
    "STAGE_ROLE_ACCESS_POLICY",
    "ROLE_STUDENT_FINAL_EVALUATION",
    "ROLE_CALIBRATION_FIT",
    "ROLE_CALIBRATION_EVAL",
    "FrameDescriptor",
    "load_population_descriptors",
    "derive_minimum_support_per_role",
    "plan_stratified_partition",
    "build_access_partition_contract",
    "write_access_partition_contract",
    "validate_access_partition_contract",
    "materialize_partition_slice",
    "enforce_and_materialize",
    "resolve_partition_fingerprints",
    "assert_stage_partition_access",
]
