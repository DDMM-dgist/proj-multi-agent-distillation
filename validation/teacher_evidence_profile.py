"""Generic, evidence-driven Teacher validation ADMISSIBLE DECISION SPACE.

No material name (SiO2, Allegro, ...), dataset size, or campaign-specific constant belongs in
this module. A campaign's actual evidence -- what assets are genuinely available for a given
Teacher -- is captured in a `TeacherEvidenceProfile`; `derive_admissible_decision_space` maps
that profile to the FULL SET of validation components whose evidence requirements are
satisfied. Components are evidence requirements, not mutually-exclusive strategy choices: they
are additive, so more than one is routinely admissible at once (e.g. whenever
``ORIGINAL_HELDOUT_FIDELITY``'s evidence holds, ``TRAINING_CORPUS_CONSISTENCY``'s strictly
weaker requirement is trivially also satisfied). Which admissible component(s) a campaign
actually USES is a separate, later, objective-driven decision made by the validation planner
(see ``runtimes.pydantic_ai.teacher_validation_plan``) -- this module only establishes what
evidence makes possible, never which of it gets selected.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

OPERATIONAL_ROBUSTNESS = "OPERATIONAL_ROBUSTNESS"
TRAINING_CORPUS_CONSISTENCY = "TRAINING_CORPUS_CONSISTENCY"
ORIGINAL_HELDOUT_FIDELITY = "ORIGINAL_HELDOUT_FIDELITY"
INDEPENDENT_REFERENCE_FIDELITY = "INDEPENDENT_REFERENCE_FIDELITY"
DEPLOYMENT_APPLICABILITY = "DEPLOYMENT_APPLICABILITY"

VALIDATION_COMPONENTS = (
    OPERATIONAL_ROBUSTNESS, TRAINING_CORPUS_CONSISTENCY, ORIGINAL_HELDOUT_FIDELITY,
    INDEPENDENT_REFERENCE_FIDELITY, DEPLOYMENT_APPLICABILITY,
)

# The floor: no component's evidence requirement is satisfied at all -- there is nothing
# admissible to plan around (this is distinct from, and strictly rarer than, "only
# OPERATIONAL_ROBUSTNESS is admissible", which is a perfectly plannable, non-empty outcome).
CAMPAIGN_INSUFFICIENT_EVIDENCE_FOR_PLANNING = "CAMPAIGN_INSUFFICIENT_EVIDENCE_FOR_PLANNING"

# The exact evidence fields (on TeacherEvidenceProfile) each component's admissibility reads --
# documentation/introspection only; _component_evidence_satisfied is the single authoritative
# check.
REQUIRED_EVIDENCE = {
    OPERATIONAL_ROBUSTNESS: (
        "teacher_model_available", "operational_evaluation_population_available"),
    TRAINING_CORPUS_CONSISTENCY: (
        "original_training_db_available", "original_labels_available"),
    ORIGINAL_HELDOUT_FIDELITY: (
        "original_training_db_available", "original_labels_available",
        "original_split_recovered", "genuine_holdout_test_available"),
    INDEPENDENT_REFERENCE_FIDELITY: (
        "independent_external_reference_available",),
    DEPLOYMENT_APPLICABILITY: (
        "deployment_domain_population_available",
        "deployment_domain_matches_original_test_distribution"),
}

# Claims each component licenses, and claims it explicitly does NOT license, when admissible AND
# selected. A claim in one component's prohibited list is never implied just because some OTHER
# component is also selected -- prohibitions are per-component, not aggregate.
_ALLOWED_CLAIMS = {
    OPERATIONAL_ROBUSTNESS: ("operational_robustness_assessed",),
    TRAINING_CORPUS_CONSISTENCY: ("training_corpus_consistency",),
    ORIGINAL_HELDOUT_FIDELITY: ("held_out_fidelity",),
    INDEPENDENT_REFERENCE_FIDELITY: ("independent_reference_fidelity",),
    DEPLOYMENT_APPLICABILITY: ("deployment_applicability_assessed",),
}
_PROHIBITED_CLAIMS = {
    OPERATIONAL_ROBUSTNESS: ("held_out_fidelity", "generalization_accuracy"),
    TRAINING_CORPUS_CONSISTENCY: ("held_out_fidelity", "generalization_accuracy"),
    ORIGINAL_HELDOUT_FIDELITY: (),
    INDEPENDENT_REFERENCE_FIDELITY: (),
    DEPLOYMENT_APPLICABILITY: (),
}

# Unconditional protected-data restrictions: apply regardless of which component(s) end up
# selected. Mirrors validation.protected_reference.RECOVERED_HOLDOUT_REQUIRED_PROHIBITIONS -- a
# genuine held-out/reference population is never usable for these roles no matter how the
# validation plan otherwise reads.
PROTECTED_DATA_RESTRICTIONS = (
    "student_training", "student_validation_tuning", "acquisition_seed",
    "augmentation_parent", "recovery_training",
)

# Generic approval condition(s) governing costly downstream reliance on a committed plan. This
# module states the condition; enforcing it against a specific downstream dispatch (Teacher
# labeling / Student training) is the Controller's job (see
# workflow.controller.RunController.authorize_downstream_teacher_reliance), not this module's.
APPROVAL_CONDITIONS = (
    "costly_downstream_reliance_on_a_committed_teacher_validation_plan_that_does_not_include_"
    "ORIGINAL_HELDOUT_FIDELITY_or_INDEPENDENT_REFERENCE_FIDELITY_requires_a_distinct_human_"
    "downstream_reliance_approval_bound_to_the_committed_plan",
)


@dataclass(frozen=True)
class TeacherEvidenceProfile:
    """What is actually, demonstrably available for ONE Teacher in ONE campaign.

    Every field is a plain evidence fact (or confidence/status string), never a material,
    dataset, or campaign identity. ``*_confidence`` fields are free-form provenance-status
    strings (e.g. "VERIFIED", "PLAUSIBLE_NOT_CONFIRMED", "NOT_AVAILABLE") -- callers may record
    whatever provenance status they actually established; only the paired boolean drives
    ``derive_admissible_decision_space``.

    ``operational_evaluation_population_available`` means a real, provenance-bound population
    exists to run the Teacher against operationally -- drawn from the original Teacher corpus, a
    user-provided deployment population, or another frozen, admissible structure source. It is
    never satisfied merely because ``teacher_model_available`` is True: a Teacher with nothing
    to evaluate against has no operational-robustness evidence, regardless of the model itself
    being loadable.
    """

    teacher_model_available: bool
    operational_evaluation_population_available: bool = False
    original_training_db_available: bool = False
    original_labels_available: bool = False
    original_split_recovered: bool = False
    genuine_holdout_test_available: bool = False
    independent_external_reference_available: bool = False
    deployment_domain_population_available: bool = False
    original_split_confidence: str = "NOT_AVAILABLE"
    genuine_holdout_test_frame_count: Optional[int] = None
    deployment_domain_matches_original_test_distribution: Optional[bool] = None


def _component_evidence_satisfied(component: str, profile: TeacherEvidenceProfile) -> bool:
    if component == OPERATIONAL_ROBUSTNESS:
        return bool(profile.teacher_model_available and
                    profile.operational_evaluation_population_available)
    if component == TRAINING_CORPUS_CONSISTENCY:
        return bool(profile.original_training_db_available and profile.original_labels_available)
    if component == ORIGINAL_HELDOUT_FIDELITY:
        return bool(profile.original_training_db_available and profile.original_labels_available and
                    profile.original_split_recovered and profile.genuine_holdout_test_available)
    if component == INDEPENDENT_REFERENCE_FIDELITY:
        return bool(profile.independent_external_reference_available)
    if component == DEPLOYMENT_APPLICABILITY:
        return bool(profile.deployment_domain_population_available and
                    profile.deployment_domain_matches_original_test_distribution is False)
    raise ValueError(f"unknown validation component: {component!r}")


def derive_admissible_decision_space(profile: TeacherEvidenceProfile) -> dict:
    """Return the FULL SET of validation components ``profile``'s evidence admits.

    Returns a dict:
      * ``admissible_components``: list[str], the (possibly multi-member, possibly empty)
        subset of ``VALIDATION_COMPONENTS`` whose evidence requirement is satisfied.
      * ``components``: {component: {"required_evidence", "allowed_claims",
        "prohibited_claims"}} for each admissible component only.
      * ``protected_data_restrictions``: unconditional prohibited roles (see
        ``PROTECTED_DATA_RESTRICTIONS``), independent of which components are admissible.
      * ``approval_conditions``: generic downstream-reliance approval condition(s) (see
        ``APPROVAL_CONDITIONS``).
      * ``insufficient_evidence``: True iff ``admissible_components`` is empty -- the one true
        "nothing is admissible" floor (``CAMPAIGN_INSUFFICIENT_EVIDENCE_FOR_PLANNING``).
    """
    admissible = [component for component in VALIDATION_COMPONENTS
                  if _component_evidence_satisfied(component, profile)]
    components = {
        component: {
            "required_evidence": list(REQUIRED_EVIDENCE[component]),
            "allowed_claims": list(_ALLOWED_CLAIMS[component]),
            "prohibited_claims": list(_PROHIBITED_CLAIMS[component]),
        }
        for component in admissible
    }
    return {
        "admissible_components": admissible,
        "components": components,
        "protected_data_restrictions": list(PROTECTED_DATA_RESTRICTIONS),
        "approval_conditions": list(APPROVAL_CONDITIONS),
        "insufficient_evidence": not admissible,
        "floor": CAMPAIGN_INSUFFICIENT_EVIDENCE_FOR_PLANNING if not admissible else None,
    }


def _profile_sha256(profile: TeacherEvidenceProfile) -> str:
    payload = json.dumps(asdict(profile), sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_frames(path: Union[str, Path], *, label: str):
    """Fail-closed structure-file reader: a caller-declared path that does not exist or does not
    parse as a structure file is a hard error, never a silently-empty population -- this is a
    frozen run input, not an optional hint."""
    from ase.io import read

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    frames = read(str(resolved), index=":")
    if not frames:
        raise ValueError(f"{label} contains no frames: {resolved}")
    return frames


def _frame_split_key(atoms, *, label: str):
    cat = atoms.info.get("source_category")
    local_index = atoms.info.get("source_local_index")
    if cat is None or local_index is None:
        return None
    return (str(cat), int(local_index))


def _frame_has_finite_ase_calculator_labels(atoms) -> bool:
    """Fall back to the standard ASE calculator-attached label convention
    (``atoms.calc``/``atoms.calc.results`` via ``get_potential_energy()``/``get_forces()``) for a
    frame that carries no CONFIGURED custom-key label. This is a storage-representation fact
    only: whether the resulting values are genuine DFT/reference labels is established by the
    caller's frozen evidence source/provenance role (e.g. ``original_training_db_path`` being the
    attested original training corpus), never guessed here merely because an ASE calculator
    happens to be attached. Fails closed (returns False) on: no calculator attached;
    ``PropertyNotImplementedError`` for either property; non-finite energy; non-finite forces; a
    force array shape other than ``(len(atoms), 3)``.
    """
    import numpy as np
    from ase.calculators.calculator import PropertyNotImplementedError

    if atoms.calc is None:
        return False
    try:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
    except PropertyNotImplementedError:
        return False
    if not np.isfinite(float(energy)):
        return False
    forces = np.asarray(forces, dtype=float)
    if forces.shape != (len(atoms), 3):
        return False
    return bool(np.all(np.isfinite(forces)))


def _frame_has_finite_labels(atoms, *, energy_key: str, forces_key: str) -> bool:
    """Whether ``atoms`` carries a genuinely finite energy/forces label. Tries the CONFIGURED
    custom-key storage convention (``atoms.info[energy_key]``/``atoms.arrays[forces_key]``)
    FIRST -- unchanged from this function's original behavior. Only when that convention is
    genuinely ABSENT (neither key present at all, not merely malformed) does it fall back to the
    standard ASE calculator-attached convention (see ``_frame_has_finite_ase_calculator_labels``).
    A frame that declares a custom-key label but stores it malformed still fails closed on that
    path -- it never silently falls through to a different label source.
    """
    import numpy as np

    custom_energy = atoms.info.get(energy_key)
    custom_forces = atoms.arrays.get(forces_key)
    if custom_energy is None and custom_forces is None:
        return _frame_has_finite_ase_calculator_labels(atoms)
    if not isinstance(custom_energy, (int, float)) or not np.isfinite(float(custom_energy)):
        return False
    if custom_forces is None or not np.all(np.isfinite(np.asarray(custom_forces, dtype=float))):
        return False
    return True


def _load_positional_split_manifest(path: Union[str, Path]):
    """Load one manifest as a dataset-scoped positional-index split source, IFF it explicitly
    self-declares that semantics via top-level ``index_semantics ==
    "source_dataset_positional_index"`` and a bound ``source_dataset_sha256``. A manifest is
    never treated as a positional-index source merely because its records happen to carry a
    field named ``global_index`` -- a field's NAME is not evidence of its MEANING; only this
    explicit declaration is.

    Returns ``None`` if ``path`` is not a JSON file, does not parse, is not a dict, lacks the
    declaration, or its records are malformed/self-contradictory (e.g. a duplicate positional
    index within the manifest itself). Otherwise returns ``{"source_dataset_sha256": str,
    "index_to_split": {int: str}, "total_frames": int}`` -- WITHOUT yet verifying that the
    declared sha256/count match any real dataset file; that verification is
    ``_verified_positional_split_join``'s job, the only gate allowed to actually mark
    ``original_split_recovered`` True from this representation.
    """
    p = Path(path)
    if p.suffix.lower() != ".json" or not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("index_semantics") != "source_dataset_positional_index":
        return None
    source_dataset_sha256 = payload.get("source_dataset_sha256")
    records = payload.get("records")
    total_frames = payload.get("total_frames")
    if not isinstance(source_dataset_sha256, str) or not source_dataset_sha256:
        return None
    if not isinstance(records, list) or not records:
        return None
    if not isinstance(total_frames, int) or total_frames <= 0:
        return None
    index_to_split: dict = {}
    for record in records:
        if not isinstance(record, dict):
            return None
        index = record.get("global_index")
        split = record.get("split")
        if not isinstance(index, int) or isinstance(index, bool) or split is None:
            return None
        if index in index_to_split:
            return None
        index_to_split[index] = split
    return {
        "source_dataset_sha256": source_dataset_sha256,
        "index_to_split": index_to_split,
        "total_frames": total_frames,
    }


def _verified_positional_split_join(manifest_data: dict, *, dataset_path, db_frames):
    """Fail-closed verification gate for a loaded positional-split manifest (see
    ``_load_positional_split_manifest``) against the REAL training-DB file: returns the
    ``{positional_index: split}`` mapping only when ALL of the following hold, and ``None``
    (fail closed) otherwise --

      * the LIVE sha256 of ``dataset_path`` equals the manifest's declared
        ``source_dataset_sha256`` (never trust a stale/asserted digest);
      * the manifest's declared ``total_frames`` equals its own record count AND the actual
        parsed frame count of ``db_frames``;
      * the set of positional indices across records is EXACTLY ``{0, ..., total_frames - 1}`` --
        no duplicates (already excluded when the manifest was loaded) and no omissions or
        out-of-range values.

    This join is never partially trusted: any single mismatch voids the whole mapping.
    """
    from workflow.integrity import sha256_file

    if dataset_path is None:
        return None
    resolved_dataset_path = Path(dataset_path).resolve()
    if not resolved_dataset_path.is_file():
        return None
    live_sha256 = sha256_file(resolved_dataset_path)
    if live_sha256 != manifest_data["source_dataset_sha256"]:
        return None
    total_frames = manifest_data["total_frames"]
    index_to_split = manifest_data["index_to_split"]
    if len(index_to_split) != total_frames:
        return None
    if total_frames != len(db_frames):
        return None
    if set(index_to_split) != set(range(total_frames)):
        return None
    return index_to_split


def inspect_teacher_evidence(
    *,
    teacher_model_path: Optional[Union[str, Path]] = None,
    operational_evaluation_population_path: Optional[Union[str, Path]] = None,
    original_training_db_path: Optional[Union[str, Path]] = None,
    split_source_manifest_paths: Sequence[Union[str, Path]] = (),
    target_split: Optional[str] = None,
    independent_external_reference_path: Optional[Union[str, Path]] = None,
    deployment_domain_population_path: Optional[Union[str, Path]] = None,
    deployment_domain_matches_original_test_distribution: Optional[bool] = None,
    original_split_confidence: str = "NOT_AVAILABLE",
    label_energy_key: str = "dft_energy",
    label_forces_key: str = "dft_forces",
) -> tuple:
    """Deterministic, evidence-driven fact-finder: turns a set of FROZEN run input paths into a
    ``TeacherEvidenceProfile`` by actually reading and cross-checking them -- never by trusting a
    caller-supplied shortcut (a literal frame count, an assertion of "use the recovered holdout",
    or any other pre-computed conclusion). Returns ``(profile, evidence_profile_sha256)``: the hash
    binds the profile's own canonical field values so any later change to what this function would
    conclude from the same frozen inputs is detectable (see ``workflow.controller.
    RunController.commit_teacher_validation_plan``, which re-derives evidence independently rather
    than trusting a stored profile).

    ``target_split`` is a declared/configured split NAME to check membership against -- an
    ordinary, non-shortcut input (see this module's own docstring and
    ``validation.protected_reference._validate_recovered_holdout_reference_config``, which takes
    the identical parameter for the same reason). What is never accepted as an input is the
    RESULT of that membership check (an asserted frame count, or a list of which frames belong):
    ``genuine_holdout_test_frame_count`` below is always computed here, from the real join, never
    passed in.

    Every boolean fact is derived from real files:
      * ``teacher_model_available``: ``teacher_model_path`` is given and exists.
      * ``original_training_db_available``/``original_labels_available``: the training DB parses
        as a non-empty structure file, and (for labels) EVERY frame in it carries finite
        ``label_energy_key``/``label_forces_key`` values -- either via the CONFIGURED custom-key
        convention, or, when that convention is genuinely absent, via a standard ASE
        calculator-attached label (see ``_frame_has_finite_labels``,
        ``_frame_has_finite_ase_calculator_labels``) -- a DB with even one unlabeled frame does not
        satisfy ``original_labels_available``.
      * ``original_split_recovered``: EITHER of two independent join representations resolving
        unambiguously for the ENTIRE corpus -- (1) a crosswalk join (``bounded_evidence.
        build_split_crosswalk``) of every training-DB frame's ``(source_category,
        source_local_index)`` against ``split_source_manifest_paths``, or (2) a dataset-scoped
        positional-index join, ``(source_dataset_sha256, positional_index)`` (see
        ``_load_positional_split_manifest``/``_verified_positional_split_join``), admissible only
        after fail-closed verification that the live training-DB sha256 and frame count match the
        manifest's declared values and every positional index resolves exactly once. If only one
        representation is available and joinable, it is used. If BOTH are available, they must
        AGREE frame-for-frame -- any disagreement fails the whole check closed rather than
        silently preferring either representation. One unjoinable or ambiguous frame (in whichever
        representation(s) are in play) fails the whole check closed.
      * ``genuine_holdout_test_available``/``genuine_holdout_test_frame_count``: computed from
        that SAME join -- the count of training-DB frames whose resolved split equals
        ``target_split`` -- only when ``original_split_recovered`` holds and ``target_split`` is
        given.
      * ``operational_evaluation_population_available``: a real, provenance-bound population
        exists to run the Teacher against -- ``operational_evaluation_population_path``, the
        original training DB itself, or ``deployment_domain_population_path`` (mirrors this
        module's own docstring); never satisfied merely because ``teacher_model_path`` was given.
      * ``independent_external_reference_available``: a distinct (not the same file as the
        training DB) non-empty structure population.
      * ``deployment_domain_population_available``: a non-empty structure population.

    ``deployment_domain_matches_original_test_distribution`` is passed through unchanged: whether
    a deployment domain matches the original test distribution is a genuine scientific/statistical
    judgment this deterministic fact-finder does not itself compute; a caller who has not
    established it must leave it ``None`` (DEPLOYMENT_APPLICABILITY then never becomes admissible,
    per ``_component_evidence_satisfied``, which requires it to be exactly ``False``).
    """
    from runtimes.pydantic_ai.bounded_evidence import build_split_crosswalk

    teacher_model_available = bool(
        teacher_model_path is not None and Path(teacher_model_path).exists())

    db_frames = None
    original_training_db_available = False
    if original_training_db_path is not None:
        db_frames = _read_frames(original_training_db_path, label="original_training_db_path")
        original_training_db_available = True

    original_labels_available = False
    if original_training_db_available:
        original_labels_available = all(
            _frame_has_finite_labels(atoms, energy_key=label_energy_key,
                                    forces_key=label_forces_key)
            for atoms in db_frames
        )

    original_split_recovered = False
    genuine_holdout_test_available = False
    genuine_holdout_test_frame_count = None
    if original_training_db_available and split_source_manifest_paths:
        crosswalk = build_split_crosswalk(split_source_manifest_paths)
        resolved = crosswalk["resolved"]
        ambiguous = crosswalk["ambiguous"]
        keys = []
        category_joinable = True
        for atoms in db_frames:
            key = _frame_split_key(atoms, label="original_training_db_path")
            if key is None or key in ambiguous or key not in resolved:
                category_joinable = False
                break
            keys.append(key)
        category_splits = (
            [resolved[key] for key in keys] if category_joinable and keys else None)

        positional_splits = None
        for manifest_path in split_source_manifest_paths:
            manifest_data = _load_positional_split_manifest(manifest_path)
            if manifest_data is None:
                continue
            verified = _verified_positional_split_join(
                manifest_data, dataset_path=original_training_db_path, db_frames=db_frames)
            if verified is not None:
                positional_splits = [verified[i] for i in range(len(db_frames))]
                break

        if category_splits is not None and positional_splits is not None:
            # Both representations resolved independently -- they must AGREE, frame for frame,
            # or this fails closed rather than silently preferring either one.
            effective_splits = category_splits if category_splits == positional_splits else None
        elif category_splits is not None:
            effective_splits = category_splits
        elif positional_splits is not None:
            effective_splits = positional_splits
        else:
            effective_splits = None

        original_split_recovered = effective_splits is not None
        if original_split_recovered and target_split:
            genuine_holdout_test_frame_count = sum(
                1 for split in effective_splits if split == target_split)
            genuine_holdout_test_available = genuine_holdout_test_frame_count > 0
            if not genuine_holdout_test_available:
                genuine_holdout_test_frame_count = None

    operational_evaluation_population_available = False
    if operational_evaluation_population_path is not None:
        _read_frames(operational_evaluation_population_path,
                    label="operational_evaluation_population_path")
        operational_evaluation_population_available = True
    elif original_training_db_available:
        operational_evaluation_population_available = True
    elif deployment_domain_population_path is not None:
        _read_frames(deployment_domain_population_path,
                    label="deployment_domain_population_path")
        operational_evaluation_population_available = True

    independent_external_reference_available = False
    if independent_external_reference_path is not None:
        _read_frames(independent_external_reference_path,
                    label="independent_external_reference_path")
        same_as_db = (
            original_training_db_path is not None and
            Path(independent_external_reference_path).resolve() ==
            Path(original_training_db_path).resolve()
        )
        independent_external_reference_available = not same_as_db

    deployment_domain_population_available = False
    if deployment_domain_population_path is not None:
        _read_frames(deployment_domain_population_path, label="deployment_domain_population_path")
        deployment_domain_population_available = True

    profile = TeacherEvidenceProfile(
        teacher_model_available=teacher_model_available,
        operational_evaluation_population_available=operational_evaluation_population_available,
        original_training_db_available=original_training_db_available,
        original_labels_available=original_labels_available,
        original_split_recovered=original_split_recovered,
        genuine_holdout_test_available=genuine_holdout_test_available,
        independent_external_reference_available=independent_external_reference_available,
        deployment_domain_population_available=deployment_domain_population_available,
        original_split_confidence=(
            original_split_confidence if original_split_recovered else "NOT_AVAILABLE"),
        genuine_holdout_test_frame_count=genuine_holdout_test_frame_count,
        deployment_domain_matches_original_test_distribution=(
            deployment_domain_matches_original_test_distribution),
    )
    return profile, _profile_sha256(profile)
