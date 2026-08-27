"""Reusable Stage-8 primary-domain acceptance and alignment contracts.

The contract here is deliberately about deterministic evidence handling, not
model execution.  It provides three pieces needed by a Stage-8 evaluator:

* authoritative source-category -> frozen-domain population slicing;
* composition-aware elemental-reference energy alignment on one common PRIMARY
  fit scope;
* per-domain Student-vs-Teacher caps plus a zero-margin triangle-inequality
  Student-vs-DFT consistency envelope.
"""
from __future__ import annotations

from collections import Counter
from enum import Enum
from math import sqrt
from typing import Any, Iterable, Mapping

import numpy as np
from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase


STUDENT_VS_TEACHER = "student_vs_teacher"
TEACHER_VS_DFT = "teacher_vs_dft"
STUDENT_VS_DFT = "student_vs_dft"


class Stage8Role(str, Enum):
    PRIMARY_CLAIM = "PRIMARY_CLAIM"
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
    AMBIGUOUS = "AMBIGUOUS"


class DomainAssignment(ContractBase):
    """One authoritative raw source-category mapping."""

    source_category: str
    domain: str
    role: Stage8Role
    rationale: str = ""


class Stage8PrimaryPopulationPolicy(ContractBase):
    """Authoritative domain slicing policy for a Stage-8 population."""

    policy_id: str
    primary_domains: list[str]
    assignments: list[DomainAssignment]
    authority: str = "source_category"
    diagnostic_domains: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _well_formed(self):
        if self.authority != "source_category":
            raise ValueError("Stage-8 domain authority must be source_category")
        if len(set(self.primary_domains)) != len(self.primary_domains):
            raise ValueError("primary_domains contains duplicates")
        seen: set[str] = set()
        for assignment in self.assignments:
            if assignment.source_category in seen:
                raise ValueError(
                    f"duplicate source_category assignment: {assignment.source_category}"
                )
            seen.add(assignment.source_category)
            if (
                assignment.role == Stage8Role.PRIMARY_CLAIM
                and assignment.domain not in self.primary_domains
            ):
                raise ValueError(
                    f"PRIMARY_CLAIM assignment {assignment.source_category!r} maps "
                    f"to non-primary domain {assignment.domain!r}"
                )
        return self

    def assignment_for(self, source_category: str) -> DomainAssignment:
        mapping = {a.source_category: a for a in self.assignments}
        try:
            assignment = mapping[source_category]
        except KeyError as exc:
            raise ValueError(
                f"unmapped Stage-8 source_category {source_category!r}; fail closed"
            ) from exc
        if assignment.role == Stage8Role.AMBIGUOUS:
            raise ValueError(
                f"ambiguous Stage-8 source_category {source_category!r}; fail closed"
            )
        return assignment


class FrameDomainRecord(ContractBase):
    frame_index: int
    frame_id: str
    source_category: str
    domain: str
    role: Stage8Role


class Stage8PopulationDomainManifest(ContractBase):
    """Hash-bindable result of applying a population policy to a population."""

    manifest_id: str
    population_id: str
    source_population_path: str
    source_population_sha256: str
    policy_sha256: str
    primary_domains: list[str]
    frame_records: list[FrameDomainRecord]
    primary_frame_count: int
    diagnostic_frame_count: int
    primary_counts_by_domain: dict[str, int]
    diagnostic_counts_by_domain: dict[str, int]

    @model_validator(mode="after")
    def _counts_are_consistent(self):
        primary = [r for r in self.frame_records if r.role == Stage8Role.PRIMARY_CLAIM]
        diagnostic = [r for r in self.frame_records if r.role == Stage8Role.DIAGNOSTIC_ONLY]
        if self.primary_frame_count != len(primary):
            raise ValueError("primary_frame_count does not match frame_records")
        if self.diagnostic_frame_count != len(diagnostic):
            raise ValueError("diagnostic_frame_count does not match frame_records")
        counts = Counter(r.domain for r in primary)
        expected = {domain: int(counts.get(domain, 0)) for domain in self.primary_domains}
        if self.primary_counts_by_domain != expected:
            raise ValueError("primary_counts_by_domain does not match frame_records")
        missing = [domain for domain, n in expected.items() if n <= 0]
        if missing:
            raise ValueError(
                "Stage-8 PRIMARY population lacks required domain(s): "
                + ", ".join(missing)
            )
        return self

    @property
    def primary_indices(self) -> list[int]:
        return [r.frame_index for r in self.frame_records if r.role == Stage8Role.PRIMARY_CLAIM]

    @property
    def diagnostic_indices(self) -> list[int]:
        return [r.frame_index for r in self.frame_records if r.role == Stage8Role.DIAGNOSTIC_ONLY]


class ElementalAlignmentPolicy(ContractBase):
    """Composition-aware energy-alignment policy.

    ``species`` fixes the columns of the elemental-fraction design matrix.  The
    same design matrix and PRIMARY fit rows are used for every pairwise channel;
    fitted coefficients are deterministic nuisance values produced by the metric.
    """

    policy_id: str
    species: list[str]
    design: str = "elemental_fraction"
    fit_scope: str = "common_PRIMARY_CLAIM_population"
    projection: str = "least_squares_orthogonal_projection"

    @model_validator(mode="after")
    def _well_formed(self):
        if len(set(self.species)) != len(self.species):
            raise ValueError("alignment species contains duplicates")
        if self.design != "elemental_fraction":
            raise ValueError("only elemental_fraction alignment is supported")
        if self.fit_scope != "common_PRIMARY_CLAIM_population":
            raise ValueError("Stage-8 alignment fit scope must be the common PRIMARY population")
        if self.projection != "least_squares_orthogonal_projection":
            raise ValueError("Stage-8 alignment must use least-squares projection")
        return self


class Stage8AcceptancePolicy(ContractBase):
    """Per-domain Stage-8 acceptance policy."""

    policy_id: str
    primary_population_policy_sha256: str
    alignment_policy_sha256: str
    required_primary_domains: list[str]
    energy_rmse_cap_meV_per_atom: float = 25.0
    force_rmse_cap_eV_per_angstrom: float = 0.30
    teacher_vs_dft_semantics: str = "required_reference_evidence_only"
    student_vs_dft_envelope_margin: float = 0.0

    @model_validator(mode="after")
    def _no_extra_margin(self):
        if self.energy_rmse_cap_meV_per_atom != 25.0:
            raise ValueError("Stage-8 energy cap must remain the frozen 25.0 meV/atom")
        if self.force_rmse_cap_eV_per_angstrom != 0.30:
            raise ValueError("Stage-8 force cap must remain the frozen 0.30 eV/Angstrom")
        if self.teacher_vs_dft_semantics != "required_reference_evidence_only":
            raise ValueError("Teacher-vs-DFT must remain evidence-only in Stage 8")
        if self.student_vs_dft_envelope_margin != 0.0:
            raise ValueError("Stage-8 DFT consistency envelope permits no empirical margin")
        if len(set(self.required_primary_domains)) != len(self.required_primary_domains):
            raise ValueError("required_primary_domains contains duplicates")
        return self


def _metadata_value(frame: Mapping[str, Any], key: str) -> Any:
    if key not in frame or frame[key] in (None, ""):
        raise ValueError(f"Stage-8 frame is missing required provenance field {key!r}")
    return frame[key]


def build_population_domain_manifest(
    *,
    manifest_id: str,
    population_id: str,
    source_population_path: str,
    source_population_sha256: str,
    policy: Stage8PrimaryPopulationPolicy,
    frames: Iterable[Mapping[str, Any]],
) -> Stage8PopulationDomainManifest:
    """Apply authoritative source-category mapping to frame metadata."""

    records: list[FrameDomainRecord] = []
    for index, frame in enumerate(frames):
        source_category = str(_metadata_value(frame, "source_category"))
        assignment = policy.assignment_for(source_category)
        records.append(
            FrameDomainRecord(
                frame_index=index,
                frame_id=str(frame.get("structure_id", frame.get("frame_id", index))),
                source_category=source_category,
                domain=assignment.domain,
                role=assignment.role,
            )
        )

    primary_counts = Counter(r.domain for r in records if r.role == Stage8Role.PRIMARY_CLAIM)
    diagnostic_counts = Counter(
        r.domain for r in records if r.role == Stage8Role.DIAGNOSTIC_ONLY
    )
    return Stage8PopulationDomainManifest(
        manifest_id=manifest_id,
        population_id=population_id,
        source_population_path=source_population_path,
        source_population_sha256=source_population_sha256,
        policy_sha256=policy.content_sha256(),
        primary_domains=list(policy.primary_domains),
        frame_records=records,
        primary_frame_count=sum(primary_counts.values()),
        diagnostic_frame_count=sum(diagnostic_counts.values()),
        primary_counts_by_domain={
            domain: int(primary_counts.get(domain, 0)) for domain in policy.primary_domains
        },
        diagnostic_counts_by_domain={k: int(v) for k, v in sorted(diagnostic_counts.items())},
    )


def elemental_fraction_design(frames: list[Mapping[str, Any]], species: list[str]) -> np.ndarray:
    """Build an elemental-fraction design matrix from symbol counts."""

    rows = []
    for index, frame in enumerate(frames):
        symbols = list(_metadata_value(frame, "symbols"))
        n_atoms = len(symbols)
        if n_atoms <= 0:
            raise ValueError(f"frame {index} has no atoms")
        counts = Counter(str(s) for s in symbols)
        unknown = sorted(set(counts) - set(species))
        if unknown:
            raise ValueError(f"frame {index} contains species outside frozen set: {unknown}")
        rows.append([counts.get(sp, 0) / n_atoms for sp in species])
    return np.asarray(rows, dtype=float)


def aligned_energy_residuals_meV(
    ref_energies_eV: Iterable[float],
    pred_energies_eV: Iterable[float],
    frames: list[Mapping[str, Any]],
    alignment: ElementalAlignmentPolicy,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return per-frame aligned per-atom residuals in meV/atom.

    The residual vector is ``(E_pred - E_ref) / N_atoms``.  Its least-squares
    projection onto the elemental-fraction design matrix is removed.
    """

    ref = np.asarray(list(ref_energies_eV), dtype=float)
    pred = np.asarray(list(pred_energies_eV), dtype=float)
    if ref.shape != pred.shape or ref.shape != (len(frames),):
        raise ValueError("energy arrays must match the frame count")
    natoms = np.asarray([len(_metadata_value(frame, "symbols")) for frame in frames], dtype=float)
    residual = (pred - ref) / natoms
    design = elemental_fraction_design(frames, alignment.species)
    coeff, *_ = np.linalg.lstsq(design, residual, rcond=None)
    aligned = residual - design @ coeff
    return aligned * 1000.0, {
        "alignment_policy_sha256": alignment.content_sha256(),
        "species": list(alignment.species),
        "design_matrix_shape": list(design.shape),
        "fit_scope": alignment.fit_scope,
        "coefficients_eV_per_atom": [float(x) for x in coeff],
    }


def force_component_rmse(
    ref_forces: Iterable[np.ndarray],
    pred_forces: Iterable[np.ndarray],
) -> float:
    ref = [np.asarray(x, dtype=float) for x in ref_forces]
    pred = [np.asarray(x, dtype=float) for x in pred_forces]
    if len(ref) != len(pred):
        raise ValueError("force frame counts differ")
    for index, (a, b) in enumerate(zip(ref, pred)):
        if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
            raise ValueError(f"force arrays differ or are invalid at frame {index}")
    delta = np.concatenate([b - a for a, b in zip(ref, pred)], axis=0)
    return float(np.sqrt(np.mean(delta ** 2)))


def rmse(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("cannot compute RMSE over an empty set")
    return float(sqrt(float(np.mean(arr ** 2))))


def per_domain_aligned_energy_rmse(
    aligned_residuals_meV: np.ndarray,
    records: list[FrameDomainRecord],
    primary_domains: list[str],
) -> dict[str, float]:
    if aligned_residuals_meV.shape != (len(records),):
        raise ValueError("aligned residual vector does not match domain records")
    out = {}
    for domain in primary_domains:
        idx = [
            record.frame_index
            for record in records
            if record.role == Stage8Role.PRIMARY_CLAIM and record.domain == domain
        ]
        if not idx:
            raise ValueError(f"no PRIMARY frames for domain {domain}")
        out[domain] = rmse(aligned_residuals_meV[idx])
    return out


def evaluate_stage8_acceptance(
    *,
    policy: Stage8AcceptancePolicy,
    student_vs_teacher_energy_rmse: Mapping[str, float],
    student_vs_teacher_force_rmse: Mapping[str, float],
    teacher_vs_dft_energy_rmse: Mapping[str, float],
    teacher_vs_dft_force_rmse: Mapping[str, float],
    student_vs_dft_energy_rmse: Mapping[str, float],
    student_vs_dft_force_rmse: Mapping[str, float],
) -> dict[str, Any]:
    """Evaluate Stage-8 PASS/FAIL semantics from per-domain metrics."""

    domains = list(policy.required_primary_domains)
    missing = []
    for name, metric in {
        "student_vs_teacher_energy_rmse": student_vs_teacher_energy_rmse,
        "student_vs_teacher_force_rmse": student_vs_teacher_force_rmse,
        "teacher_vs_dft_energy_rmse": teacher_vs_dft_energy_rmse,
        "teacher_vs_dft_force_rmse": teacher_vs_dft_force_rmse,
        "student_vs_dft_energy_rmse": student_vs_dft_energy_rmse,
        "student_vs_dft_force_rmse": student_vs_dft_force_rmse,
    }.items():
        absent = [d for d in domains if d not in metric]
        if absent:
            missing.append({"metric": name, "domains": absent})
    if missing:
        raise ValueError(f"Stage-8 metrics missing required PRIMARY domains: {missing}")

    svt = {}
    dft = {}
    for domain in domains:
        e_svt = float(student_vs_teacher_energy_rmse[domain])
        f_svt = float(student_vs_teacher_force_rmse[domain])
        e_tdf = float(teacher_vs_dft_energy_rmse[domain])
        f_tdf = float(teacher_vs_dft_force_rmse[domain])
        e_sdf = float(student_vs_dft_energy_rmse[domain])
        f_sdf = float(student_vs_dft_force_rmse[domain])
        svt[domain] = {
            "aligned_energy_rmse_meV_per_atom": e_svt,
            "force_rmse_eV_per_angstrom": f_svt,
            "energy_pass": e_svt <= policy.energy_rmse_cap_meV_per_atom,
            "force_pass": f_svt <= policy.force_rmse_cap_eV_per_angstrom,
        }
        dft[domain] = {
            "aligned_energy_rmse_meV_per_atom": e_sdf,
            "force_rmse_eV_per_angstrom": f_sdf,
            "energy_bound_meV_per_atom": e_tdf + policy.energy_rmse_cap_meV_per_atom,
            "force_bound_eV_per_angstrom": f_tdf + policy.force_rmse_cap_eV_per_angstrom,
            "energy_pass": e_sdf <= e_tdf + policy.energy_rmse_cap_meV_per_atom,
            "force_pass": f_sdf <= f_tdf + policy.force_rmse_cap_eV_per_angstrom,
        }
    return {
        "student_vs_teacher": svt,
        "teacher_vs_dft": {
            "semantics": policy.teacher_vs_dft_semantics,
            "complete_reference_evidence": True,
        },
        "student_vs_dft_consistency": dft,
        "overall_pass": all(v["energy_pass"] and v["force_pass"] for v in svt.values())
        and all(v["energy_pass"] and v["force_pass"] for v in dft.values()),
        "no_empirical_margin": policy.student_vs_dft_envelope_margin == 0.0,
    }


__all__ = [
    "STUDENT_VS_TEACHER",
    "TEACHER_VS_DFT",
    "STUDENT_VS_DFT",
    "Stage8Role",
    "DomainAssignment",
    "Stage8PrimaryPopulationPolicy",
    "FrameDomainRecord",
    "Stage8PopulationDomainManifest",
    "ElementalAlignmentPolicy",
    "Stage8AcceptancePolicy",
    "build_population_domain_manifest",
    "elemental_fraction_design",
    "aligned_energy_residuals_meV",
    "force_component_rmse",
    "per_domain_aligned_energy_rmse",
    "evaluate_stage8_acceptance",
]
