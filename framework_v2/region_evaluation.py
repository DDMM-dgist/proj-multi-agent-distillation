"""Region-aware V2 evaluation compatibility helpers."""
from __future__ import annotations

import math
from typing import Any

from pydantic import Field, model_validator

from framework_v2.contracts import ContractBase
from framework_v2.stage8_acceptance import Stage8PopulationDomainManifest, Stage8Role
from framework_v2.structural_regions import (
    StructuralRegion,
    StructuralRegionManifest,
    StructuralRegionProviderType,
)


def structural_regions_from_stage8_manifest(
    manifest: Stage8PopulationDomainManifest,
    *,
    manifest_id: str,
) -> StructuralRegionManifest:
    """Adapt historical explicit-domain Stage-8 evidence to V2 regions.

    PRIMARY_CLAIM and DIAGNOSTIC_ONLY semantics remain in the historical
    manifest.  The V2 surface exposes provider-neutral regions without
    promoting diagnostic frames into primary pass/fail logic.
    """

    frame_to_region: dict[str, str] = {}
    counts: dict[str, int] = {}
    for record in manifest.frame_records:
        if record.role != Stage8Role.PRIMARY_CLAIM:
            continue
        frame_to_region[record.frame_id] = record.domain
        counts[record.domain] = counts.get(record.domain, 0) + 1

    regions = [
        StructuralRegion(
            region_id=domain,
            provider_type=StructuralRegionProviderType.EXPLICIT_METADATA,
            membership_provenance=[manifest.content_sha256(), manifest.policy_sha256],
            population_size=count,
            semantic_annotation=domain,
            membership_manifest_sha256=manifest.content_sha256(),
        )
        for domain, count in sorted(counts.items())
    ]
    return StructuralRegionManifest(
        manifest_id=manifest_id,
        provider_type=StructuralRegionProviderType.EXPLICIT_METADATA,
        regions=regions,
        frame_to_region=frame_to_region,
        source_sha256=manifest.source_population_sha256,
    )


class EvaluationPopulationRegionBinding(ContractBase):
    binding_id: str
    structural_region_manifest_sha256: str
    evaluation_population_sha256: str
    frame_to_region: dict[str, str]
    required_region_ids: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _valid(self):
        if not self.frame_to_region:
            raise ValueError("evaluation binding requires at least one frame")
        if len(set(self.frame_to_region)) != len(self.frame_to_region):
            raise ValueError("duplicate evaluation frame IDs")
        return self


def bind_evaluation_population_to_regions(
    *,
    region_manifest: StructuralRegionManifest,
    evaluation_frame_ids: list[str],
    evaluation_population_sha256: str,
    binding_id: str,
    required_region_ids: list[str] | None = None,
) -> EvaluationPopulationRegionBinding:
    mapping: dict[str, str] = {}
    for fid in evaluation_frame_ids:
        region = region_manifest.frame_to_region.get(fid)
        if region is None:
            raise ValueError(f"evaluation frame {fid!r} has no structural-region assignment")
        mapping[fid] = region
    return EvaluationPopulationRegionBinding(
        binding_id=binding_id,
        structural_region_manifest_sha256=region_manifest.content_sha256(),
        evaluation_population_sha256=evaluation_population_sha256,
        frame_to_region=mapping,
        required_region_ids=list(required_region_ids or sorted(set(mapping.values()))),
        provenance=[region_manifest.content_sha256()],
    )


class FrameEvaluationRecord(ContractBase):
    frame_id: str
    n_atoms: int
    reference_channel: str
    energy_error_eV: float | None = None
    force_component_errors: list[float] | None = None
    uncertainty_metrics: dict[str, float | int | str] = Field(default_factory=dict)
    target_metrics: dict[str, float | int | str] = Field(default_factory=dict)
    coverage_metrics: dict[str, float | int | str] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)


class RegionEvaluationRecord(ContractBase):
    region_id: str
    region_membership_sha256: str
    evaluation_binding_sha256: str
    n_frames: int
    reference_channel: str
    energy_rmse_meV_per_atom: float | None = None
    energy_mae_meV_per_atom: float | None = None
    force_component_rmse_eV_per_angstrom: float | None = None
    namespaced_signals: dict[str, float | int | str | None] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)


def _rmse(values: list[float], scale: float) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) * scale


def _mae(values: list[float], scale: float) -> float:
    return (sum(abs(v) for v in values) / len(values)) * scale


def _namespace(prefix: str, rows: list[FrameEvaluationRecord], attr: str) -> dict[str, Any]:
    keys = sorted({k for r in rows for k in getattr(r, attr)})
    out: dict[str, Any] = {}
    for k in keys:
        vals = [getattr(r, attr)[k] for r in rows if k in getattr(r, attr)]
        nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if nums and len(nums) == len(vals):
            out[f"{prefix}.{k}"] = sum(nums) / len(nums)
        elif len({str(v) for v in vals}) == 1:
            out[f"{prefix}.{k}"] = vals[0]
        else:
            out[f"{prefix}.{k}"] = None
    return out


def aggregate_region_metrics(
    binding: EvaluationPopulationRegionBinding,
    frame_records: list[FrameEvaluationRecord],
) -> list[RegionEvaluationRecord]:
    by_frame: dict[str, FrameEvaluationRecord] = {}
    for row in frame_records:
        if row.frame_id in by_frame:
            raise ValueError(f"duplicate frame evaluation record {row.frame_id!r}")
        by_frame[row.frame_id] = row

    grouped: dict[str, list[FrameEvaluationRecord]] = {rid: [] for rid in binding.required_region_ids}
    for fid, rid in binding.frame_to_region.items():
        if fid not in by_frame:
            raise ValueError(f"required evaluation frame {fid!r} missing")
        grouped.setdefault(rid, []).append(by_frame[fid])

    out: list[RegionEvaluationRecord] = []
    for rid in sorted(grouped):
        rows = grouped[rid]
        if not rows:
            raise ValueError(f"required region {rid!r} has no evaluation evidence")
        channels = {r.reference_channel for r in rows}
        if len(channels) != 1:
            raise ValueError(f"mixed reference channels in region {rid!r}")

        e = [r.energy_error_eV / r.n_atoms for r in rows if r.energy_error_eV is not None]
        f = [x for r in rows for x in (r.force_component_errors or [])]

        signals: dict[str, Any] = {
            "energy.rmse_meV_per_atom": _rmse(e, 1000.0) if e else None,
            "energy.mae_meV_per_atom": _mae(e, 1000.0) if e else None,
            "force.component_rmse_eV_per_angstrom": _rmse(f, 1.0) if f else None,
        }
        signals.update(_namespace("target", rows, "target_metrics"))
        signals.update(_namespace("uncertainty", rows, "uncertainty_metrics"))
        signals.update(_namespace("coverage", rows, "coverage_metrics"))

        out.append(
            RegionEvaluationRecord(
                region_id=rid,
                region_membership_sha256=binding.structural_region_manifest_sha256,
                evaluation_binding_sha256=binding.content_sha256(),
                n_frames=len(rows),
                reference_channel=next(iter(channels)),
                energy_rmse_meV_per_atom=signals["energy.rmse_meV_per_atom"],
                energy_mae_meV_per_atom=signals["energy.mae_meV_per_atom"],
                force_component_rmse_eV_per_angstrom=signals[
                    "force.component_rmse_eV_per_angstrom"
                ],
                namespaced_signals=signals,
                evidence_refs=[
                    binding.content_sha256(),
                    *[ref for r in rows for ref in r.evidence_refs],
                ],
            )
        )
    return out


__all__ = [
    "EvaluationPopulationRegionBinding",
    "FrameEvaluationRecord",
    "RegionEvaluationRecord",
    "aggregate_region_metrics",
    "bind_evaluation_population_to_regions",
    "structural_regions_from_stage8_manifest",
]
