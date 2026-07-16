#!/usr/bin/env python3
"""Surface excess energy from composition-matched slab and bulk references."""
import argparse
import csv
import json
import math
from pathlib import Path

import yaml

from validation.report import validate_validation_report


def surface_energy(slab_energy_ev, bulk_reference_ev, area_a2, n_surfaces=2):
    values = (float(slab_energy_ev), float(bulk_reference_ev), float(area_a2),
              float(n_surfaces))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("surface-energy inputs must be finite")
    slab_energy_ev, bulk_reference_ev, area_a2, n_surfaces = values
    if area_a2 <= 0 or n_surfaces <= 0:
        raise ValueError("area and n_surfaces must be positive")
    return (slab_energy_ev - bulk_reference_ev) / (n_surfaces * area_a2)


def _validate_surface_payload(payload, required_methods=None):
    """Validate comparable static surface-excess-energy values."""
    if payload.get("quantity") != "static_surface_excess_energy":
        raise ValueError("surface manifest quantity must be static_surface_excess_energy")
    if payload.get("unit") not in {"J/m2", "J/m^2"}:
        raise ValueError("surface manifest unit must be J/m2")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("surface manifest requires non-empty entries")
    methods = [entry.get("method") for entry in entries]
    missing_methods = set(required_methods or []) - set(methods)
    if missing_methods:
        raise ValueError("surface manifest is missing methods: " + ", ".join(sorted(missing_methods)))
    if len(methods) != len(set(methods)) or any(not method for method in methods):
        raise ValueError("surface methods must be unique and non-empty")

    common_fields = ("orientation", "termination", "area_a2", "n_surfaces",
                     "geometry_protocol", "reference_convention")
    baseline = {key: entries[0].get(key) for key in common_fields}
    if any(value in (None, "") for value in baseline.values()):
        raise ValueError("surface entries require orientation, termination, area, surfaces, geometry protocol, and reference convention")
    for entry in entries:
        if any(entry.get(key) != value for key, value in baseline.items()):
            raise ValueError("surface methods must use the same geometry and reference convention")
        if entry.get("composition_balance_confirmed") is not True:
            raise ValueError("surface composition balance must be explicitly confirmed")
        if entry.get("nonstoichiometric") is True and not entry.get("chemical_potentials"):
            raise ValueError("non-stoichiometric surfaces require chemical potentials")
        for key in ("slab_energy_ev", "bulk_reference_ev", "surface_energy_J_m2"):
            if key not in entry:
                raise ValueError(f"surface entry is missing {key}")
        reported = float(entry["surface_energy_J_m2"])
        if not math.isfinite(reported):
            raise ValueError(f"surface energy must be finite for method {entry['method']}")
        calculated = surface_energy(float(entry["slab_energy_ev"]),
                                    float(entry["bulk_reference_ev"]),
                                    float(entry["area_a2"]), int(entry["n_surfaces"])) * 16.02176634
        if abs(calculated - reported) > 1e-8:
            raise ValueError(f"surface energy is inconsistent for method {entry['method']}")
    return payload


def validate_surface_manifest(manifest_path, required_methods=None):
    """Validate the legacy value-only surface manifest."""
    return _validate_surface_payload(json.loads(Path(manifest_path).read_text()),
                                     required_methods)


def validate_surface_report(manifest_path, profile_path, required_methods=None,
                            submitted_artifacts=None, allowed_evidence=None,
                            enforce_required_pass=False):
    """Validate a surface result through the common evidence-bound report.

    The profile supplies the acceptance thresholds. Every slab and bulk energy
    entry names evidence roles that must exist in the report, and the controller
    can require those raw files to be submitted as stage artifacts.
    """
    manifest_path = Path(manifest_path).resolve()
    profile_path = Path(profile_path).expanduser().resolve()
    profile = yaml.safe_load(profile_path.read_text())
    thresholds = profile.get("surface_energetics", {}).get("thresholds", {})
    threshold_names = {
        "surface_delta:student_teacher": "student_teacher_max_abs_J_m2",
        "surface_delta:student_dft": "student_dft_max_abs_J_m2",
    }
    unresolved = [name for name in threshold_names.values()
                  if thresholds.get(name) is None]
    if unresolved:
        raise ValueError("surface validation thresholds are unresolved: " +
                         ", ".join(unresolved))
    required_pass = profile.get("required_pass_observables", [])
    if set(required_pass) != set(threshold_names):
        raise ValueError(
            "surface profile required_pass_observables must name both surface deltas"
        )

    payload = json.loads(manifest_path.read_text())
    if Path(payload.get("profile", "")).expanduser().resolve() != profile_path:
        raise ValueError("surface report profile does not match the bound validation profile")
    delta_methods = {"teacher", "student", "dft"}
    surface = _validate_surface_payload(
        payload.get("surface", {}), sorted(delta_methods | set(required_methods or []))
    )
    entries = {entry["method"]: entry for entry in surface["entries"]}
    evidence_roles = {item.get("role") for item in payload.get("evidence", [])
                      if isinstance(item, dict)}
    for method, entry in entries.items():
        for field in ("slab_evidence_role", "bulk_evidence_role"):
            role = entry.get(field)
            if not role or role not in evidence_roles:
                raise ValueError(f"surface {method} entry is missing bound evidence role {field}")

    required_observables = list(threshold_names)
    report = validate_validation_report(
        manifest_path,
        required_observables=required_observables,
        required_pass_observables=required_pass,
        submitted_artifacts=submitted_artifacts,
        require_submitted_evidence=True,
        allowed_evidence=allowed_evidence,
        enforce_required_pass=enforce_required_pass,
    )
    checks = {check["observable"]: check for check in report["checks"]}
    expected_values = {
        "surface_delta:student_teacher": (
            float(entries["student"]["surface_energy_J_m2"]) -
            float(entries["teacher"]["surface_energy_J_m2"])
        ),
        "surface_delta:student_dft": (
            float(entries["student"]["surface_energy_J_m2"]) -
            float(entries["dft"]["surface_energy_J_m2"])
        ),
    }
    for observable, threshold_name in threshold_names.items():
        check = checks[observable]
        expected_criterion = {"operator": "max_abs",
                              "threshold": float(thresholds[threshold_name])}
        if check.get("criterion") != expected_criterion:
            raise ValueError(f"surface criterion does not match profile for {observable}")
        if (check.get("unit") not in {"J/m2", "J/m^2"} or
                not math.isclose(float(check["value"]), expected_values[observable],
                                 rel_tol=0.0, abs_tol=1e-10)):
            raise ValueError(f"surface delta is inconsistent for {observable}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("table", help="CSV: method,slab_energy_ev,bulk_reference_ev,area_a2[,n_surfaces]")
    args = p.parse_args()
    with open(args.table) as f:
        rows = list(csv.DictReader(f))
    required = {"method", "slab_energy_ev", "bulk_reference_ev", "area_a2"}
    if not rows:
        raise SystemExit("input table has no data rows")
    missing = required - set(rows[0])
    if missing:
        raise SystemExit("missing required CSV columns: " + ", ".join(sorted(missing)))
    values = {}
    for row in rows:
        gamma = surface_energy(float(row["slab_energy_ev"]), float(row["bulk_reference_ev"]),
                               float(row["area_a2"]), int(row.get("n_surfaces") or 2))
        values[row["method"]] = gamma
        print(f"{row['method']}: {gamma:.8f} eV/A^2 = {gamma * 16.02176634:.6f} J/m^2")
    if "teacher" in values and "student" in values:
        print(f"student-teacher delta: {(values['student']-values['teacher'])*16.02176634:+.6f} J/m^2")


if __name__ == "__main__":
    main()
