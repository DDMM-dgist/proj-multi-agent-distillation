#!/usr/bin/env python3
"""Surface excess energy from composition-matched slab and bulk references."""
import argparse
import csv
import json
from pathlib import Path


def surface_energy(slab_energy_ev, bulk_reference_ev, area_a2, n_surfaces=2):
    if area_a2 <= 0 or n_surfaces <= 0:
        raise ValueError("area and n_surfaces must be positive")
    return (slab_energy_ev - bulk_reference_ev) / (n_surfaces * area_a2)


def validate_surface_manifest(manifest_path, required_methods=None):
    """Validate comparable static surface-excess-energy evidence."""
    payload = json.loads(Path(manifest_path).read_text())
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
        calculated = surface_energy(float(entry["slab_energy_ev"]),
                                    float(entry["bulk_reference_ev"]),
                                    float(entry["area_a2"]), int(entry["n_surfaces"])) * 16.02176634
        if abs(calculated - float(entry["surface_energy_J_m2"])) > 1e-8:
            raise ValueError(f"surface energy is inconsistent for method {entry['method']}")
    return payload


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
