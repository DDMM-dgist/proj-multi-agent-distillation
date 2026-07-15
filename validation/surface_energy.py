#!/usr/bin/env python3
"""Surface excess energy from composition-matched slab and bulk references."""
import argparse
import csv


def surface_energy(slab_energy_ev, bulk_reference_ev, area_a2, n_surfaces=2):
    if area_a2 <= 0 or n_surfaces <= 0:
        raise ValueError("area and n_surfaces must be positive")
    return (slab_energy_ev - bulk_reference_ev) / (n_surfaces * area_a2)


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
