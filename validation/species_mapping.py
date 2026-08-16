"""Resolves and validates the LAMMPS integer-``type`` -> chemical-species ordering
(``specorder``) that a raw LAMMPS dump needs before ``ase.io.read`` can be trusted to
report real chemical symbols. Without an explicit ``specorder``, ASE treats bare
integer atom types as literal atomic numbers (type 1 -> H, type 2 -> He, ...), which
silently corrupts species-dependent observables (RDF/coordination naming, density
mass) for any campaign whose LAMMPS atom-type order does not coincidentally match the
periodic table by index. This module never guesses or hardcodes a species order
itself -- callers must supply one resolved from an authoritative source (e.g. the
Student deployment configuration's ``deploy.elements``, the same ordering used to
build the LAMMPS ``pair_coeff`` line).
"""
from ase.data import chemical_symbols
from ase.io.formats import filetype

LAMMPS_DUMP_FORMATS = {"lammps-dump-text", "lammps-dump-binary"}


def requires_specorder(frames_path):
    """True if ``frames_path`` is a LAMMPS dump whose per-atom records carry only an
    integer ``type`` column (no ``element``/``species`` column), so ASE cannot resolve
    chemical species from the file alone. False for self-describing formats (e.g.
    extxyz) where species are never overwritten or required."""
    fmt = filetype(str(frames_path), read=True)
    if fmt not in LAMMPS_DUMP_FORMATS:
        return False
    if fmt == "lammps-dump-binary":
        # Binary dumps carry the same column layout as their text counterpart but
        # aren't line-sniffable here; treat as requiring an explicit mapping.
        return True
    columns = _lammps_dump_atoms_columns(frames_path)
    return "type" in columns and not ({"element", "species"} & set(columns))


def _lammps_dump_atoms_columns(frames_path):
    with open(frames_path) as fh:
        for line in fh:
            if line.startswith("ITEM: ATOMS"):
                return line.split()[2:]
    return []


def validate_specorder(specorder):
    """Validate that ``specorder`` is a non-empty ordered sequence of unique, real
    element symbols, and return it as a plain list. Raises ValueError otherwise --
    callers must fail closed rather than pass a malformed mapping to ASE."""
    if not specorder or not isinstance(specorder, (list, tuple)):
        raise ValueError("specorder must be a non-empty ordered sequence of element symbols")
    specorder = list(specorder)
    if len(set(specorder)) != len(specorder):
        raise ValueError(f"specorder must not contain duplicate species: {specorder}")
    unknown = [el for el in specorder if el not in chemical_symbols]
    if unknown:
        raise ValueError(f"specorder contains invalid element symbols: {unknown}")
    return specorder
