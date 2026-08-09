"""Stage D-2 C3 TRUSTED Allegro/NequIP teacher adapter.

Replaces the "arbitrary injected callable" with a trusted, committed, deterministic, provenance-recorded
loading path:  trusted executor -> TrustedAllegroAdapter -> compiled model -> one structure -> E/F.
The adapter (not an agent, not an ad-hoc callable) constructs the forward function. torch/nequip are
imported LAZILY inside the load/forward methods so this module imports without them (pure guards +
species mapping + conversion contract are always testable). NO forward pass is performed at import or
preflight; the model forward is invoked ONLY at approved execution (build_forward_fn's callable, called
inside the C3 executor after approval).

Model identity (source-grounded, gpu_finetune_handoff/models/MODEL_NOTES.txt + HANDOFF.md): the
compiled TorchScript deploy-only copy of the CURRENT teacher PRE-fine-tune (the base/original KISTI
Allegro teacher), symbols [O, Si], r_max 5.0, float32.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


class AdapterGuardError(RuntimeError):
    """A trusted-adapter guard refused (path allow-list, sha mismatch, species mapping, load)."""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TrustedAllegroAdapter:
    """Trusted loader/adapter for the compiled Allegro teacher. Construction is committed + deterministic;
    the model path is allow-listed and pinned by sha256 (immutability). The species mapping is derived
    from the MODEL's own ``type_names`` metadata, not assumed from numeric LAMMPS types."""

    def __init__(self, model_path: str, *, expected_sha256: str, allow_prefixes):
        real = str(Path(model_path).resolve())
        if not any(real.startswith(str(Path(a).resolve())) for a in allow_prefixes):
            raise AdapterGuardError(f"model outside allow-list: {model_path}")
        actual = sha256_file(model_path)
        if actual != expected_sha256:
            raise AdapterGuardError("teacher model sha256 mismatch (immutability check)")
        self.model_path = real
        self.model_sha256 = actual
        self.type_names = None          # filled by load(): e.g. ["O", "Si"]
        self.r_max = None
        self.model_dtype = None
        self._model = None
        self._loaded = False

    # ---- model-load-only preflight (lazy torch; NO forward) ----
    def load(self, device: str = "cpu") -> dict:
        """Load the compiled TorchScript teacher and read its embedded metadata. NO forward pass.
        Returns a provenance dict (versions, device, type_names, r_max, dtype)."""
        import platform
        import torch  # lazy
        extra = {k: "" for k in ("r_max", "type_names", "model_dtype", "config", "metadata")}
        model = torch.jit.load(self.model_path, map_location=device, _extra_files=extra)
        meta = {k: (v.decode(errors="replace") if isinstance(v, bytes) else str(v)) for k, v in extra.items() if v}
        self.type_names = meta.get("type_names", "").split() or None
        self.r_max = float(meta["r_max"]) if meta.get("r_max") else None
        self.model_dtype = meta.get("model_dtype")
        self._model = model
        self._loaded = True
        try:
            import nequip; nqv = getattr(nequip, "__version__", "?")
        except Exception:  # noqa: BLE001
            nqv = "?"
        try:
            import allegro; alv = getattr(allegro, "__version__", "?")
        except Exception:  # noqa: BLE001
            alv = "?"
        return {"model_sha256": self.model_sha256, "python": platform.python_version(),
                "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                "nequip": nqv, "allegro": alv, "device": device, "type_names": self.type_names,
                "r_max": self.r_max, "model_dtype": self.model_dtype,
                "required_input_fields": ["pos", "cell", "pbc", "atom_types", "edge_index"]}

    # ---- species / type mapping (pure; fail-closed) ----
    def species_index(self, symbol: str) -> int:
        """Model type index for a chemical symbol, from the model's own type_names. Fail-closed."""
        if not self.type_names:
            raise AdapterGuardError("type_names unknown — call load() first")
        if symbol not in self.type_names:
            raise AdapterGuardError(f"unexpected species '{symbol}' (model type_names={self.type_names})")
        return self.type_names.index(symbol)

    def map_lammps_types(self, lammps_types, type_symbol_map: dict):
        """Map LAMMPS integer atom types -> model type indices, VIA chemical symbol (never assume the
        numeric LAMMPS type equals the model's internal index). Fail-closed on unknown/reversed/
        unexpected species. Returns the per-atom model type-index list."""
        idx = []
        for t in lammps_types:
            sym = type_symbol_map.get(str(t))
            if sym is None:
                raise AdapterGuardError(f"unknown LAMMPS atom type {t} (map={type_symbol_map})")
            idx.append(self.species_index(sym))
        return idx

    def structure_conversion_contract(self, positions, lammps_types, box_L, type_symbol_map: dict):
        """The committed STRUCTURE CONVERSION CONTRACT: assemble the model-input fields from the parsed
        structure, preserving N/composition/cell/PBC. Records exactly how the structure becomes input.
        No relaxation / coordinate modification. Does NOT run the model."""
        n = len(positions)
        atom_type_index = self.map_lammps_types(lammps_types, type_symbol_map)
        symbols = [type_symbol_map[str(t)] for t in lammps_types]
        composition = {s: symbols.count(s) for s in sorted(set(symbols))}
        cell = [[box_L, 0.0, 0.0], [0.0, box_L, 0.0], [0.0, 0.0, box_L]]   # cubic
        return {"n_atoms": n, "composition": composition, "atom_type_index": atom_type_index,
                "cell": cell, "pbc": [True, True, True], "positions_len": len(positions),
                "units": {"length": "Angstrom", "energy": "eV", "force": "eV/Angstrom"},
                "atom_ordering": "by LAMMPS id (ascending)", "relaxation": "none"}

    # ---- the ONE trusted forward (invoked ONLY at approved execution, via the C3 executor) ----
    def build_forward_fn(self):
        """Return the trusted forward callable used by the generic C3 executor:
        forward_fn(positions, types, box_L, type_symbol_map) -> (energy_eV, forces_eV_A[N][3]).
        Constructs the neighbor list at r_max and runs exactly ONE model forward. Requires load()."""
        if not self._loaded:
            raise AdapterGuardError("call load() before build_forward_fn()")

        def forward_fn(positions, lammps_types, box_L, type_symbol_map):
            import torch  # lazy
            conv = self.structure_conversion_contract(positions, lammps_types, box_L, type_symbol_map)
            # NOTE: neighbor-list + AtomicDataDict assembly + model(input) run at approved execution in
            # the matching nequip/allegro env; kept behind this trusted callable. The output contract is
            # (total_energy_eV: float, forces_eV_A: list[[fx,fy,fz]] length N).
            from nequip.data import AtomicDataDict  # lazy; execution-time
            pos = torch.tensor(positions, dtype=torch.float32)
            cell = torch.tensor(conv["cell"], dtype=torch.float32)
            data = {AtomicDataDict.POSITIONS_KEY: pos,
                    AtomicDataDict.CELL_KEY: cell.unsqueeze(0),
                    AtomicDataDict.PBC_KEY: torch.tensor([[True, True, True]]),
                    AtomicDataDict.ATOM_TYPE_KEY: torch.tensor(conv["atom_type_index"], dtype=torch.long)}
            data = AtomicDataDict.with_edge_vectors(  # neighbor list at r_max
                AtomicDataDict.compute_neighborlist(data, r_max=self.r_max))
            out = self._model(data)                                        # the ONE forward pass
            energy = float(out[AtomicDataDict.TOTAL_ENERGY_KEY].reshape(-1)[0].item())
            forces = out[AtomicDataDict.FORCE_KEY].detach().cpu().tolist()
            return energy, forces

        return forward_fn
