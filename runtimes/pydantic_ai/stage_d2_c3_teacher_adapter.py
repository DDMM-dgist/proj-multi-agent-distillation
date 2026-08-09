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


def device_consistency_report(model_devices, input_devices, target_device: str) -> dict:
    """Pure, model-free device-consistency check (source-grounded on the attempt-2 failure:
    ``Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cpu``).

    ``model_devices`` / ``input_devices`` are iterables of torch-device strings (e.g. ``"cuda:1"``,
    ``"cpu"``). Returns ``ok`` = every model buffer/param AND every model input tensor sits on exactly
    ``target_device`` (no CPU/CUDA mix). Testable with synthetic device sets — no torch, no model call."""
    tgt = str(target_device)
    md = sorted({str(d) for d in model_devices})
    idd = sorted({str(d) for d in input_devices})
    all_dev = sorted(set(md) | set(idd))
    model_offenders = [d for d in md if d != tgt]
    input_offenders = [d for d in idd if d != tgt]
    ok = (len(all_dev) > 0 and all_dev == [tgt])
    return {"ok": ok, "target_device": tgt, "model_devices": md, "input_devices": idd,
            "all_devices": all_dev, "mixed": len(all_dev) > 1,
            "model_offenders": model_offenders, "input_offenders": input_offenders}


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
        self._device = "cpu"            # the load() device request (real placement is read from the model)
        # forward-phase flags — the execution state machine reads these to classify a failure as
        # BEFORE / DURING / AFTER the model forward (attempt-2 was a DURING-forward device mismatch that
        # the old "teacher_ef.json exists?" heuristic mislabeled BEFORE). Reset in build_forward_fn().
        self.forward_invoked = False
        self.forward_completed = False

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
        self._device = device
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

    # ---- device introspection (NO forward; reads the model's ACTUAL param/buffer placement) ----
    def _target_device(self):
        """The device the model actually lives on (read from a real param/buffer, NOT assumed from the
        load arg). Falls back to CPU when no model is loaded (pure-logic unit tests)."""
        import torch  # lazy
        if self._model is not None:
            for p in self._model.parameters():
                return p.device
            for b in self._model.buffers():
                return b.device
        return torch.device("cpu")

    def model_device_report(self) -> dict:
        """Enumerate the device of every model parameter and buffer (esp. edge-normalization buffers like
        ``_rmax_recip``, the cuda:1 side of the attempt-2 mismatch). NO forward pass. Requires load()."""
        if not self._loaded or self._model is None:
            raise AdapterGuardError("call load() before model_device_report()")
        params = {n: str(p.device) for n, p in self._model.named_parameters()}
        buffers = {n: str(b.device) for n, b in self._model.named_buffers()}
        return {"parameters": params, "buffers": buffers,
                "devices": sorted(set(params.values()) | set(buffers.values())),
                "target_device": str(self._target_device())}

    @staticmethod
    def input_device_report(data) -> dict:
        """Device/shape/dtype of every tensor field in a built model-input dict. NO forward pass."""
        import torch  # lazy
        fields = {k: {"device": str(v.device), "shape": list(v.shape), "dtype": str(v.dtype)}
                  for k, v in data.items() if torch.is_tensor(v)}
        return {"fields": fields, "devices": sorted({f["device"] for f in fields.values()})}

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

    # ---- model INPUT construction (nequip 0.16.1 API; NO model call — used by the input-build preflight) ----
    def build_model_input(self, positions, lammps_types, box_L, type_symbol_map):
        """Build the exact nequip 0.16.1 model input (AtomicDataDict) + neighbor list for one structure,
        WITHOUT calling the model. Returns (data, conv). Uses the real installed API
        ``nequip.data.compute_neighborlist_`` (the deprecated ``AtomicDataDict.with_edge_vectors`` from
        the attempt-1 failure does NOT exist in 0.16.1). Requires load() (for r_max) + torch/nequip."""
        if not self._loaded:
            raise AdapterGuardError("call load() before build_model_input()")
        import torch  # lazy
        from nequip.data import AtomicDataDict as A, compute_neighborlist_  # lazy; nequip 0.16.1 API
        conv = self.structure_conversion_contract(positions, lammps_types, box_L, type_symbol_map)
        # Build on CPU first: compute_neighborlist_ uses a CPU (ASE/matscipy) neighbor search, so its
        # edge_index / edge_cell_shift come back on CPU regardless of the position device.
        data = {A.POSITIONS_KEY: torch.tensor(positions, dtype=torch.float32),
                A.CELL_KEY: torch.tensor(conv["cell"], dtype=torch.float32).unsqueeze(0),
                A.PBC_KEY: torch.tensor([[True, True, True]]),
                A.ATOM_TYPE_KEY: torch.tensor(conv["atom_type_index"], dtype=torch.long)}
        data = compute_neighborlist_(data, r_max=self.r_max)   # adds edge_index + edge_cell_shift (CPU)
        # DEVICE-PLACEMENT FIX (attempt-2 root cause): the compiled model's buffers (e.g. edge-norm
        # `_rmax_recip`) live on the load device (cuda:1), but the input built above is on CPU — so the
        # in-model `r * rmax_recip` mixed cuda:1 and cpu. Move EVERY input tensor onto the model's actual
        # device (read from the model, not assumed) so model and input are consistently placed.
        dev = self._target_device()
        data = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in data.items()}
        return data, conv

    # ---- the ONE trusted forward (invoked ONLY at approved execution, via the C3 executor) ----
    def build_forward_fn(self):
        """Return the trusted forward callable used by the generic C3 executor:
        forward_fn(positions, types, box_L, type_symbol_map) -> (energy_eV, forces_eV_A[N][3]).
        Builds the input (build_model_input) and runs exactly ONE model forward. Requires load()."""
        if not self._loaded:
            raise AdapterGuardError("call load() before build_forward_fn()")
        self.forward_invoked = False
        self.forward_completed = False

        def forward_fn(positions, lammps_types, box_L, type_symbol_map):
            from nequip.data import AtomicDataDict as A  # lazy
            # PRE_FORWARD: building the (device-consistent) input is NOT a model invocation.
            data, _conv = self.build_model_input(positions, lammps_types, box_L, type_symbol_map)
            # FORWARD_STARTED: mark the deployed-model invocation as begun BEFORE the call, so an
            # exception INSIDE the model (attempt-2 device mismatch) is classified DURING_FORWARD.
            self.forward_invoked = True
            out = self._model(data)                                        # the ONE forward pass
            # FORWARD_COMPLETED: the model returned E/F.
            self.forward_completed = True
            energy = float(out[A.TOTAL_ENERGY_KEY].reshape(-1)[0].item())
            forces = out[A.FORCE_KEY].detach().cpu().tolist()
            return energy, forces

        return forward_fn
