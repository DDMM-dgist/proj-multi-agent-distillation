import re, json, sys
from pathlib import Path

KISTI = Path("/home/hyunjin/CLAUDE/SIO2_FT/gpu_finetune_handoff/kisti_assets/kisti_pack")
SEEDPOOL = Path("/home/hyunjin/distill-real-user/local_inputs/sio2_fresh/seed_pool_11424")

CT_RE = re.compile(r"config_type=(\S+)")
E_RE = re.compile(r"(?:^|\s)energy=(-?[0-9.eE+-]+)")
DFTE_RE = re.compile(r"(?:^|\s)dft_energy=(-?[0-9.eE+-]+)")

def parse_xyz_headers(path):
    """Return list of (natoms:int, config_type:str, energy:float) in file order, fast line-based parse."""
    out = []
    with open(path, "r") as f:
        while True:
            line1 = f.readline()
            if not line1:
                break
            line1 = line1.strip()
            if not line1:
                continue
            natoms = int(line1)
            header = f.readline()
            ct_m = CT_RE.search(header)
            e_m = E_RE.search(header) or DFTE_RE.search(header)
            ct = ct_m.group(1) if ct_m else None
            e = float(e_m.group(1)) if e_m else None
            out.append((natoms, ct, e))
            # skip natoms atom lines
            for _ in range(natoms):
                f.readline()
    return out

print("Parsing dataset.xyz (kisti_pack) ...", file=sys.stderr)
full = parse_xyz_headers(KISTI / "dataset.xyz")
print(f"dataset.xyz frames: {len(full)}", file=sys.stderr)

# Build per-category ordered list of global indices (0-based, file order)
cat_indices = {}
for gidx, (natoms, ct, e) in enumerate(full):
    cat_indices.setdefault(ct, []).append(gidx)

print("Category counts in dataset.xyz:", file=sys.stderr)
for k in sorted(cat_indices):
    print(f"  {k}: {len(cat_indices[k])}", file=sys.stderr)

# Now verify against seed_pool_11424 per-category files: local_index N should match global frame content
mismatches = 0
checked = 0
for catdir in sorted(SEEDPOOL.iterdir()):
    if not catdir.is_dir():
        continue
    cat = catdir.name
    xyz_files = list(catdir.glob("*.xyz"))
    if not xyz_files:
        continue
    seed_frames = parse_xyz_headers(xyz_files[0])
    if cat not in cat_indices:
        print(f"CATEGORY MISSING IN dataset.xyz: {cat}", file=sys.stderr)
        continue
    gidxs = cat_indices[cat]
    if len(seed_frames) != len(gidxs):
        print(f"COUNT MISMATCH {cat}: seed_pool={len(seed_frames)} dataset.xyz={len(gidxs)}", file=sys.stderr)
    n = min(len(seed_frames), len(gidxs))
    for local_idx in range(n):
        s_natoms, s_ct, s_e = seed_frames[local_idx]
        g_natoms, g_ct, g_e = full[gidxs[local_idx]]
        checked += 1
        ok = (s_natoms == g_natoms) and (s_e is not None and g_e is not None and abs(s_e - g_e) < 1e-6)
        if not ok:
            mismatches += 1
            if mismatches <= 5:
                print(f"MISMATCH {cat} local_idx={local_idx}: seed(natoms={s_natoms},e={s_e}) vs dataset(natoms={g_natoms},e={g_e}) at gidx={gidxs[local_idx]}", file=sys.stderr)

print(f"Checked {checked} frame correspondences, mismatches={mismatches}", file=sys.stderr)

# Save mapping for reuse
mapping = {"cat_indices": cat_indices, "total_frames": len(full)}
with open("/tmp/kisti_verify/cat_indices.json", "w") as f:
    json.dump(mapping, f)
print("Saved mapping.", file=sys.stderr)
