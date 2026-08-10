#!/usr/bin/env python3
"""SMOKE: real SIMPLE-NN preprocessing+1-epoch training via the FIXED kind:simple-nn adapter path,
on 3 small SMALL_002-labeled frames. Classifies whether the simple-nn env drift is BLOCKING.
NOT a scientific run."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ase.io import read, write
from adapters import load_config
from adapters.student import train_student

PROJ = Path(__file__).resolve().parents[1]
W = PROJ/"work"/"devv6_smoke"
LAB = PROJ/"runs/SIO2_DISTILLATION_DEV_V6_SMALL_002/artifacts/teacher_labeled.extxyz"
frames = read(str(LAB), index=":")
elig = sorted([a for a in frames if "teacher_energy" in a.info and "teacher_forces" in a.arrays], key=lambda a: len(a))
small = [elig[0], elig[len(elig)//2], elig[-1]] if len(elig)>=3 else elig  # varied sizes -> weight spread
smoke = W/"smoke3.extxyz"; write(str(smoke), small, format="extxyz")
print("smoke set:", [len(a) for a in small], "atoms")
cfg = load_config(str(W/"student.smoke.yaml"))
out = W/"smoke_out_seed234"
try:
    art = train_student(cfg, str(smoke), out, 234)
    print("SMOKE_RESULT: TRAIN_OK ->", art)
except Exception as e:
    import traceback; traceback.print_exc()
    print("SMOKE_RESULT: FAILED ->", type(e).__name__, str(e)[:300])
