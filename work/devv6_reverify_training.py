import sys, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
A = Path("runs/SIO2_DISTILLATION_DEV_V6_SMALL_R1/artifacts")
d = json.loads((A/"student_committee.manifest.json").read_text())
models = d.get("models", [])          # FIXED key (was 'members' in buggy det_check)
ok = [m for m in models if os.path.exists(m.get("path",""))]
print("REVERIFY_TRAINING models=%d checkpoints_ok=%d verdict=%s" % (
    len(models), len(ok), "PASS" if (len(ok)>=1 and len(ok)==len(models)) else "REVISE"))
for m in models: print("  seed", m.get("seed"), "exists", os.path.exists(m.get("path","")), "sha", m.get("integrity",{}).get("sha256","")[:12])
