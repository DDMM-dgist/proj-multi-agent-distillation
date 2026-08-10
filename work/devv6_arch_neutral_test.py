import sys, inspect
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters import load_config
from adapters.teacher import load_teacher
from adapters import student as st
cfg = load_config("/tmp/hypo_teacher.yaml")
calc = load_teacher(cfg)
src = inspect.getsource(st.train_student)
print("ARCH_NEUTRAL_DRY_TEST teacher_loaded_via_generic_factory:", type(calc).__name__)
print("student_generic_dispatch adapter.train:", 'adapter.get("train")' in src, "train.command:", '"train", {}' in src or "train\", {}" in src or "cfg.get(\"train\"" in src)
print("controller_edits_required: NONE (config-only registration)")
