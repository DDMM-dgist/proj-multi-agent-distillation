import sys
try:
    import yaml, numpy, ase, nequip
    from adapters.acquisition import label_with_teacher
    from adapters.teacher import load_teacher
    from adapters import load_config
    cfg = load_config("configs/runs/SIO2_DISTILLATION_DEV_V6_001/teacher.v6.yaml")
    calc = load_teacher(cfg)
    print("PREFLIGHT_OK: imports + v6 NequIPCalculator loaded:", type(calc).__name__)
except Exception as e:
    import traceback; traceback.print_exc(); print("PREFLIGHT_FAIL:", type(e).__name__, str(e)[:200])
