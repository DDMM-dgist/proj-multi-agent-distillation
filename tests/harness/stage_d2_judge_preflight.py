#!/usr/bin/env python3
"""Stage D-2 C1 advisory-Judge EVIDENCE-READ PREFLIGHT (network-free; NO LLM).

Exercises the SAME read-only tool + path-resolution + allow-list layer the Judge uses
(runtimes.pydantic_ai.tool_registry.ReadOnlyToolset), to confirm — BEFORE authorizing a live attempt —
that the Judge task's evidence paths resolve INSIDE the declared read allow-list and are readable. This
directly guards against the attempt-1 READ_ALLOW_PATH_RESOLUTION_LOOP (bare filenames resolving against
repo-root CWD, outside the run-dir allow-list). Fails CLOSED if any required evidence path cannot be
read. Uses cwd = repo root and allow-list = the C1 run dir, exactly as the Judge runner does.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_REL = "tests/fixtures/stage_d2/d2c1-posthoc-msd-random_x006"
REQUIRED = [(f"{RUN_REL}/msd_summary.json", "read_json"),
            (f"{RUN_REL}/msd.csv", "read_csv_summary")]
OPTIONAL = [(f"{RUN_REL}/provenance.json", "read_json")]


def preflight(repo_root=None, include_optional=True):
    root = Path(repo_root or ROOT).resolve()
    from runtimes.pydantic_ai.tool_registry import ReadOnlyToolset, ToolAccessError
    allow = [str(root / RUN_REL)]                       # allow-list = the C1 run dir (exactly as runner)
    checks, ok_all = [], True
    cwd = os.getcwd()
    try:
        os.chdir(root)                                  # cwd = repo root (exactly as the Judge CLI)
        for rel, tool in REQUIRED + (OPTIONAL if include_optional else []):
            ts = ReadOnlyToolset(allow)                 # fresh toolset per read (avoid dup-guard noise)
            required = (rel, tool) in REQUIRED
            try:
                getattr(ts, tool)(rel)                  # read_json / read_csv_summary
                inside = str((root / rel).resolve()).startswith(str(Path(allow[0]).resolve()))
                checks.append({"path": rel, "tool": tool, "readable": True,
                               "inside_allow_list": inside, "required": required})
                ok_all = ok_all and inside
            except ToolAccessError as exc:
                checks.append({"path": rel, "tool": tool, "readable": False,
                               "inside_allow_list": False, "required": required, "error": str(exc)})
                if required:
                    ok_all = False
            except Exception as exc:  # noqa: BLE001  (parse/OS error -> fail closed if required)
                checks.append({"path": rel, "tool": tool, "readable": False,
                               "required": required, "error": f"{type(exc).__name__}: {exc}"})
                if required:
                    ok_all = False
    finally:
        os.chdir(cwd)
    return ok_all, checks


def main():
    ok, checks = preflight()
    import json
    print(json.dumps({"preflight_ok": ok, "checks": checks}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
