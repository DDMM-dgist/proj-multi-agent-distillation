#!/usr/bin/env bash
# L4B setup + inspection on jbnu-gpu2. Run from inside the transferred repo:
#     cd <repo>   &&   bash work/l4b_gpu2_setup.sh
# Creates TWO ISOLATED conda envs (vllm-mad, mad-client), installs deps, and RECORDS exact
# resolved versions + `vllm serve --help` tool/guided-decoding support into work/l4b_gpu2_report.txt.
#
# HARD LIMITS (this script obeys them):
#   - NEVER modifies/uninstalls anything in conda BASE.
#   - NEVER launches a vLLM server (only `--help`), NEVER downloads a model, NEVER runs inference,
#     NEVER touches the GPU. (pip install + --help do not allocate GPU.)
set +e
EXPECT_HEAD="2432ebb"   # informational; real check prints the actual HEAD below
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REPORT="$REPO/work/l4b_gpu2_report.txt"
: > "$REPORT"
say(){ echo "$@" | tee -a "$REPORT"; }
sec(){ printf '\n===== %s =====\n' "$1" | tee -a "$REPORT"; }
run(){ echo "\$ $*" | tee -a "$REPORT"; "$@" 2>&1 | tee -a "$REPORT"; }

sec "0 context"
run hostname; run date; say "repo: $REPO"; say "conda: $(command -v conda)"

sec "1 transferred branch integrity"
run git -C "$REPO" rev-parse HEAD
run git -C "$REPO" log --oneline -1
say "(expected HEAD starts with $EXPECT_HEAD)"

sec "2 create isolated envs (base untouched)"
conda env list 2>&1 | tee -a "$REPORT"
for e in vllm-mad mad-client; do
  if conda env list 2>/dev/null | grep -qE "(^|/)$e[[:space:]]"; then say "env $e already exists (skip create)";
  else run conda create -y -n "$e" python=3.12; fi
done

sec "3 install vLLM into vllm-mad (records resolved version; NO launch)"
run conda run -n vllm-mad python -m pip install --upgrade pip
run conda run -n vllm-mad python -m pip install vllm

sec "4 record vllm-mad versions"
conda run -n vllm-mad python - <<'PY' 2>&1 | tee -a "$REPORT"
import importlib.metadata as M
def v(p):
    try: return M.version(p)
    except Exception as e: return f"absent ({type(e).__name__})"
print("python :", __import__("sys").version.split()[0])
print("vllm   :", v("vllm"))
print("torch  :", v("torch"))
try:
    import torch; print("torch.cuda_build:", torch.version.cuda, "| torch.cuda.is_available:", torch.cuda.is_available())
except Exception as e: print("torch import:", type(e).__name__, e)
print("transformers:", v("transformers"))
PY

sec "5 vllm serve --help : tool parser + guided/structured-output support"
# try the modern CLI, then the module entrypoint
HELP="$(conda run -n vllm-mad vllm serve --help 2>&1)"
[ -z "$HELP" ] && HELP="$(conda run -n vllm-mad python -m vllm.entrypoints.openai.api_server --help 2>&1)"
echo "$HELP" | grep -iE -- "--tool-call-parser|hermes|--enable-auto-tool-choice|--guided-decoding|--structured|xgrammar|outlines|lm-format|json" | tee -a "$REPORT"
say "--- (full help saved to work/l4b_gpu2_vllm_help.txt) ---"
echo "$HELP" > "$REPO/work/l4b_gpu2_vllm_help.txt"
# best-effort: list the exact choices offered for --tool-call-parser
echo "$HELP" | grep -iA3 -- "--tool-call-parser" | tee -a "$REPORT"

sec "6 install client into mad-client (our runtime + openai provider; NO GPU)"
run conda run -n mad-client python -m pip install --upgrade pip
run conda run -n mad-client python -m pip install -e "$REPO"'[pydantic-ai,local-openai]'

sec "7 record mad-client versions"
conda run -n mad-client python - <<'PY' 2>&1 | tee -a "$REPORT"
import importlib.metadata as M, sys
def v(p):
    try: return M.version(p)
    except Exception as e: return f"absent ({type(e).__name__})"
print("python       :", sys.version.split()[0])
print("pydantic     :", v("pydantic"))
print("pydantic-ai-slim:", v("pydantic-ai-slim"))
print("openai       :", v("openai"))
# prove the local provider path imports offline (no server, no key)
try:
    from runtimes.pydantic_ai.provider import select_provider_kind, build_local_model  # noqa
    print("runtime import: OK (select_provider_kind/build_local_model importable)")
except Exception as e:
    print("runtime import FAILED:", type(e).__name__, e)
PY

sec "8 env paths"
run conda env list

sec "DONE — paste work/l4b_gpu2_report.txt (and work/l4b_gpu2_vllm_help.txt if tool-parser unclear)"
