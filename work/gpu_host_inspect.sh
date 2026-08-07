#!/usr/bin/env bash
# L4A read-only inspection of the target GPU host (gpu2-via-cpu1).
# STRICTLY READ-ONLY: no installs, no downloads, no model pulls, no vLLM launch, no bind, no
# inference. The only network use is a bounded HEAD connectivity probe to PyPI/Hugging Face.
# Run from wherever `ssh gpu2-via-cpu1` resolves:
#     ssh gpu2-via-cpu1 'bash -s' < work/gpu_host_inspect.sh
# then paste the full output back.
set +e
line(){ printf '\n===== %s =====\n' "$1"; }
have(){ command -v "$1" >/dev/null 2>&1; }

line "0 identity"
echo "hostname: $(hostname 2>/dev/null)"; echo "user: $(id -un 2>/dev/null)"; echo "HOME=$HOME"
uname -a 2>/dev/null

line "1-7 nvidia-smi / GPU / driver / CUDA"
if have nvidia-smi; then
  nvidia-smi 2>&1 | head -25
  echo "--- query ---"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version --format=csv 2>&1
else echo "nvidia-smi: NOT FOUND"; fi
if have nvcc; then nvcc --version 2>&1 | tail -2; else echo "nvcc: absent"; fi

line "8-9 python / conda"
for p in python3 python; do have $p && echo "$p: $($p --version 2>&1) @ $(command -v $p)"; done
have conda && { echo "conda: $(command -v conda)"; conda --version 2>&1; } || echo "conda: absent"
have mamba && echo "mamba: $(command -v mamba)" || echo "mamba: absent"

line "10-12 torch / vllm / transformers / openai / pydantic-ai (base python3, best-effort)"
python3 - <<'PY' 2>&1
import importlib, importlib.util as u
for m in ("torch","vllm","transformers","openai","pydantic_ai","pydantic"):
    try:
        if u.find_spec(m) is None: print(f"{m}: not installed"); continue
        mod=importlib.import_module(m); v=getattr(mod,"__version__","?")
        extra=""
        if m=="torch":
            try: extra=f" | cuda_build={mod.version.cuda} avail={mod.cuda.is_available()}"
            except Exception: pass
        print(f"{m}: {v}{extra}")
    except Exception as e: print(f"{m}: import-error {type(e).__name__}: {e}")
PY

line "13 disk space"
df -h "$HOME" /scratch /tmp . 2>/dev/null | sort -u

line "14 Hugging Face cache + cached models"
echo "HF_HOME=${HF_HOME:-<unset>}  HF_HUB_CACHE=${HF_HUB_CACHE:-<unset>}"
for d in "${HF_HOME:-$HOME/.cache/huggingface}/hub" "$HOME/.cache/huggingface/hub"; do
  [ -d "$d" ] && { echo "cache $d:"; ls -1 "$d" 2>/dev/null | grep '^models--' | head -30; } || echo "absent: $d"
done

line "15 outbound connectivity (HEAD only; NO download)"
if have curl; then
  for url in https://pypi.org/simple/ https://huggingface.co; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -I --max-time 8 "$url" 2>/dev/null)
    echo "$url -> HTTP ${code:-timeout/blocked}"
  done
else echo "curl: absent (cannot probe)"; fi

line "16 loopback port 8000 (read-only, no bind)"
if have ss; then ss -ltnH 2>/dev/null | awk '{print $4}' | grep -E '(:|\.)8000$' >/dev/null && echo "127.0.0.1:8000 IN USE" || echo "127.0.0.1:8000 appears FREE"; \
elif have netstat; then netstat -ltn 2>/dev/null | grep -q ':8000 ' && echo ":8000 IN USE" || echo ":8000 appears FREE"; \
else echo "ss/netstat absent"; fi

line "17-19 shared filesystem visibility (JBNU /home)"
P=/home/hyunjin/mad-pydanticai-persist/proj-mad-pydanticai-full-runtime
if [ -d "$P" ]; then
  echo "persistent worktree VISIBLE: $P"
  have git && echo "  HEAD: $(git -C "$P" rev-parse HEAD 2>/dev/null)  (JBNU expects da54f6a...)" || echo "  git absent (cannot read HEAD)"
else echo "persistent worktree NOT visible at $P  -> /home is NOT shared here"; fi
V=/tmp/mad-impl-venv-ma0Qre/venv/bin/python
[ -x "$V" ] && echo "JBNU /tmp venv visible: $V" || echo "JBNU /tmp venv NOT visible (expected: /tmp is node-local)"

line "20 direct GPU box vs scheduler/login gateway"
for c in sinfo squeue sbatch qsub qstat pbsnodes; do have $c && echo "scheduler client present: $c"; done
if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then echo "GPUs directly visible here -> looks like a DIRECT interactive GPU box"; else echo "no direct GPUs here -> may be a login/gateway; GPUs likely behind a scheduler"; fi

line "DONE"
