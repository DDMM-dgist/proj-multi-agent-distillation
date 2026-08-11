#!/usr/bin/env bash
# STAGE D-1 — AUDITABLE FROZEN SCIENTIFIC DECISION SHADOW REPLAY on jbnu-gpu2.
# Default model = Qwen2.5-7B-Instruct (the validated scientific model). ONE vLLM session; the frozen
# replay checkpoints (all Judge decisions over metrics-only evidence) run SEQUENTIALLY, each EXACTLY
# ONCE, provider retries=0, request_limit=6 runtime guard, duplicate-read guard active. SHADOW ONLY:
# no controller mutation, no scientific side effects, no Anthropic/paid API. Semantic PASS/FAIL +
# AGREE/JUSTIFIED_DIFFERENCE/UNJUSTIFIED_DIFFERENCE are decided OFFLINE by tests/harness/stage_d1_evaluate.py
# against golden_decisions.json (historical verdicts are a REFERENCE, never read by the agent).
#   EXPECT_HEAD=<sha> bash tests/harness/stage_d1_shadow_replay.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
FIX="$REPO/tests/fixtures/stage_d1_replay"
OUT="$FIX/out"
LOG="$OUT/stage_d1_vllm.log"
EXPECT_HEAD="${EXPECT_HEAD:-}"
# Model = only run variable. Default 7B (validated scientific model). Frozen runtime otherwise.
STAGE_D1_MODEL_PATH="${STAGE_D1_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
STAGE_D1_SERVED_MODEL_NAME="${STAGE_D1_SERVED_MODEL_NAME:-qwen2.5-7b-instruct}"
STAGE_D1_CUDA_DEVICE="${STAGE_D1_CUDA_DEVICE:-1}"
# Reuse of the EMPIRICALLY-VALIDATED Stage C 7B BF16 co-scheduled profile (a resource-policy reuse,
# NOT a scientific/runtime-semantic change): Stage C ran Qwen2.5-7B end-to-end on this RTX 6000 Ada
# co-scheduled with VASP at util 0.36 / free 18683 MiB, completed all 12 tasks, vLLM shut down
# cleanly, VASP process/memory intact. Fail closed below MIN_FREE; selected GPU only; no auto-switch.
STAGE_D1_GPU_MEM_UTIL="${STAGE_D1_GPU_MEM_UTIL:-0.36}"      # ~17GB of 48; validated 7B co-scheduled
MIN_FREE_MIB="${STAGE_D1_MIN_FREE_MIB:-18000}"             # validated co-scheduled floor
DEV="$STAGE_D1_CUDA_DEVICE"

PGID=""; STOPPED=0
terminate_group(){ [ -n "$PGID" ] || return 0; [ "$STOPPED" = 1 ] && return 0
  kill -TERM -"$PGID" 2>/dev/null; sleep 5; kill -KILL -"$PGID" 2>/dev/null
  STOPPED=1; echo "vLLM process group $PGID terminated."; }
trap terminate_group INT TERM EXIT

cd "$REPO" || { echo "repo missing"; exit 1; }
echo "== [1] HEAD =="; H=$(git rev-parse HEAD); echo "$H"
echo "config: model=$STAGE_D1_MODEL_PATH served=$STAGE_D1_SERVED_MODEL_NAME gpu=$DEV util=$STAGE_D1_GPU_MEM_UTIL min_free=$MIN_FREE_MIB"
if [ -n "$EXPECT_HEAD" ] && [ "$H" != "$EXPECT_HEAD" ]; then echo "HEAD != $EXPECT_HEAD — sync first"; exit 1; fi

echo "== [2] network-free fixture validation =="
conda run -n mad-client --no-capture-output python tests/harness/stage_d1_validate.py "$REPO" >/dev/null || {
  echo "fixture validation FAILED -> not launching."; exit 1; }
echo "fixtures OK"

echo "== [3] output dirs =="
mapfile -t IDS < <(ls "$FIX/tasks" | sed 's/\.json$//' | sort)
for cid in "${IDS[@]}"; do mkdir -p "$OUT/$cid/exchange"; done

echo "== [4] GPU$DEV snapshot (pre-launch) =="
nvidia-smi -i "$DEV" --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi -i "$DEV" --query-compute-apps=pid,process_name,used_memory --format=csv
FREE=$(nvidia-smi -i "$DEV" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
echo "GPU$DEV free MiB=$FREE (gate >= $MIN_FREE_MIB)"
[ "${FREE:-0}" -ge "$MIN_FREE_MIB" ] || { echo "INSUFFICIENT GPU$DEV FREE VRAM -> stop (no retry, no GPU switch)."; exit 2; }

echo "== [4b] fail closed if port 8000 occupied =="
if ss -ltnH 2>/dev/null | awk '{print $4}' | grep -qE '(:|\.)8000$'; then echo "port 8000 in use -> stop."; exit 4; fi

echo "== [5] launch ONE vLLM server on GPU$DEV ($STAGE_D1_SERVED_MODEL_NAME) =="
CUDA_VISIBLE_DEVICES="$DEV" setsid conda run -n vllm-mad --no-capture-output \
  vllm serve "$STAGE_D1_MODEL_PATH" --served-model-name "$STAGE_D1_SERVED_MODEL_NAME" \
    --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --max-model-len 8192 --max-num-seqs 1 --enforce-eager \
    --gpu-memory-utilization "$STAGE_D1_GPU_MEM_UTIL" \
    --enable-auto-tool-choice --tool-call-parser hermes \
  > "$LOG" 2>&1 &
VP=$!; PGID=$(ps -o pgid= -p "$VP" | tr -d ' '); echo "vLLM pid=$VP pgid=$PGID log=$LOG"

echo "== [6] wait for /v1/models (<=180s) =="
READY=0
for i in $(seq 1 90); do
  kill -0 "$VP" 2>/dev/null || { echo "vLLM exited early"; break; }
  curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { READY=1; break; }
  grep -qiE "out of memory|CUDA error|torch.OutOfMemory|insufficient|Failed to infer" "$LOG" && { echo "init failure in log"; break; }
  sleep 2
done
if [ "$READY" != 1 ]; then echo "== vLLM NOT ready — preserving failure, NO retry =="; tail -50 "$LOG"; terminate_group; exit 3; fi
curl -s http://127.0.0.1:8000/v1/models >/dev/null && echo "/v1/models OK"

echo "== [7] run replay checkpoints SEQUENTIALLY (Judge; one each, retries=0, continue-on-failure) =="
declare -A RC
for cid in "${IDS[@]}"; do
  exdir="$OUT/$cid/exchange"
  echo "---- $cid ----"
  env -u ANTHROPIC_API_KEY PYDANTIC_AI_PROVIDER=local-openai PYDANTIC_AI_MODEL="$STAGE_D1_SERVED_MODEL_NAME" \
      PYDANTIC_AI_BASE_URL=http://127.0.0.1:8000/v1 PYDANTIC_AI_SMOKE_CONFIRM=yes \
    conda run -n mad-client --no-capture-output python -m runtimes.pydantic_ai.cli run-task \
      --runtime pydantic-ai --agent judge --agent-specs-dir "$REPO/agent_specs" \
      --task "$FIX/tasks/$cid.json" --exchange-dir "$exdir" --read-allow "$FIX/evidence" \
      --repo-root "$REPO" --mode shadow --correlation-id "$cid" \
      2>&1 | tee "$OUT/$cid/stdout.log"
  RC[$cid]=${PIPESTATUS[0]}; echo "exit[$cid]=${RC[$cid]}"
done

echo "== [8] stop vLLM immediately =="; terminate_group
echo "== [impact] GPU$DEV after run + VASP still present? =="
nvidia-smi -i "$DEV" --query-gpu=memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi -i "$DEV" --query-compute-apps=pid,process_name,used_memory --format=csv
echo "== [9] per-checkpoint CLI exit (semantic + AGREE/JUSTIFIED/UNJUSTIFIED decided OFFLINE) =="
for cid in "${IDS[@]}"; do echo "  $cid: exit=${RC[$cid]:-NA}"; done
echo "Copy back tests/fixtures/stage_d1_replay/out/ (stdout.log + provenance/*.json) for offline evaluation."
