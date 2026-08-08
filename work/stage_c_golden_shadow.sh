#!/usr/bin/env bash
# STAGE C — GOLDEN-TASK SHADOW VALIDATION on jbnu-gpu2 (Qwen2.5-3B / vLLM 0.26.0 / hermes /
# local-openai / PydanticAI 0.8.1). ONE vLLM session; the frozen golden tasks run SEQUENTIALLY,
# each EXACTLY ONCE, provider retries = 0, request_limit=6 runtime guard. Producers shadow/dry-run.
# Run from the transferred repo:  EXPECT_HEAD=<sha> bash work/stage_c_golden_shadow.sh
#
# CONTINUE/STOP POLICY (explicit): a task that fails is preserved and NOT retried, and its model/
# prompt is NOT changed; the runner CONTINUES to the next task so all golden attempts exist in one
# session. Semantic PASS/FAIL + metrics are computed OFFLINE by work/stage_c_evaluate.py against the
# FROZEN golden_expectations.json (never edited to match results).
#
# SAFETY: GPU1 only; relaxed co-scheduled smoke policy (MIN_FREE_MIB=12000, gpu-mem-util 0.18);
# VASP never signaled/reniced/modified; fail closed on low VRAM or busy port; only THIS run's
# process group is terminated (trap). No Anthropic/paid API. No scientific side effects.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
FIX="$REPO/examples/stage_c_golden"
OUT="$FIX/out"
RUNDIR="$OUT/run"
LOG="$OUT/stage_c_vllm.log"
EXPECT_HEAD="${EXPECT_HEAD:-}"
# MODEL-ONLY escalation config: the ONLY independent variable is the model. With NO env overrides
# these defaults reproduce the exact frozen 3B configuration byte-for-byte. Every frozen benchmark
# input (fixtures, golden_expectations.json, committed evaluator, Judge prompt, producer prompts,
# duplicate-read guard, request_limit=6, authorization, routes, tools, semantic rules) is UNCHANGED.
STAGE_C_MODEL_PATH="${STAGE_C_MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"            # vLLM served model repo id
STAGE_C_SERVED_MODEL_NAME="${STAGE_C_SERVED_MODEL_NAME:-qwen2.5-3b-instruct}"  # --served-model-name + PYDANTIC_AI_MODEL
STAGE_C_CUDA_DEVICE="${STAGE_C_CUDA_DEVICE:-1}"                                # GPU index (no auto-switch)
STAGE_C_GPU_MEM_UTIL="${STAGE_C_GPU_MEM_UTIL:-0.18}"                           # vLLM --gpu-memory-utilization
MIN_FREE_MIB="${STAGE_C_MIN_FREE_MIB:-12000}"                                  # pre-launch free-VRAM gate (MiB)
DEV="$STAGE_C_CUDA_DEVICE"

PGID=""; STOPPED=0
terminate_group(){ [ -n "$PGID" ] || return 0; [ "$STOPPED" = 1 ] && return 0
  kill -TERM -"$PGID" 2>/dev/null; sleep 5; kill -KILL -"$PGID" 2>/dev/null
  STOPPED=1; echo "vLLM process group $PGID terminated."; }
trap terminate_group INT TERM EXIT

cd "$REPO" || { echo "repo missing"; exit 1; }
echo "== [1] HEAD =="; H=$(git rev-parse HEAD); echo "$H"
echo "config: model=$STAGE_C_MODEL_PATH served=$STAGE_C_SERVED_MODEL_NAME gpu=$DEV util=$STAGE_C_GPU_MEM_UTIL min_free=$MIN_FREE_MIB (defaults reproduce the 3B run)"
if [ -n "$EXPECT_HEAD" ] && [ "$H" != "$EXPECT_HEAD" ]; then echo "HEAD != $EXPECT_HEAD — sync first"; exit 1; fi

echo "== [2] network-free golden fixture validation =="
conda run -n mad-client --no-capture-output python work/stage_c_validate.py "$REPO" >/dev/null || {
  echo "golden fixture validation FAILED -> not launching."; exit 1; }
echo "golden fixtures OK"

echo "== [3] output dirs + v7 controller manifest (producers) =="
mkdir -p "$RUNDIR"
mapfile -t IDS < <(ls "$FIX/tasks" | sed 's/\.json$//' | sort)
for tid in "${IDS[@]}"; do mkdir -p "$OUT/$tid/exchange"; done
cat > "$RUNDIR/manifest.json" <<'JSON'
{
  "schema_version": 7, "run_id": "stageC-golden",
  "created_at": "2026-08-08T00:00:00+00:00", "updated_at": "2026-08-08T00:00:00+00:00",
  "workflow_config": "w", "artifacts": [], "project_dir": "p", "inputs": [],
  "code_revision": "x", "events": [],
  "stages": [{"name": "s", "status": "pending", "gate": "pending", "artifacts": []}],
  "iterations": [{"id": 1, "parent_iteration": null, "status": "active",
                  "started_at": "2026-08-08T00:00:00+00:00", "trigger": null}],
  "recoveries": [], "pending_recovery": null,
  "runtime_attempts": [], "idempotency": {}, "action_approvals": {}, "scheduler_jobs": {}
}
JSON

echo "== [4] GPU$DEV snapshot (pre-launch) =="
nvidia-smi -i "$DEV" --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi -i "$DEV" --query-compute-apps=pid,process_name,used_memory --format=csv
FREE=$(nvidia-smi -i "$DEV" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
echo "GPU$DEV free MiB=$FREE (gate >= $MIN_FREE_MIB)"
[ "${FREE:-0}" -ge "$MIN_FREE_MIB" ] || { echo "INSUFFICIENT GPU$DEV FREE VRAM -> stop (no retry, no GPU switch)."; exit 2; }

echo "== [4b] fail closed if port 8000 occupied =="
if ss -ltnH 2>/dev/null | awk '{print $4}' | grep -qE '(:|\.)8000$'; then echo "port 8000 in use -> stop."; exit 4; fi

echo "== [5] launch ONE vLLM server on GPU$DEV (model=$STAGE_C_SERVED_MODEL_NAME, util=$STAGE_C_GPU_MEM_UTIL) =="
CUDA_VISIBLE_DEVICES="$DEV" setsid conda run -n vllm-mad --no-capture-output \
  vllm serve "$STAGE_C_MODEL_PATH" --served-model-name "$STAGE_C_SERVED_MODEL_NAME" \
    --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --max-model-len 8192 --max-num-seqs 1 --enforce-eager \
    --gpu-memory-utilization "$STAGE_C_GPU_MEM_UTIL" \
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

echo "== [7] run golden tasks SEQUENTIALLY (one each, retries=0, continue-on-failure) =="
declare -A RC
run_one(){
  local tid="$1"; local role; role=$(conda run -n mad-client --no-capture-output \
    python -c "import json;print(json.load(open('$FIX/tasks/$tid.json'))['agent'])")
  local exdir="$OUT/$tid/exchange"; local extra=()
  case "$role" in
    judge) extra=(--read-allow "$FIX/artifacts");;
    data-curator|ml-trainer|simulation|analyst) extra=(--run-dir "$RUNDIR");;
  esac
  echo "---- $tid (role=$role) ----"
  env -u ANTHROPIC_API_KEY PYDANTIC_AI_PROVIDER=local-openai PYDANTIC_AI_MODEL="$STAGE_C_SERVED_MODEL_NAME" \
      PYDANTIC_AI_BASE_URL=http://127.0.0.1:8000/v1 PYDANTIC_AI_SMOKE_CONFIRM=yes \
    conda run -n mad-client --no-capture-output python -m runtimes.pydantic_ai.cli run-task \
      --runtime pydantic-ai --agent "$role" --agent-specs-dir "$REPO/agent_specs" \
      --task "$FIX/tasks/$tid.json" --exchange-dir "$exdir" \
      --repo-root "$REPO" --mode shadow --correlation-id "$tid" "${extra[@]}" \
      2>&1 | tee "$OUT/$tid/stdout.log"
  RC[$tid]=${PIPESTATUS[0]}; echo "exit[$tid]=${RC[$tid]}"
}
for tid in "${IDS[@]}"; do run_one "$tid"; done

echo "== [8] stop vLLM immediately =="; terminate_group
echo "== [impact] GPU$DEV after run + VASP still present? =="
nvidia-smi -i "$DEV" --query-gpu=memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi -i "$DEV" --query-compute-apps=pid,process_name,used_memory --format=csv
echo "== [9] per-task CLI exit summary (semantic PASS/FAIL is decided OFFLINE by stage_c_evaluate.py) =="
for tid in "${IDS[@]}"; do echo "  $tid: exit=${RC[$tid]:-NA}"; done
echo "Copy back the whole examples/stage_c_golden/out/ tree (stdout.log + provenance/*.json)."
