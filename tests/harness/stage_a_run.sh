#!/usr/bin/env bash
# L4C Stage A runner (jbnu-gpu2): conservative co-scheduling on GPU1 while VASP runs.
# Run from the transferred repo:  bash tests/harness/stage_a_run.sh
# Optional strict sync gate:      EXPECT_HEAD=<sha> bash tests/harness/stage_a_run.sh
#
# SAFETY (enforced below): GPU1 only; fixed conservative profile; single request; NO retry;
# server stopped after; VASP never signaled/reniced/modified; fails closed on low VRAM or busy
# port; only THIS run's process group is ever terminated (trap on INT/TERM/EXIT).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/tests/fixtures/stage_a_judge_smoke/out"
LOG="$OUT/stage_a_vllm.log"
MIN_FREE_MIB=16000        # co-scheduling headroom above vLLM's ~20% (0.20*48G) allocation
EXPECT_HEAD="${EXPECT_HEAD:-}"

mkdir -p "$OUT"           # ensure output dir exists BEFORE tee (fixes stdout.log persistence)

PGID=""; STOPPED=0
# Terminates ONLY this run's vLLM process group. Never signals any other process (VASP untouched).
terminate_group(){
  [ -n "$PGID" ] || return 0
  [ "$STOPPED" = 1 ] && return 0
  kill -TERM -"$PGID" 2>/dev/null; sleep 5; kill -KILL -"$PGID" 2>/dev/null
  STOPPED=1; echo "vLLM process group $PGID terminated."
}
trap terminate_group INT TERM EXIT     # Ctrl+C / TERM / any exit cleans up only the recorded PGID

cd "$REPO" || { echo "repo missing"; exit 1; }
echo "== [1] HEAD =="; H=$(git rev-parse HEAD); echo "$H"
if [ -n "$EXPECT_HEAD" ] && [ "$H" != "$EXPECT_HEAD" ]; then
  echo "HEAD != EXPECT_HEAD ($EXPECT_HEAD) — sync first"; exit 1
fi

echo "== [2] ensure model cached (GPU-free) =="
conda run -n vllm-mad hf download Qwen/Qwen2.5-3B-Instruct || { echo "download failed"; exit 1; }

echo "== [3] GPU1 snapshot (pre-launch) =="
nvidia-smi -i 1 --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
echo "-- GPU1 compute processes (VASP etc.; read-only) --"
nvidia-smi -i 1 --query-compute-apps=pid,process_name,used_memory --format=csv
FREE=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
echo "GPU1 free MiB=$FREE (gate >= $MIN_FREE_MIB)"
[ "${FREE:-0}" -ge "$MIN_FREE_MIB" ] || { echo "INSUFFICIENT GPU1 FREE VRAM -> stop (no retry, no setting change, no GPU switch)."; exit 2; }

echo "== [3b] fail closed if port 8000 already occupied =="
if ss -ltnH 2>/dev/null | awk '{print $4}' | grep -qE '(:|\.)8000$'; then
  echo "port 8000 already in use -> stop."; exit 4
fi

echo "== [4] launch vLLM on GPU1 (conservative, background) =="
CUDA_VISIBLE_DEVICES=1 setsid conda run -n vllm-mad --no-capture-output \
  vllm serve Qwen/Qwen2.5-3B-Instruct --served-model-name qwen2.5-3b-instruct \
    --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --max-model-len 4096 --max-num-seqs 1 --enforce-eager \
    --gpu-memory-utilization 0.20 \
    --enable-auto-tool-choice --tool-call-parser hermes \
  > "$LOG" 2>&1 &
VP=$!; PGID=$(ps -o pgid= -p "$VP" | tr -d ' '); echo "vLLM pid=$VP pgid=$PGID log=$LOG"

echo "== [5] wait for /v1/models (<=180s); detect OOM/early-exit =="
READY=0
for i in $(seq 1 90); do
  kill -0 "$VP" 2>/dev/null || { echo "vLLM exited early"; break; }
  curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { READY=1; break; }
  grep -qiE "out of memory|CUDA error|torch.OutOfMemory|insufficient|Failed to infer" "$LOG" && { echo "init failure in log"; break; }
  sleep 2
done
if [ "$READY" != 1 ]; then
  echo "== vLLM NOT ready — preserving first failure, NO retry =="; tail -50 "$LOG"; terminate_group
  echo "== post GPU1 =="; nvidia-smi -i 1 --query-gpu=memory.used,memory.free --format=csv
  nvidia-smi -i 1 --query-compute-apps=pid,process_name,used_memory --format=csv; exit 3
fi
echo "-- /v1/models --"; curl -s http://127.0.0.1:8000/v1/models; echo

echo "== [6] Stage A (one frozen task, shadow, retries=0) =="
env -u ANTHROPIC_API_KEY PYDANTIC_AI_PROVIDER=local-openai PYDANTIC_AI_MODEL=qwen2.5-3b-instruct \
    PYDANTIC_AI_BASE_URL=http://127.0.0.1:8000/v1 PYDANTIC_AI_SMOKE_CONFIRM=yes \
  conda run -n mad-client --no-capture-output python -m runtimes.pydantic_ai.cli run-task \
    --runtime pydantic-ai --agent judge --agent-specs-dir "$PWD/agent_specs" \
    --task "$PWD/tests/fixtures/stage_a_judge_smoke/task.json" \
    --exchange-dir "$OUT/exchange" \
    --read-allow "$PWD/tests/fixtures/stage_a_judge_smoke/artifacts" \
    --repo-root "$PWD" --mode shadow --correlation-id stageA-judge-0001 --probe-server \
    2>&1 | tee "$OUT/stdout.log"
echo "Stage A CLI exit=${PIPESTATUS[0]}"

echo "== [7] stop vLLM immediately =="; terminate_group
echo "== [impact] GPU1 after run + VASP still present? =="
nvidia-smi -i 1 --query-gpu=memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi -i 1 --query-compute-apps=pid,process_name,used_memory --format=csv
echo "== provenance files =="
ls -1 "$OUT"/exchange/provenance/ 2>/dev/null
