#!/usr/bin/env bash
# L4 Stage B — seven-role LOCAL smoke on jbnu-gpu2 (Qwen2.5-3B / vLLM 0.26.0 / hermes), ONE vLLM
# session, seven frozen tasks run SEQUENTIALLY (never parallel). Run from the transferred repo:
#     bash work/stage_b_local_smoke.sh
#     EXPECT_HEAD=<sha> bash work/stage_b_local_smoke.sh   # optional strict sync gate
#
# CONTINUE/STOP POLICY (explicit): each role runs EXACTLY ONCE with provider retries = 0. A role
# that fails is NOT retried and its model/prompt is NOT changed; its attempt is preserved and the
# runner CONTINUES to the next role, so all seven attempts exist in one server session. Per-role
# exit codes are reported at the end; aggregate PASS is decided OFF-LINE from the provenance.
#
# SAFETY: GPU1 only; conservative co-scheduling; producers shadow (dispatch dry_run -> no side
# effects); VASP never signaled/reniced/modified; fail closed on low VRAM or busy port; ONLY this
# run's vLLM process group is ever terminated (trap on INT/TERM/EXIT). No Anthropic/paid API.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
FIX="$REPO/examples/stage_b_smoke"
OUT="$FIX/out"
RUNDIR="$OUT/run"                      # controller manifest dir for producer roles
LOG="$OUT/stage_b_vllm.log"
MIN_FREE_MIB=16000
EXPECT_HEAD="${EXPECT_HEAD:-}"

PGID=""; STOPPED=0
terminate_group(){                     # terminates ONLY this run's process group; never VASP
  [ -n "$PGID" ] || return 0
  [ "$STOPPED" = 1 ] && return 0
  kill -TERM -"$PGID" 2>/dev/null; sleep 5; kill -KILL -"$PGID" 2>/dev/null
  STOPPED=1; echo "vLLM process group $PGID terminated."
}
trap terminate_group INT TERM EXIT

cd "$REPO" || { echo "repo missing"; exit 1; }
echo "== [1] HEAD =="; H=$(git rev-parse HEAD); echo "$H"
if [ -n "$EXPECT_HEAD" ] && [ "$H" != "$EXPECT_HEAD" ]; then echo "HEAD != $EXPECT_HEAD — sync first"; exit 1; fi

echo "== [2] network-free fixture validation (all 7) =="
conda run -n mad-client --no-capture-output python work/stage_b_validate.py "$REPO" || {
  echo "fixture validation FAILED -> not launching."; exit 1; }

echo "== [3] prepare output dirs + v7 controller manifest =="
mkdir -p "$RUNDIR"
for r in orchestrator literature data-curator ml-trainer simulation analyst judge; do
  mkdir -p "$OUT/$r/exchange"
done
cat > "$RUNDIR/manifest.json" <<'JSON'
{
  "schema_version": 7, "run_id": "stageB-smoke",
  "created_at": "2026-08-07T00:00:00+00:00", "updated_at": "2026-08-07T00:00:00+00:00",
  "workflow_config": "w", "artifacts": [], "project_dir": "p", "inputs": [],
  "code_revision": "x", "events": [],
  "stages": [{"name": "s", "status": "pending", "gate": "pending", "artifacts": []}],
  "iterations": [{"id": 1, "parent_iteration": null, "status": "active",
                  "started_at": "2026-08-07T00:00:00+00:00", "trigger": null}],
  "recoveries": [], "pending_recovery": null,
  "runtime_attempts": [], "idempotency": {}, "action_approvals": {}, "scheduler_jobs": {}
}
JSON

echo "== [4] GPU1 snapshot (pre-launch) =="
nvidia-smi -i 1 --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi -i 1 --query-compute-apps=pid,process_name,used_memory --format=csv
FREE=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
echo "GPU1 free MiB=$FREE (gate >= $MIN_FREE_MIB)"
[ "${FREE:-0}" -ge "$MIN_FREE_MIB" ] || { echo "INSUFFICIENT GPU1 FREE VRAM -> stop (no retry, no setting change, no GPU switch)."; exit 2; }

echo "== [4b] fail closed if port 8000 already occupied =="
if ss -ltnH 2>/dev/null | awk '{print $4}' | grep -qE '(:|\.)8000$'; then echo "port 8000 in use -> stop."; exit 4; fi

echo "== [5] launch ONE vLLM server on GPU1 (conservative) =="
# max-model-len raised 4096 (Stage A / Stage B attempt-1) -> 8192 for attempt-2: 4096 was our
# conservative smoke limit, not a real ceiling. This is a documented smoke-runtime config change,
# NOT the fix for the tool loop — the loop is bounded deterministically by the runtime's
# request_limit (RuntimeContext.request_limit, pydantic_ai UsageLimits). Attempt-1's 4096 config
# is preserved in provenance/docs.
CUDA_VISIBLE_DEVICES=1 setsid conda run -n vllm-mad --no-capture-output \
  vllm serve Qwen/Qwen2.5-3B-Instruct --served-model-name qwen2.5-3b-instruct \
    --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --max-model-len 8192 --max-num-seqs 1 --enforce-eager \
    --gpu-memory-utilization 0.20 \
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
if [ "$READY" != 1 ]; then
  echo "== vLLM NOT ready — preserving failure, NO retry =="; tail -50 "$LOG"; terminate_group; exit 3
fi
echo "-- /v1/models --"; curl -s http://127.0.0.1:8000/v1/models; echo

echo "== [7] run the seven roles SEQUENTIALLY (one task each, retries=0, no auto-retry) =="
ROLES="orchestrator literature data-curator ml-trainer simulation analyst judge"
declare -A RC
run_role(){
  local role="$1"; shift
  local exdir="$OUT/$role/exchange"
  echo "---- role: $role ----"
  env -u ANTHROPIC_API_KEY PYDANTIC_AI_PROVIDER=local-openai PYDANTIC_AI_MODEL=qwen2.5-3b-instruct \
      PYDANTIC_AI_BASE_URL=http://127.0.0.1:8000/v1 PYDANTIC_AI_SMOKE_CONFIRM=yes \
    conda run -n mad-client --no-capture-output python -m runtimes.pydantic_ai.cli run-task \
      --runtime pydantic-ai --agent "$role" --agent-specs-dir "$REPO/agent_specs" \
      --task "$FIX/$role.json" --exchange-dir "$exdir" \
      --repo-root "$REPO" --mode shadow --correlation-id "stageB-$role" "$@" \
      2>&1 | tee "$OUT/$role/stdout.log"
  RC[$role]=${PIPESTATUS[0]}
  echo "exit[$role]=${RC[$role]}"
}
# producers need --run-dir (controller); judge needs --read-allow (evidence); others need neither.
run_role orchestrator
run_role literature
run_role data-curator --run-dir "$RUNDIR"
run_role ml-trainer   --run-dir "$RUNDIR"
run_role simulation   --run-dir "$RUNDIR"
run_role analyst      --run-dir "$RUNDIR"
run_role judge        --read-allow "$FIX/artifacts"

echo "== [8] stop vLLM immediately =="; terminate_group

echo "== [impact] GPU1 after run + VASP still present? =="
nvidia-smi -i 1 --query-gpu=memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi -i 1 --query-compute-apps=pid,process_name,used_memory --format=csv

echo "== [9] per-role exit summary =="
for r in $ROLES; do echo "  $r: exit=${RC[$r]:-NA}"; done
echo "== provenance files =="
for r in $ROLES; do
  echo "-- $r --"; ls -1 "$OUT/$r/exchange/provenance/" 2>/dev/null || echo "  (none)"
done
echo "Copy back each out/<role>/stdout.log and out/<role>/exchange/provenance/*.json for verification."
