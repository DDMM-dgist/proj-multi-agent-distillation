#!/usr/bin/env bash
# STAGE D-2 C1 — ADVISORY SEMANTIC JUDGE (separately-approvable step; NOT run by preparation).
# Reads ONLY the generated msd.csv + msd_summary.json (+ provenance) for the C1 run; performs NO
# scientific compute; modifies NO scientific artifact. One advisory Judge (deterministic_authoritative
# =false -> genuine LLM verdict, not bound), local-openai/qwen2.5-7b-instruct on the local vLLM.
# Requires GPU. NO scheduler. NO paid API. Preserves the authoritative Axis-A PASS. Writes only
# judge_interpretation.json + the Judge provenance. One attempt, retries=0.
#   EXPECT_HEAD=<sha> bash work/stage_d2_judge_run.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="$REPO/runs/stage_d2/d2c1-posthoc-msd-random_x006"
TASK="$REPO/examples/stage_d2/judge_interpretation_task.json"
JEXCH="$RUN_DIR/judge_exchange"
LOG="$RUN_DIR/judge_vllm.log"
EXPECT_HEAD="${EXPECT_HEAD:-}"
MODEL_PATH="${STAGE_D2_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
SERVED="${STAGE_D2_SERVED_MODEL_NAME:-qwen2.5-7b-instruct}"
DEV="${STAGE_D2_CUDA_DEVICE:-1}"
UTIL="${STAGE_D2_GPU_MEM_UTIL:-0.36}"
MIN_FREE="${STAGE_D2_MIN_FREE_MIB:-18000}"

PGID=""; STOPPED=0
term(){ [ -n "$PGID" ] || return 0; [ "$STOPPED" = 1 ] && return 0
  kill -TERM -"$PGID" 2>/dev/null; sleep 5; kill -KILL -"$PGID" 2>/dev/null; STOPPED=1; echo "vLLM $PGID down"; }
trap term INT TERM EXIT

cd "$REPO" || exit 1
H=$(git rev-parse HEAD); echo "HEAD=$H"
[ -n "$EXPECT_HEAD" ] && [ "$H" != "$EXPECT_HEAD" ] && { echo "HEAD != $EXPECT_HEAD"; exit 1; }
# guardrails: the run + its scientific artifacts must already exist and are read-only inputs here
for f in msd.csv msd_summary.json provenance.json criterion_results.json; do
  [ -f "$RUN_DIR/$f" ] || { echo "missing $f — run C1 first"; exit 1; }
done
grep -q '"STAGE_D2_C1_AXIS_A": "PASS"' "$RUN_DIR/provenance.json" || { echo "Axis-A not PASS; refuse"; exit 1; }

echo "== GPU$DEV free gate (>= $MIN_FREE) =="
FREE=$(nvidia-smi -i "$DEV" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
echo "GPU$DEV free=$FREE"; [ "${FREE:-0}" -ge "$MIN_FREE" ] || { echo "insufficient VRAM"; exit 2; }
if ss -ltnH 2>/dev/null | awk '{print $4}' | grep -qE '(:|\.)8000$'; then echo "port 8000 busy"; exit 4; fi

echo "== launch ONE vLLM on GPU$DEV =="
CUDA_VISIBLE_DEVICES="$DEV" setsid conda run -n vllm-mad --no-capture-output \
  vllm serve "$MODEL_PATH" --served-model-name "$SERVED" --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --max-model-len 8192 --max-num-seqs 1 --enforce-eager \
    --gpu-memory-utilization "$UTIL" --enable-auto-tool-choice --tool-call-parser hermes \
  > "$LOG" 2>&1 &
VP=$!; PGID=$(ps -o pgid= -p "$VP" | tr -d ' ')
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 2; done

echo "== one advisory Judge (retries=0); reads ONLY the run dir; writes judge_interpretation.json =="
mkdir -p "$JEXCH/exchange"
env -u ANTHROPIC_API_KEY PYDANTIC_AI_PROVIDER=local-openai PYDANTIC_AI_MODEL="$SERVED" \
    PYDANTIC_AI_BASE_URL=http://127.0.0.1:8000/v1 PYDANTIC_AI_SMOKE_CONFIRM=yes \
  conda run -n mad-client --no-capture-output python -m runtimes.pydantic_ai.cli run-task \
    --runtime pydantic-ai --agent judge --agent-specs-dir "$REPO/agent_specs" \
    --task "$TASK" --exchange-dir "$JEXCH/exchange" --read-allow "$RUN_DIR" \
    --repo-root "$REPO" --mode shadow --correlation-id d2c1-judge 2>&1 | tee "$RUN_DIR/judge_stdout.log"
RC=${PIPESTATUS[0]}
term
echo "== extract JudgeVote -> judge_interpretation.json (advisory; Axis-A PASS preserved) =="
conda run -n mad-client python - "$JEXCH/exchange" "$RUN_DIR/judge_interpretation.json" <<'PY'
import json, sys, glob
exch, out = sys.argv[1], sys.argv[2]
provs = sorted(glob.glob(f"{exch}/provenance/*.json"), key=lambda f: json.load(open(f)).get("recorded_at",""))
p = json.load(open(provs[-1])) if provs else {}
vote = (p.get("parsed_result") or {})
rec = {"status": "COMPLETED", "deterministic_authoritative": False,
       "advisory_verdict": vote.get("verdict"), "criteria_checked": vote.get("criteria_checked"),
       "rationale": vote.get("rationale"), "criterion_contradictions": len(p.get("criterion_contradictions") or []),
       "provider": p.get("provider"), "model_id": p.get("model_id"),
       "prompt_tokens": p.get("prompt_tokens"), "completion_tokens": p.get("completion_tokens"),
       "latency_s": p.get("latency_s"), "axis_a_authoritative_verdict": "PASS (preserved)"}
open(out, "w").write(json.dumps(rec, indent=2) + "\n")
print("wrote", out, "advisory_verdict=", rec["advisory_verdict"])
PY
echo "judge exit=$RC. NOTE: this script is PREPARATION — run only after separate explicit approval."
