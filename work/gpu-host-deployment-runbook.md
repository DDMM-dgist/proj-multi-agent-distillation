# L4 GPU-host deployment runbook — vLLM + PydanticAI Stage A (portable)

Purpose: run the frozen `stageA-judge-0001` single-Judge smoke through **real vLLM inference** on a
GPU host, validating the SAME OpenAI-compatible architecture the local runtime will use. The
JBNU cluster (master/j001) is CPU-only, so inference runs on a **separate GPU host**.

**Portability rules (do not violate):**
- Do NOT assume JBNU `/tmp` or `/home` paths exist on the GPU host. Every path below is a
  placeholder resolved ON that host.
- Transfer the branch WITHOUT push/PR/merge (git bundle, offline).
- Prefer vLLM server + PydanticAI client on the SAME GPU host → `PYDANTIC_AI_BASE_URL=
  http://127.0.0.1:8000/v1`. If they must be split, use `127.0.0.1` + an SSH tunnel; never expose
  the endpoint publicly (avoid `0.0.0.0` unless cluster policy forces it).
- `ANTHROPIC_API_KEY` is neither set nor required. No paid/hosted API call, ever.

Placeholders: `$GPU_HOST`, `$WORK` (writable dir on the GPU host, e.g. `/scratch/$USER/mad`),
`$ENV` (python env path/name), `$PY` (that env's python), `$MODEL_DIR` (HF cache on the host),
`$SERVED` (the `--served-model-name` string), `$PORT` (default 8000).

---

## 1. Transfer the exact branch (all 16 commits) — no push/PR/merge

The branch is preserved on JBNU shared FS at
`/home/hyunjin/mad-pydanticai-persist/proj-mad-pydanticai-full-runtime`
(branch `feat/pydanticai-full-runtime`, HEAD **b383ffa**, 16 commits ahead of GitHub `main`).

Create a single offline bundle (no network, no push) and copy it to the GPU host:
```
# on JBNU (source of truth for the branch):
cd /home/hyunjin/mad-pydanticai-persist/proj-mad-pydanticai-full-runtime
git bundle create /home/hyunjin/mad-pydanticai-persist/mad-runtime.bundle feat/pydanticai-full-runtime
git bundle verify /home/hyunjin/mad-pydanticai-persist/mad-runtime.bundle    # sanity
# transfer (choose one that fits your access): scp / rsync / manual copy
scp /home/hyunjin/mad-pydanticai-persist/mad-runtime.bundle  $GPU_HOST:$WORK/
```
On the GPU host, materialise the worktree from the bundle and verify integrity:
```
cd $WORK
git clone -b feat/pydanticai-full-runtime mad-runtime.bundle proj-mad-pydanticai-full-runtime
cd proj-mad-pydanticai-full-runtime
git rev-parse HEAD          # MUST equal b383ffa1acfc411cdb65c6aed1b25685aa36f17d
git log --oneline -1        # "feat(pydantic-ai): local OpenAI-compatible (vLLM/Ollama) backend ..."
```
A matching HEAD SHA proves all 16 commits + trees arrived byte-identical. No remote is contacted.

## 2. Python environment on the GPU host

vLLM dictates the Python/torch/CUDA versions, so build a DEDICATED env on the GPU host's own
filesystem (conda or venv). Recommended: conda, Python 3.11 or 3.12 (vLLM supports 3.9–3.12).
```
conda create -y -p $ENV python=3.12       # or: python -m venv $ENV
# activate, then upgrade pip
$PY -m pip install --upgrade pip
```
Keep this env SEPARATE from any DFT/MD envs. It holds the runtime client + vLLM only.

## 3. Exact packages

Two concerns, installed into the same GPU-host env:
```
# (a) our runtime client + OpenAI-compatible provider (pure-python; pulls pydantic-ai + openai SDK):
$PY -m pip install -e ".[pydantic-ai,local-openai]"
#     -> pydantic, pydantic-ai-slim[openai] (==openai SDK), opentelemetry-api, + project deps.
# (b) vLLM (the GPU inference server). Its version MUST match the host CUDA/driver — pick AFTER
#     step 4. Example (do NOT assume the version; confirm against the host's CUDA):
$PY -m pip install "vllm==<version-matching-host-CUDA>"
```
Notes:
- `.[local-openai]` installs `pydantic-ai-slim[openai]` (the `openai` client). NO provider key is
  needed — vLLM is keyless locally; pydantic_ai injects the non-secret placeholder `api-key-not-set`.
- vLLM brings its own pinned `torch`/CUDA wheels; do not pre-install a conflicting torch.
- If the host has no outbound internet, you'll need a wheelhouse/mirror for vllm+torch and a
  pre-staged model (tell me and I'll adjust to an offline install).

## 4. Verify GPU / CUDA / driver compatibility (before choosing vLLM version + model)
```
nvidia-smi                                   # GPU model(s), VRAM, DRIVER version, CUDA runtime
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv
$PY -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
$PY -c "import vllm; print('vllm', vllm.__version__)"
```
Decision rule: the vLLM/torch CUDA build must be ≤ the driver's supported CUDA. Note total/free
VRAM and compute capability (bf16 needs Ampere+ / cc≥8.0; else use fp16).

## 5. Smallest suitable Stage A smoke model (chosen AFTER seeing VRAM)

Stage A validates integration (tool calling + typed JudgeVote + schema + provenance), NOT
reasoning quality — so the smallest reliable tool-calling model wins. Candidates (open license,
vLLM-supported, known tool parser):
- **Qwen2.5-3B-Instruct** (Apache-2.0; vLLM tool parser `hermes`; ~6–7 GB bf16) — default choice.
- Qwen2.5-1.5B-Instruct (Apache-2.0; ~3–4 GB) — if VRAM is very tight; weaker tool calling.
- Llama-3.2-3B-Instruct (Llama-3.2 license, HF-gated → needs `HF_TOKEN`; parser `llama3_json`).
Record exactly once chosen: repo id, revision/commit, params, precision, expected VRAM, license,
context length, tool parser, guided-decoding backend.

## 6. vLLM launch (template — confirm flags against the INSTALLED vLLM on the host)

Do NOT guess flags; check `vllm serve --help` on the host (flag names shift between versions).
Template for Qwen2.5-3B-Instruct, server bound to loopback only:
```
$PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --served-model-name $SERVED \
  --host 127.0.0.1 --port $PORT \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice --tool-call-parser hermes
  # add guided-decoding backend if the installed version needs it, e.g. --guided-decoding-backend xgrammar
```
(`vllm serve Qwen/Qwen2.5-3B-Instruct ...` is the newer equivalent CLI — use whichever the
installed version documents.) Then, BEFORE any Stage A call, health-check the OpenAI endpoint
only (no inference beyond listing):
```
curl -s http://127.0.0.1:$PORT/v1/models        # must list $SERVED
```

## 7. Exact Stage A command (single frozen task, shadow, retries=0)
```
cd $WORK/proj-mad-pydanticai-full-runtime
env -u ANTHROPIC_API_KEY \
    PYDANTIC_AI_PROVIDER=local-openai \
    PYDANTIC_AI_MODEL=$SERVED \
    PYDANTIC_AI_BASE_URL=http://127.0.0.1:$PORT/v1 \
    PYDANTIC_AI_SMOKE_CONFIRM=yes \
  $PY -m runtimes.pydantic_ai.cli run-task \
    --runtime pydantic-ai --agent judge \
    --agent-specs-dir "$PWD/agent_specs" \
    --task "$PWD/examples/stage_a_judge_smoke/task.json" \
    --exchange-dir "$PWD/examples/stage_a_judge_smoke/out/exchange" \
    --read-allow "$PWD/examples/stage_a_judge_smoke/artifacts" \
    --repo-root "$PWD" --mode shadow --correlation-id stageA-judge-0001 --probe-server \
    2>&1 | tee "$PWD/examples/stage_a_judge_smoke/out/stdout.log"
```
The CLI path: production CLI → production router `judge_gate` → PydanticAI Agent (local model) →
`read_json` tool on `evidence.json` → typed `JudgeVote` → canonical `validate_agent_response` →
provenance. CLI defaults on this path give **max_total_calls=1, provider_retries=0** (one task,
no auto-retry, no model switching). Do not re-run or switch models on failure — preserve the one
result and analyse.

## 8. Output / provenance paths (on the GPU host)
- Provenance JSON: `$PWD/examples/stage_a_judge_smoke/out/exchange/provenance/stageA-judge-0001.*.json`
- Stdout log:      `$PWD/examples/stage_a_judge_smoke/out/stdout.log`
- `.../out/exchange/results/` must NOT appear (shadow never accepts → controller mutation = 0).
To verify Stage A here, copy those two files back to JBNU `/home` (shared) or paste their content;
I'll check: local inference occurred; Anthropic/paid calls = 0; tool call read evidence.json;
JudgeVote typed-parse PASS; canonical validation PASS; role/lens correct; no nonexistent-artifact
citation; unauthorized tools = 0; controller mutation = 0; raw+parsed preserved; provenance
complete (+ provider/base-url-no-secret/served-model/revision/vLLM/GPU/dtype/tool-parser if
recordable); latency + token usage (if vLLM exposes usage).

---

## Information I need from you about the GPU host (before any launch/download)
1. **Access model:** will YOU run these commands on the GPU host (I provide the runbook), or is
   the host reachable from this Claude session? If you run them, how do results come back to me
   (copy the 2 files to JBNU `/home`, or paste)?
2. **`nvidia-smi` output:** GPU model(s), total/free VRAM, driver version, CUDA runtime.
3. **Python:** conda or venv available? which Python versions?
4. **Internet:** can the host reach PyPI (for vllm/torch) and Hugging Face (to pull the smoke
   model)? If not, I'll switch to an offline wheelhouse + pre-staged model plan.
5. **Filesystem:** a writable dir (`$WORK`) + free disk (need room for vLLM/torch wheels ≈ several
   GB and the smoke model ≈ 6–7 GB); and the HF cache location.
6. **Scheduler:** is the GPU host behind a scheduler (must I `qsub`/`srun` onto it) or a direct box?
7. **Ports / policy:** is loopback `127.0.0.1:8000` usable for same-host server+client? Any port
   restriction? Any `HF_TOKEN` needed (only if you want a gated model like Llama)?

Nothing is launched or downloaded until you answer 1–7 and approve.
