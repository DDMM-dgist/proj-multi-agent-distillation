# L4A findings (jbnu-gpu2) + exact L4B plan

Inspection date 2026-08-07, host `jbnu-gpu2`, reached only from the user's LOCAL PC
(`ssh gpu2-via-cpu1`). This Claude session (on cpu1/master) has **no route** to gpu2, and gpu2
does not mount JBNU's `/home` — so all gpu2 steps are run by the user; results copied back.

## L4A — confirmed facts

| Item | Value |
|---|---|
| Host | `jbnu-gpu2` |
| GPU | **2× NVIDIA RTX 6000 Ada Generation**, 49140 MiB (~48 GB) each; Ada → compute cap 8.9 → **bf16 OK** |
| Driver / CUDA | **580.95.05** / driver supports **CUDA 13.0** (backward-compatible with cu124/cu128 vLLM wheels) |
| **GPU load NOW** | GPU0: 39394/49140 MiB used (VASP 29818 + python3 9562), 100% util → **~9.7 GB free**. GPU1: 29857/49140 used (VASP), 99% util → **~19.3 GB free**. Both actively running VASP DFT. |
| Python (base) | 3.13.5, `/opt/anaconda3` (conda at `/opt/anaconda3/bin/conda`) |
| torch (base) | **2.12.1 (CUDA 13 build)** in `~/.local` — **required-by nequip, mace-torch, chgnet, e3nn, ssneb_fm, torchvision…** (the MLIP/teacher stack) |
| transformers | 5.12.1 |
| vLLM | **NOT installed** |
| openai / pydantic-ai | not reported (assume absent; installed via our client env) |
| Disk | gpu2 `/home` = local `/dev/nvme0n1p1` 3.5 T, **1.3 T free** (NOT the JBNU 66 T `/home` → not shared) |
| Port 8000 | **free** on gpu2 loopback |
| Topology | direct interactive GPU box (no scheduler seen); server+client co-locate → `127.0.0.1:8000` |

## ⚠️ Safety-critical constraint (drives the whole install plan)

gpu2's base env has **torch 2.12.1** that **nequip / mace-torch / chgnet / e3nn** depend on — the
user's scientific MLIP/teacher toolchain. `pip install vllm` pulls vLLM's **own pinned torch**,
which would **downgrade/replace torch in that env and break the MLIP stack**. Therefore vLLM and
our client MUST go into **fresh, isolated conda envs**, never the base. This also keeps the
scientific baseline untouched (a standing project rule).

Second constraint: both GPUs are busy with VASP. The smoke model must fit in the **free** VRAM of
one GPU (target GPU1, ~19 GB free) and keep a small footprint so it never OOMs a VASP job.
Re-check `nvidia-smi` immediately before launch and pin the GPU with the most free memory.

## Recommended smoke model (verify tool parser against the installed vLLM at L4B)

**`Qwen/Qwen2.5-3B-Instruct`** — Apache-2.0, **ungated** (no HF token, no key), ~6–7 GB bf16 → fits
GPU1's free VRAM with margin. Documented vLLM tool-calling via `--tool-call-parser hermes`;
guided/structured JSON output via vLLM's guided-decoding — matching our `output_type` (JudgeVote)
tool-based structured output. **Tighter-VRAM fallback:** `Qwen/Qwen2.5-1.5B-Instruct` (~3–4 GB) if
GPU1 free drops (e.g., forced onto GPU0's ~9.7 GB). Final pick + parser to be confirmed against
the actual installed vLLM `vllm serve --help` on gpu2 (do not assume the flag name survives).

## Recommended vLLM version/config

Install the **latest stable `vllm`** into the isolated py3.12 env and **record whatever it
resolves to** (vLLM pins its own torch/CUDA; a cu124/cu128 wheel runs fine under driver 580). Do
not hard-code a version blindly — pin the resolved one after install. Python 3.12 (not the base's
3.13) for broad vLLM wheel support.
Launch profile (small, shared-GPU-safe; finalize flags from `vllm serve --help`):
single GPU (`CUDA_VISIBLE_DEVICES=<gpu with most free>`), `--dtype bfloat16`,
`--gpu-memory-utilization` chosen so `util*48GB ≤ (free − 2GB safety)` (e.g. ~0.35 on GPU1),
`--max-model-len 8192`, `--max-num-seqs 1`, `--enforce-eager` (skip CUDA-graph capture → lower
VRAM), `--host 127.0.0.1 --port 8000`, `--enable-auto-tool-choice --tool-call-parser hermes`,
`--served-model-name <SERVED>`.

## Exactly what needs to be installed on gpu2 (L4B — NOT executed yet)

1. **Isolated vLLM server env** (do NOT use base):
   `conda create -y -n vllm-mad python=3.12` → `pip install vllm` → record resolved vllm+torch;
   `vllm serve --help | grep -i tool-call-parser` to confirm `hermes` is offered.
2. **Isolated client env** (do NOT use base):
   `conda create -y -n mad-client python=3.12`; after the branch is transferred,
   `pip install -e ".[pydantic-ai,local-openai]"` (PydanticAI + openai client + our runtime).
3. **Smoke model**: `Qwen/Qwen2.5-3B-Instruct` (~6–7 GB) — downloaded at L4B (HF, ungated) or
   auto-pulled by vLLM at first launch. **One model only.**

## Prerequisites I still need confirmed for gpu2 (before any install)
- **Outbound internet from gpu2** to `pypi.org` (for vllm) and `huggingface.co` (for the model)?
  If blocked, we switch to an offline wheelhouse + pre-staged model. (Run the connectivity section
  of `work/gpu_host_inspect.sh`, or `curl -sI --max-time 8 https://pypi.org https://huggingface.co`.)
- Confirm it's acceptable to create the two isolated conda envs on gpu2 (base env untouched).

## Branch transfer to gpu2 (prepared, NOT executed — cpu1↔gpu2 not connected)
gpu2 `/home` is not shared with cpu1, so use a git bundle relayed via the LOCAL PC:
`scp cpu1:/home/hyunjin/mad-pydanticai-persist/mad-runtime.bundle .` then
`scp mad-runtime.bundle gpu2:<WORK>/`, then on gpu2
`git clone -b feat/pydanticai-full-runtime mad-runtime.bundle proj-mad-pydanticai-full-runtime`
and verify `git rev-parse HEAD` == the current branch tip. (Bundle is created in L4B, not now.)

## Stage A (L4C) command + provenance — unchanged from the runbook, on gpu2
`env -u ANTHROPIC_API_KEY PYDANTIC_AI_PROVIDER=local-openai PYDANTIC_AI_MODEL=<SERVED>
PYDANTIC_AI_BASE_URL=http://127.0.0.1:8000/v1 PYDANTIC_AI_SMOKE_CONFIRM=yes <client-py> -m
runtimes.pydantic_ai.cli run-task --runtime pydantic-ai --agent judge --agent-specs-dir
$PWD/agent_specs --task $PWD/examples/stage_a_judge_smoke/task.json --exchange-dir
$PWD/examples/stage_a_judge_smoke/out/exchange --read-allow
$PWD/examples/stage_a_judge_smoke/artifacts --repo-root $PWD --mode shadow --correlation-id
stageA-judge-0001 --probe-server` → provenance at
`examples/stage_a_judge_smoke/out/exchange/provenance/stageA-judge-0001.*.json` + `out/stdout.log`.

## L4B RESULTS — confirmed on jbnu-gpu2 (2026-08-07)
Setup ran; base env UNTOUCHED (both envs under ~/.conda/envs; base = /opt/anaconda3).
Repo on gpu2: `/home/hyunjin/proj-mad-pydanticai-full-runtime`, HEAD **dff0af8** (verified).

Pinned resolved versions (RECORD for reproducibility of THIS experiment):
- **vllm-mad** (`/home/hyunjin/.conda/envs/vllm-mad`, Python 3.12):
  vllm **0.26.0**; torch **2.11.0** (CUDA-13 build: nvidia-cudnn-cu13 9.19.0.56, nvidia-nccl-cu13
  2.28.9, nvidia-cublas 13.1.0.3, cuda-toolkit 13.0.2, triton 3.6.0); transformers **5.14.1**;
  tokenizers 0.22.2; numpy 2.3.5; flashinfer-python 0.6.14.
  Structured-output backends present: **xgrammar 0.2.3**, **lm-format-enforcer 0.11.3**,
  **outlines_core 0.2.14**, **llguidance 1.7.6** (+ help shows `StructuredOutputsConfig`).
  (vllm also pulled anthropic 0.120.2 / openai 2.53.0 as its own deps — unused by our runtime.)
- **mad-client** (`/home/hyunjin/.conda/envs/mad-client`, Python 3.12):
  distillation-agents 0.1.0 (editable); pydantic **2.13.4**; **pydantic-ai-slim 0.8.1**
  (pydantic-graph 0.8.1); **openai 2.53.0**; opentelemetry-api 1.43.0; ase 3.29.0; numpy 2.5.1.
  Local-provider import check expected OK (re-capture with the fixed recorder if blank).

Compatibility notes:
- Structured/guided JSON output: SUPPORTED (xgrammar/outlines/llguidance/lm-format-enforcer all
  installed; engine has StructuredOutputsConfig).
- **Tool-call parser: NOT YET CONFIRMED.** `vllm serve --help` grep matched only
  `StructuredOutputsConfig` (no `--tool-call-parser`/`hermes` string). vLLM 0.26.0 reorganized the
  CLI help; the parser set must be enumerated from the registry (see follow-up command). Our Judge
  Stage A REQUIRES tool calling (the `read_json` tool), so the exact parser name for Qwen2.5 must be
  confirmed before finalizing the model — do NOT assume `hermes`.
- torch in vllm-mad (2.11.0) is isolated from the base env's torch 2.12.1 → MLIP/teacher stack safe.

Smoke model: Qwen/Qwen2.5-3B-Instruct remains the candidate, FINAL pending the tool-parser
enumeration. Pinned experiment stack = vllm 0.26.0 / torch 2.11.0(cu13) / transformers 5.14.1.
