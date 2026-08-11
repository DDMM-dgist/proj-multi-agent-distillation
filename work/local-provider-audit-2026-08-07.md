# Local LLM backend — source-grounded audit + L1/L2/L3 (2026-08-07)

Pivot: the PydanticAI production runtime's inference backend moves from the hosted Anthropic API
to a **local, OpenAI-compatible server (vLLM first)**. Anthropic is no longer a prerequisite and
is kept only as an optional backend. Scope of this pass: **L1–L3** (abstraction, fail-closed
local preflight, model-suitability audit). No server launch, no model download, no inference.

Architecture (unchanged control flow):
`local LLM → vLLM/OpenAI-compatible server → PydanticAI → typed seven-role runtime →
production router → authorization/approval → workflow/controller.py (sole durable authority)`.

## Source-grounded audit (installed pydantic-ai-slim 0.8.1)

Read directly from the installed package (not guessed):

- `pydantic_ai/models/openai.py` — `class OpenAIChatModel(Model)` (`OpenAIModel` is an alias).
  `__init__(model_name, *, provider: Literal[... 'ollama','openai', ...] | Provider = 'openai',
  profile=None, system_prompt_role=None, settings=None)`. A `Provider` **instance** may be passed.
- `pydantic_ai/providers/openai.py` — `class OpenAIProvider(Provider[AsyncOpenAI])`,
  `__init__(base_url=None, api_key=None, openai_client=None, http_client=None)`. Lines 58–61
  (verbatim intent): *"a workaround for the OpenAI client requiring an API key, whilst locally
  served, openai compatible models do not always need an API key, but a placeholder (non-empty)
  key is required"* → when `base_url` is set and no key/env is present it uses `api_key =
  'api-key-not-set'`.
- `pydantic_ai/providers/ollama.py` — `class OllamaProvider(Provider[AsyncOpenAI])`,
  `__init__(base_url=None, api_key=None, ...)`; requires `base_url` (or `OLLAMA_BASE_URL`) and
  applies the same `'api-key-not-set'` placeholder. Always paired with `OpenAIChatModel`.
- Both providers `from openai import AsyncOpenAI` → the **`openai` SDK is required** (extra
  `pydantic-ai-slim[openai]`). Constructing the provider/client is **lazy**: no network request
  until an actual model call.

### Report items 1–7

1. **Exact local mechanism (pinned 0.8.1):**
   `OpenAIChatModel(model_id, provider=OpenAIProvider(base_url="http://HOST:PORT/v1"))` for any
   OpenAI-compatible server; `OpenAIChatModel(model_id, provider=OllamaProvider(base_url=...))`
   for Ollama. Verified constructible **offline** with a placeholder key and no network call.
2. **vLLM:** supported via the OpenAI-compatible path (vLLM serves `/v1`). No dedicated vLLM
   class exists or is needed — use `OpenAIProvider(base_url=<vllm>/v1)`. **Priority-1 backend.**
3. **Ollama:** supported (`OllamaProvider`, or `provider='ollama'`); OpenAI-compat endpoint
   `http://127.0.0.1:11434/v1`. Priority-3 fallback.
4. **Typed structured output / tool calling:** `OpenAIChatModel` drives tool calls + JSON-schema
   structured output (`OpenAIJsonSchemaTransformer`, `supports_json_object_output`,
   tool-choice). Our runtime's `output_type` (JudgeVote / role models) + `@agent.tool_plain`
   read tools **build offline** against a local model (verified). Runtime **reliability** then
   depends on the *served model* supporting tool/function calling and on vLLM being launched with
   the matching tool-call parser + guided-decoding (see L3 caveat).
5. **No real API key required for local:** confirmed — the provider injects a non-secret
   placeholder. The local preflight does **not** read or require `ANTHROPIC_API_KEY`.
6. **Placeholder vs credential:** `'api-key-not-set'` is a **local, public placeholder** the SDK
   sets to satisfy the OpenAI client's non-empty-key requirement. It is not an authentication
   credential, is never sourced from the environment, and is not treated as a secret by
   redaction (verified). A *real* remote key (e.g. `OPENAI_API_KEY`) is still redacted.
7. **Production router:** **no change required.** The router selects the acceptance strategy from
   the role/typed output and is provider-agnostic; only provider *construction* (provider.py /
   CLI) changed. The CLI caller still does not hand-pick an internal path.

## L1 — provider abstraction (implemented)

`runtimes/pydantic_ai/provider.py`:
- `PROVIDER_KINDS = ("test", "local-openai", "ollama", "anthropic")`; `LOCAL_KINDS`.
- `select_provider_kind(env)` — explicit `PYDANTIC_AI_PROVIDER` wins; legacy `anthropic:<model>`
  in `PYDANTIC_AI_MODEL` infers `anthropic`; local kinds are **never** inferred from the model
  string (an ollama id `qwen2.5:7b` must not be misread as `provider:model`).
- `build_local_model(kind, model_id, base_url)` — returns `OpenAIChatModel` bound to
  `OpenAIProvider`/`OllamaProvider`. Lazy import; RuntimeError if the `openai` extra is missing.
- Anthropic path (`preflight_credentials`, `build_provider_model`) kept unchanged (optional).
- TestModel/FunctionModel tests unchanged and still pass.

### Environment variables
| var | meaning | example |
|---|---|---|
| `PYDANTIC_AI_PROVIDER` | backend kind | `local-openai` \| `ollama` \| `anthropic` |
| `PYDANTIC_AI_MODEL` | served model id | `Qwen/Qwen2.5-7B-Instruct` (vLLM) / `qwen2.5:7b` (ollama) |
| `PYDANTIC_AI_BASE_URL` | local server base URL | `http://127.0.0.1:8000/v1` |

No `ANTHROPIC_API_KEY` on the local path. `OPENAI_API_KEY` is **not** required for a local
server (placeholder is injected); if a remote OpenAI-compatible service is ever used, its key is
env-only and redacted.

### Required dependency
`pip install -e ".[pydantic-ai,local-openai]"` → adds `pydantic-ai-slim[openai]` (the `openai`
SDK, 2.53.0 in this env). No torch/vLLM/transformers needed on the *runtime* host — those live on
the GPU server the runtime connects to.

## L2 — fail-closed local preflight (implemented)

`preflight_local(env, *, probe=False)` inspects env + optionally TCP-probes the server. It
**never** calls the model / runs inference. Statuses (operational, not scientific/runtime
failures): `LOCAL_PROVIDER_NOT_SELECTED`, `LOCAL_MODEL_NOT_CONFIGURED`,
`LOCAL_BASE_URL_NOT_CONFIGURED`, `LOCAL_SDK_NOT_INSTALLED`, `LOCAL_PROVIDER_NOT_CONSTRUCTIBLE`,
`LOCAL_PROVIDER_NOT_RUNNING`, `LOCAL_PROVIDER_READY`. Reachability is a bounded TCP connect only
(no request). Bounded-call guard preserved (timeout 120 s, provider_retries 0,
structured_output_retries 0, **max_total_calls 1** on the CLI path). Secret redaction unchanged;
shadow-mode zero-mutation is provider-agnostic (already enforced by the production router).
Demonstrated network-free: server absent → `LOCAL_PROVIDER_NOT_RUNNING`, exit 3,
`anthropic_key_required: False`, nothing written.

CLI additions: `--provider`, `--base-url`, `--probe-server`; the `--runtime pydantic-ai` path
dispatches local vs anthropic and keeps the explicit `PYDANTIC_AI_SMOKE_CONFIRM=yes` live-call
gate for local too.

## L3 — model suitability audit (shortlist; NO download/run)

This host: **no GPU visible in this shell**, no vLLM/torch/transformers, only small embedding
models cached (`~/.cache/huggingface/hub`: ModernBERT-base, siglip/onnx). ⇒ the vLLM server runs
in the user's **separate GPU environment**; the runtime host only needs `openai` + a base URL.
GPU availability on that box is **unconfirmed here**. The shortlist is grounded in model
capabilities / license / vLLM support, not in confirmed local availability; no single model is
fixed.

Required properties: instruction following · reliable JSON/structured output · tool/function
calling · adequate context · vLLM support · permissive (commercial/research) license ·
single-GPU feasible · suitable for the 7 roles.

### SMOKE_MODEL (fast/small — verify the PydanticAI structured-output + tool path)
| candidate | license | tool calling | ~bf16 VRAM | notes |
|---|---|---|---|---|
| **Qwen2.5-3B-Instruct** (recommended) | Apache-2.0 | yes | ~6–7 GB | reliable JSON/tools for size; vLLM parser `hermes` |
| Llama-3.2-3B-Instruct | Llama 3.2 | yes | ~7 GB | vLLM parser `llama3_json` |
| Phi-3.5-mini-instruct (3.8B) | MIT | partial | ~8 GB | permissive; tool calling less consistent |
| Qwen2.5-1.5B-Instruct | Apache-2.0 | limited | ~3–4 GB | minimal smoke only |

### PRODUCTION_CANDIDATE (judge/analyst reasoning quality)
| candidate | license | ~bf16 VRAM | 4-bit (AWQ/GPTQ) | fit |
|---|---|---|---|---|
| **Qwen2.5-32B-Instruct** (recommended) | Apache-2.0 | ~65 GB | ~20 GB | 1×80 GB / 2×40 GB bf16; 4-bit on 24–48 GB |
| Qwen2.5-14B-Instruct | Apache-2.0 | ~28 GB | ~10 GB | single 40–48 GB bf16; 4-bit on 16–24 GB |
| Llama-3.3-70B-Instruct | Llama 3.3 | ~140 GB | ~40 GB | 2×80 GB bf16; 4-bit on 48–80 GB — strongest reasoning |
| Mixtral-8x7B-Instruct | Apache-2.0 | ~90 GB (MoE) | ~24 GB | weaker structured output than Qwen2.5 |

Estimates exclude vLLM KV-cache/`gpu-memory-utilization` overhead (add headroom).
Recommendation: **Qwen2.5-3B-Instruct** (smoke) + **Qwen2.5-32B-Instruct**, or **-14B** for a
single mid-range GPU (production); Llama-3.3-70B if a larger GPU budget and top reasoning wanted.

### Typed-output / tool-calling compatibility caveat (real)
vLLM must be launched with OpenAI tool calling enabled and the **model-matched parser**, e.g.:
`--enable-auto-tool-choice --tool-call-parser hermes` (Qwen2.5) or `llama3_json` (Llama), plus
guided-decoding (`--guided-decoding-backend outlines`/`xgrammar`) for JSON-schema structured
output. Our runtime uses `output_type=<BaseModel>` (tool-based structured output by default), so
this vLLM config is a prerequisite for L4 reliability — flagged, not yet exercised.

## Remaining blockers (before L4)
1. No GPU/vLLM in this environment → server launched in the user's GPU env (L4, approval-gated).
2. vLLM tool-call parser + guided-decoding flags must match the chosen model.
3. Structured-output reliability is model-dependent; the smoke model verifies the path first.
4. No golden reference set yet (L6).

## Status
- `ANTHROPIC_LIVE_PROVIDER = BILLING_UNAVAILABLE / NOT_REQUIRED_FOR_LOCAL_RUNTIME`
- Overall: `NETWORK_FREE_FULL_RUNTIME_READY` · `LOCAL_PROVIDER_IMPLEMENTATION_PENDING → L1/L2
  DONE, L3 SHORTLISTED` · `ANTHROPIC_BILLING_BLOCKED_NON_FATAL`.

## Exact next command for local Stage A (L4 — only after approval + a running server)
Run in the environment where the vLLM/OpenAI-compatible server is up (no Anthropic key needed):
```
cd <worktree>
env PYDANTIC_AI_PROVIDER=local-openai \
    PYDANTIC_AI_MODEL=<served-model-id> \
    PYDANTIC_AI_BASE_URL=http://127.0.0.1:8000/v1 \
    PYDANTIC_AI_SMOKE_CONFIRM=yes \
  <venv-python> -m runtimes.pydantic_ai.cli run-task \
    --runtime pydantic-ai --agent judge \
    --agent-specs-dir "$PWD/agent_specs" \
    --task "$PWD/examples/stage_a_judge_smoke/task.json" \
    --exchange-dir "$PWD/examples/stage_a_judge_smoke/out/exchange" \
    --read-allow "$PWD/examples/stage_a_judge_smoke/artifacts" \
    --repo-root "$PWD" --mode shadow --correlation-id stageA-judge-0001 --probe-server \
    2>&1 | tee "$PWD/examples/stage_a_judge_smoke/out/stdout.log"
```
Not executed now (L4 is approval-gated; no server, no inference in this pass).
