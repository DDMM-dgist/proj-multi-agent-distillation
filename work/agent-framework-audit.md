# Agent-framework 감사 (코드 근거 기반)

**감사일 2026-08-03. 대상: `DDMM-dgist/proj-multi-agent-distillation` @ `dfb67e7`.**
모든 주장은 실제 파일·라인 근거. 추측 없음.

## 1. 표준 agent package 사용 여부 — 코드 근거

**결론: LangGraph / PydanticAI / LangChain / CrewAI / AutoGen / LlamaIndex 중 무엇도 사용하지 않는다.**

근거 (전 파일 grep, `.git` 제외):
```
grep -rniE "langgraph|pydantic_ai|pydantic-ai|langchain|crewai|autogen|llamaindex|llama_index" \
     --include=*.py --include=*.toml --include=*.txt --include=*.yml --include=*.yaml \
     --include=*.md --include=*.js .
→ (빈 결과)
```
- `pyproject.toml` dependencies: `ase, numpy, scipy, pyyaml` 뿐. **Pydantic조차 직접 의존성이 아니다.**
- `requirements.txt`, `environment.yml` 동일 (모델-독립 핵심 4개만).
- Agent 실행은 **Claude Code의 Agent tool**(암묵적)과 `gates/gate_vote.workflow.js`(Claude Code Workflow runtime)이 담당. 순수 자체 구현.

## 2. 책임별 실제 소유권 (file:line 근거)

| 책임 | 구현 파일 | 상태의 실제 소유자 | 외부 package | 문제점 |
|---|---|---|---|---|
| Agent role 정의 | `agent_specs/*.yaml` + `agents/*.md` + `.claude/agents/*.md` | repo (typed spec) | 없음 | 정의 3곳 분산(spec/canonical md/Claude adapter) — 의도적이나 동기화 부담 |
| Prompt/context 조립 | `orchestration/specs.py:AgentSpec.prompt`(48), `.claude/agents/*.md` | repo | 없음 | Claude 경로는 Agent tool이 암묵 조립 |
| **Model invocation** | Claude Code Agent tool(암묵), `gate_vote.workflow.js:agent()`(142) | **Claude Code (암묵)** | 없음 | **provider-neutral 아님 — 여기가 표준화 대상** |
| **Structured output** | `gate_vote.workflow.js` StructuredOutput(118), `orchestration/schema/*.json` | Claude Code(강제) + repo(검증) | 없음 | Claude 밖에선 수동 JSON. typed 강제는 스키마+`validate_*`가 이중 담당 |
| Task lifecycle | `orchestration/exchange.py:make_task/dispatch/accept/collect` | repo | 없음 | 건전 |
| **Workflow state** | `workflow/controller.py` state dict `schema_version:6`(211), stages/iterations/events | **controller.py (유일)** | 없음 | 건전 — 단일 authority |
| Gate/approval | `controller.py:_validate_vote_bundle`(537)+`record_gate`(622); `gate_vote.workflow.js`는 dispatch만 | **controller.py** | 없음 | JS는 무상태 dispatcher(124 `return{}` → `controller gate --votes`로 기록). **상태 중복 없음** |
| Retry/recovery | `controller.py` `attempts`(190/397/528), `rebind_inputs`(264), iterations(215) | **controller.py** | 없음 | 건전 |
| **HPC 제출/상태** | `run_stage:subprocess.run`(405) [저비용]; `complete_external_stage`(501) [고비용 sbatch 외부] | **SLURM + 외부 agent** | 없음 | agent framework가 HPC 미소유(정합). 단 external 완료 등록은 사람/agent 신뢰 |
| Artifact 저장 | run_dir 파일; `exchange/{tasks,results,raw}/` | 파일시스템 | 없음 | 건전 |
| Provenance/hash | `workflow/integrity.py`(artifact_digest/sha256_file/verify), `git_revision`(43) | **controller + integrity** | 없음 | 강함 — SHA-256 hash-binding, code_revision 고정 |
| Validation | `orchestration/exchange.py:validate_*`(schema/계약) + `validation/*`(physics) | repo | 없음 | **type validation과 physics validation 분리됨(양호)** |
| Observability | `run_dir/logs/*.log`, `events[]`, 산문 coordination log | repo(파일) | 없음 | **표준 tracing(OpenTelemetry) 없음 — 향후 gap** |

## 3. 특정 확인 질문 (코드 근거)

1. **controller.py와 gate_vote.workflow.js가 상태를 중복 소유하는가?** — **아니다.** JS(`gate_vote.workflow.js`)는 judge를 dispatch하고 vote bundle을 `return`할 뿐(124), 실제 기록은 전부 `workflow.controller gate --votes`(주석 33)로 controller가 함. `_validate_vote_bundle`(537)이 lens/criteria/hash를 재검증. **상태 authority는 controller 단일.**
2. **Claude Code Agent tool이 암묵 제공하는 것:** model invocation, prompt 조립, tool 실행(Read/Bash/…), StructuredOutput 강제, 병렬 subagent(`parallel`). 이게 provider-lock의 실체.
3. **agent가 실제 쓰는 tool:** reviewer(analyst/judge/literature)=`Read,Grep,Glob,Bash`(+literature는 WebSearch/WebFetch); producer(data-curator/ml-trainer/simulation)=`Read,Write,Edit,Bash,Glob,Grep`.
4. **planner/reviewer vs executor:** reviewer는 read-only 성격, **producer는 Write/Edit/Bash로 side-effect 가능**. 단 무거운 계산 자체는 controller가 adapters/SLURM로 실행. → **첫 PoC는 read-only 역할(judge/analyst)이 안전**(side-effect 없음).
5. **task/result canonical schema 위치:** `orchestration/schema/{agent_task,agent_result,judge_vote,agent_spec}.schema.json` (JSON Schema).
6. **raw + parsed 둘 다 보존?** — **그렇다(방금 추가됨).** `exchange.py:accept()`가 `_preserve_raw`로 검증 전 원문을 `raw/{task_id}.json`에 먼저 쓰고, 검증 성공분만 `results/`에 기록.
7. **중복 accept 방지:** raw는 append-only(재제출 시 `.1.json`/`.2.json`, 덮어쓰기 없음). `dispatch`는 중복 task를 `FileExistsError`. **과학적 결정의 중복 방지는 `controller.record_gate`(622)+recovery 게이트가 담당** (results/는 last-wins이나 controller 상태가 authority).
8. **repo 밖/secret 접근 가능성:** ⚠️ **있음.** 모든 Claude agent가 **무제한 `Bash`** 보유 → 파일시스템 어디든 읽기 가능. controller는 *출력*만 run 내부로 제한(132 "output must stay inside the run")하나, **agent tool 자체엔 path/secret allow-list 없음.** → PydanticAI가 typed read-only tool + allow-list로 **개선할 실질 가치.**
9. **attempt별 기록:** `attempts` 카운터 증가(397/528), attempt별 로그 `logs/{name}.attempt-{n}.log`(399). recovery는 iteration으로 lineage 기록.

## 4. Claude Code 종속 요소 (전환 시 표준화 대상)

- `gate_vote.workflow.js` — Claude Code Workflow runtime 전용(`agent()`, `parallel()`, `StructuredOutput`, `agentType`).
- `.claude/agents/*.md`, `.claude/skills/*` — Claude Code frontend 전용.
- model invocation 전체가 Claude Code Agent tool에 암묵 결합.
→ 단, **`orchestration/`(specs/exchange/cli) + `agent_specs/` + `agents/*.md`(정본 프롬프트) + `runtimes/{claude,codex,manual}/`는 이미 provider-neutral** — 표준 runtime 추가의 발판.

## 5. 중복 상태 / 기술 부채

- **중복 상태: 사실상 없음.** controller가 단일 authority, JS는 무상태 dispatcher.
- 부채: (a) role 정의가 spec.yaml / canonical md / .claude md 3곳 — 동기화는 test(`test_claude_onboarding`)가 일부 강제하나 표준 registration 아님. (b) model invocation이 provider-locked. (c) 표준 observability(trace) 부재.

## 6. Security / reproducibility 위험

- 🔴 **agent Bash 무제한** → path traversal / secret 접근 가능(§3-8). Claude Code 신뢰에 의존.
- 🟠 external HPC 완료 등록(`complete_external_stage`)은 사람/agent 신뢰 — hash 검증은 있으나 "정말 그 job이 그 결과를 냈는가"는 밖에서 보장.
- ✅ 강점: SHA-256 hash-binding, code_revision 고정, raw 원문 보존, type/physics validation 분리 — reproducibility 기반은 견고.

## 7. PydanticAI vs LangGraph (이 repo 기준)

| 판단축 | 이 repo의 사실 | 함의 |
|---|---|---|
| durable state(task/gate/retry/recovery/resume) 소유 | **controller.py가 이미 안정적으로 소유**(§2) | LangGraph의 graph-checkpoint는 **controller와 중복** |
| agent state가 분산·불안정한가 | 아니다 — JS는 무상태, controller 단일 authority | LangGraph 도입 근거(상태 분산) **해당 없음** |
| 새 framework에 실제 필요한 것 | model invocation, typed tool, structured output, provider abstraction | **정확히 PydanticAI의 범위** |
| pause/resume/branch/approval | controller 게이트 + `approval_boundaries`가 이미 강제 | agent framework에 durable interrupt 불필요 |

→ **PydanticAI 우선 조건(스펙 2단계)을 코드가 충족.** LangGraph 검토 조건(상태 분산·불안정)은 **미해당.**

**LangGraph에 대한 정확한 표현**: LangGraph 사용 자체가 원칙 위반이거나 부적절한 것은
**아니다**. 다만 이 repository에서는 `controller.py`가 durable scientific workflow state
(run/gate/retry/recovery/resume)를 **이미 단일 소유**하므로, 동일 상태를 LangGraph가 다시
소유하면 **두 개의 source of truth**가 생긴다. 따라서 현 단계에서는 PydanticAI 기반 runtime
표준화가 최소 변경 경로이며, 향후 agent decision state가 여러 runtime에 분산되거나
pause/resume·conditional branch·human approval 관리가 controller 밖에서 실제 병목이 될 때
LangGraph를 재평가한다. "절대 도입 안 함"이 아니라 "현재는 도입 근거가 부족함".

## 8. 최종 권고

**PydanticAI runtime adapter를 additive로 추가한다. LangGraph로 decision state를 이전하지 않는다.**

근거: (1) controller.py가 durable state를 단일 소유(§2,§7) — LangGraph는 중복. (2) 필요한 건 provider-neutral model invocation + typed tool + structured output뿐 — PydanticAI 범위. (3) runtime-neutral 발판(`orchestration/`, `runtimes/`)이 이미 존재. (4) PDF #1 원칙("LLM framework가 workflow engine을 대체 말라")과 정합 — controller가 engine으로 남는다. (5) 부수 이득: typed read-only tool + allow-list로 §6의 Bash-무제한 위험 개선.

**단, 패키지 추가 자체를 "표준 전환 완료"나 논문 novelty로 주장하지 않는다.** novelty는 validation-gated workflow에 남는다(두 PDF 9.2 정합).

## 9. 코드 근거 파일·함수 목록

- `pyproject.toml` / `requirements.txt` / `environment.yml` — 의존성(표준 agent pkg 없음)
- `workflow/controller.py` — `initialize`/state(211), `run_stage`(379), `complete_external_stage`(501), `_validate_vote_bundle`(537), `record_gate`(622), `rebind_inputs`(264), `git_revision`(43)
- `workflow/integrity.py` — `artifact_digest`/`sha256_file`/`verify_artifact`
- `orchestration/exchange.py` — `make_task`/`validate_task`/`validate_result`/`validate_judge_vote`/`validate_agent_response`/`FileExchangeRuntime.{dispatch,collect,accept,_preserve_raw}`
- `orchestration/specs.py` — `AgentSpec`, `load_agent_specs`
- `orchestration/schema/*.json` — 계약 canonical source
- `orchestration/cli.py` — `make-task`/`validate-result`/`accept-result`
- `gates/gate_vote.workflow.js` — Claude Code 게이트 dispatch(무상태)
- `agent_specs/*.yaml`, `agents/*.md`, `.claude/agents/*.md`, `runtimes/{claude,codex,manual}/`
- `validation/*` — physics/domain validation (type validation과 분리)

## 10. 후속 정리 (2026-08-03, review 반영)

- **정확한 PoC 명칭**: 현재 구현은 **PydanticAI-compatible runtime PoC**다. 기존 scientific
  workflow를 유지하면서 PydanticAI를 수용하는 runtime boundary·typed contract·read-only
  tool policy·provenance·validation pipeline·shadow mode를 **mock 및 실제 pydantic_ai
  package-level 실행(TestModel)**으로 검증했다. 실제 외부 LLM provider 호출과 Claude
  runtime 대비 과학적 동등성 평가는 **아직 수행하지 않았다**.
- **실제 package 통합 검증**: `pydantic-ai-slim 0.8.1`을 설치하고 실제 `pydantic_ai.Agent`를
  `TestModel`로 구동(네트워크·API key 없음). 유효 출력이 기존 `validate_agent_response`를
  다시 통과해 accept됨, 잘못된 lens 출력은 거부됨, Agent가 호출한 tool이 allow-list에
  차단되고 refusal이 기록됨을 테스트로 확인. 미검증: 실제 provider tool calling,
  structured-output retry, timeout/429/provider failure, 실제 token usage, 모델 evidence
  준수율, Claude runtime과의 과학적 동등성.
- **CI**: core CI(`pip install -e .`)는 optional 미설치 → runtime test skip(코어 호환 확인).
  별도 job `pydantic-ai-runtime`(`pip install -e .[pydantic-ai]`)이 runtime test를 실제
  실행하며, **전부 skip되면 실패**하도록 강제.
- **보안 강화**: allow-list containment는 `os.path.realpath`(symlink 추종·미존재 경로 parent
  해석)로, secret 차단은 **경로 component 단위**(.env*/.ssh/.aws/.gnupg/id_rsa/*.pem/*.key/
  token*/secrets*), 텍스트 확장자 화이트리스트+binary 차단, 파일당·invocation당 byte budget,
  거부 기록, prompt-injection 경계(파일 내용=untrusted data) 명시. 관련 테스트 8종 통과
  (symlink escape 포함, 미지원 OS에선 명시 skip).
- **usage/retry 정확성**: provenance에 `usage_source`(mock/test-model/provider/estimated/
  unavailable) 추가 — mock 토큰을 실제 비용으로 오인하지 않게. `parent_attempt_id`,
  `retry_category`(none/agent_invocation/model_output/provider/controller_task/
  scientific_recovery) 추가. PoC가 실제 구현한 retry 계층은 재실행 시 별도 attempt 기록뿐;
  나머지 계층은 README에 "미구현"으로 명시.
- **mock/real 공통 경로**: 두 runtime이 `interface.build_invocation` 단일 provenance/parse
  경로를 공유(mock 우회 없음). 테스트로 강제.
- **기존 RDF/ASE 실패 base 재현**: base commit `dfb67e7`(PoC 이전)의 clean 트리에서 동일
  환경·명령으로 `test_rdf_uses_supported_ase_api`가 **동일 assertion으로 재현**됨 →
  사전존재 환경 실패(agox ASE 버전), 이번 변경의 회귀 아님. (CI의 고정 ASE에선 통과.)
