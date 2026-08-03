# Agent-framework 전환 계획 — PydanticAI runtime (additive)

**2026-08-03. 대상 `dfb67e7`.** 근거: `work/agent-framework-audit.md`.
결론: **controller.py를 workflow engine으로 유지하고, provider-neutral agent 실행만
PydanticAI로 표준화하는 additive runtime을 추가한다. LangGraph로 상태 이전은 하지 않는다.**

## 1. 유지할 구성 (변경 없음)
- `workflow/controller.py` — durable workflow/gate/retry/recovery/resume 상태의 단일 authority
- `validation/*`, `adapters/*` — 과학 로직·계산 어댑터
- `orchestration/{exchange,specs,cli}.py`, `orchestration/schema/*.json` — 계약·검증(canonical)
- `agent_specs/*.yaml`, `agents/*.md` — 역할 등록·정본 프롬프트
- `gates/gate_vote.workflow.js`, `runtimes/{claude,codex,manual}/` — 기존 runtime(삭제 안 함)

## 2. 변경할 구성 (최소)
- `pyproject.toml` — optional-dependency group `[pydantic-ai]` + `runtimes.pydantic_ai` 패키지
  등록. **핵심 의존성은 불변**(ase/numpy/scipy/pyyaml).

## 3. 새로 추가한 구성 (이번 PoC로 이미 구현)
- `runtimes/pydantic_ai/`
  - `models.py` — AgentResult/JudgeVote/EvidenceReference + provenance 모델
    (RuntimeContext, RuntimeInvocationRecord, ToolInvocationRecord, ValidationErrorRecord).
    **방안 B**: JSON Schema가 canonical, 이 모델은 typed parsing 계층.
  - `interface.py` — `AgentRuntime` Protocol + `AgentInvocation`.
  - `tool_registry.py` — `ReadOnlyToolset`(allow-list + secret 차단 + 호출 감사).
  - `mock_runtime.py` — 의존성 0 test double runtime.
  - `pydantic_ai_runtime.py` — 실제 provider runtime(pydantic_ai lazy import).
  - `driver.py` — 수용 파이프라인(runtime→validate_agent_response→accept, shadow 모드).
  - `README.md`.
- `tests/test_pydantic_ai_runtime.py` — mock 기반 10 테스트(pydantic 없으면 skip).
- `work/agent-framework-audit.md`, 이 문서.

## 4. 단계별 구현 계획
- **P0 repository audit** — 완료(`work/agent-framework-audit.md`).
- **P1 Pydantic contract + runtime boundary** — 완료. 계약 모델(방안 B), `AgentRuntime`
  Protocol, RuntimeContext/provenance.
- **P2 mock-validated PydanticAI-compatible PoC** — 완료. mock runtime + 제한 tool +
  수용 파이프라인 + shadow + 보안/파이프라인 테스트. read-only `judge` 역할로 실증.
- **P3 package-level local integration test** — 완료. 실제 `pydantic_ai.Agent` + `TestModel`
  로 네트워크 없이 accept/거부/tool-차단을 검증. optional CI job이 실제 실행.
- **P4 actual provider read-only smoke test** — 미완. provider 결정 후 실제 provider로
  read-only 1회 호출(키는 환경변수, 커밋 금지). timeout/429/failure 처리 확인.
- **P5 Claude vs PydanticAI golden shadow comparison** — 미완. shadow 모드로 동일 task에
  대해 두 runtime의 PASS/FAIL 일치·false-PASS·evidence 완전성 비교(§6 지표).
- **P6 reviewer roles migration** — 미완. analyst/literature 확장(여전히 read-only tool).
- **P7 typed producer action proposal** — 미완. producer(data-curator/ml-trainer/simulation)는
  tool 표면 확대가 필요하므로 별도 설계(allow-list·dry-run). 무거운 계산은 계속
  controller/adapters가 실행(agent tool 미노출).
- **P8 LangGraph 재평가** — 조건부(§7). 현재는 미해당.

## 5. 위험과 rollback
- **위험**: (a) provider 비용/키 관리 — 환경변수·비용 상한으로 제한. (b) producer 역할의
  tool 확대 시 side-effect 위험 — P4에서 allow-list·dry-run으로 통제. (c) pydantic_ai API
  변동 — lazy import + 버전 핀.
- **rollback**: 전부 additive. `runtimes/pydantic_ai/` + optional dep + 테스트 파일만
  되돌리면 코어는 그대로. 기존 runtime·controller·계약 무손상.

## 6. Acceptance criteria (전환 전 충족해야)
- 계약 지표: schema 검증 성공률, 존재하지 않는 artifact 인용 0, 허용 밖 path/secret 접근
  0, duplicate accept 0, raw/parsed 보존 100%.
- 과학 지표: 동일 task에서 Claude vs PydanticAI의 PASS/FAIL 일치, false-PASS 0,
  evidence 완전성, unsupported claim 0.
- 운영 지표: token/latency/provider-failure/retry 성공률/attempt provenance/idempotency/비용.
- 문자열 exact-match는 핵심 지표로 쓰지 않음.

## 7. LangGraph 재평가 조건 (P8)
LangGraph 사용 자체가 잘못은 아니다. 다만 현재는 controller와 상태 중복 때문에 보류한다.
다음이 **실제로** 관찰되면 재평가한다:
- agent decision state가 여러 runtime에 분산되어 controller만으로 재현 불가해질 때
- 장기 pause/resume이 controller 밖에서 durable해야 할 때
- human approval 상태가 중복·유실될 때
- conditional branch가 controller에 과도하게 결합될 때
- replay/checkpoint가 실질적 bottleneck이 될 때
- controller의 scientific state와 agent conversational state를 분리해야 할 때

단, 재평가하더라도 LangGraph가 HPC scheduler·scientific execution을 직접 소유하지 않도록,
LangGraph 소유 상태와 controller 소유 상태를 명시 분리한 별도 migration plan을 먼저 작성한다.

## 8. RAG·계산 인프라 전환을 제외한 이유
- RAG: 교수님 지시로 이번 범위 제외. 현재 워크플로에 검색 병목이 실증되지 않았고,
  provenance 붙은 구조화 검색은 별도 트랙(문헌 agent)에서 필요 시.
- atomate2/AiiDA/PostgreSQL/pgvector: 계산·데이터 인프라 전환은 workflow engine 교체이며,
  audit 결과 controller가 이미 안정적으로 그 역할을 수행 → 이번 agent-framework 표준화와
  결합하면 위험·범위가 과대. 독립 트랙으로 분리.

---
**요약**: agent 실행 계층만 provider-neutral(PydanticAI)로 표준화. 코어(controller/
validation/adapters/계약)와 기존 runtime은 무손상. novelty가 아니라 credibility·채택성
작업. 되돌리기 쉬운 additive PoC로 시작해 provider 결정 후 shadow 비교로 확장.
