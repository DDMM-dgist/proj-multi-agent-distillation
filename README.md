# Multi-Agent MLIP Distillation

Teacher MLIP의 지식을 더 가벼운 student potential로 옮기고, 그 결과를
teacher·DFT·실제 MD 영역에서 검증하기 위한 **human-in-the-loop multi-agent
workflow**입니다.

이 저장소는 연구자가 Director와 대화하면서 다음 과정을 진행하도록 설계됐습니다.

```text
연구 목표 정리
→ 구조 생성·증강
→ teacher labeling과 dataset 검토
→ student committee 학습
→ 정확도·불확실도 평가
→ MD·surface·DFT 검증
→ 독립 judge gate
→ 결과 해석과 필요한 경우 재증류
```

비싼 학습, production MD, DFT 제출이나 중요한 과학적 선택에서는 연구자가
설정과 비용을 확인합니다. 승인된 범위 안의 실행·기록·검증·수정 반복은 Director와
전문 agent가 조율합니다.

## 시작하기

저장소를 받은 뒤 루트에서 Claude Code를 실행합니다.

```bash
git clone https://github.com/DDMM-dgist/proj-multi-agent-distillation.git
cd proj-multi-agent-distillation
claude
```

Claude 안에서 새 증류 run을 시작합니다.

```text
/distill-start
```

설명을 함께 입력해도 됩니다.

```text
/distill-start MACE-MH-1을 teacher로 사용해 Mo-Nb-Ta 3원계를
GRACE/FS로 증류하고 싶습니다. Surface energetics가 중요합니다.
```

자연어로 “이 저장소의 multi-agent workflow로 새 증류를 시작해 주세요”라고
요청해도 됩니다.

프로젝트 설정은 Claude main session을 Director로 시작하고 다음 전문 agent를
자동 등록합니다.

- Literature: 문헌과 검증 기준
- Data Curator: acquisition, teacher labeling, split과 provenance
- ML Trainer: student committee 학습과 평가
- Simulation: MD, DFT, surface와 deployment 검증
- Analyst: 네 가지 error channel과 물성 결과 해석
- Judge: producer와 분리된 독립 gate 평가

사용자가 agent 파일을 복사하거나 controller 명령을 직접 실행할 필요는 없습니다.
Director가 필요한 정보만 질문하고 active config와 run 기록을 준비합니다.

## 대화형 진행 방식

처음 시작할 때 Director는 보통 다음을 확인합니다.

- teacher 종류, model/checkpoint와 head
- student architecture와 학습 config
- 대상 원소와 LAMMPS atom-type 순서
- 초기 구조 위치
- augment-atoms, teacher MD 또는 두 방법의 조합
- DFT/MD reference와 주요 validation observable
- surface orientation·termination 및 acceptance threshold

정보가 모이면 Director가 작은 pilot 계획을 제시합니다. 큰 계산은 바로 실행하지
않고 예상 계산량과 설정을 먼저 공유합니다.

진행 상황 확인:

```text
/distill-status <run-name>
```

새 Claude session에서 이어서 진행:

```text
/distill-resume <run-name>
```

## Dataset acquisition

두 acquisition 방식을 config로 선택할 수 있습니다.

1. `augment-atoms`: 기존 seed 주변의 distorted structure 생성
2. `teacher-md`: foundation teacher를 이용한 ASE MD snapshot 생성

두 방법으로 만든 구조와 별도의 surface/defect pool을 provenance를 유지한 채 함께
사용할 수도 있습니다. Acquisition과 teacher labeling은 분리되어 있으며, 모든
구조는 teacher model/head, label 단위, source와 structure ID가 기록된 뒤
dataset gate를 통과해야 합니다.

## 정확도 진단

결과는 하나의 MAE로만 판단하지 않습니다.

| 채널 | 의미 |
|---|---|
| teacher vs DFT | teacher 자체의 정확도 한계 |
| student vs teacher | 증류 과정에서 추가된 오차 |
| student vs DFT | student의 절대 정확도 |
| student-MD trajectory vs DFT | 실제 배포 영역의 정확도 |

Committee disagreement는 student가 teacher를 잘 재현하지 못하는 구조를 찾는
fidelity indicator로 사용합니다. Teacher의 DFT 정확도를 직접 나타내는 calibrated
uncertainty로 해석하지 않습니다.

## Surface energetics

Validation은 EOS에 고정되지 않습니다. 예제 3원계 workflow는 surface energetics를
선택할 수 있으며 teacher, student, DFT에 동일한 slab·relaxation·reference
convention을 적용합니다.

현재 공통 계산기는 정적 surface excess energy를 지원합니다. 유한온도 surface
free energy가 필요한 경우 진동, 배치 및 chemical-potential 항을 validation
profile에 명시하고 별도로 계산해야 합니다.

## Audit와 gate

각 주요 artifact는 세 명의 독립 Judge가 같은 기준으로 검토합니다.

- 세 명 모두 PASS해야 다음 단계 진행
- 한 명이라도 FAIL이면 전체 FAIL
- 누락되거나 잘못된 vote는 REVISE
- dataset, model, validation artifact와 vote는 run manifest에 기록

Controller는 Director가 내부적으로 사용합니다. Stage 상태, attempt, log,
artifact hash와 gate 결과를 `runs/<run-name>/`에 저장하고, PASS하지 않은 앞 단계가
있으면 다음 단계를 차단합니다.

## 현재 지원 범위

### 구현된 경로

- Allegro teacher → SIMPLE-NN student reference path
- MACE/MACE-MH-1 teacher factory
- GRACE/FS committee training, export, ASE prediction, LAMMPS deployment adapter
- augment-atoms command adapter와 teacher-driven ASE MD
- teacher pseudo-labeling과 provenance manifest
- persistent run state, artifact hashing, gate blocking과 resume
- teacher/student/DFT error audit와 committee fidelity ranking
- RDF, coordination, density, MSD, NVE drift
- 정적 surface excess energy

### 첫 적용에서 확인할 부분

- 서버에 설치된 augment-atoms와 GRACE/FS의 실제 명령·버전
- 대상 물질의 완전한 GRACE/FS input과 validation threshold
- 3원계 MACE-MH-1 → GRACE/FS end-to-end pilot
- 물질별 DFT 및 surface protocol

따라서 현재 저장소는 다른 연구자가 Director와 소통하면서 새 증류 pilot을 시작하고
단계별 산출물을 검토·수정할 수 있는 수준입니다. 새로운 architecture는 공통
adapter 계약에 맞는 training, prediction, deployment 구현이 한 번 필요합니다.

## 디렉토리 구성

```text
.claude/       Director, subagent, /distill-* skills
agents/        역할별 canonical 지침
adapters/      teacher, student, acquisition, MD, DFT adapter
workflow/      run controller와 공통 stage 실행기
configs/       interface 문서와 예제 config
validation/    정확도, uncertainty, 구조·동역학·surface 검증
gates/         judge gate 규칙과 audit schema
templates/     LAMMPS, DFT, student 입력 template
tests/         adapter, controller, Claude onboarding 테스트
```

## 주의 사항

- Model checkpoint, production dataset와 run 결과는 저장소에 포함하지 않습니다.
- VASP `POTCAR`는 배포하지 않습니다.
- MACE-MH-1 head와 DFT reference의 차이를 dataset manifest와 결과 해석에
  명시해야 합니다.
- 지원된 interface와 여러 architecture에서의 empirical validation은 구분해서
  보고합니다.
