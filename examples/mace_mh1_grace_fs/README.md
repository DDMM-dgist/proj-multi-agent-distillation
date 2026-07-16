# MACE-MH-1 → GRACE/FS ternary pilot

이 예제는 실제 서버 integration을 위한 pilot 절차입니다. 모델, 구조, GRACE/FS
version-specific input과 surface threshold는 저장소에 임의로 고정하지 않습니다.

저장소 루트에서 Claude Code를 실행한 뒤 다음처럼 시작합니다.

```text
/distill-start MACE-MH-1을 teacher로 사용해 <A-B-C> 3원계를 GRACE/FS로
증류하고 싶습니다. <surface orientation/termination>의 static surface
energetics를 teacher, student, DFT로 검증해 주세요. 먼저 작은 pilot만 계획하고
학습·MD·DFT 제출 전에는 비용과 설정을 확인받아 주세요.
```

Run-specific config를 만들 때 다음 항목을 확인합니다.

- MACE-MH-1 model path와 head
- 정확한 ternary element/LAMMPS type 순서
- 설치된 `gracemaker`가 생성한 완전한 input
- augment-atoms, teacher MD 또는 혼합 acquisition
- seed structure와 `parent_structure_id`
- surface orientation, termination, slab/reference convention
- teacher/student/DFT acceptance threshold

## Pilot 통과 기준

1. MACE teacher가 소수 구조에서 energy/force를 생성하고 model/head가 manifest에 남음
2. acquisition 결과의 lineage가 완전하고 parent-group split이 생성됨
3. GRACE/FS 한 seed가 학습·export·ASE reload되고 held-out prediction을 생성함
4. 통과 후에만 나머지 committee seed를 학습함
5. 선택 checkpoint로 작은 LAMMPS MD가 실행되고 `md.manifest.json` binding을 통과함
6. teacher/student/DFT surface manifest가 동일 protocol과 reference convention을 사용함
7. Judge gate 결과와 연구자의 PASS/REVISE 결정이 run manifest에 남음
