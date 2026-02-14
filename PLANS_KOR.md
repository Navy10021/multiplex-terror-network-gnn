# 온톨로지 중심 고도화 계획서 (PLANS_KOR)

본 문서는 현재 레포의 상태를 점검한 뒤,
**온톨로지 기반 테러 네트워크 생성·학습·리포팅 고도화**를 위한 실행 계획을 구체화한다.

---


## 0) 실행 상태

- [x] **Phase A 시작/반영**: validator 심화(규칙 단위 위반, severity, affected IDs, histogram, 시간/관계-역할/provenance 점검)
- [x] **Phase B 시작/반영**: ontology 제약기반 재시도 생성 + telemetry를 generator/run 파이프라인에 연동
- [x] **Phase C 시작/반영**: PyG builder ontology 브릿지 텐서/마스크 + payload 일관성 점검 추가
- [x] **Phase D 시작/반영**: 멀티태스크 학습에 ontology 정규화 loss + ablation 플래그 + metrics 기록 연동
- [x] **Phase E 시작/반영**: ontology-aware 리포팅 지표 + 노드 설명 산출물(explanations) 연동

---

## 1) 현재 상태 점검 요약

### 1.1 핵심 코드 위치
- 데이터 생성기(v3): `src/data/multiplex_generator_v3.py`
- 매니페스트 스키마/검증: `src/validation/schema.py`
- 온톨로지 검증 체계:
  - `src/ontology/load.py`
  - `src/ontology/validator.py`
  - `src/cli/validate_ontology.py`
- 전체 파이프라인 실행: `src/run_all.py`
- PyG 변환기(v3): `src/data/build_pyg_dataset_v3.py`
- 모델 학습:
  - `src/models/train_multitask_gnn_v3.py`
  - `src/models/train_hvt_gnn_v3.py`
  - `src/models/train_linkpred_layer_v3.py`
- 진단/리포팅:
  - `src/data/basic_diagnostics_v3.py`
  - `src/analysis/plot_multitask_linkpred_summary.py`

### 1.2 기본 건전성 확인
- 전체 테스트 통과.
- `run_all` / `validate_ontology` CLI 진입점 정상.
- generator/run 파이프라인에 온톨로지 검증 결과 산출물 연동됨.

---

## 2) 다음 목표 (핵심)

현재의 “생성 후 온톨로지 검증” 단계를 넘어,
**온톨로지 지식이 생성/학습/평가 전 구간에 작동**하도록 확장한다.

1. 생성 단계: 규칙 위반을 사후탐지만 하지 말고 생성 중 억제
2. 학습 단계: 온톨로지 의미론을 loss/regularization으로 반영
3. 평가 단계: 성능 지표와 규칙 정합성·근거를 함께 리포팅

---

## 3) 단계별 구현 계획

## Phase A — Validator 심화

### 작업 범위
- 관계-역할 적합성 규칙 강화
- layer interaction 기반 시간 제약 점검 추가
- provenance 일관성 검사 강화(confidence/false/copy)
- 리포트 구조 고도화(위반 rule histogram, 영향 ID 목록)

### 대상 파일
- `src/ontology/validator.py`
- `ontology/constraints.shacl.ttl`
- `tests/test_ontology_validator.py`

### 완료 조건
- validator 리포트가 모델/리포팅에서 바로 활용 가능한 구조를 가짐
- 신규 규칙이 테스트로 pass/fail 검증됨

### 테스트 커맨드
- `pytest -q tests/test_ontology_validator.py tests/test_ontology_cli.py`

---

## Phase B — Generator v3 제약기반 생성

### 작업 범위
- 제약기반 생성 모드 추가(rejection/resampling)
- 최대 재시도/백오프 정책 정의
- repair telemetry(재시도 횟수, 위반 유형)를 산출물에 기록

### 대상 파일
- `src/data/multiplex_generator_v3.py`
- `src/run_all.py`
- `tests/test_generator_ontology_integration.py`

### 완료 조건
- strict 모드에서 온톨로지 위반 없이 생성 완료
- 위반/복구 통계가 산출물에 기록됨

### 테스트 커맨드
- `pytest -q tests/test_generator_ontology_integration.py tests/test_manifest_validation.py`

---

## Phase C — PyG Builder 온톨로지 브릿지 강화

### 작업 범위
- ontology 기반 node/edge 텐서 확장
- relation-role 허용 마스크 텐서 생성
- manifest ontology payload와 텐서 결과 일관성 검증

### 대상 파일
- `src/data/build_pyg_dataset_v3.py`
- `tests/test_build_pyg_ontology_bridge.py`

### 완료 조건
- Data 객체에 ontology 텐서/마스크가 안정적으로 저장
- shape/value 계약이 테스트로 고정

### 테스트 커맨드
- `pytest -q tests/test_build_pyg_ontology_bridge.py`

---

## Phase D — Model Zoo 온톨로지 Loss 주입

### 작업 범위
- 멀티태스크 모델에 온톨로지 정규화 loss 추가
  - role-relation compatibility
  - hierarchy transitivity
  - temporal consistency
- ablation 플래그/가중치 CLI 옵션 추가

### 대상 파일
- `src/models/train_multitask_gnn_v3.py`
- (필요시) `src/models/` 하위 유틸 모듈

### 완료 조건
- 기존 학습 파이프라인과 호환 유지
- metrics JSON에 ontology-loss 설정 기록

### 테스트 커맨드
- `python -m src.models.train_multitask_gnn_v3 --help`
- 소규모 데이터 1 epoch 스모크 학습

---

## Phase E — Experiment/Reporting 연동 강화

### 작업 범위
- summary CSV/시각화에 ontology 지표 추가
  - conformance rate
  - violations per 1k edges/events
  - provenance false/copy rate
- 노드별 설명 산출물(JSON) 규격화
  - model evidence + ontology evidence + conflict flags

### 대상 파일
- `src/analysis/plot_multitask_linkpred_summary.py`
- `src/run_all.py`
- `README.md`

### 완료 조건
- 성능과 규칙정합성을 동시에 비교 가능한 리포트 제공
- README에 결과 해석 예시 포함

### 테스트 커맨드
- summary script 스모크 실행(최소 1 run)

---

## 4) 공통 품질 기준

1. 하위 호환성 유지: 기존 CLI 기본 동작 유지
2. 재현성 확보: ontology/shapes 경로+해시+버전 기록
3. 테스트 우선: 각 단계별 테스트 추가 후 전체 suite green 유지

---

## 5) 실행 우선순위

1. Phase A (validator 심화)
2. Phase B (제약기반 생성)
3. Phase C (builder 브릿지)
4. Phase D (학습 loss 주입)
5. Phase E (리포팅/설명)

위 순서로 진행하면 리스크를 낮추고 단계별 성과를 정량 확인하기 쉽다.


## 6) 다음 발전 제안 (Phase E 이후)

- [x] **F1 완료/반영**: ontology_mode 프리셋 + strict 실패 시 상위 위반 체크 요약 + 실행 가이드

### F1 — 온톨로지 정합성 모드 UX/기본값 고도화

**배경**
- 최근 스모크 점검에서 baseline 설정은 `run_all` strict 모드에서 위반으로 중단될 수 있어, 초회 사용자 경험이 불안정할 수 있음.

**작업**
- strict 실패 메시지/가이드 강화(권장 플래그/재시도 경로 안내).
- 실행 모드 프리셋 정리: 연구용 strict vs 탐색용 non-strict.
- 터미널에 상위 위반 체크/카운트 요약 출력.

**완료 기준(DoD)**
- 신규 사용자가 혼란 없이 end-to-end 1회 실행 가능.
- strict 실패 시 바로 실행 가능한 대응 커맨드가 안내됨.

- [x] **F2 완료/반영**: explanation JSON에 rule_chains + confidence_alignment 반영

### F2 — 설명 품질 고도화 (모델 attribution + 규칙 체인)

**작업**
- 현재 degree proxy 중심 설명을 모델 attribution 기반 근거로 확장.
- 노드별로 “만족/위반 규칙 체인”을 구조화.
- 모델 확률과 규칙 점수의 `confidence_alignment` 지표 추가.

**완료 기준(DoD)**
- explanation JSON에 모델 근거 + 온톨로지 규칙 체인이 함께 기록.
- README에 실제 설명 예시 1건 이상 제공.

### F3 — 리포팅/벤치마크 표준화

**작업**
- base vs +ontology_loss, strict vs non-strict 생성 비교 템플릿 추가.
- seed/난이도별 ontology 지표(conformance, violations_per_1k_*) 집계 자동화.
- Phase E 플래그 포함 `run_all` CI 스모크 잡 추가.

**완료 기준(DoD)**
- 단일 명령으로 재현 가능한 비교표 생성.
- ontology 리포팅 산출물 회귀를 CI에서 조기 탐지.
