# STEP 11 Implementation Plan (세분화)

## 목표
- 데이터/온톨로지/그래프 모델 고도화 항목을 작은 티켓으로 쪼개 점진 적용
- 각 티켓은 파일 범위와 테스트 기준을 명시

---

## STEP11-1 (이번 PR)
### 제목
Generator config validation 계층 추가

### 변경 파일
- `src/data/multiplex_generator_v3.py`
- `tests/test_generator_config_validation.py` (new)

### 작업
- `validate_generator_config(cfg)` 추가
  - 확률/비율 파라미터 [0,1] 강제
  - 이벤트 min/max 관계 검증
  - size, num_days, campaign_count 등 양수 검증
  - `cross_layer_copy` 구조 및 rate 범위 검증
- `load_generator_config` 및 기본 config 경로에서 validator 호출

### 완료 조건
- invalid config가 명확한 에러 메시지로 실패
- valid config는 통과

### 테스트
- `pytest -q tests/test_generator_config_validation.py`

---

## STEP11-2
### 제목
Ontology rule registry externalization + report schema versioning

### 변경 파일
- `src/ontology/validator.py`
- `src/ontology/report_schema.py`
- `docs/ONTOLOGY_CONTRACT.md`
- `tests/test_ontology_report_schema.py`

---

## STEP11-3 (이번 PR)
### 제목
Link prediction protocol v2 (temporal split + advanced negatives)

### 변경 파일
- `src/models/train_linkpred_layer_v3.py`
- `tests/test_linkpred_protocol_v2.py` (new)
- `README.md`

### 작업
- link prediction split protocol에 `split_mode`(random/temporal) 추가
- temporal split 시 edge timestamp 기반 순차 분할 적용(없으면 random fallback)
- negative sampling 고도화(`degree`, `hybrid`) 추가
- protocol v2 유닛 테스트 추가

### 완료 조건
- temporal split 경로가 테스트로 검증
- advanced negative sampler가 기본 제약(self-loop/true-edge 회피)을 만족

### 테스트
- `pytest -q tests/test_linkpred_protocol.py tests/test_linkpred_protocol_v2.py`

---

## STEP11-4
### 제목
Model calibration 요약을 MODEL_CARD에 자동 반영

### 변경 파일
- `src/models/train_multitask_gnn_v3.py`
- `src/reporting/cards.py`
- `tests/test_reporting_cards.py`
