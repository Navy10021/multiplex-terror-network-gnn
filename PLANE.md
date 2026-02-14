# Ontology-based Terror Network: Forward Plan (PLANE)

> 목적: 이미 구현된 항목(온톨로지 파일/기본 validator/CLI 통합) 반복이 아니라,
> **연구 기여를 키우는 다음 발전 요소**를 테러 네트워크 도메인 관점으로 정리한다.

## 0) Focus Reset (What to optimize next)

1. **도메인 정합성 강화**: 생성 그래프가 “테러 네트워크로서 말이 되는가”를 더 엄밀히 보장
2. **모델 주입 강화**: 온톨로지 의미론을 feature뿐 아니라 loss/학습전략에 반영
3. **설명/감사 가능성 강화**: 예측 결과에 규칙 기반 근거를 함께 남기고 실험 레벨에서 집계

---

## 1) Ontology Schema Deepening (Domain-specific)

### 1.1 Role/Relation 적합성 고도화
- `hasRole(Actor, Role)` 기반으로 다중 역할 + 시간가변 역할 지원
- 역할 전이(Event-driven role transitions) 규칙 추가
  - 예: `courier -> operative` 가능, `support -> leader`는 낮은 prior/추가 증거 필요
- Relation별 domain/range 제약 세분화
  - `commands`: source=leader|financier, target=operative|support
  - `finance`: source=financier 비율 최소 제약

### 1.2 Evidence/Provenance 온톨로지 확장
- `Evidence` 타입 확장: HUMINT/SIGINT/FININT/OSINT synthetic source taxonomy
- `EdgeProvenance`에 `confidence`, `source_layer`, `is_false`, `is_copied`, `generation_rule_id` 추가 정규화
- “모형 가정”과 “관측 사실” 분리 필드 정의

### 1.3 Temporal/Interaction 규칙 강화
- layer interaction rule을 단순 correlation에서 조건부 규칙으로 확장
  - 예: `finance burst` + `comm burst` -> `operation` window 내 발생 확률 상승
- 이벤트 시계열 제약 (lead-lag, min/max lag) SHACL/validator 규칙으로 내재화

---

## 2) Generator v3 Upgrades (Ontology-constrained generation)

### 2.1 Constraint-aware sampling
- 현재 “생성 후 검증” 중심에서 “생성 중 제약 유지”로 전환
- hard constraint 위반 시 rejection-resampling 정책 + 최대 반복/백오프 정책 명시

### 2.2 Ontology scenario presets
- 연구 시나리오 preset 제공
  - `command_heavy`, `finance_fragmented`, `cellular_covert`, `cross_border_ops`
- preset별 ontology payload(roles/relations/interactions priors) 자동 주입

### 2.3 Violation telemetry
- 어떤 제약이 generation에서 자주 위반되는지 통계화
- `ontology_validation_report.json`에 `violation_histogram`, `repair_actions` 추가

---

## 3) PyG Dataset Builder v3 Upgrades (Ontology-to-tensor bridge)

### 3.1 Node/Edge ontology embeddings
- node: role one-hot 외 `role hierarchy depth`, `evidence coverage`, `provenance reliability`
- edge: relation logical props + temporal props + provenance confidence를 통합한 벡터화

### 3.2 Constraint masks for training
- relation별 허용 role-pair mask tensor 생성
- temporal violation candidate mask 생성 (학습 시 penalty 대상 지정)

### 3.3 Dataset card 강화
- ontology 요약: 활성 규칙 수, 위반율, provenance 통계, 시나리오 preset 기록

---

## 4) Model Zoo Upgrades (Ontology-aware learning)

### 4.1 Loss-level knowledge injection (우선순위 높음)
- Role-Relation compatibility loss
- Hierarchy transitivity consistency loss
- Temporal consistency loss (event-order/window)

### 4.2 Multi-task coupling
- HVT head와 role head 사이에 ontology priors 반영 (co-regularization)
- importance regression에 rule-based prior score를 residual target으로 활용

### 4.3 Robustness/Ablation protocol
- `base vs +ontology_features vs +ontology_losses vs full`
- noise 레벨( false/copy edge )별 강건성 비교

---

## 5) Experiment & Reporting Suite Upgrades

### 5.1 Ontology-aware experiment matrix
- difficulty(easy/baseline/hard) × ontology_mode(off/feature/loss/full) 매트릭스 표준화

### 5.2 Reporting artifacts
- run summary에 ontology 지표 추가
  - `ontology_conforms`, `violations_per_1k_edges`, `provenance_false_rate`, `copied_rate`
- 예측 설명 산출물
  - 노드별 `model_evidence + ontology_evidence + conflict_flags`

### 5.3 Reproducibility hardening
- ontology/shapes checksum 기록
- ruleset versioning (`ontology.version`, `constraints.version`) 강제

---

## 6) Suggested Delivery Order (next PR chunks)

1. **PR-A (Generator/Validator 강화)**
   - constraint-aware sampling + violation telemetry
2. **PR-B (Builder bridge 강화)**
   - ontology mask/feature tensor 추가 + dataset card 확장
3. **PR-C (Model loss 주입)**
   - compatibility/transitivity/temporal losses + ablation flags
4. **PR-D (Reporting/Explanation)**
   - ontology-aware summary columns + explanation json export

---

## 7) Done Criteria for “Ontology-centered maturity”

- 생성물의 ontology violations가 strict 모드에서 **0%**
- builder가 ontology tensors/masks를 기본 제공
- model이 ontology-loss ablation에서 일관된 이점(정확도/강건성/칼리브레이션)을 보여줌
- reporting에서 성능 + 규칙정합성 + 설명근거가 함께 비교 가능
