# 🔬 Multiplex Terror Network GNN (Ontology-First)

> 합법적·방어적 목적의 **합성(Synthetic) 데이터 기반** 멀티플렉스 네트워크 연구 저장소입니다.
>
> ⚠️ 실제 작전/타게팅 용도로 사용하지 마세요.

---

## 1) 프로젝트 개요

이 저장소는 다음 파이프라인을 제공합니다.

1. 합성 멀티플렉스 네트워크 생성
2. 온톨로지(OWL/SHACL + 런타임 룰) 검증
3. PyG 데이터셋 변환
4. 멀티태스크 GNN 학습(온톨로지 정규화 옵션)
5. 리포팅/설명 아티팩트 생성

```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 1500 \
  --seed 2025 \
  --out_root results
```

---

## 2) 온톨로지 중심 설계가 중요한 이유

멀티플렉스 위협 네트워크는 레이어별 의미가 다르고, 관측 노이즈·결측·오탐이 섞이며, 시간/역할/관계 제약이 성능에 큰 영향을 줍니다.

이 프로젝트는 온톨로지를 **데이터 계약(data contract)** 으로 사용해 아래를 일관되게 맞춥니다.

- 생성 단계 품질 통제 (strict / constrained / report_only)
- 피처 엔지니어링(ontology bridge tensor)
- 학습 손실 설계(ontology-aware loss)
- 평가/설명(위반율, rule-chain 근거)

---

## 3) 온톨로지 구성요소 (구체 설명)

### 3.1 OWL 도메인 모델 (`ontology/terror.ttl`)

- 핵심 클래스: `Actor`, `Role`, `Relation`, `Event`, `Evidence`, `EdgeProvenance`, `InteractionRule`
- 역할 타입: `Leader`, `Financier`, `Courier`, `Operative`, `Support`
- 관계 타입: `HierarchyRelation`, `FinanceRelation`, `CommunicationRelation`, `OperationRelation`, `IdeologyRelation`
- 주요 속성:
  - 구조: `source`, `target`, `hasRole`
  - 이벤트: `timestamp`, `eventType`
  - 금융: `txnAmountSum`, `txnCount`
  - provenance: `isFalseEdge`, `isCopiedEdge`, `copiedFromLayer`, `confidence`
  - 레이어 상호작용: `fromLayer`, `toLayer`, `temporalWindowDays`, `ruleType`, `strength`

### 3.2 SHACL 제약 (`ontology/constraints.shacl.ttl`)

- `ActorRoleShape`: 역할 카드inality
- `EventTimeShape`: timestamp는 0 이상
- `EventTypeShape`: `comm|txn|op`
- `FinanceEdgeAmountShape`: 거래 합계 양수
- `HierarchyNoSelfLoopShape`: self-loop 금지
- `ProvenanceConfidenceShape`: confidence는 [0,1]

### 3.3 런타임 검증기 (`src/ontology/validator.py`)

정적 SHACL만으로 부족한 제약을 실제 manifest 데이터로 검증합니다.

- 역할 화이트리스트
- 계층(hierarchy) 명령 역할 소스 제약
- 금융값 유효성(양수/음수/비정상값)
- 관계-역할 호환성
- provenance 유효성 (`is_false`, `copied_from`, `confidence`)
- `ontology.layer_interactions` 기반 시간 순서/지연 제약

리포트 산출:

- `conforms`
- `violations`, `violations_by_check`
- `violation_histogram`
- `errors_by_check`
- 카운트(노드/레이어/이벤트/위반)

---

## 4) 실행 모드

| 모드 | 동작 | 권장 사용 |
|---|---|---|
| `strict` (기본) | 위반 시 즉시 실패 | 품질 보장 실험 |
| `constrained` | seed 이동 재시도 | 제약 만족 manifest 확보 |
| `report_only` | 위반 기록 후 계속 | 탐색/분석/디버깅 |

예시:

```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 800 --seed 2025 --out_root results \
  --ontology_mode constrained
```

---

## 5) 주요 산출물

- `multiplex.json`
- `ontology_validation_report.json`
- `run_metadata.json`
- `pyg_data.pt`
- `multitask_metrics.json`
- `reporting_summary/multitask_linkpred_summary.csv` (옵션)
- `explanations/ontology_explanations.json` (옵션)

---

## 6) 설치

```bash
git clone https://github.com/Navy10021/multiplex-terror-network-gnn.git
cd multiplex-terror-network-gnn
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
```

CUDA/OS 환경에 따라 PyTorch/PyG 선설치가 필요할 수 있습니다.

---

## 7) 빠른 사용법

### 7.1 E2E 실행

```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 1200 --seed 2025 --out_root results
```

### 7.2 검증 전용 실행

```bash
python -m src.cli.validate_ontology \
  --manifest data/multiplex_baseline/multiplex.json \
  --ontology ontology/terror.ttl \
  --shapes ontology/constraints.shacl.ttl \
  --json
```

### 7.3 온톨로지 손실 포함 학습

```bash
python -m src.models.train_multitask_gnn_v3 \
  --data_path results/<run>/pyg_data.pt \
  --encoder transformer --epochs 20 \
  --ontology_loss \
  --ontology_loss_role_weight 0.2 \
  --ontology_loss_transitivity_weight 0.1 \
  --ontology_loss_temporal_weight 0.1
```

---

## 8) 보안/견고성 점검 포인트

- `pyg_data.pt`는 pickle 기반 로딩 경로(`torch.load(weights_only=False)`)를 사용하므로 **신뢰 가능한 파일만** 로드하세요.
- `strict` 모드 실패는 시스템 오류가 아니라 제약 위반일 수 있습니다. 필요 시 `report_only` 또는 `constrained`로 전환하세요.
- 온톨로지 검증기는 비정상 수치(`NaN`, `inf`, 비숫자 문자열)도 위반으로 안전하게 처리합니다.

---

## 9) 개발자 검증 명령

```bash
pytest -q
python -m src.run_all --help
python -m src.cli.validate_ontology --help
```

---

## 10) 로드맵

- English: `PLANS.md`
- 한국어: `PLANS_KOR.md`

