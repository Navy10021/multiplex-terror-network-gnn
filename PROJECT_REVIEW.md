# PROJECT_REVIEW — Multiplex Terror Network GNN (v3)

## 1. Project Goal

This project provides a **synthetic multiplex network benchmark** designed to evaluate GNN performance under conditions that resemble real operational constraints:
- Partial observability / missing edges
- Noisy observations (false edges)
- Cross-layer copying / provenance effects
- Temporal event streams aggregated into static training graphs
- Multi-task supervision on nodes + link prediction on selected layers

The emphasis is on **reproducible experimentation** and **controlled difficulty scaling**.

---

## 2. Current Pipeline (v3)

### 2.1 Data Generation (multiplex_generator_v3.py)
**Input**
- `--size`, `--seed`, `--out_dir`
- `--config configs/generator_*.json`

**Output**
- `multiplex.json` (single manifest containing nodes, layer edges, events, labels, and metadata)

**Design intent**
- Generator configs provide difficulty knobs (structure strength, randomness, missingness, false/copy edges, observation bias, activity gating).
- `multiplex.json` is treated as the “single source of truth” for all subsequent steps.

---

### 2.2 Dataset Build (build_pyg_dataset_v3.py)
Converts `multiplex.json` → `pyg_data.pt` (PyTorch Geometric `Data`).

**Key contents**
- `x`: node features (categorical one-hot + continuous attributes)
- `y_role`: role class labels
- `y_hvt`: binary HVT labels
- `y_imp`: continuous importance score
- `train_mask`, `val_mask`, `test_mask`
- `edge_index`, `edge_type` (multiplex relation types)
- `edge_attr` (aggregated edge statistics, if present)
- Optional edge provenance flags (if generated), e.g.:
  - `edge_is_false`: injected false edge
  - `edge_is_copied`: copied edge (cross-layer provenance)

**Rationale**
- Keeping provenance flags in `Data` allows:
  - Diagnostics to quantify noise/copy behavior
  - Models to optionally incorporate edge reliability signals

---

### 2.3 Diagnostics (basic_diagnostics_v3.py)
Purpose: confirm that **knobs → measurable statistical changes**.

**Outputs**
- `*.png`: degree distributions per layer, role/region/group counts, role-wise degree boxplots, etc.
- `*.csv`: overlap, noise summaries, burstiness, activity/observability bias, copy provenance breakdowns, etc.

**What to look for**
- Degree distributions shift appropriately as structure/randomness knobs change.
- Layer overlap/Jaccard behaves as expected as cross-layer copying increases.
- False/copy rates match config within tolerance.
- Observability bias summaries reflect intended skew.

---

## 3. Modeling & Training (v3)

### 3.1 Multi-task Node Prediction (train_multitask_gnn_v3.py)
Jointly predicts:
- Role classification
- HVT classification
- Importance regression

**Typical artifacts**
- `multitask_metrics.json`
- `multitask_plots/` (curves & summaries)

**Evaluation**
- Role: macro-F1 / accuracy
- HVT: ROC-AUC / PR-AUC / F1 (threshold strategy supported)
- Importance: R² / RMSE

---

### 3.2 Layer-wise Link Prediction (train_linkpred_layer_v3.py)
Targets: `finance` or `communication`.

**Key design choice: leakage-safe message passing**
For the target layer LP task, validation/test positive edges are removed from the encoder message-passing graph.  
This ensures the model cannot “see” held-out positives during representation learning.

**Edge attributes & flags**
Two-stage handling:
1) `--edge_attr_agg`: aggregates edge attributes into node-level signals (training edges only for leakage safety)
2) `--include_edge_flags`: also aggregate edge provenance flags (`edge_is_false`, `edge_is_copied`) into the node signals

**Negatives**
- `--neg_mode uniform`: standard negative sampling
- `--neg_mode hard_region`: region-aware hard negatives (more realistic confusability)

**Artifacts**
- `linkpred_<layer>_<neg_mode>_v3.json`

---

## 4. Known Gaps / Technical Debt

1) **Config hash standardization**
- Some scripts can pass through generator metadata; ensure consistent, explicit hashing of generator configs for result tracking.

2) **Unified experiment runner**
- A single `run_all_sanity_checks.py` or `run_experiments.py` would reduce friction and improve reproducibility.

3) **Result summarization**
- Provide one canonical summarizer that merges:
  - `multitask_metrics.json`
  - `linkpred_*_v3.json`
  into one CSV + one clean plot (publication-ready).

4) **Testing**
- Add unit tests for:
  - generator output schema validation
  - build_pyg_dataset conversions (masks, edge flags, sizes)
  - leakage-safe edge removal logic
  - deterministic outputs under fixed seeds

---

## 5. Recommended Next Improvements (High ROI)

### A) Difficulty calibration suite
Automate “difficulty curves”:
- Sweep key knobs (missingness/false/copy rates, comm_randomness, structure strength)
- Run diagnostics + models
- Produce a single comparative report for easy/baseline/hard (or custom grids)

### B) Stronger realism for link prediction
- Add time-aware splitting for edges (temporal holdout)
- Evaluate generalization across time, not just random edge splits

### C) Ablations tied to operational constraints
Examples:
- Without edge flags vs with edge flags
- Uniform negatives vs hard-region negatives
- With/without edge_attr_agg
- With/without cross-layer copy mechanisms in the generator

---

## 6. Suggested “What to Claim” (Paper/Report Positioning)

- A controllable synthetic multiplex benchmark with explicit operational constraints
- A leakage-safe link prediction protocol for multiplex layers
- Evidence that noise/observability/copy knobs produce measurable graph shifts (diagnostics)
- Empirical results showing which modeling choices are robust under degraded observability

---

## 7. How to Reproduce (Minimal)

1) Generate:
```bash
python src/multiplex_generator_v3.py --size 1500 --seed 2025 --out_dir data/multiplex_baseline --config configs/generator_baseline.json
```

2) Build:
```bash
python src/build_pyg_dataset_v3.py --manifest data/multiplex_baseline/multiplex.json --out_path data/multiplex_baseline/pyg_data.pt --seed 2025
```

3) Diagnostics:
```bash
python src/basic_diagnostics_v3.py --manifest data/multiplex_baseline/multiplex.json --out_dir data/multiplex_baseline/diagnostics
```

4) Train multi-task:
```bash
python src/models/train_multitask_gnn_v3.py --data_path data/multiplex_baseline/pyg_data.pt
```

5) Train link prediction:
```bash
python src/models/train_linkpred_layer_v3.py --data_path data/multiplex_baseline/pyg_data.pt --layer finance --neg_mode hard_region --edge_attr_agg --include_edge_flags
```

---

## 8. Notes on Responsible Use

This project uses synthetic data and is intended for defensive research and benchmarking.
Avoid interpreting it as operational intelligence or guidance.

---

## 9. 추가 개선 제안 (실행 우선순위 포함)

### A) 재현성/결과 추적 강화를 위한 운영 자동화 (우선순위 ★★★)
- **단일 엔트리포인트 스크립트**: `python -m src.run_all` 형식의 드라이버를 추가해 `generator → build → diagnostics → train`을 일괄 실행하도록 하고, 실행에 사용된 **config 경로/해시, seed, Git commit**을 JSON 로그에 남기기.
- **환경 고정**: `requirements.txt`의 버전 범위를 상한/하한으로 고정하고, `pip-tools`나 `uv lock` 등을 사용해 `requirements.lock`을 커밋해 동일 환경 재현성을 높이기.
- **아티팩트 네이밍 표준화**: 출력 폴더를 `<date>_<config-hash>_<seed>` 규칙으로 생성하여 비교 실험 간 충돌을 방지하고, 요약 리포트에서 이 메타데이터를 자동 수집하도록 수정.

**단계별 실행(우선 적용 순)**
1) **잠금 파일 도입 및 실행 메타데이터 로그**
   - `requirements.in → requirements.lock` 파이프라인을 추가하고, CI에서 `pip install -r requirements.lock` 사용.
   - `src/utils/exp_logging.py`(신규)로 **config 해시, seed, Git commit, 실행 명령어**를 표준 JSON(`run_metadata.json`)에 기록.
2) **아티팩트 디렉터리 표준화**
   - 드라이버 스크립트에서 `<UTC date>_<config-hash>_<seed>` 규칙으로 모든 산출물 저장 루트 생성.
   - `basic_diagnostics_v3.py`와 학습 스크립트가 공통 메타데이터를 읽어 요약 리포트에 병기하도록 경로 인수 추가.
3) **엔드투엔드 드라이버 제공**
   - `python -m src.run_all --config ... --seed ... --out_root results/` 형태로 생성→빌드→진단→훈련을 호출.
   - 주요 단계 성공/실패 상태와 산출물 경로를 단일 `run_summary.json`으로 모아 후속 실험 비교에 활용.

### B) 데이터 품질 및 안전장치 확장 (우선순위 ★★☆)
- **스키마 검증**: `pydantic` 기반의 `multiplex.json` 스키마 검증기를 추가해 필수 필드 누락, 타입 불일치, 레이어별 edge flag 존재 여부 등을 파이프라인 초입에서 차단.
- **노이즈/편향 점검 규칙**: diagnostics 단계에 "허용 오차"를 명시(예: false-edge 비율 ±2%p)하고, 벗어날 경우 경고를 발생시켜 잘못된 설정을 조기에 발견.
- **데이터 카드**: 각 생성된 데이터셋 폴더에 자동으로 `DATASET_CARD.md`를 생성해 생성 명령어, config 요약, 주요 통계(노드/엣지 수, 레이어별 missing/false/copy 비율)를 기록.

**단계별 실행(우선 적용 순)**
1) **스키마 검증기 + CI 게이트**
   - `src/validation/schema.py`에 Pydantic 모델 정의 후 `basic_diagnostics_v3.py` 및 `build_pyg_dataset_v3.py` 진입 전에 실행.
   - CI에서 샘플 `multiplex.json`으로 검증기를 호출하는 테스트 추가하여 구조 변경 시 조기 감지.
2) **노이즈/편향 허용 범위 알림**
   - `basic_diagnostics_v3.py` 출력에 **허용 오차 대비 초과 여부**(warning)와 세부 항목(예: false-edge rate, copy rate, layer overlap)을 JSON+stdout 모두에 기록.
   - 파라미터는 `configs/generator_*.json`에서 `tolerance` 섹션으로 읽어 유연하게 조정.
3) **DATASET_CARD 자동 생성**
   - 드라이버 실행 종료 시 `DATASET_CARD.md`를 생성하여 **생성 명령어, config 해시, seed, 주요 통계(노드/엣지/레이어별 노이즈 비율)**를 표로 기록.
   - 재생산성을 위해 `run_metadata.json`과 동일 경로에 저장하고, 결과 수집 스크립트가 카드 내용을 요약 테이블에 병합하도록 확장.

### C) 테스트 커버리지 및 CI 구축 (우선순위 ★★☆)
- **단위 테스트**: `tests/`에 다음 검증을 추가
  - `build_pyg_dataset_v3`의 mask/label 크기 일관성 및 `edge_is_false/edge_is_copied` 전달 여부 확인
  - 링크 예측 학습 시 **target layer held-out positive 제거** 로직이 동작하는지 확인
  - 고정 seed에서 `multiplex_generator_v3` 출력이 결정적임을 보증
- **샘플 데이터 기반 회귀 테스트**: 작은 `toy` 설정(예: size=40)을 추가해 CI에서 빠르게 생성→빌드→간단한 forward pass까지 실행하도록 구성.
- **GitHub Actions 워크플로우**: `python -m pip install -r requirements.txt` 후 위 테스트와 `ruff/black` 체크를 수행하는 기본 CI를 추가.

### D) 문서화 & 온보딩 개선 (우선순위 ★☆☆)
- **실험 요약 템플릿**: `results/summary_all`에 실험별 핵심 지표(역할 F1, HVT AUC, importance R², layer별 LP AUC)를 표 형태로 자동 갱신하는 스크립트와 README 섹션을 추가.
- **사용 사례 예제 노트북**: `notebooks/`에 "difficulty sweep" 예제 노트북을 추가해 config 파라미터가 결과 지표에 미치는 영향을 시각화.
- **안전 가이드 고도화**: `README`의 윤리/안전 섹션에 허용/금지 사용 사례 체크리스트와 모델/데이터 재배포 시 요구사항을 명시.
