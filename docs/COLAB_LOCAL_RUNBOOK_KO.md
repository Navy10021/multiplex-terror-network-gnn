# 로컬/Colab 실행 가이드 (최종 점검본)

이 문서는 질문에서 제시한 흐름을 기준으로, **누락/순서/실행 안정성**을 반영해 정리한 최종 실행 순서입니다.

---

## 0) 사전 준비

```bash
# 0-1) 저장소 준비
git clone https://github.com/Navy10021/multiplex-terror-network-gnn.git
cd multiplex-terror-network-gnn

# 0-2) 가상환경(권장)
python -m venv .venv
source .venv/bin/activate

# 0-3) 의존성 설치
pip install -r requirements.lock

# 0-4) (선택) 패키지 모드 CLI
pip install -e .
```

> Colab에서는 각 명령 앞에 `!`를 붙여 실행하면 됩니다.

> 주의: `requirements.lock`만으로는 환경에 따라 `torch/torch-geometric`이 설치되지 않을 수 있습니다.
> 반드시 `docs/INSTALL.md`의 CPU/CUDA/Colab 설치 레시피를 먼저 확인하세요.

### 0-A) 최소 헬스체크 (권장)

```bash
python -m src.run_all --help
python -m src.cli.validate_ontology --help
```

---

## 1) 권장 실행 순서 (수동 단계 실행)

아래 순서는 **생성 → 검증 → 변환 → 진단 → 학습 → 요약** 흐름으로 구성됩니다.

### 1-1. Baseline 데이터 생성

```bash
python -m src.data.multiplex_generator_v3 \
  --size 1500 \
  --seed 1024 \
  --out_dir data/multiplex_baseline \
  --config configs/generator_baseline.json
```

> 참고: strict 모드에서 온톨로지 제약으로 실패할 수 있습니다. 이 경우 아래 중 하나를 사용하세요.
>
> - 제약 만족 재시도: `--ontology_constrained`
> - 리포트만 남기고 진행: `--no_ontology_strict`

### 1-2. Ontology Validation (생성 직후 확인 권장)

```bash
python -m src.cli.validate_ontology \
  --manifest data/multiplex_baseline/multiplex.json \
  --ontology ontology/terror.ttl \
  --shapes ontology/constraints.shacl.ttl \
  --json
```

### 1-3. PyG 데이터 변환

```bash
python -m src.data.build_pyg_dataset_v3 \
  --manifest data/multiplex_baseline/multiplex.json \
  --out_path data/multiplex_baseline/pyg_data.pt
```

### 1-4. 데이터 통계/진단

```bash
python -m src.data.basic_diagnostics_v3 \
  --manifest data/multiplex_baseline/multiplex.json \
  --out_dir data/analysis/multiplex_baseline
```

---

## 2) End-to-End 실행 (run_all)

수동 실행 대신 아래로 한 번에 실행할 수 있습니다.

### 2-1. 기본(strict)

```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 1500 \
  --seed 2025 \
  --out_root results
```

### 2-2. constrained 모드

```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 800 \
  --seed 2025 \
  --out_root results \
  --ontology_mode constrained
```

### 2-3. report_only 모드(실험 진행 우선)

```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 800 \
  --seed 2025 \
  --out_root results \
  --ontology_mode report_only
```

---

## 3) GNN 모델 학습

> `pyg_data.pt`가 준비된 뒤 실행합니다.

### 3-1. HVT Classification (v3)

```bash
python -m src.models.train_hvt_gnn_v3 \
  --data_path data/multiplex_baseline/pyg_data.pt \
  --hidden_dim 128 --num_layers 2 --dropout 0.3 \
  --lr 1e-3 --weight_decay 1e-4 --epochs 500 \
  --seed 1024 --pos_weight 10 \
  --edge_attr_agg --edge_attr_transform none \
  --include_edge_flags
```

### 3-2. HVT Single Task (v2)

```bash
python -m src.models.train_hvt_gnn_v2 \
  --data_path data/multiplex_baseline/pyg_data.pt \
  --hidden_dim 64 --num_layers 2 --dropout 0.5 \
  --lr 1e-3 --weight_decay 1e-4 --epochs 500 \
  --seed 1024 --pos_weight 10
```

> `--pos_weight` 중복 입력은 제거(1회만 유지)했습니다.

### 3-3. Multi-task

```bash
python -m src.models.train_multitask_gnn_v3 \
  --data_path data/multiplex_baseline/pyg_data.pt \
  --encoder transformer \
  --hidden_dim 128 --num_layers 2 --dropout 0.3 \
  --lr 2e-3 --weight_decay 1e-4 --epochs 300 \
  --seed 2025 --patience 50 \
  --edge_attr_transform none \
  --include_edge_flags
```

### 3-4. Layer별 Link Prediction

#### Finance

```bash
python -m src.models.train_linkpred_layer_v3 \
  --data_path data/multiplex_baseline/pyg_data.pt \
  --layer finance \
  --hidden_dim 192 --num_layers 2 --dropout 0.3 \
  --lr 1e-3 --weight_decay 1e-4 --epochs 500 \
  --seed 1024 --patience 50 --min_delta 1e-3 \
  --edge_attr_agg --include_edge_flags
```

#### Communication

```bash
python -m src.models.train_linkpred_layer_v3 \
  --data_path data/multiplex_baseline/pyg_data.pt \
  --layer communication \
  --hidden_dim 128 --num_layers 2 --dropout 0.3 \
  --lr 1e-3 --weight_decay 1e-4 --epochs 500 \
  --seed 1024 --patience 50 --min_delta 1e-3 \
  --edge_attr_agg --include_edge_flags
```

---

## 4) 결과 시각화 / 요약

> `multitask_metrics.json`이 존재하는 run 디렉터리(또는 data 디렉터리)를 `--run_dirs`로 전달합니다.

```bash
python -m src.analysis.plot_multitask_linkpred_summary \
  --run_dirs data/multiplex_baseline data/multiplex_hard \
  --out_dir results/summary_all \
  --difficulty_mode auto \
  --aggregate \
  --save_runs_csv \
  --write_benchmark_table
```

---

## 5) 재현 실험(난이도 3종) 빠른 실행

```bash
bash scripts/run_easy_baseline_hard.sh 2025 1500 results/repro_runs
bash scripts/summarize_all.sh results/repro_runs results/summary_all
```

---

## 6) Colab용 초간단 스모크 테스트 (저비용)

긴 학습 전에 아래 순서로 환경/파이프라인을 빠르게 확인하세요.

```bash
# 생성 (report-only 성격으로 진행)
python -m src.data.multiplex_generator_v3 \
  --size 120 --seed 1024 \
  --out_dir data/smoke/multiplex_baseline \
  --config configs/generator_baseline.json \
  --no_ontology_strict

# 변환/진단
python -m src.data.build_pyg_dataset_v3 \
  --manifest data/smoke/multiplex_baseline/multiplex.json \
  --out_path data/smoke/multiplex_baseline/pyg_data.pt

python -m src.data.basic_diagnostics_v3 \
  --manifest data/smoke/multiplex_baseline/multiplex.json \
  --out_dir data/smoke/analysis_baseline
```

---

## 7) 실무 팁 (누락 보완)

1. **실험 기록 고정**: seed, config, out_dir, git commit hash를 함께 저장.
2. **온톨로지 정책 명시**: strict/constrained/report_only 중 무엇으로 돌렸는지 결과 파일과 함께 기록.
3. **경로 일관성 유지**: 항상 `python -m ...` 형태로 실행해 Colab/로컬 경로 차이 리스크 최소화.

