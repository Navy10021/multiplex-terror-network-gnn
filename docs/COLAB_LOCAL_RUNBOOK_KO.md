# 로컬/Colab 실행 가이드 (Baseline 중심)

아래는 질문에서 제시한 실행 순서를 **실제 CLI 옵션 기준으로 정리/보완**한 버전입니다.

## 0) 사전 준비

```bash
# (권장) 가상환경
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock

# (선택) 패키지 모드로 실행하려면
pip install -e .
```

> Colab에서는 `!` 셀 매직을 붙여서 실행하면 됩니다.

---

## 1) Data Generation

### 1-1. Baseline 데이터 생성

```bash
python -m src.data.multiplex_generator_v3 \
  --size 1500 \
  --seed 1024 \
  --out_dir data/multiplex_baseline \
  --config configs/generator_baseline.json
```

### 1-2. PyG 데이터 변환

```bash
python -m src.data.build_pyg_dataset_v3 \
  --manifest data/multiplex_baseline/multiplex.json \
  --out_path data/multiplex_baseline/pyg_data.pt
```

---

## 2) Data Statistics

```bash
python -m src.data.basic_diagnostics_v3 \
  --manifest data/multiplex_baseline/multiplex.json \
  --out_dir data/analysis/multiplex_baseline
```

---

## 3) Ontology Mode

### 3-1. End-to-End (기본 strict)

```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 1500 \
  --seed 2025 \
  --out_root results
```

### 3-2. Ontology constrained 모드

```bash
python -m src.run_all \
  --config configs/generator_baseline.json \
  --size 800 \
  --seed 2025 \
  --out_root results \
  --ontology_mode constrained
```

### 3-3. Ontology Validation

```bash
python -m src.cli.validate_ontology \
  --manifest data/multiplex_baseline/multiplex.json \
  --ontology ontology/terror.ttl \
  --shapes ontology/constraints.shacl.ttl \
  --json
```

---

## 4) GNN Models

### 4-1. HVT Classification (v3 권장)

```bash
python -m src.models.train_hvt_gnn_v3 \
  --data_path data/multiplex_baseline/pyg_data.pt \
  --hidden_dim 128 --num_layers 2 --dropout 0.3 \
  --lr 1e-3 --weight_decay 1e-4 --epochs 500 \
  --seed 1024 --pos_weight 10 \
  --edge_attr_agg --edge_attr_transform none \
  --include_edge_flags
```

### 4-2. HVT Single Task (v2)

```bash
python -m src.models.train_hvt_gnn_v2 \
  --data_path data/multiplex_baseline/pyg_data.pt \
  --hidden_dim 64 --num_layers 2 --dropout 0.5 \
  --lr 1e-3 --weight_decay 1e-4 --epochs 500 \
  --seed 1024 --pos_weight 10
```

> 기존 초안에서 `--pos_weight`가 중복되어 2회 들어가 있었는데 1회로 정리했습니다.

### 4-3. Multi-task

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

### 4-4. Layer별 Link Prediction

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

## 5) Result Visualization

```bash
python -m src.analysis.plot_multitask_linkpred_summary \
  --run_dirs \
    data/multiplex_baseline \
    data/multiplex_hard \
  --out_dir results/summary_all
```

> `--aggregate`, `--save_runs_csv`, `--write_benchmark_table`를 함께 쓰면 요약 산출물이 더 풍부해집니다.

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

## 추가 권장사항 (빠진 부분)

1. **패키지 실행(`python -m ...`) 일관화**
   - 로컬/Colab의 작업 디렉터리 차이로 인한 import 오류를 줄입니다.
2. **재현성 로그 고정**
   - seed, config, out_dir를 실험노트에 함께 기록하세요.
3. **난이도 3종 자동 실행 사용**

```bash
bash scripts/run_easy_baseline_hard.sh 2025 1500 results/repro_runs
bash scripts/summarize_all.sh results/repro_runs results/summary_all
```

