# REPRO Guide (STEP 8)

이 문서는 **한 줄 실행으로 재현 실험 + 요약표/그림 생성**하는 표준 절차를 제공한다.

## 1) 난이도 3종 실행 (easy/baseline/hard)

```bash
bash scripts/run_easy_baseline_hard.sh 2025 1500 results/repro_runs
```

인자:
- `SEED` (default `2025`)
- `SIZE` (default `1500`)
- `OUT_ROOT` (default `results/repro_runs`)

## 2) 결과 폴더 전체 요약 (CSV + plot)

```bash
bash scripts/summarize_all.sh results/repro_runs results/summary_all
```

생성 산출물(`results/summary_all`):
- `multitask_linkpred_summary_runs.csv`
- `multitask_linkpred_summary_agg.csv`
- `ontology_benchmark_table.csv`
- `ontology_benchmark_table.md`
- `hvt_f1_by_difficulty.png`, `hvt_auc_by_difficulty.png`, ...
- `difficulty_vs_performance_curve.png`

## 3) 권장 결과 구조

```text
results/
  repro_runs/
    easy/run_.../
      multiplex.json
      ontology_validation_report.json
      pyg_data.pt
      multitask_metrics.json
      linkpred_finance_uniform.json
      linkpred_communication_uniform.json
      run_metadata.json
    baseline/run_.../
    hard/run_.../
  summary_all/
    multitask_linkpred_summary_runs.csv
    multitask_linkpred_summary_agg.csv
    ontology_benchmark_table.csv
    *.png
```

위 구조는 README의 "Typical outputs"와 일치하도록 설계했다.
