from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from src.validation.schema import Manifest


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _cfg_value(cfg: Dict[str, Any], key: str, default: Any = "n/a") -> Any:
    v = cfg.get(key, default)
    return v if v is not None else default


def _delta(before: Any, after: Any) -> str:
    try:
        b = float(before)
        a = float(after)
        return f"{b:.4f} -> {a:.4f} (Δ={a - b:+.4f})"
    except Exception:
        return "n/a"


def render_dataset_card(
    manifest: Manifest,
    *,
    run_dir: str,
    dataset_path: Optional[str],
    diagnostics_dir: Optional[str],
) -> str:
    cfg = manifest.meta.config if isinstance(manifest.meta.config, dict) else {}
    lines = ["# DATASET_CARD", ""]
    lines.append("## Run overview")
    lines.append(f"- generator: `{manifest.meta.generator}`")
    lines.append(f"- seed: `{manifest.meta.seed}`")
    lines.append(f"- num_nodes: `{manifest.meta.num_nodes}`")
    lines.append("")

    lines.append("## Generator knobs (key)")
    lines.append(
        "- missing edge rates: "
        f"h={_cfg_value(cfg, 'missing_edge_rate_hierarchy')}, "
        f"f={_cfg_value(cfg, 'missing_edge_rate_finance')}, "
        f"c={_cfg_value(cfg, 'missing_edge_rate_communication')}, "
        f"o={_cfg_value(cfg, 'missing_edge_rate_operation')}, "
        f"i={_cfg_value(cfg, 'missing_edge_rate_ideology')}"
    )
    lines.append(
        "- false edge rates: "
        f"h={_cfg_value(cfg, 'false_edge_rate_hierarchy')}, "
        f"f={_cfg_value(cfg, 'false_edge_rate_finance')}, "
        f"c={_cfg_value(cfg, 'false_edge_rate_communication')}, "
        f"o={_cfg_value(cfg, 'false_edge_rate_operation')}, "
        f"i={_cfg_value(cfg, 'false_edge_rate_ideology')}"
    )
    lines.append(f"- cross_layer_copy: `{_cfg_value(cfg, 'cross_layer_copy')}`")
    lines.append("")

    lines.append("## Split / task notes")
    lines.append("- Node tasks: HVT classification / role classification / importance regression")
    lines.append("- Link prediction: per-layer positives with leakage-safe encoder graph and configurable negatives")
    lines.append("")

    lines.append("## Layer summary")
    lines.append("| layer | edges | false_rate | copied_rate |")
    lines.append("| --- | ---: | ---: | ---: |")
    for lname, layer in manifest.layers.items():
        total = len(layer.edges)
        if total == 0:
            false_rate = 0.0
            copied_rate = 0.0
        else:
            false_rate = sum(1 for e in layer.edges if (e.is_false or 0) != 0) / total
            copied_rate = sum(1 for e in layer.edges if e.copied_from) / total
        lines.append(f"| {lname} | {total} | {false_rate:.3f} | {copied_rate:.3f} |")
    lines.append("")

    lines.append("## Ethics / use limitations")
    lines.append("- Synthetic research dataset only; do not use for real-world targeting/surveillance.")
    lines.append("- Respect legal/ethical review when adapting methods to real data.")
    lines.append("")

    lines.append("## Artifacts")
    if dataset_path:
        lines.append(f"- PyG dataset: `{os.path.abspath(dataset_path)}`")
    if diagnostics_dir:
        lines.append(f"- Diagnostics: `{os.path.abspath(diagnostics_dir)}`")
    lines.append(f"- Run directory: `{os.path.abspath(run_dir)}`")

    return "\n".join(lines) + "\n"


def render_model_card(run_dir: str) -> str:
    mt = _read_json(os.path.join(run_dir, "multitask_metrics.json"))
    lp_fin_u = _read_json(os.path.join(run_dir, "linkpred_finance_uniform.json"))
    lp_fin_h = _read_json(os.path.join(run_dir, "linkpred_finance_hard_region.json"))
    lp_com_u = _read_json(os.path.join(run_dir, "linkpred_communication_uniform.json"))
    lp_com_h = _read_json(os.path.join(run_dir, "linkpred_communication_hard_region.json"))

    def _m(obj: Dict[str, Any], *keys: str) -> Any:
        cur: Any = obj
        for k in keys:
            if not isinstance(cur, dict):
                return "n/a"
            cur = cur.get(k)
        return cur if cur is not None else "n/a"

    lines = ["# MODEL_CARD", ""]
    lines.append("## Task metrics")
    lines.append(f"- HVT auc (tuned/test): `{_m(mt, 'hvt_threshold_tuned', 'test', 'auc')}`")
    lines.append(f"- HVT f1 (tuned/test): `{_m(mt, 'hvt_threshold_tuned', 'test', 'f1')}`")
    lines.append(f"- Role f1_macro (fixed/test): `{_m(mt, 'fixed_threshold', 'test', 'role_f1_macro')}`")
    lines.append(f"- Importance r2 (fixed/test): `{_m(mt, 'fixed_threshold', 'test', 'imp_r2')}`")
    lines.append("")

    lines.append("## Link prediction metrics")
    lines.append(f"- finance uniform auc/ap: `{_m(lp_fin_u, 'test_at_best_val_auc', 'auc')}` / `{_m(lp_fin_u, 'test_at_best_val_auc', 'ap')}`")
    lines.append(f"- finance hard_region auc/ap: `{_m(lp_fin_h, 'test_at_best_val_auc', 'auc')}` / `{_m(lp_fin_h, 'test_at_best_val_auc', 'ap')}`")
    lines.append(f"- communication uniform auc/ap: `{_m(lp_com_u, 'test_at_best_val_auc', 'auc')}` / `{_m(lp_com_u, 'test_at_best_val_auc', 'ap')}`")
    lines.append(f"- communication hard_region auc/ap: `{_m(lp_com_h, 'test_at_best_val_auc', 'auc')}` / `{_m(lp_com_h, 'test_at_best_val_auc', 'ap')}`")
    lines.append("")

    lines.append("## Calibration / threshold")
    lines.append("- HVT metrics use threshold-tuned and fixed-threshold reporting when available in multitask_metrics.json.")

    calib = mt.get("hvt_calibration") if isinstance(mt, dict) else None
    if isinstance(calib, dict):
        enabled = bool(calib.get("enabled", False))
        lines.append(f"- calibration enabled: `{enabled}`")
        lines.append(f"- method / temperature: `{calib.get('method', 'n/a')}` / `{calib.get('temperature', 'n/a')}`")
        lines.append(f"- validation ECE (before -> after): `{_delta(calib.get('ece_val_before'), calib.get('ece_val_after'))}`")
        lines.append(f"- validation Brier (before -> after): `{_delta(calib.get('brier_val_before'), calib.get('brier_val_after'))}`")
        lines.append(
            "- threshold strategy: "
            f"`{calib.get('thr_strategy', 'n/a')}` "
            f"(prevalence=`{calib.get('thr_prevalence', 'n/a')}`, fpr=`{calib.get('thr_fpr', 'n/a')}`)"
        )
        thr_iqr = calib.get("thr_iqr")
        if isinstance(thr_iqr, dict):
            lines.append(f"- threshold IQR: p25=`{thr_iqr.get('p25', 'n/a')}`, p75=`{thr_iqr.get('p75', 'n/a')}`")
    else:
        lines.append("- calibration summary: `n/a`")
    lines.append("")

    lines.append("## Limitations / cautions")
    lines.append("- Metrics may be unavailable (`n/a`) if training/evaluation artifacts were not generated in this run.")
    lines.append("- Synthetic-data performance does not directly transfer to real-world operational settings.")

    return "\n".join(lines) + "\n"


def write_run_cards(
    manifest: Manifest,
    *,
    run_dir: str,
    dataset_path: Optional[str],
    diagnostics_dir: Optional[str],
) -> Dict[str, str]:
    os.makedirs(run_dir, exist_ok=True)

    dataset_card_path = os.path.join(run_dir, "DATASET_CARD.md")
    model_card_path = os.path.join(run_dir, "MODEL_CARD.md")

    dataset_card = render_dataset_card(
        manifest,
        run_dir=run_dir,
        dataset_path=dataset_path,
        diagnostics_dir=diagnostics_dir,
    )
    model_card = render_model_card(run_dir)

    with open(dataset_card_path, "w", encoding="utf-8") as f:
        f.write(dataset_card)
    with open(model_card_path, "w", encoding="utf-8") as f:
        f.write(model_card)

    return {
        "dataset_card": dataset_card_path,
        "model_card": model_card_path,
    }
