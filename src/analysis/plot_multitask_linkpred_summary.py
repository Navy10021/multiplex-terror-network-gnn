import argparse
import json
import os
import re
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------
# Helpers
# --------------------------

def safe_get(d: Dict[str, Any], *keys, default=None):
    """
    Helper to safely retrieve values from nested dictionaries.
    Returns default if any key is missing.
    """
    cur: Any = d
    for k in keys:
        if cur is None or not isinstance(cur, dict):
            return default
        cur = cur.get(k, None)
    return cur if cur is not None else default


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[!] Failed to read json: {path} ({e})")
        return None


def _float_or_nan(v: Any) -> float:
    try:
        if v is None:
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def _parse_seed_from_name(run_name: str) -> Optional[int]:
    """
    Try to parse a seed from folder name.
    Examples:
      - multiplex_easy_seed2025 -> 2025
      - ..._s2025 -> 2025
    """
    m = re.search(r"(?:seed|s)(\d{1,6})", run_name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


# --------------------------
# Loading per-run artifacts
# --------------------------

def load_multitask_metrics(run_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load multitask_metrics.json in run_dir and extract key test metrics.
    Returns:
      - metrics (flat floats for plotting)
      - raw json object (for provenance fields)
    """
    mt_path = os.path.join(run_dir, "multitask_metrics.json")
    mt = _read_json(mt_path)
    if mt is None:
        print(f"[!] multitask_metrics.json not found in {run_dir}")
        return {}, {}

    tuned_test = safe_get(mt, "hvt_threshold_tuned", "test", default={}) or {}
    fixed_test = safe_get(mt, "fixed_threshold", "test", default={}) or {}

    metrics = {
        "hvt_f1": _float_or_nan(tuned_test.get("f1", np.nan)),
        "hvt_auc": _float_or_nan(tuned_test.get("auc", np.nan)),
        "role_f1_macro": _float_or_nan(fixed_test.get("role_f1_macro", np.nan)),
        "imp_r2": _float_or_nan(fixed_test.get("imp_r2", np.nan)),
    }
    return metrics, mt


def load_linkpred_metrics(run_dir: str, layer: str, neg_mode: str) -> Dict[str, Any]:
    """
    Load linkpred_{layer}_{neg_mode}.json in run_dir and extract
    test_at_best_val_auc AUC/AP.
    """
    fname = f"linkpred_{layer}_{neg_mode}.json"
    obj = _read_json(os.path.join(run_dir, fname))
    if obj is None:
        print(f"[!] {fname} not found in {run_dir}")
        return {"auc": np.nan, "ap": np.nan}

    test = obj.get("test_at_best_val_auc", {}) or {}
    return {
        "auc": _float_or_nan(test.get("auc", np.nan)),
        "ap": _float_or_nan(test.get("ap", np.nan)),
    }




def load_ontology_report_metrics(run_dir: str) -> Dict[str, Any]:
    """Load ontology_validation_report.json and expose summary-friendly metrics."""
    rep = _read_json(os.path.join(run_dir, "ontology_validation_report.json")) or {}
    mani = _read_json(os.path.join(run_dir, "multiplex.json")) or {}

    layers = mani.get("layers", {}) if isinstance(mani.get("layers"), dict) else {}
    total_edges = 0
    for _, layer_obj in layers.items():
        if isinstance(layer_obj, dict) and isinstance(layer_obj.get("edges"), list):
            total_edges += len(layer_obj.get("edges") or [])

    events = mani.get("events", []) if isinstance(mani.get("events"), list) else []

    violations_total = _float_or_nan(rep.get("violations_total", np.nan))
    if not np.isfinite(violations_total):
        violations_total = _float_or_nan(len(rep.get("violations", []) if isinstance(rep.get("violations"), list) else []))

    denom_edges = max(float(total_edges), 1.0)
    denom_events = max(float(len(events)), 1.0)

    return {
        "ontology_conforms": 1.0 if bool(rep.get("conforms", False)) else 0.0,
        "ontology_violations_total": violations_total,
        "ontology_violations_per_1k_edges": float(1000.0 * violations_total / denom_edges) if np.isfinite(violations_total) else np.nan,
        "ontology_violations_per_1k_events": float(1000.0 * violations_total / denom_events) if np.isfinite(violations_total) else np.nan,
    }


def load_ontology_loss_metrics(mt_raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Expose ontology-loss training settings from multitask_metrics.json."""
    if not isinstance(mt_raw, dict):
        return {
            "ontology_loss_enabled": np.nan,
            "ontology_loss_role": np.nan,
            "ontology_loss_transitivity": np.nan,
            "ontology_loss_temporal": np.nan,
        }

    onto = mt_raw.get("ontology_loss", {}) if isinstance(mt_raw.get("ontology_loss"), dict) else {}
    final_losses = onto.get("final_epoch_losses", {}) if isinstance(onto.get("final_epoch_losses"), dict) else {}
    return {
        "ontology_loss_enabled": 1.0 if bool(onto.get("enabled", False)) else 0.0,
        "ontology_loss_role": _float_or_nan(final_losses.get("role_relation_compatibility", np.nan)),
        "ontology_loss_transitivity": _float_or_nan(final_losses.get("hierarchy_transitivity", np.nan)),
        "ontology_loss_temporal": _float_or_nan(final_losses.get("temporal_ordering", np.nan)),
    }

def load_generator_config(run_dir: str, mt_raw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Prefer config snapshot embedded into multitask_metrics.json (if available),
    else fall back to multiplex.json(meta.config).
    """
    cfg: Dict[str, Any] = {}

    if mt_raw:
        mt_cfg = mt_raw.get("generator_config", None)
        if isinstance(mt_cfg, dict):
            cfg = mt_cfg

    if not cfg:
        manifest = _read_json(os.path.join(run_dir, "multiplex.json")) or {}
        cfg = safe_get(manifest, "meta", "config", default={}) or {}

    # normalize to a fixed set of keys (so CSV columns are consistent)
    keys = [
        "finance_structure_strength",
        "comm_structure_strength",
        "comm_randomness",
        "hvt_ratio",
        "finance_w_group",
        "finance_w_region",
        "finance_w_ideo",
        "finance_w_tier_dist",
        "finance_base_bias",
        "comm_avg_degree",
        "comm_alpha0",
        "comm_alpha_group",
        "comm_alpha_region",
        "comm_alpha_hier",
        "comm_alpha_fin",
        "ideo_threshold",
        "op_num_cells",
        "op_cell_size",
    ]
    out = {k: (_float_or_nan(cfg.get(k)) if k in cfg else np.nan) for k in keys}
    return out


# --------------------------
# Difficulty inference / checks
# --------------------------

def infer_difficulty_from_folder(run_name: str) -> str:
    low = run_name.lower()
    if "easy" in low:
        return "easy"
    if "hard" in low:
        return "hard"
    if "base" in low or "baseline" in low:
        return "baseline"
    return "unknown"


def infer_difficulty_from_config(cfg: Dict[str, Any]) -> str:
    """
    Heuristic mapping based on the intended presets:
      - easy: higher structure strengths and low randomness
      - hard: lower structure strengths and/or higher randomness
      - baseline: middle
    """
    f = cfg.get("finance_structure_strength", np.nan)
    c = cfg.get("comm_structure_strength", np.nan)
    r = cfg.get("comm_randomness", np.nan)

    # NaN-safe comparisons
    if np.isfinite(r) and r >= 0.2:
        return "hard"
    if np.isfinite(f) and np.isfinite(c):
        if f >= 1.1 and c >= 1.1:
            return "easy"
        if f <= 0.8 or c <= 0.8:
            return "hard"
        return "baseline"
    return "unknown"


def difficulty_sort_key(diff: str) -> int:
    order = {"easy": 0, "baseline": 1, "hard": 2, "unknown": 99}
    return order.get(diff, 99)


# --------------------------
# Build DataFrames
# --------------------------

def build_runs_dataframe(run_dirs: List[str], difficulty_mode: str = "auto") -> pd.DataFrame:
    """
    Build a per-run DataFrame.
    difficulty_mode:
      - folder: use folder name only
      - config: use config heuristic only
      - auto: use config if it resolves to easy/baseline/hard else folder
    """
    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        run_dir = os.path.abspath(run_dir)
        run_name = os.path.basename(run_dir.rstrip("/"))
        mt_metrics, mt_raw = load_multitask_metrics(run_dir)
        gen_cfg = load_generator_config(run_dir, mt_raw=mt_raw)

        diff_folder = infer_difficulty_from_folder(run_name)
        diff_cfg = infer_difficulty_from_config(gen_cfg)

        if difficulty_mode == "folder":
            difficulty = diff_folder
        elif difficulty_mode == "config":
            difficulty = diff_cfg
        else:
            # auto
            difficulty = diff_cfg if diff_cfg != "unknown" else diff_folder

        if diff_cfg != "unknown" and diff_folder != "unknown" and diff_cfg != diff_folder:
            print(f"[WARN] difficulty mismatch: folder={diff_folder}, config={diff_cfg} ({run_dir})")

        # seed (prefer multitask_metrics.json)
        seed = None
        if isinstance(mt_raw, dict) and "seed" in mt_raw:
            try:
                seed = int(mt_raw["seed"])
            except Exception:
                seed = None
        if seed is None:
            seed = _parse_seed_from_name(run_name)

        # linkpred combinations
        lp_fin_uniform = load_linkpred_metrics(run_dir, layer="finance", neg_mode="uniform")
        lp_fin_hard = load_linkpred_metrics(run_dir, layer="finance", neg_mode="hard_region")
        lp_comm_uniform = load_linkpred_metrics(run_dir, layer="communication", neg_mode="uniform")
        lp_comm_hard = load_linkpred_metrics(run_dir, layer="communication", neg_mode="hard_region")

        onto_report = load_ontology_report_metrics(run_dir)
        onto_loss = load_ontology_loss_metrics(mt_raw)

        row = {
            "run_dir": run_dir,
            "run_name": run_name,
            "seed": seed,
            "difficulty": difficulty,
            **gen_cfg,
            # multitask
            "hvt_f1": mt_metrics.get("hvt_f1", np.nan),
            "hvt_auc": mt_metrics.get("hvt_auc", np.nan),
            "role_f1_macro": mt_metrics.get("role_f1_macro", np.nan),
            "imp_r2": mt_metrics.get("imp_r2", np.nan),
            # ontology reporting
            **onto_report,
            # ontology loss diagnostics
            **onto_loss,
            # linkpred - finance
            "finance_auc_uniform": lp_fin_uniform["auc"],
            "finance_ap_uniform": lp_fin_uniform["ap"],
            "finance_auc_hard_region": lp_fin_hard["auc"],
            "finance_ap_hard_region": lp_fin_hard["ap"],
            # linkpred - communication
            "comm_auc_uniform": lp_comm_uniform["auc"],
            "comm_ap_uniform": lp_comm_uniform["ap"],
            "comm_auc_hard_region": lp_comm_hard["auc"],
            "comm_ap_hard_region": lp_comm_hard["ap"],
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["difficulty_order"] = df["difficulty"].apply(difficulty_sort_key)
    df = df.sort_values(["difficulty_order", "run_name"]).reset_index(drop=True)
    return df


def build_aggregated_dataframe(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across seeds/runs per difficulty: mean±std.
    Keeps generator config columns as the first (they should be identical by design).
    """
    if df_runs.empty:
        return df_runs

    cfg_cols = [
        "finance_structure_strength",
        "comm_structure_strength",
        "comm_randomness",
        "hvt_ratio",
        "finance_w_group",
        "finance_w_region",
        "finance_w_ideo",
        "finance_w_tier_dist",
        "finance_base_bias",
        "comm_avg_degree",
        "comm_alpha0",
        "comm_alpha_group",
        "comm_alpha_region",
        "comm_alpha_hier",
        "comm_alpha_fin",
        "ideo_threshold",
        "op_num_cells",
        "op_cell_size",
    ]
    metric_cols = [
        "hvt_f1",
        "hvt_auc",
        "role_f1_macro",
        "imp_r2",
        "ontology_conforms",
        "ontology_violations_total",
        "ontology_violations_per_1k_edges",
        "ontology_violations_per_1k_events",
        "ontology_loss_enabled",
        "ontology_loss_role",
        "ontology_loss_transitivity",
        "ontology_loss_temporal",
        "finance_auc_uniform",
        "finance_ap_uniform",
        "finance_auc_hard_region",
        "finance_ap_hard_region",
        "comm_auc_uniform",
        "comm_ap_uniform",
        "comm_auc_hard_region",
        "comm_ap_hard_region",
    ]

    # protect against missing columns
    cfg_cols = [c for c in cfg_cols if c in df_runs.columns]
    metric_cols = [c for c in metric_cols if c in df_runs.columns]

    grouped = df_runs.groupby("difficulty", dropna=False)
    agg_mean = grouped[metric_cols].mean(numeric_only=True).add_suffix("_mean")
    agg_std = grouped[metric_cols].std(numeric_only=True).add_suffix("_std")
    agg_n = grouped.size().rename("n_runs")

    # take first config snapshot per difficulty (they should match)
    cfg_first = grouped[cfg_cols].first()

    df_agg = pd.concat([cfg_first, agg_mean, agg_std, agg_n], axis=1).reset_index()
    df_agg["difficulty_order"] = df_agg["difficulty"].apply(difficulty_sort_key)
    df_agg = df_agg.sort_values(["difficulty_order"]).reset_index(drop=True)
    return df_agg


# --------------------------
# Plotting
# --------------------------

def _get_series(df: pd.DataFrame, name: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Return (y, yerr) for a metric name.
    Supports either raw columns (name) or aggregated columns (name_mean / name_std).
    """
    if name in df.columns:
        return df[name].to_numpy(), None
    if f"{name}_mean" in df.columns:
        y = df[f"{name}_mean"].to_numpy()
        yerr = df.get(f"{name}_std", pd.Series([0.0] * len(df))).to_numpy()
        return y, yerr
    return np.array([np.nan] * len(df)), None


def plot_bar_hvt_metrics(df: pd.DataFrame, out_dir: str):
    if df.empty:
        print("[!] Empty DataFrame, skip plot_bar_hvt_metrics")
        return

    labels = df["difficulty"].tolist()
    x = np.arange(len(labels))

    for metric, ylabel, title, fname in [
        ("hvt_f1", "HVT F1 (threshold tuned)", "HVT F1 by difficulty", "hvt_f1_by_difficulty.png"),
        ("hvt_auc", "HVT AUC", "HVT AUC by difficulty", "hvt_auc_by_difficulty.png"),
        ("role_f1_macro", "Role F1 (macro)", "Role F1 (macro) by difficulty", "role_f1_macro_by_difficulty.png"),
        ("imp_r2", "Importance R2", "Importance R2 by difficulty", "imp_r2_by_difficulty.png"),
        ("ontology_conforms", "Ontology conformance rate", "Ontology conformance by difficulty", "ontology_conformance_by_difficulty.png"),
        ("ontology_violations_per_1k_edges", "Violations / 1k edges", "Ontology violations per 1k edges", "ontology_violations_per_1k_edges.png"),
    ]:
        y, yerr = _get_series(df, metric)
        plt.figure()
        if yerr is None:
            plt.bar(x, y)
        else:
            plt.bar(x, y, yerr=yerr, capsize=5)
        plt.xticks(x, labels)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path)
        plt.close()
        print(f"[*] Saved plot: {out_path}")


def plot_bar_linkpred_layer(df: pd.DataFrame, out_dir: str, layer: str):
    if df.empty:
        print("[!] Empty DataFrame, skip plot_bar_linkpred_layer")
        return

    labels = df["difficulty"].tolist()
    x = np.arange(len(labels))

    for neg_mode in ["uniform", "hard_region"]:
        for metric_key, metric_label in [("auc", "AUC"), ("ap", "AP")]:
            col = f"{'finance' if layer=='finance' else 'comm'}_{metric_key}_{neg_mode}"
            y, yerr = _get_series(df, col)
            plt.figure()
            if yerr is None:
                plt.bar(x, y)
            else:
                plt.bar(x, y, yerr=yerr, capsize=5)
            plt.xticks(x, labels)
            plt.ylabel(f"{layer} {metric_label} ({neg_mode})")
            plt.title(f"{layer} link prediction {metric_label} by difficulty ({neg_mode})")
            plt.tight_layout()
            out_path = os.path.join(out_dir, f"{layer}_{metric_key}_{neg_mode}_by_difficulty.png")
            plt.savefig(out_path)
            plt.close()
            print(f"[*] Saved plot: {out_path}")


def plot_difficulty_vs_performance_curve(df: pd.DataFrame, out_dir: str):
    """
    Difficulty index: easy=1, baseline=2, hard=3. Plot mean curves (with optional error bars).
    """
    if df.empty:
        print("[!] Empty DataFrame, skip plot_difficulty_vs_performance_curve")
        return

    difficulty_to_x = {"easy": 1, "baseline": 2, "hard": 3}
    df = df[df["difficulty"].isin(difficulty_to_x.keys())].copy()
    if df.empty:
        print("[!] No recognized difficulty labels (easy/baseline/hard). Skip curve plot.")
        return

    df["diff_idx"] = df["difficulty"].map(difficulty_to_x)
    df = df.sort_values("diff_idx")

    curves = [
        ("hvt_f1", "HVT F1"),
        ("hvt_auc", "HVT AUC"),
        ("role_f1_macro", "Role F1 (macro)"),
        ("imp_r2", "Importance R2"),
        ("finance_auc_uniform", "Finance AUC (uniform)"),
        ("comm_auc_uniform", "Comm AUC (uniform)"),
    ]

    plt.figure()
    for metric, label in curves:
        y, yerr = _get_series(df, metric)
        x = df["diff_idx"].to_numpy()
        if yerr is None:
            plt.plot(x, y, marker="o", label=label)
        else:
            plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4, label=label)

    plt.xticks([1, 2, 3], ["easy", "baseline", "hard"])
    plt.xlabel("Difficulty")
    plt.ylabel("Performance")
    plt.title("Difficulty vs Performance (mean across seeds; ±1 std if available)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "difficulty_vs_performance_curve.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[*] Saved plot: {out_path}")


# --------------------------
# Main
# --------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Summarize multitask + link prediction metrics across difficulties (supports multi-seed averaging)."
    )
    parser.add_argument(
        "--run_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to experiment result folders (can include multiple seeds per difficulty).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to save plots and CSV (default: first run_dir).",
    )
    parser.add_argument(
        "--difficulty_mode",
        type=str,
        default="auto",
        choices=["auto", "folder", "config"],
        help="How to determine difficulty label: folder-name / config-heuristic / auto (default).",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="If set, plot and save aggregated (mean±std) results by difficulty.",
    )
    parser.add_argument(
        "--save_runs_csv",
        action="store_true",
        help="If set, also save the per-run CSV (useful for debugging).",
    )

    args = parser.parse_args()

    if args.out_dir is None:
        out_dir = os.path.abspath(args.run_dirs[0])
    else:
        out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df_runs = build_runs_dataframe(args.run_dirs, difficulty_mode=args.difficulty_mode)
    if df_runs.empty:
        print("[!] No runs were loaded. Check --run_dirs paths.")
        return

    if args.save_runs_csv:
        runs_csv = os.path.join(out_dir, "multitask_linkpred_summary_runs.csv")
        df_runs.to_csv(runs_csv, index=False)
        print(f"[*] Saved per-run CSV: {runs_csv}")

    if args.aggregate:
        df_plot = build_aggregated_dataframe(df_runs)
        agg_csv = os.path.join(out_dir, "multitask_linkpred_summary_agg.csv")
        df_plot.to_csv(agg_csv, index=False)
        print(f"[*] Saved aggregated CSV: {agg_csv}")
    else:
        df_plot = df_runs
        csv_path = os.path.join(out_dir, "multitask_linkpred_summary.csv")
        df_plot.to_csv(csv_path, index=False)
        print(f"[*] Saved summary CSV: {csv_path}")

    plot_bar_hvt_metrics(df_plot, out_dir)
    plot_bar_linkpred_layer(df_plot, out_dir, layer="finance")
    plot_bar_linkpred_layer(df_plot, out_dir, layer="communication")
    plot_difficulty_vs_performance_curve(df_plot, out_dir)


if __name__ == "__main__":
    main()