import argparse
import json
import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_get(d: Dict[str, Any], *keys, default=None):
    """
    Helper to safely retrieve values from nested dictionaries.
    Returns default if any key is missing.
    """
    cur = d
    for k in keys:
        if cur is None:
            return default
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, None)
    return cur if cur is not None else default


def load_multitask_metrics(run_dir: str) -> Dict[str, Any]:
    """
    Load multitask_metrics.json in run_dir and extract key test metrics.
    - HVT: tuned threshold (hvt_threshold_tuned) test f1/auc
    - Role/Importance: fixed_threshold.test values
    """
    mt_path = os.path.join(run_dir, "multitask_metrics.json")
    if not os.path.exists(mt_path):
        print(f"[!] multitask_metrics.json not found in {run_dir}")
        return {}

    with open(mt_path, "r") as f:
        mt = json.load(f)

    # 1) HVT: threshold-tuned results (acc/f1/auc)
    tuned_test = safe_get(mt, "hvt_threshold_tuned", "test", default={})
    hvt_f1 = tuned_test.get("f1", np.nan)
    hvt_auc = tuned_test.get("auc", np.nan)

    # 2) Role / Importance: multitask metrics at threshold=0.5
    fixed_test = safe_get(mt, "fixed_threshold", "test", default={})
    role_f1_macro = fixed_test.get("role_f1_macro", np.nan)
    imp_r2 = fixed_test.get("imp_r2", np.nan)

    metrics = {
        "hvt_f1": float(hvt_f1) if hvt_f1 is not None else np.nan,
        "hvt_auc": float(hvt_auc) if hvt_auc is not None else np.nan,
        "role_f1_macro": float(role_f1_macro) if role_f1_macro is not None else np.nan,
        "imp_r2": float(imp_r2) if imp_r2 is not None else np.nan,
    }
    return metrics



def load_linkpred_metrics(run_dir: str, layer: str, neg_mode: str) -> Dict[str, Any]:
    """
    Load linkpred_{layer}_{neg_mode}.json in run_dir and extract
    test_at_best_val_auc AUC/AP.
    """
    fname = f"linkpred_{layer}_{neg_mode}.json"
    metrics_path = os.path.join(run_dir, fname)
    if not os.path.exists(metrics_path):
        print(f"[!] {fname} not found in {run_dir}")
        return {"auc": np.nan, "ap": np.nan}

    with open(metrics_path, "r") as f:
        obj = json.load(f)

    test = obj.get("test_at_best_val_auc", {}) or {}

    def get_float(key: str):
        v = test.get(key, np.nan)
        try:
            return float(v)
        except Exception:
            return np.nan

    return {
        "auc": get_float("auc"),
        "ap": get_float("ap"),
    }


def load_generator_config(run_dir: str) -> Dict[str, Any]:
    """
    Read multiplex.json(meta.config) in run_dir to capture generator settings
    such as finance_structure_strength, comm_structure_strength, comm_randomness, etc.
    Fill NaN when missing.
    """
    manifest_path = os.path.join(run_dir, "multiplex.json")
    if not os.path.exists(manifest_path):
        print(f"[!] multiplex.json not found in {run_dir}")
        return {
            "finance_structure_strength": np.nan,
            "comm_structure_strength": np.nan,
            "comm_randomness": np.nan,
            "hvt_ratio": np.nan,
        }

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    cfg = safe_get(manifest, "meta", "config", default={})
    return {
        "finance_structure_strength": cfg.get("finance_structure_strength", np.nan),
        "comm_structure_strength": cfg.get("comm_structure_strength", np.nan),
        "comm_randomness": cfg.get("comm_randomness", np.nan),
        "hvt_ratio": cfg.get("hvt_ratio", np.nan),
    }


def infer_difficulty_label(run_name: str) -> str:
    """
    Infer difficulty label from folder name.
    Example: multiplex_easy / multiplex_baseline / multiplex_hard
    """
    name = run_name.lower()
    if "easy" in name:
        return "easy"
    if "baseline" in name:
        return "baseline"
    if "hard" in name:
        return "hard"
    return run_name  # keep as-is if no rule matches


def difficulty_sort_key(label: str) -> int:
    """
    Define sorting order for difficulty labels.
    easy -> baseline -> hard -> everything else
    """
    l = label.lower()
    if "easy" in l:
        return 0
    if "baseline" in l:
        return 1
    if "hard" in l:
        return 2
    return 99


def build_summary_dataframe(run_dirs: List[str]) -> pd.DataFrame:
    """
    Combine multitask + link prediction metrics + generator settings
    from multiple run directories into one DataFrame.
    """
    rows = []

    for run_dir in run_dirs:
        run_dir = os.path.abspath(run_dir)
        run_name = os.path.basename(run_dir.rstrip("/"))
        difficulty = infer_difficulty_label(run_name)

        print(f"[*] Loading metrics from {run_dir} (difficulty={difficulty})")

        mt_metrics = load_multitask_metrics(run_dir)
        gen_cfg = load_generator_config(run_dir)

        # layer × neg_mode combinations
        lp_fin_uniform = load_linkpred_metrics(run_dir, layer="finance", neg_mode="uniform")
        lp_fin_hard = load_linkpred_metrics(run_dir, layer="finance", neg_mode="hard_region")
        lp_comm_uniform = load_linkpred_metrics(run_dir, layer="communication", neg_mode="uniform")
        lp_comm_hard = load_linkpred_metrics(run_dir, layer="communication", neg_mode="hard_region")

        row = {
            "run_dir": run_dir,
            "run_name": run_name,
            "difficulty": difficulty,
            # generator config
            **gen_cfg,
            # multitask
            "hvt_f1": mt_metrics.get("hvt_f1", np.nan),
            "hvt_auc": mt_metrics.get("hvt_auc", np.nan),
            "role_f1_macro": mt_metrics.get("role_f1_macro", np.nan),
            "imp_r2": mt_metrics.get("imp_r2", np.nan),
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
    # sort by difficulty order
    df["difficulty_order"] = df["difficulty"].apply(difficulty_sort_key)
    df = df.sort_values(["difficulty_order", "run_name"]).reset_index(drop=True)
    return df


# --------------------------
# Plot functions
# --------------------------

def plot_bar_hvt_metrics(df: pd.DataFrame, out_dir: str):
    """
    Bar plots for HVT F1 / HVT AUC / Role F1 / Importance R2 by difficulty.
    (Requirement #1 – simple bar plots)
    """
    if df.empty:
        print("[!] Empty DataFrame, skip plot_bar_hvt_metrics")
        return

    labels = df["difficulty"].tolist()
    x = np.arange(len(labels))

    # 1) HVT F1
    plt.figure()
    plt.bar(x, df["hvt_f1"].values)
    plt.xticks(x, labels)
    plt.ylabel("HVT F1 (threshold tuned)")
    plt.title("HVT F1 by difficulty")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "hvt_f1_by_difficulty.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[*] Saved plot: {out_path}")

    # 2) HVT AUC
    plt.figure()
    plt.bar(x, df["hvt_auc"].values)
    plt.xticks(x, labels)
    plt.ylabel("HVT AUC (threshold tuned)")
    plt.title("HVT AUC by difficulty")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "hvt_auc_by_difficulty.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[*] Saved plot: {out_path}")

    # 3) Role macro-F1
    plt.figure()
    plt.bar(x, df["role_f1_macro"].values)
    plt.xticks(x, labels)
    plt.ylabel("Role macro-F1 (test)")
    plt.title("Role classification macro-F1 by difficulty")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "role_f1_macro_by_difficulty.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[*] Saved plot: {out_path}")

    # 4) Importance R^2
    plt.figure()
    plt.bar(x, df["imp_r2"].values)
    plt.xticks(x, labels)
    plt.ylabel("Importance $R^2$ (test)")
    plt.title("Importance regression $R^2$ by difficulty")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "importance_r2_by_difficulty.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[*] Saved plot: {out_path}")


def plot_bar_linkpred_layer(df: pd.DataFrame, out_dir: str, layer: str):
    """
    Grouped bar plot of uniform vs hard_region AUC for each layer
    (Finance / Communication).
    """
    if df.empty:
        print("[!] Empty DataFrame, skip plot_bar_linkpred_layer")
        return

    labels = df["difficulty"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    if layer == "finance":
        auc_uniform = df["finance_auc_uniform"].values
        auc_hard = df["finance_auc_hard_region"].values
        title = "Finance link prediction AUC by difficulty"
        fname = "finance_link_auc_by_difficulty.png"
    elif layer == "communication":
        auc_uniform = df["comm_auc_uniform"].values
        auc_hard = df["comm_auc_hard_region"].values
        title = "Communication link prediction AUC by difficulty"
        fname = "communication_link_auc_by_difficulty.png"
    else:
        raise ValueError(f"Unknown layer: {layer}")

    plt.figure()
    plt.bar(x - width / 2, auc_uniform, width, label="uniform")
    plt.bar(x + width / 2, auc_hard, width, label="hard_region")
    plt.xticks(x, labels)
    plt.ylabel("Test AUC")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path)
    plt.close()
    print(f"[*] Saved plot: {out_path}")


def plot_difficulty_vs_performance_curve(df: pd.DataFrame, out_dir: str):
    """
    Requirement #2: show the following across difficulty (easy/baseline/hard)
    on a single line plot:
    - HVT F1
    - Finance link AUC (hard_region)
    - Communication link AUC (hard_region)
    """
    if df.empty:
        print("[!] Empty DataFrame, skip plot_difficulty_vs_performance_curve")
        return

    labels = df["difficulty"].tolist()
    x = np.arange(len(labels))

    plt.figure()
    plt.plot(x, df["hvt_f1"].values, marker="o", label="HVT F1 (tuned)")
    plt.plot(x, df["finance_auc_hard_region"].values, marker="o", label="Finance AUC (hard_region)")
    plt.plot(x, df["comm_auc_hard_region"].values, marker="o", label="Comm AUC (hard_region)")
    plt.xticks(x, labels)
    plt.xlabel("Difficulty")
    plt.ylabel("Metric value")
    plt.title("Difficulty vs Performance (HVT + Link prediction)")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "difficulty_vs_performance_curve.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[*] Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize multitask + link prediction metrics across difficulties."
    )
    parser.add_argument(
        "--run_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to experiment result folders (e.g., data/multiplex_easy data/multiplex_baseline data/multiplex_hard)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to save plots and CSV (default: first run_dir)",
    )

    args = parser.parse_args()

    if args.out_dir is None:
        out_dir = os.path.abspath(args.run_dirs[0])
    else:
        out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 1) aggregate metrics from multiple run_dirs into a DataFrame
    df = build_summary_dataframe(args.run_dirs)
    if df.empty:
        print("[!] No data loaded. Check run_dirs.")
        return

    # save as CSV (tabular results)
    csv_path = os.path.join(out_dir, "multitask_linkpred_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"[*] Saved summary CSV: {csv_path}")

    # Requirement 1: bar plots (HVT/Role/Importance + layer-wise link AUC)
    plot_bar_hvt_metrics(df, out_dir)
    plot_bar_linkpred_layer(df, out_dir, layer="finance")
    plot_bar_linkpred_layer(df, out_dir, layer="communication")

    # Requirement 2: difficulty (easy/baseline/hard) vs performance curves
    plot_difficulty_vs_performance_curve(df, out_dir)


if __name__ == "__main__":
    main()
