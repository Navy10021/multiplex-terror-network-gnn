from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional, Any, List

from src.data.basic_diagnostics_v3 import (
    activity_observability_diagnostics,
    basic_stats,
    copy_provenance_diagnostics,
    cross_layer_correlations,
    degree_distributions,
    edge_attribute_diagnostics,
    edge_noise_diagnostics,
    ensure_dir,
    event_burstiness_diagnostics,
    false_edge_observability_diagnostics,
    label_diagnostics,
    load_multiplex,
    operation_cell_purity,
    print_meta,
    layer_overlap_diagnostics,
    rolewise_degree_stats,
)
from src.data.build_pyg_dataset_v3 import build_pyg_data
from src.analysis.plot_multitask_linkpred_summary import build_runs_dataframe
from src.data.multiplex_generator_v3 import (
    generate_multiplex_with_config,
    generate_with_ontology_constraints,
    load_generator_config,
)
from src.ontology.validator import (
    OntologyValidationError,
    validate_manifest_dict_with_ontology,
    write_ontology_report,
)
from src.utils.exp_logging import build_artifact_dir, collect_run_metadata, write_run_metadata
from src.validation.schema import Manifest, validate_manifest_dict




def _build_node_explanations(manifest: Dict[str, Any], ontology_report: Dict[str, Any], top_k: int = 25) -> List[Dict[str, Any]]:
    nodes = manifest.get("nodes", []) if isinstance(manifest.get("nodes"), list) else []
    layers = manifest.get("layers", {}) if isinstance(manifest.get("layers"), dict) else {}
    violations = ontology_report.get("violations", []) if isinstance(ontology_report.get("violations"), list) else []

    degree: Dict[int, int] = {}
    neighbors: Dict[int, set] = {}
    for layer_obj in layers.values():
        if not isinstance(layer_obj, dict):
            continue
        for e in (layer_obj.get("edges", []) or []):
            try:
                u = int(e.get("source"))
                v = int(e.get("target"))
            except Exception:
                continue
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
            neighbors.setdefault(u, set()).add(v)
            neighbors.setdefault(v, set()).add(u)

    node_viol: Dict[int, List[Dict[str, Any]]] = {}
    for v in violations:
        affected = v.get("affected_ids", []) if isinstance(v, dict) else []
        if not isinstance(affected, list):
            continue
        for aid in affected:
            try:
                nid = int(aid)
            except Exception:
                continue
            node_viol.setdefault(nid, []).append({
                "check": v.get("check"),
                "rule_id": v.get("rule_id"),
                "severity": v.get("severity", "error"),
                "message": v.get("message", ""),
            })

    ranked = sorted(nodes, key=lambda n: degree.get(int(n.get("id", n.get("node_id", -1))), 0), reverse=True)
    selected = ranked[:max(1, int(top_k))]
    out: List[Dict[str, Any]] = []
    for n in selected:
        nid = int(n.get("id", n.get("node_id", -1)))
        neigh = sorted(list(neighbors.get(nid, set())))[:15]
        nv = node_viol.get(nid, [])
        out.append({
            "target": nid,
            "task": "hvt_risk_screening",
            "model_evidence": {
                "proxy_signal": "local_degree",
                "local_degree": int(degree.get(nid, 0)),
                "top_neighbors": neigh,
            },
            "ontology_evidence": {
                "conforms_global": bool(ontology_report.get("conforms", False)),
                "violations_for_target": nv,
                "violation_count_for_target": len(nv),
            },
            "conflict_flags": {
                "rule_violation_for_target": bool(nv),
                "global_nonconformance": not bool(ontology_report.get("conforms", False)),
            },
        })
    return out


def _write_explanations(run_dir: str, manifest: Dict[str, Any], ontology_report: Dict[str, Any], top_k: int = 25) -> str:
    out_dir = os.path.join(run_dir, "explanations")
    os.makedirs(out_dir, exist_ok=True)
    exps = _build_node_explanations(manifest, ontology_report, top_k=top_k)
    out_path = os.path.join(out_dir, "ontology_explanations.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"explanations": exps}, f, indent=2)
    return out_path


def _run_reporting_summary(run_dir: str) -> str:
    out_dir = os.path.join(run_dir, "reporting_summary")
    os.makedirs(out_dir, exist_ok=True)
    df = build_runs_dataframe([run_dir], difficulty_mode="auto")
    csv_path = os.path.join(out_dir, "multitask_linkpred_summary.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def _write_manifest(manifest: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _run_diagnostics(manifest_path: str, out_dir: str) -> None:
    ensure_dir(out_dir)
    mani, nodes, labels, layers, df_events = load_multiplex(manifest_path)
    print_meta(mani)
    basic_stats(nodes, layers, out_dir=os.path.join(out_dir, "1_basic_stats"))
    degree_distributions(layers, out_dir=os.path.join(out_dir, "2_degree_dists"))
    rolewise_degree_stats(nodes, layers, out_dir=os.path.join(out_dir, "3_rolewise"))
    cross_layer_correlations(nodes, layers, out_dir=os.path.join(out_dir, "4_cross_layer"))
    layer_overlap_dir = os.path.join(out_dir, "4b_overlap")
    ensure_dir(layer_overlap_dir)
    layer_overlap_diagnostics(layers, out_dir=layer_overlap_dir)
    edge_noise_diagnostics(layers, out_dir=os.path.join(out_dir, "6_edge_noise"))
    false_edge_observability_diagnostics(labels, layers, out_dir=os.path.join(out_dir, "6b_false_edge_obs"))
    copy_provenance_diagnostics(mani, layers, out_dir=os.path.join(out_dir, "6c_copy"))
    edge_attribute_diagnostics(layers, out_dir=os.path.join(out_dir, "7_edge_attr"))
    operation_cell_purity(labels, out_dir=os.path.join(out_dir, "8_op_cells"))
    event_burstiness_diagnostics(df_events, out_dir=os.path.join(out_dir, "9_events_burst"))
    label_diagnostics(labels, out_dir=os.path.join(out_dir, "5_labels"))


def _write_dataset_card(manifest: Manifest, run_dir: str, dataset_path: Optional[str], diagnostics_dir: Optional[str]) -> str:
    lines = ["# DATASET_CARD", ""]
    lines.append(f"- generator: `{manifest.meta.generator}`")
    lines.append(f"- seed: `{manifest.meta.seed}`")
    lines.append(f"- num_nodes: `{manifest.meta.num_nodes}`")
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
        lines.append(
            f"| {lname} | {total} | {false_rate:.3f} | {copied_rate:.3f} |"
        )

    lines.append("")
    lines.append("## Artifacts")
    if dataset_path:
        lines.append(f"- PyG dataset: `{os.path.abspath(dataset_path)}`")
    if diagnostics_dir:
        lines.append(f"- Diagnostics: `{os.path.abspath(diagnostics_dir)}`")

    card_path = os.path.join(run_dir, "DATASET_CARD.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return card_path



def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end runner: generate → build → diagnostics")
    parser.add_argument("--config", type=str, required=True, help="Generator config JSON")
    parser.add_argument("--size", type=int, default=1500, help="Number of nodes")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--out_root", type=str, default="results", help="Root folder for runs")
    parser.add_argument("--skip_diagnostics", action="store_true", help="Skip diagnostics stage")
    parser.add_argument("--skip_build", action="store_true", help="Skip PyG dataset build stage")
    parser.add_argument("--ontology", type=str, default="ontology/terror.ttl", help="Ontology TTL path")
    parser.add_argument("--shapes", type=str, default="ontology/constraints.shacl.ttl", help="SHACL constraints path")
    parser.add_argument("--no_ontology_strict", action="store_true", help="Do not fail run on ontology-rule violations")
    parser.add_argument("--ontology_constrained", action="store_true", help="Retry generation with shifted seeds until ontology validation conforms")
    parser.add_argument("--ontology_max_retries", type=int, default=3, help="Max generation attempts in ontology_constrained mode")
    parser.add_argument("--ontology_retry_seed_stride", type=int, default=1, help="Seed increment per attempt in ontology_constrained mode")
    parser.add_argument("--run_reporting_summary", action="store_true", help="Write ontology-aware reporting summary CSV for this run")
    parser.add_argument("--write_explanations", action="store_true", help="Write node-level ontology explanation artifacts")
    parser.add_argument("--explanations_top_k", type=int, default=25, help="Number of top-degree nodes to include in explanations")
    args = parser.parse_args()

    run_dir = build_artifact_dir(args.out_root, args.config, args.seed, prefix="run")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[*] Run directory: {run_dir}")

    cfg = load_generator_config(args.config, size=args.size, seed=args.seed)

    ontology_telemetry = {"mode": "single_pass", "attempts": 1, "failed_attempts": 0}
    if args.ontology_constrained:
        manifest, ontology_report, ontology_telemetry = generate_with_ontology_constraints(
            cfg=cfg,
            ontology_path=args.ontology,
            shapes_path=args.shapes,
            max_retries=args.ontology_max_retries,
            retry_seed_stride=args.ontology_retry_seed_stride,
        )
        if ontology_report.get("conforms", False):
            print(f"[*] Ontology-constrained generation passed at attempt {ontology_telemetry.get('successful_attempt')}")
        else:
            msg = "ontology-constrained generation exhausted retries without conformance"
            if not args.no_ontology_strict:
                raise OntologyValidationError(msg)
            print(f"[!] Ontology validation warning (strict disabled): {msg}")
    else:
        manifest = generate_multiplex_with_config(cfg)
        ontology_report = validate_manifest_dict_with_ontology(
            manifest_dict=manifest,
            ontology_path=args.ontology,
            shapes_path=args.shapes,
            strict=False,
        )
        if ontology_report.get("conforms", False):
            print("[*] Ontology validation passed")
        elif not args.no_ontology_strict:
            raise OntologyValidationError("ontology validation failed: " + "; ".join(ontology_report.get("errors", [])) + " | hint: use --no_ontology_strict to continue with reports only, or --ontology_constrained to retry generation")
        else:
            print("[!] Ontology validation warning (strict disabled)")

    ontology_report["generation_telemetry"] = ontology_telemetry
    manifest_model = validate_manifest_dict(manifest)

    ontology_report_path = os.path.join(run_dir, "ontology_validation_report.json")
    write_ontology_report(ontology_report, ontology_report_path)

    manifest_path = os.path.join(run_dir, "multiplex.json")
    _write_manifest(manifest, manifest_path)
    print(f"[*] Saved manifest: {manifest_path}")

    dataset_path: Optional[str] = None
    if not args.skip_build:
        dataset_path = os.path.join(run_dir, "pyg_data.pt")
        data = build_pyg_data(manifest_path)
        import torch

        torch.save(data, dataset_path)
        print(f"[*] Saved PyG dataset: {dataset_path}")

    diagnostics_dir: Optional[str] = None
    if not args.skip_diagnostics:
        diagnostics_dir = os.path.join(run_dir, "diagnostics")
        _run_diagnostics(manifest_path, diagnostics_dir)
        print(f"[*] Saved diagnostics under: {diagnostics_dir}")

    explanation_path: Optional[str] = None
    if args.write_explanations:
        explanation_path = _write_explanations(run_dir, manifest, ontology_report, top_k=args.explanations_top_k)
        print(f"[*] Wrote ontology explanations: {explanation_path}")

    reporting_summary_csv: Optional[str] = None
    if args.run_reporting_summary:
        reporting_summary_csv = _run_reporting_summary(run_dir)
        print(f"[*] Wrote reporting summary: {reporting_summary_csv}")

    metadata = collect_run_metadata(
        out_dir=run_dir,
        config_path=args.config,
        seed=args.seed,
        extra={
            "manifest_path": os.path.abspath(manifest_path),
            "dataset_path": os.path.abspath(dataset_path) if dataset_path else None,
            "diagnostics_dir": os.path.abspath(diagnostics_dir) if diagnostics_dir else None,
            "ontology_report": os.path.abspath(ontology_report_path),
            "ontology_conforms": bool(ontology_report.get("conforms", False)),
            "ontology_generation_telemetry": ontology_telemetry,
            "ontology_explanations": os.path.abspath(explanation_path) if explanation_path else None,
            "reporting_summary_csv": os.path.abspath(reporting_summary_csv) if reporting_summary_csv else None,
        },
    )
    meta_path = write_run_metadata(run_dir, metadata)
    print(f"[*] Logged metadata: {meta_path}")

    card_path = _write_dataset_card(manifest_model, run_dir, dataset_path, diagnostics_dir)
    print(f"[*] Wrote dataset card: {card_path}")


if __name__ == "__main__":
    main()
