"""
Report generation for ToolFailBench results.
"""
import json
from pathlib import Path
from typing import List, Dict
from .metrics import compute_all_metrics, compute_metrics_by_domain, compute_metrics_by_mode


def generate_summary_table(results: List[Dict], model_name: str = "unknown") -> str:
    metrics = compute_all_metrics(results)
    lines = [
        f"{'='*60}",
        f"  ToolFailBench Results: {model_name}",
        f"{'='*60}",
        f"  Total tasks:            {metrics['total_tasks']} "
        f"({metrics['tool_required_tasks']} tool-required, {metrics['ctrl_tasks']} CTRL)",
        f"",
        f"  --- Tool-Required Tasks ---",
        f"  Tool-Skip Rate (TSR):   {metrics['tsr']:.2%}",
        f"  Result-Ignore Rate:     {metrics['rir']:.2%}",
        f"  Output-Fabrication:     {metrics['ofr']:.2%}",
        f"  Clean Tool-Use Rate:    {metrics['ctur']:.2%}",
        f"",
        f"  --- CTRL Tasks ---",
        f"  Unnecessary Tool Use:   {metrics['utr']:.2%}",
        f"  CTRL Accuracy:          {metrics['ctrl_accuracy']:.2%}",
        f"{'='*60}",
        f"  Distribution: {metrics['distribution']}",
    ]
    return "\n".join(lines)


def generate_domain_breakdown(results: List[Dict]) -> str:
    by_domain = compute_metrics_by_domain(results)
    lines = ["Domain Breakdown:", "-" * 50]
    for domain, metrics in by_domain.items():
        lines.append(
            f"  {domain}: TSR={metrics['tsr']:.2%} RIR={metrics['rir']:.2%} "
            f"OFR={metrics['ofr']:.2%} CTUR={metrics['ctur']:.2%} "
            f"UTR={metrics['utr']:.2%}"
        )
    return "\n".join(lines)


def save_results_json(results: List[Dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def plot_failure_mode_heatmap(results: List[Dict], output_path: str):
    """TODO: Generate heatmap of failure modes across models/domains using seaborn."""
    pass
