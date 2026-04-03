"""
Metric computation for ToolFailBench.
"""
from typing import List, Dict
from collections import Counter


def compute_tsr(results: List[Dict]) -> float:
    """Tool-Skip Rate."""
    requiring = [r for r in results if r["task"]["evaluation_criteria"]["tool_must_be_called"]]
    if not requiring:
        return 0.0
    skipped = [r for r in requiring if r["classification"] == "tool_skip"]
    return len(skipped) / len(requiring)


def compute_rir(results: List[Dict]) -> float:
    """Result-Ignore Rate."""
    called = [r for r in results if r["classification"] not in ("tool_skip",)]
    if not called:
        return 0.0
    ignored = [r for r in called if r["classification"] == "result_ignore"]
    return len(ignored) / len(called)


def compute_ofr(results: List[Dict]) -> float:
    """Output-Fabrication Rate."""
    called = [r for r in results if r["classification"] not in ("tool_skip",)]
    if not called:
        return 0.0
    fabricated = [r for r in called if r["classification"] == "output_fabrication"]
    return len(fabricated) / len(called)


def compute_ctur(results: List[Dict]) -> float:
    """Clean Tool-Use Rate."""
    if not results:
        return 0.0
    correct = [r for r in results if r["classification"] == "correct"]
    return len(correct) / len(results)


def compute_all_metrics(results: List[Dict]) -> Dict:
    return {
        "tsr": round(compute_tsr(results), 4),
        "rir": round(compute_rir(results), 4),
        "ofr": round(compute_ofr(results), 4),
        "ctur": round(compute_ctur(results), 4),
        "total_tasks": len(results),
        "distribution": dict(Counter(r["classification"] for r in results)),
    }


def compute_metrics_by_domain(results: List[Dict]) -> Dict:
    domains = set(r["task"]["domain"] for r in results)
    return {
        d: compute_all_metrics([r for r in results if r["task"]["domain"] == d])
        for d in sorted(domains)
    }


def compute_metrics_by_mode(results: List[Dict]) -> Dict:
    modes = set(r["task"]["target_failure_mode"] for r in results)
    return {
        m: compute_all_metrics([r for r in results if r["task"]["target_failure_mode"] == m])
        for m in sorted(modes)
    }
