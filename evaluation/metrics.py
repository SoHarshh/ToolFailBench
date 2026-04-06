"""
Metric computation for ToolFailBench.

Non-CTRL metrics (TSR, RIR, OFR, CTUR) are computed only over tasks where
tool_must_be_called=True, so CTRL tasks don't pollute failure mode rates.

CTRL metrics (UTR, CTRL accuracy) are computed separately over CTRL tasks only.
"""
from typing import List, Dict
from collections import Counter


def _non_ctrl(results: List[Dict]) -> List[Dict]:
    """Tasks where a tool call was required."""
    return [r for r in results if r["task"]["evaluation_criteria"]["tool_must_be_called"]]


def _ctrl(results: List[Dict]) -> List[Dict]:
    """Tasks where no tool call was required (CTRL tasks)."""
    return [r for r in results if not r["task"]["evaluation_criteria"]["tool_must_be_called"]]


def compute_tsr(results: List[Dict]) -> float:
    """Tool-Skip Rate — fraction of tool-required tasks where agent skipped the tool."""
    subset = _non_ctrl(results)
    if not subset:
        return 0.0
    return sum(1 for r in subset if r["classification"] == "tool_skip") / len(subset)


def compute_rir(results: List[Dict]) -> float:
    """Result-Ignore Rate — of tool-required tasks where tool was called, fraction where result was ignored."""
    called = [r for r in _non_ctrl(results) if r["classification"] != "tool_skip"]
    if not called:
        return 0.0
    return sum(1 for r in called if r["classification"] == "result_ignore") / len(called)


def compute_ofr(results: List[Dict]) -> float:
    """Output-Fabrication Rate — of tool-required tasks where tool was called, fraction where output was fabricated."""
    called = [r for r in _non_ctrl(results) if r["classification"] != "tool_skip"]
    if not called:
        return 0.0
    return sum(1 for r in called if r["classification"] == "output_fabrication") / len(called)


def compute_ctur(results: List[Dict]) -> float:
    """Clean Tool-Use Rate — fraction of tool-required tasks fully correct."""
    subset = _non_ctrl(results)
    if not subset:
        return 0.0
    return sum(1 for r in subset if r["classification"] == "correct") / len(subset)


def compute_utr(results: List[Dict]) -> float:
    """Unnecessary Tool Use Rate — fraction of CTRL tasks where agent called a tool anyway."""
    subset = _ctrl(results)
    if not subset:
        return 0.0
    return sum(1 for r in subset if r["classification"] == "unnecessary_tool_use") / len(subset)


def compute_ctrl_accuracy(results: List[Dict]) -> float:
    """CTRL Accuracy — fraction of CTRL tasks answered correctly without tool use."""
    subset = _ctrl(results)
    if not subset:
        return 0.0
    return sum(1 for r in subset if r["classification"] == "correct") / len(subset)


def compute_all_metrics(results: List[Dict]) -> Dict:
    return {
        # Tool-required task metrics
        "tsr": round(compute_tsr(results), 4),
        "rir": round(compute_rir(results), 4),
        "ofr": round(compute_ofr(results), 4),
        "ctur": round(compute_ctur(results), 4),
        # CTRL task metrics
        "utr": round(compute_utr(results), 4),
        "ctrl_accuracy": round(compute_ctrl_accuracy(results), 4),
        # Counts
        "total_tasks": len(results),
        "tool_required_tasks": len(_non_ctrl(results)),
        "ctrl_tasks": len(_ctrl(results)),
        "distribution": dict(Counter(r["classification"] for r in results)),
    }


def compute_metrics_by_domain(results: List[Dict]) -> Dict:
    domains = sorted(set(r["task"]["domain"] for r in results))
    return {
        d: compute_all_metrics([r for r in results if r["task"]["domain"] == d])
        for d in domains
    }


def compute_metrics_by_mode(results: List[Dict]) -> Dict:
    modes = sorted(set(r["task"]["target_failure_mode"] for r in results))
    return {
        m: compute_all_metrics([r for r in results if r["task"]["target_failure_mode"] == m])
        for m in modes
    }
