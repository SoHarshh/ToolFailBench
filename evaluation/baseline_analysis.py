"""
RIR–Baseline Correlation Analysis for ToolFailBench.

Compares model answers WITH tools (eval) against answers WITHOUT tools
(parametric baseline) to test the hypothesis:

    "Models that ignore tool results fall back to parametric memory."

If RIR answers ≈ baseline answers, it proves models regress to training
data even after calling the tool and receiving correct results.

Usage:
    python evaluation/baseline_analysis.py \
        --eval-file results/llama3.1-8b_20260413_022850.json \
        --baseline-file results/baselines/llama3.1-8b_baseline_20260413_085015.json

    python evaluation/baseline_analysis.py --all
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalize text for comparison: strip special tokens, lower, collapse whitespace."""
    # Gemma4 thought channel tokens
    text = re.sub(r'<\|channel>thought\s*\n?<channel\|>', '', text)
    # Qwen/deepseek think blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'</think>\s*', '', text)
    # Strip markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'[`#*_]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def _tokenize(text: str) -> set[str]:
    """Simple whitespace + punctuation tokenizer."""
    return set(re.findall(r'[a-z0-9]+(?:\.[0-9]+)?', text.lower()))


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Token-level Jaccard similarity between two texts."""
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a and not tokens_b:
        return 1.0  # both empty
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def containment_similarity(text_a: str, text_b: str) -> float:
    """
    Asymmetric containment: what fraction of text_a's tokens appear in text_b?
    Useful when text_b is much longer (baseline answers tend to be verbose).
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


def ngram_overlap(text_a: str, text_b: str, n: int = 3) -> float:
    """Character n-gram overlap (Jaccard over character n-grams)."""
    a = _clean_text(text_a)
    b = _clean_text(text_b)
    if len(a) < n and len(b) < n:
        return 1.0 if a == b else 0.0
    if len(a) < n or len(b) < n:
        return 0.0
    ngrams_a = set(a[i:i+n] for i in range(len(a) - n + 1))
    ngrams_b = set(b[i:i+n] for i in range(len(b) - n + 1))
    intersection = ngrams_a & ngrams_b
    union = ngrams_a | ngrams_b
    return len(intersection) / len(union) if union else 0.0


def value_match(answer: str, values: list) -> dict:
    """
    Check which specific values from a list appear in the answer text.
    Returns dict with match details.
    """
    answer_lower = answer.lower()
    matches = []
    for v in values:
        v_str = str(v).lower()
        matches.append({
            "value": str(v),
            "found": v_str in answer_lower,
        })
    matched = sum(1 for m in matches if m["found"])
    return {
        "matches": matches,
        "matched_count": matched,
        "total": len(values),
        "match_rate": matched / len(values) if values else 0.0,
    }


# ---------------------------------------------------------------------------
# Core analysis: per-task correlation
# ---------------------------------------------------------------------------

def analyze_task(eval_result: dict, baseline: dict) -> dict:
    """
    Analyze one task: compare eval answer vs baseline answer,
    and check both against ground truth and mock return values.
    """
    task = eval_result["task"]
    eval_answer = eval_result["agent_answer"]
    baseline_answer = baseline["parametric_answer"]
    classification = eval_result["classification"]

    eval_clean = _clean_text(eval_answer)
    baseline_clean = _clean_text(baseline_answer)
    eval_empty = len(eval_clean.strip()) == 0
    baseline_empty = len(baseline_clean.strip()) == 0

    # Similarity between eval and baseline answers
    sim = {}
    if eval_empty or baseline_empty:
        sim = {"jaccard": 0.0, "containment": 0.0, "ngram3": 0.0}
    else:
        sim = {
            "jaccard": round(jaccard_similarity(eval_clean, baseline_clean), 4),
            "containment": round(containment_similarity(eval_clean, baseline_clean), 4),
            "ngram3": round(ngram_overlap(eval_clean, baseline_clean, n=3), 4),
        }

    # Ground truth check: does the answer contain required values?
    gt = task["ground_truth"]
    gt_values = gt["answer_must_contain"]
    eval_gt_match = value_match(eval_answer, gt_values)
    baseline_gt_match = value_match(baseline_answer, gt_values)

    # Mock return values: extract key leaf values from mock_tool_return
    mock_return = task.get("mock_tool_return", {})
    mock_values = _extract_key_values(mock_return)
    eval_mock_match = value_match(eval_answer, mock_values) if mock_values else None
    baseline_mock_match = value_match(baseline_answer, mock_values) if mock_values else None

    # Parametric trap detection: does the task have expected_parametric_answer?
    expected_parametric = task.get("metadata", {}).get("expected_parametric_answer", "")

    return {
        "task_id": task["task_id"],
        "domain": task["domain"],
        "classification": classification,
        "target_failure_mode": task["target_failure_mode"],
        "conflict_type": task["conflict_type"],
        "tool_must_be_called": task["evaluation_criteria"]["tool_must_be_called"],
        "eval_answer_empty": eval_empty,
        "baseline_answer_empty": baseline_empty,
        "similarity": sim,
        "eval_gt_match": eval_gt_match,
        "baseline_gt_match": baseline_gt_match,
        "eval_mock_match": eval_mock_match,
        "baseline_mock_match": baseline_mock_match,
        "expected_parametric": expected_parametric,
        "eval_answer_preview": eval_answer[:200],
        "baseline_answer_preview": baseline_answer[:200],
    }


def _extract_key_values(obj, values=None, depth=0) -> list:
    """Extract unique leaf values from mock return, filtering out noisy/generic ones."""
    if values is None:
        values = []
    if depth > 5:
        return values
    if isinstance(obj, dict):
        for k, v in obj.items():
            # Skip keys that are generic metadata
            if k in ("timestamp", "source", "api_version", "request_id"):
                continue
            _extract_key_values(v, values, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            _extract_key_values(v, values, depth + 1)
    elif isinstance(obj, (int, float)):
        # Only include numbers that are distinctive (not 0, 1, etc.)
        if abs(obj) > 1 and obj not in values:
            values.append(obj)
    elif isinstance(obj, str) and len(obj) > 2 and obj not in ("true", "false", "null", "none"):
        if obj not in values:
            values.append(obj)
    return values


# ---------------------------------------------------------------------------
# Aggregate analysis
# ---------------------------------------------------------------------------

def run_analysis(eval_file: str, baseline_file: str) -> dict:
    """Run the full RIR–baseline correlation analysis for one model."""
    with open(eval_file) as f:
        eval_results = json.load(f)
    with open(baseline_file) as f:
        baselines = json.load(f)

    model_id = eval_results[0]["model_id"] if eval_results else "unknown"
    baseline_map = {b["task_id"]: b for b in baselines}

    # Analyze each task
    per_task = []
    skipped = 0
    for result in eval_results:
        task_id = result["task"]["task_id"]
        if task_id not in baseline_map:
            skipped += 1
            continue
        analysis = analyze_task(result, baseline_map[task_id])
        per_task.append(analysis)

    if skipped:
        print(f"  Warning: {skipped} tasks in eval had no matching baseline entry")

    # Group by classification
    by_class = defaultdict(list)
    for t in per_task:
        by_class[t["classification"]].append(t)

    # Compute aggregate similarity stats per classification
    class_stats = {}
    for cls, tasks in sorted(by_class.items()):
        non_empty = [t for t in tasks if not t["eval_answer_empty"]]
        empty = [t for t in tasks if t["eval_answer_empty"]]

        # Mean similarities (only for non-empty eval answers)
        if non_empty:
            mean_jaccard = sum(t["similarity"]["jaccard"] for t in non_empty) / len(non_empty)
            mean_containment = sum(t["similarity"]["containment"] for t in non_empty) / len(non_empty)
            mean_ngram = sum(t["similarity"]["ngram3"] for t in non_empty) / len(non_empty)
        else:
            mean_jaccard = mean_containment = mean_ngram = 0.0

        # GT match rates
        eval_gt_rates = [t["eval_gt_match"]["match_rate"] for t in tasks]
        baseline_gt_rates = [t["baseline_gt_match"]["match_rate"] for t in tasks]

        class_stats[cls] = {
            "count": len(tasks),
            "empty_eval_answers": len(empty),
            "non_empty_eval_answers": len(non_empty),
            "mean_jaccard": round(mean_jaccard, 4),
            "mean_containment": round(mean_containment, 4),
            "mean_ngram3": round(mean_ngram, 4),
            "eval_gt_match_rate": round(sum(eval_gt_rates) / len(eval_gt_rates), 4) if eval_gt_rates else 0.0,
            "baseline_gt_match_rate": round(sum(baseline_gt_rates) / len(baseline_gt_rates), 4) if baseline_gt_rates else 0.0,
        }

    # RIR-specific deep analysis
    rir_analysis = _analyze_rir_fallback(by_class.get("result_ignore", []))

    # Domain breakdown for RIR tasks
    rir_by_domain = defaultdict(list)
    for t in by_class.get("result_ignore", []):
        rir_by_domain[t["domain"]].append(t)

    domain_stats = {}
    for domain, tasks in sorted(rir_by_domain.items()):
        non_empty = [t for t in tasks if not t["eval_answer_empty"]]
        domain_stats[domain] = {
            "count": len(tasks),
            "empty": len(tasks) - len(non_empty),
            "non_empty": len(non_empty),
            "mean_jaccard": round(
                sum(t["similarity"]["jaccard"] for t in non_empty) / len(non_empty), 4
            ) if non_empty else 0.0,
        }

    return {
        "model_id": model_id,
        "eval_file": str(eval_file),
        "baseline_file": str(baseline_file),
        "total_tasks": len(per_task),
        "classification_stats": class_stats,
        "rir_fallback_analysis": rir_analysis,
        "rir_by_domain": domain_stats,
        "per_task": per_task,
    }


def _analyze_rir_fallback(rir_tasks: list) -> dict:
    """
    Deep analysis of RIR tasks: do models fall back to parametric memory?

    For each RIR task, checks:
    1. Does the eval answer look like the baseline answer? (similarity)
    2. Does the baseline answer contain the ground truth values? (parametric knowledge)
    3. Does the eval answer contain mock return values? (tool result usage)
    """
    if not rir_tasks:
        return {"count": 0, "message": "No RIR tasks to analyze"}

    empty_answer = [t for t in rir_tasks if t["eval_answer_empty"]]
    non_empty = [t for t in rir_tasks if not t["eval_answer_empty"]]

    # For non-empty RIR answers: parametric fallback evidence
    fallback_evidence = []
    for t in non_empty:
        # Does eval answer match baseline more than it matches ground truth?
        eval_baseline_sim = t["similarity"]["jaccard"]
        eval_has_gt = t["eval_gt_match"]["match_rate"]
        baseline_has_gt = t["baseline_gt_match"]["match_rate"]

        # Does eval answer contain mock values?
        eval_has_mock = t["eval_mock_match"]["match_rate"] if t["eval_mock_match"] else 0.0
        baseline_has_mock = t["baseline_mock_match"]["match_rate"] if t["baseline_mock_match"] else 0.0

        fallback_evidence.append({
            "task_id": t["task_id"],
            "domain": t["domain"],
            "eval_baseline_similarity": eval_baseline_sim,
            "eval_has_ground_truth": eval_has_gt,
            "baseline_has_ground_truth": baseline_has_gt,
            "eval_has_mock_values": eval_has_mock,
            "baseline_has_mock_values": baseline_has_mock,
            # If eval answer is similar to baseline AND doesn't contain
            # mock values → strong evidence of parametric fallback
            "parametric_fallback": eval_has_mock < 0.5 and eval_baseline_sim > 0.1,
        })

    fallback_count = sum(1 for e in fallback_evidence if e["parametric_fallback"])

    # For empty-answer RIR tasks: these are "result dropped" — the model
    # called the tool but produced no answer at all. Different from fallback.
    return {
        "count": len(rir_tasks),
        "empty_answer_count": len(empty_answer),
        "non_empty_count": len(non_empty),
        "parametric_fallback_count": fallback_count,
        "parametric_fallback_rate": round(
            fallback_count / len(non_empty), 4
        ) if non_empty else 0.0,
        "evidence": fallback_evidence,
        "summary": _generate_fallback_summary(
            len(rir_tasks), len(empty_answer), len(non_empty), fallback_count
        ),
    }


def _generate_fallback_summary(total: int, empty: int, non_empty: int, fallback: int) -> str:
    """Generate a human-readable summary of the RIR fallback analysis."""
    lines = [
        f"Of {total} RIR tasks:",
        f"  - {empty} ({empty/total:.0%}) produced empty eval answers "
        f"(tool result dropped entirely)",
    ]
    if non_empty:
        lines.append(
            f"  - {non_empty} ({non_empty/total:.0%}) produced non-empty answers"
        )
        lines.append(
            f"  - {fallback} of {non_empty} non-empty answers "
            f"({fallback/non_empty:.0%}) show parametric fallback "
            f"(low mock-value usage + baseline similarity)"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------

def print_report(analysis: dict):
    """Print a formatted report of the analysis results."""
    model = analysis["model_id"]

    print(f"\n{'='*70}")
    print(f"  RIR–Baseline Correlation Analysis: {model}")
    print(f"{'='*70}")
    print(f"  Total tasks analyzed: {analysis['total_tasks']}")

    # Classification breakdown with similarity
    print(f"\n  --- Similarity by Classification ---")
    print(f"  {'Classification':<25s} {'N':>4s} {'Empty':>5s} "
          f"{'Jaccard':>8s} {'Contain':>8s} {'3-gram':>8s} "
          f"{'Eval→GT':>8s} {'Base→GT':>8s}")
    print(f"  {'-'*25} {'----':>4s} {'-----':>5s} "
          f"{'--------':>8s} {'--------':>8s} {'--------':>8s} "
          f"{'--------':>8s} {'--------':>8s}")

    for cls, stats in sorted(analysis["classification_stats"].items()):
        print(f"  {cls:<25s} {stats['count']:4d} {stats['empty_eval_answers']:5d} "
              f"{stats['mean_jaccard']:8.4f} {stats['mean_containment']:8.4f} "
              f"{stats['mean_ngram3']:8.4f} "
              f"{stats['eval_gt_match_rate']:8.4f} {stats['baseline_gt_match_rate']:8.4f}")

    # RIR fallback analysis
    rir = analysis["rir_fallback_analysis"]
    if rir["count"] > 0:
        print(f"\n  --- RIR Parametric Fallback Analysis ---")
        print(f"  {rir['summary']}")

        if rir.get("evidence"):
            print(f"\n  --- Per-Task RIR Evidence ---")
            print(f"  {'Task':<20s} {'Domain':<12s} {'Eval↔Base':>10s} "
                  f"{'Eval→GT':>8s} {'Eval→Mock':>10s} {'Base→Mock':>10s} {'Fallback':>9s}")
            print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*9}")
            for e in rir["evidence"]:
                fb = "YES" if e["parametric_fallback"] else "no"
                print(f"  {e['task_id']:<20s} {e['domain']:<12s} "
                      f"{e['eval_baseline_similarity']:10.4f} "
                      f"{e['eval_has_ground_truth']:8.4f} "
                      f"{e['eval_has_mock_values']:10.4f} "
                      f"{e['baseline_has_mock_values']:10.4f} "
                      f"{fb:>9s}")

        # Domain breakdown
        if analysis["rir_by_domain"]:
            print(f"\n  --- RIR by Domain ---")
            print(f"  {'Domain':<15s} {'Total':>5s} {'Empty':>5s} "
                  f"{'NonEmpty':>8s} {'Jaccard':>8s}")
            for domain, stats in sorted(analysis["rir_by_domain"].items()):
                print(f"  {domain:<15s} {stats['count']:5d} {stats['empty']:5d} "
                      f"{stats['non_empty']:8d} {stats['mean_jaccard']:8.4f}")

    else:
        print(f"\n  No RIR tasks found — model either skips tools or uses them correctly.")

    print(f"\n{'='*70}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_analysis(analysis: dict, output_dir: str = "results"):
    """Generate plots for the RIR–baseline correlation analysis."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    model = analysis["model_id"]
    per_task = analysis["per_task"]
    output_path = Path(output_dir)

    # -------------------------------------------------------------------------
    # Plot 1: Eval–Baseline similarity vs Eval–GT match rate, colored by class
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    color_map = {
        "correct": "#2ecc71",
        "result_ignore": "#e74c3c",
        "tool_skip": "#3498db",
        "output_fabrication": "#e67e22",
        "unnecessary_tool_use": "#9b59b6",
        "wrong_answer": "#95a5a6",
        "other_error": "#7f8c8d",
    }

    for t in per_task:
        if t["eval_answer_empty"]:
            continue  # skip empty answers — can't compute meaningful similarity
        x = t["similarity"]["jaccard"]
        y = t["eval_gt_match"]["match_rate"]
        color = color_map.get(t["classification"], "#333333")
        ax.scatter(x, y, c=color, s=60, alpha=0.7, edgecolors="white", linewidth=0.5)

    # Legend
    handles = [
        mpatches.Patch(color=c, label=cls)
        for cls, c in color_map.items()
        if any(t["classification"] == cls and not t["eval_answer_empty"] for t in per_task)
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9)

    ax.set_xlabel("Eval ↔ Baseline Similarity (Jaccard)", fontsize=12)
    ax.set_ylabel("Eval → Ground Truth Match Rate", fontsize=12)
    ax.set_title(f"RIR–Baseline Correlation: {model}", fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Annotate quadrants
    ax.text(0.75, 0.15, "High baseline sim\nLow GT match\n→ parametric fallback",
            fontsize=8, color="#e74c3c", alpha=0.6, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeaea", alpha=0.5))
    ax.text(0.25, 0.85, "Low baseline sim\nHigh GT match\n→ used tool correctly",
            fontsize=8, color="#2ecc71", alpha=0.6, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafff0", alpha=0.5))

    fig.tight_layout()
    plot_path = output_path / f"{model}_rir_baseline_scatter.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Saved scatter plot: {plot_path}")

    # -------------------------------------------------------------------------
    # Plot 2: Mean eval–baseline similarity by classification (bar chart)
    # -------------------------------------------------------------------------
    class_stats = analysis["classification_stats"]
    classes = sorted(class_stats.keys())
    # Only include classes with non-empty answers
    classes = [c for c in classes if class_stats[c]["non_empty_eval_answers"] > 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, metric in enumerate(["mean_jaccard", "mean_containment", "mean_ngram3"]):
        ax = axes[idx]
        vals = [class_stats[c][metric] for c in classes]
        colors = [color_map.get(c, "#333") for c in classes]
        bars = ax.bar(range(len(classes)), vals, color=colors, alpha=0.8, edgecolor="white")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(metric.replace("mean_", "").replace("_", " ").title(), fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
        # Value labels on bars
        for bar, val in zip(bars, vals):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{val:.3f}", ha="center", fontsize=8)

    fig.suptitle(f"Eval–Baseline Similarity by Classification: {model}", fontsize=13)
    fig.tight_layout()
    plot_path = output_path / f"{model}_rir_baseline_bars.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Saved bar chart: {plot_path}")

    # -------------------------------------------------------------------------
    # Plot 3: RIR-specific — eval answer's mock-value usage vs baseline similarity
    # -------------------------------------------------------------------------
    rir_evidence = analysis["rir_fallback_analysis"].get("evidence", [])
    if rir_evidence:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for e in rir_evidence:
            x = e["eval_baseline_similarity"]
            y = e["eval_has_mock_values"]
            color = "#e74c3c" if e["parametric_fallback"] else "#2ecc71"
            ax.scatter(x, y, c=color, s=80, alpha=0.8, edgecolors="white", linewidth=0.5)
            ax.annotate(e["task_id"].split("-")[-1], (x, y),
                        fontsize=7, ha="center", va="bottom", alpha=0.6)

        handles = [
            mpatches.Patch(color="#e74c3c", label="Parametric fallback"),
            mpatches.Patch(color="#2ecc71", label="Used tool result"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=9)
        ax.set_xlabel("Eval ↔ Baseline Similarity (Jaccard)", fontsize=11)
        ax.set_ylabel("Eval → Mock Return Match Rate", fontsize=11)
        ax.set_title(f"RIR Tasks: Parametric Fallback Evidence ({model})", fontsize=13)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # Reference lines
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(x=0.1, color="gray", linestyle="--", alpha=0.3)

        fig.tight_layout()
        plot_path = output_path / f"{model}_rir_fallback_evidence.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Saved fallback evidence plot: {plot_path}")


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def find_eval_baseline_pairs(results_dir: str = "results") -> list[dict]:
    """Auto-discover matching eval + baseline file pairs."""
    rdir = Path(results_dir)
    baseline_dir = rdir / "baselines"

    pairs = []
    if not baseline_dir.exists():
        return pairs

    # Find baseline files
    baseline_files = {}
    for bf in baseline_dir.glob("*_baseline_*.json"):
        # Extract model_id from filename: <model_id>_baseline_<timestamp>.json
        model_id = bf.name.rsplit("_baseline_", 1)[0]
        baseline_files[model_id] = bf

    # Find matching eval files (latest per model)
    for ef in sorted(rdir.glob("*.json"), reverse=True):
        if ef.parent != rdir:
            continue
        if "baseline" in ef.name or "judge" in ef.name:
            continue
        # Extract model_id: <model_id>_<timestamp>.json
        parts = ef.name.rsplit("_202", 1)
        if len(parts) != 2:
            continue
        model_id = parts[0]
        if model_id in baseline_files and model_id not in [p["model_id"] for p in pairs]:
            pairs.append({
                "model_id": model_id,
                "eval_file": str(ef),
                "baseline_file": str(baseline_files[model_id]),
            })

    return pairs


def run_all(results_dir: str = "results", output_dir: str = "results"):
    """Run analysis on all available eval+baseline pairs."""
    pairs = find_eval_baseline_pairs(results_dir)
    if not pairs:
        print("No matching eval+baseline file pairs found.")
        return []

    print(f"Found {len(pairs)} eval+baseline pairs:")
    for p in pairs:
        print(f"  {p['model_id']}: {Path(p['eval_file']).name} + {Path(p['baseline_file']).name}")

    all_analyses = []
    for p in pairs:
        analysis = run_analysis(p["eval_file"], p["baseline_file"])
        print_report(analysis)
        plot_analysis(analysis, output_dir)
        all_analyses.append(analysis)

        # Save per-model analysis JSON
        out_path = Path(output_dir) / f"{p['model_id']}_rir_analysis.json"
        # Save without per_task details for readability (full data in separate file)
        summary = {k: v for k, v in analysis.items() if k != "per_task"}
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved analysis: {out_path}")

    # Cross-model comparison
    if len(all_analyses) > 1:
        print_cross_model_comparison(all_analyses)

    return all_analyses


def print_cross_model_comparison(analyses: list[dict]):
    """Print a side-by-side comparison across models."""
    print(f"\n{'='*70}")
    print(f"  Cross-Model RIR–Baseline Comparison")
    print(f"{'='*70}")

    print(f"  {'Model':<20s} {'RIR':>4s} {'Empty':>5s} {'NonEmp':>6s} "
          f"{'Fallback':>8s} {'Jaccard':>8s} {'Eval→GT':>8s}")
    print(f"  {'-'*20} {'----':>4s} {'-----':>5s} {'------':>6s} "
          f"{'--------':>8s} {'--------':>8s} {'--------':>8s}")

    for a in analyses:
        model = a["model_id"]
        rir = a["rir_fallback_analysis"]
        rir_stats = a["classification_stats"].get("result_ignore", {})

        rir_count = rir.get("count", 0)
        empty = rir.get("empty_answer_count", 0)
        non_empty = rir.get("non_empty_count", 0)
        fallback = rir.get("parametric_fallback_count", 0)
        jaccard = rir_stats.get("mean_jaccard", 0.0)
        eval_gt = rir_stats.get("eval_gt_match_rate", 0.0)

        fb_str = f"{fallback}/{non_empty}" if non_empty else "N/A"
        print(f"  {model:<20s} {rir_count:4d} {empty:5d} {non_empty:6d} "
              f"{fb_str:>8s} {jaccard:8.4f} {eval_gt:8.4f}")

    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RIR–Baseline Correlation Analysis for ToolFailBench"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--eval-file", type=str,
        help="Path to eval results JSON"
    )
    input_group.add_argument(
        "--all", action="store_true",
        help="Auto-discover and analyze all eval+baseline pairs"
    )

    parser.add_argument(
        "--baseline-file", type=str,
        help="Path to baseline results JSON (required with --eval-file)"
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Output directory for plots and analysis JSON"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation"
    )

    args = parser.parse_args()

    if args.eval_file and not args.baseline_file:
        parser.error("--baseline-file is required when using --eval-file")

    if args.all:
        analyses = run_all(output_dir=args.output_dir)
        if not args.no_plots and analyses:
            # Already plotted in run_all
            pass
    else:
        analysis = run_analysis(args.eval_file, args.baseline_file)
        print_report(analysis)
        if not args.no_plots:
            plot_analysis(analysis, args.output_dir)

        # Save analysis JSON
        model = analysis["model_id"]
        out_path = Path(args.output_dir) / f"{model}_rir_analysis.json"
        summary = {k: v for k, v in analysis.items() if k != "per_task"}
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved analysis: {out_path}")


if __name__ == "__main__":
    main()
