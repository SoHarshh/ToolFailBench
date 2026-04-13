"""
LLM-as-judge runner for ToolFailBench.

Reads existing eval results, runs each through an LLM judge, and produces
annotated results with independent failure mode classifications and scores.

Usage:
  # Validate on small sample first
  python runners/run_judge.py --results-file results/gemma4-31b_20260413_030923.json --sample 10

  # Full run
  python runners/run_judge.py --results-file results/gemma4-31b_20260413_030923.json

  # Batch all Tier 1 results
  python runners/run_judge.py --results-dir results/ --judge-model claude-sonnet-4-5

  # Dry run (print prompts, no API calls)
  python runners/run_judge.py --results-file results/gemma4-31b_20260413_030923.json --dry-run --sample 3
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.judge import (
    run_judge_on_result,
    build_judge_prompt,
    compare_classifications,
    JUDGE_SYSTEM_PROMPT,
)

load_dotenv()

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_cost(n_tasks: int, judge_model: str) -> dict:
    """Rough cost estimate for running the judge."""
    # Average tokens per task (estimated)
    sys_prompt_tokens = 900   # system prompt (cached after first call)
    user_prompt_tokens = 500  # per task
    output_tokens = 150       # per task

    total_input = sys_prompt_tokens + (user_prompt_tokens * n_tasks)
    total_output = output_tokens * n_tasks

    # Rough pricing (per 1M tokens)
    pricing = {
        "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }
    rates = pricing.get(judge_model, {"input": 3.0, "output": 15.0})

    input_cost = (total_input / 1_000_000) * rates["input"]
    output_cost = (total_output / 1_000_000) * rates["output"]

    return {
        "n_tasks": n_tasks,
        "est_input_tokens": total_input,
        "est_output_tokens": total_output,
        "est_cost_usd": round(input_cost + output_cost, 2),
        "model": judge_model,
    }


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def judge_results(
    results: list[dict],
    judge_model: str,
    delay: float = 0.5,
    dry_run: bool = False,
    api_base: str = None,
) -> list[dict]:
    """
    Run the judge on each result. Returns list of annotated dicts.

    Each annotated dict has:
      task_id, domain, model_id, rule_based_classification, judge, agreement
    """
    annotated = []

    for result in tqdm(results, desc=f"Judging ({judge_model})"):
        task_id = result["task"]["task_id"]
        domain = result["task"]["domain"]
        model_id = result["model_id"]
        rb_class = result["classification"]

        if dry_run:
            prompt = build_judge_prompt(result)
            print(f"\n{'='*70}")
            print(f"TASK: {task_id} | MODEL: {model_id} | RULE-BASED: {rb_class}")
            print(f"{'='*70}")
            print(prompt)
            print(f"{'='*70}\n")
            annotated.append({
                "task_id": task_id,
                "domain": domain,
                "model_id": model_id,
                "rule_based_classification": rb_class,
                "judge": {"failure_mode": None, "reasoning": "dry_run"},
                "agreement": None,
            })
            continue

        verdict = run_judge_on_result(result, judge_model=judge_model, api_base=api_base)

        agreement = None
        if verdict.get("failure_mode"):
            agreement = verdict["failure_mode"] == rb_class

        annotated.append({
            "task_id": task_id,
            "domain": domain,
            "model_id": model_id,
            "rule_based_classification": rb_class,
            "judge": verdict,
            "agreement": agreement,
        })

        # Rate limiting — small delay between API calls
        if delay > 0:
            time.sleep(delay)

    return annotated


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def print_summary(annotated: list[dict], model_id: str):
    """Print a summary comparison of rule-based vs judge classifications."""
    comparison = compare_classifications(annotated)

    print(f"\n{'='*60}")
    print(f"  Judge Summary: {model_id}")
    print(f"{'='*60}")
    print(f"  Tasks judged:     {comparison['total_judged']}")
    print(f"  Agreements:       {comparison['agreements']}")
    print(f"  Disagreements:    {comparison['disagreements']}")
    print(f"  Agreement rate:   {comparison['agreement_rate']:.1%}")

    if comparison["disagreement_details"]:
        print(f"\n  --- Disagreements ---")
        for d in comparison["disagreement_details"]:
            print(f"  {d['task_id']:20s}  rule={d['rule_based']:20s}  "
                  f"judge={d['judge']:20s}  conf={d['judge_confidence']}")
            print(f"  {'':20s}  {d['reasoning'][:80]}")
            print()

    # Score distribution (for tool-required tasks)
    faithfulness_scores = [
        e["judge"].get("result_faithfulness")
        for e in annotated
        if e["judge"].get("result_faithfulness") is not None
    ]
    if faithfulness_scores:
        print(f"  --- Result Faithfulness Score Distribution ---")
        for score in range(4):
            count = faithfulness_scores.count(score)
            print(f"    {score}: {'█' * count} ({count})")

    correctness_scores = [
        e["judge"].get("answer_correctness")
        for e in annotated
        if e["judge"].get("answer_correctness") is not None
    ]
    if correctness_scores:
        print(f"\n  --- Answer Correctness Score Distribution ---")
        for score in range(4):
            count = correctness_scores.count(score)
            print(f"    {score}: {'█' * count} ({count})")

    # Judge failure mode distribution
    judge_modes = [
        e["judge"]["failure_mode"]
        for e in annotated
        if e["judge"].get("failure_mode")
    ]
    if judge_modes:
        from collections import Counter
        dist = Counter(judge_modes)
        print(f"\n  --- Judge Failure Mode Distribution ---")
        for mode, count in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"    {mode:25s} {count:3d}")

    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# W&B logging
# ---------------------------------------------------------------------------

def log_to_wandb(annotated: list[dict], judge_model: str, model_id: str):
    """Log judge results to W&B."""
    try:
        import wandb

        comparison = compare_classifications(annotated)

        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "toolfailbench"),
            entity=os.getenv("WANDB_ENTITY") or None,
            name=f"judge-{model_id}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "task": "judge",
                "judge_model": judge_model,
                "evaluated_model": model_id,
                "n_tasks": comparison["total_judged"],
            },
        )

        run.log({
            "agreement_rate": comparison["agreement_rate"],
            "total_judged": comparison["total_judged"],
            "agreements": comparison["agreements"],
            "disagreements": comparison["disagreements"],
        })

        # Log per-task table
        columns = [
            "task_id", "domain", "rule_based", "judge", "agreement",
            "confidence", "answer_correctness", "reasoning",
        ]
        table = wandb.Table(columns=columns)
        for e in annotated:
            j = e.get("judge", {})
            table.add_data(
                e["task_id"],
                e["domain"],
                e["rule_based_classification"],
                j.get("failure_mode", "error"),
                e.get("agreement"),
                j.get("confidence", ""),
                j.get("answer_correctness"),
                (j.get("reasoning") or "")[:200],
            )
        run.log({"judge_results": table})

        run.finish()
        print(f"  [W&B] Logged to {run.url}")

    except Exception as e:
        print(f"  [W&B] Failed: {e}")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_result_files(results_dir: str) -> list[Path]:
    """Find all Tier 1 eval result files (not baselines, not judge outputs)."""
    rdir = Path(results_dir)
    files = []
    for f in sorted(rdir.glob("*.json")):
        # Skip baselines and judge outputs
        if "baseline" in f.name or "judge" in f.name:
            continue
        files.append(f)
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-as-judge on ToolFailBench results"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--results-file", type=str,
        help="Path to a single results JSON file"
    )
    input_group.add_argument(
        "--results-dir", type=str,
        help="Directory containing results JSON files (judges all non-baseline files)"
    )

    parser.add_argument(
        "--judge-model", default="claude-sonnet-4-5",
        help="Judge model (litellm string). Default: claude-sonnet-4-5"
    )
    parser.add_argument(
        "--output-dir", default="results/judge",
        help="Output directory for judge results"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Only judge first N results (for validation)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between API calls in seconds (rate limiting)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print prompts without making API calls"
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging"
    )
    parser.add_argument(
        "--api-base", type=str, default=None,
        help="API base URL for local models (e.g., http://localhost:8000/v1)"
    )

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Resolve input files
    if args.results_file:
        files = [Path(args.results_file)]
    else:
        files = find_result_files(args.results_dir)
        if not files:
            print(f"No result files found in {args.results_dir}")
            return

    print(f"Judge model: {args.judge_model}")
    print(f"Files to judge: {[f.name for f in files]}")

    for filepath in files:
        print(f"\n{'='*60}")
        print(f"  Loading: {filepath.name}")
        print(f"{'='*60}")

        results = json.load(open(filepath))
        model_id = results[0]["model_id"] if results else "unknown"

        if args.sample:
            results = results[:args.sample]

        # Cost estimate
        est = estimate_cost(len(results), args.judge_model)
        print(f"  Tasks: {est['n_tasks']}")
        print(f"  Est. cost: ${est['est_cost_usd']:.2f} ({args.judge_model})")

        if args.dry_run:
            print(f"  [DRY RUN] Printing prompts only...\n")

        # Run judge
        annotated = judge_results(
            results,
            judge_model=args.judge_model,
            delay=args.delay,
            dry_run=args.dry_run,
            api_base=args.api_base,
        )

        # Summary
        if not args.dry_run:
            print_summary(annotated, model_id)

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = Path(args.output_dir) / f"{model_id}_judge_{timestamp}.json"
            with open(out_path, "w") as f:
                json.dump(annotated, f, indent=2, default=str)
            print(f"  Saved to {out_path}")

            # W&B logging
            if not args.no_wandb:
                log_to_wandb(annotated, args.judge_model, model_id)


if __name__ == "__main__":
    main()
