"""
Main evaluation runner for ToolFailBench.
Loads tasks, runs a model via litellm, captures tool calls, classifies failures.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from evaluation.detect import classify_failure_mode
from evaluation.metrics import compute_all_metrics
from evaluation.report import generate_summary_table, save_results_json

TASK_DIR = Path(__file__).parent.parent / "tasks"
DOMAINS = ["finance", "medical", "code"]


def load_tasks(domains: list[str]) -> list[dict]:
    tasks = []
    for domain in domains:
        task_file = TASK_DIR / domain / "tasks.json"
        if task_file.exists():
            with open(task_file) as f:
                domain_tasks = json.load(f)
                tasks.extend(domain_tasks)
                print(f"  Loaded {len(domain_tasks)} tasks from {domain}")
    return tasks


def run_single_task(task: dict, model: str) -> dict:
    """
    Run a single task against a model.
    Returns dict with task, agent_trace, agent_answer, classification.
    """
    try:
        import litellm

        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": {
                        "type": "object",
                        "properties": tool["parameters"],
                    },
                },
            }
            for tool in task["available_tools"]
        ]

        messages = [
            {"role": "system", "content": task["system_prompt"]},
            {"role": "user", "content": task["user_message"]},
        ]

        response = litellm.completion(model=model, messages=messages, tools=tools)

        # Parse tool calls from response
        tool_calls = []
        agent_answer = ""
        choice = response.choices[0]

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    {
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    }
                )
            # Simulate tool return and get final response
            tool_messages = messages + [choice.message]
            for tc in choice.message.tool_calls:
                tool_name = tc.function.name
                mock_return = task["mock_tool_return"]
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(mock_return),
                    }
                )
            follow_up = litellm.completion(model=model, messages=tool_messages, tools=tools)
            agent_answer = follow_up.choices[0].message.content or ""
        else:
            agent_answer = choice.message.content or ""

        agent_trace = {"tool_calls": tool_calls}
        classification = classify_failure_mode(task, agent_trace, agent_answer)

        return {
            "task": task,
            "agent_trace": agent_trace,
            "agent_answer": agent_answer,
            "classification": classification,
        }

    except Exception as e:
        return {
            "task": task,
            "agent_trace": {"tool_calls": []},
            "agent_answer": f"ERROR: {str(e)}",
            "classification": "other_error",
        }


def main():
    parser = argparse.ArgumentParser(description="Run ToolFailBench evaluation")
    parser.add_argument("--model", required=True, help="Model name (litellm format)")
    parser.add_argument("--domains", nargs="+", default=DOMAINS, help="Domains to evaluate")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max tasks to run")
    args = parser.parse_args()

    print(f"Loading tasks for domains: {args.domains}")
    tasks = load_tasks(args.domains)
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    print(f"Running {len(tasks)} tasks against {args.model}\n")

    results = []
    for task in tqdm(tasks, desc="Evaluating"):
        result = run_single_task(task, args.model)
        results.append(result)

    print("\n" + generate_summary_table(results, args.model))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{args.output_dir}/{args.model.replace('/', '_')}_{timestamp}.json"
    save_results_json(results, out_path)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
