"""
Collects parametric baselines: runs models WITHOUT tool access.
Records what the model answers from pure parametric memory.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

TASK_DIR = Path(__file__).parent.parent / "tasks"
DOMAINS = ["finance", "medical", "code"]


def load_tasks(domains: list[str]) -> list[dict]:
    tasks = []
    for domain in domains:
        task_file = TASK_DIR / domain / "tasks.json"
        if task_file.exists():
            with open(task_file) as f:
                tasks.extend(json.load(f))
    return tasks


def get_parametric_answer(task: dict, model: str) -> str:
    """Query model WITHOUT tools to get its parametric answer."""
    try:
        import litellm

        messages = [
            {"role": "system", "content": task["system_prompt"]},
            {"role": "user", "content": task["user_message"]},
        ]
        # No tools parameter = pure parametric memory
        response = litellm.completion(model=model, messages=messages)
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"ERROR: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Collect parametric baselines")
    parser.add_argument("--model", required=True, help="Model name (litellm format)")
    parser.add_argument("--domains", nargs="+", default=DOMAINS)
    parser.add_argument("--output-dir", default="results/baselines")
    args = parser.parse_args()

    tasks = load_tasks(args.domains)
    print(f"Collecting parametric baselines for {len(tasks)} tasks using {args.model}")

    baselines = []
    for task in tqdm(tasks, desc="Baseline"):
        answer = get_parametric_answer(task, args.model)
        baselines.append({"task_id": task["task_id"], "domain": task["domain"], "parametric_answer": answer})

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{args.output_dir}/{args.model.replace('/', '_')}_parametric_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"Baselines saved to {out_path}")


if __name__ == "__main__":
    main()
