"""
Main evaluation runner for ToolFailBench.
Loads tasks, runs models from the registry, logs to Weights & Biases.

Usage:
  python runners/run_eval.py --model qwen3.5-7b --domains finance medical code
  python runners/run_eval.py --tier 1 --domains finance medical code
  python runners/run_eval.py --tier 1 2 3 4
"""
import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

# Allow importing local modules from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.detect import classify_failure_mode
from evaluation.metrics import compute_all_metrics
from evaluation.report import generate_summary_table, save_results_json
from evaluation.data import load_tasks, ALL_DOMAINS
from models.registry import load_registry, get_model_config, get_models_for_tier

load_dotenv()

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "configs" / "default.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} is empty or invalid — expected a YAML mapping.")
    if "inference" not in config:
        raise ValueError(f"Config at {path} is missing required 'inference' section.")
    required_inference_keys = ["temperature", "max_tokens", "seed", "tool_choice"]
    missing = [k for k in required_inference_keys if k not in config["inference"]]
    if missing:
        raise ValueError(f"Config 'inference' section is missing keys: {missing}")
    null_keys = [k for k in required_inference_keys if config["inference"][k] is None]
    if null_keys:
        raise ValueError(f"Config 'inference' keys must not be null: {null_keys}")
    return config


# ---------------------------------------------------------------------------
# Inference dispatch
# ---------------------------------------------------------------------------

def _build_litellm_model_str(model_cfg: dict, config: dict) -> tuple[str, dict]:
    """
    Returns (litellm_model_string, extra_kwargs) for the given model config.
    vLLM models are called via the local OpenAI-compatible server.
    Closed models go directly through their provider API.
    """
    extra = {}

    if model_cfg["inference_backend"] == "vllm":
        base_url = (
            config.get("vllm", {}).get("base_url")
            or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        )
        # litellm uses "openai/<model_name>" for custom OpenAI-compatible endpoints
        litellm_model = f"openai/{model_cfg['hf_model_id']}"
        extra["api_base"] = base_url
        extra["api_key"] = "vllm"  # vLLM server doesn't check the key but litellm requires it

    elif model_cfg["family"] == "openai":
        litellm_model = "gpt-4o"

    elif model_cfg["family"] == "anthropic":
        litellm_model = "claude-sonnet-4-5"

    else:
        raise ValueError(f"Unknown inference backend for model: {model_cfg['id']}")

    return litellm_model, extra


def _build_tools_payload(task: dict) -> list[dict]:
    """Convert task tool definitions to OpenAI-compatible tool schema."""
    tools = []
    for tool in task["available_tools"]:
        raw_params = tool["parameters"]
        properties = {
            k: {sk: sv for sk, sv in v.items() if sk != "required"}
            for k, v in raw_params.items()
        }
        required = [k for k, v in raw_params.items() if v.get("required")]
        tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return tools


def _raw_http_call(messages, tools, inf, base_url, model_id):
    """Raw HTTP call bypassing all client-side pydantic validation.
    Some vLLM tool call parsers (e.g. mistral) return arguments as a dict
    instead of a JSON string, which breaks both litellm and openai clients."""
    import requests

    url = f"{base_url}/chat/completions"
    payload = {
        "model": model_id,
        "messages": messages,
        "tools": tools,
        "tool_choice": inf["tool_choice"],
        "temperature": inf["temperature"],
        "max_tokens": inf["max_tokens"],
        "seed": inf["seed"],
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _parse_tool_calls_from_dict(choice_dict: dict) -> tuple[list[dict], list[dict]]:
    """Extract tool calls from a raw JSON response choice dict.
    Returns (parsed_tool_calls, raw_tool_calls_for_followup)."""
    tool_calls = []
    raw_tcs = []
    msg = choice_dict.get("message", {})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        tool_calls.append({"name": fn.get("name", ""), "arguments": args})
        raw_tcs.append(tc)
    return tool_calls, raw_tcs


def _parse_tool_calls(choice) -> tuple[list[dict], None]:
    """Extract tool calls from a litellm/openai response choice object."""
    tool_calls = []
    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append({"name": tc.function.name, "arguments": args})
    return tool_calls, None


def run_single_task(task: dict, model_cfg: dict, config: dict) -> dict:
    """Run one task against a model. Returns result dict."""
    try:
        import litellm

        litellm_model, extra_kwargs = _build_litellm_model_str(model_cfg, config)
        inf = config.get("inference", {})
        tools = _build_tools_payload(task)

        system_prompt = task["system_prompt"]
        if model_cfg.get("no_think"):
            system_prompt = system_prompt + " /no_think"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task["user_message"]},
        ]

        call_kwargs = dict(
            model=litellm_model,
            messages=messages,
            tools=tools,
            tool_choice=inf["tool_choice"],
            temperature=inf["temperature"],
            max_tokens=inf["max_tokens"],
            seed=inf["seed"],
            **extra_kwargs,
        )

        # Try litellm first; fall back to raw HTTP if response parsing fails
        use_raw = False
        tool_calls = []
        raw_tcs = []
        agent_answer = ""

        try:
            response = litellm.completion(**call_kwargs)
            choice = response.choices[0]
            tool_calls, _ = _parse_tool_calls(choice)
        except Exception as litellm_err:
            err_str = str(litellm_err)
            if "FunctionCall" in err_str or "arguments" in err_str or "validation error" in err_str:
                base_url = extra_kwargs.get("api_base", "http://localhost:8000/v1")
                resp_json = _raw_http_call(messages, tools, inf, base_url, model_cfg["hf_model_id"])
                choice_dict = resp_json["choices"][0]
                tool_calls, raw_tcs = _parse_tool_calls_from_dict(choice_dict)
                use_raw = True
            else:
                raise

        if tool_calls:
            # Build follow-up messages with mock tool return injected
            if use_raw:
                # Build assistant message from raw dict
                msg_dict = resp_json["choices"][0].get("message", {})
                assistant_msg = {"role": "assistant", "content": msg_dict.get("content") or ""}
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"] if isinstance(tc["function"]["arguments"], str)
                            else json.dumps(tc["function"]["arguments"]),
                        },
                    }
                    for i, tc in enumerate(raw_tcs)
                ]
                tool_messages = messages + [assistant_msg]
                for tc in raw_tcs:
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_0"),
                        "content": json.dumps(task["mock_tool_return"]),
                    })
                base_url = extra_kwargs.get("api_base", "http://localhost:8000/v1")
                follow_json = _raw_http_call(tool_messages, tools, inf, base_url, model_cfg["hf_model_id"])
                agent_answer = follow_json["choices"][0].get("message", {}).get("content") or ""
            else:
                tool_messages = messages + [choice.message]
                for tc in choice.message.tool_calls:
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(task["mock_tool_return"]),
                    })
                follow_up = litellm.completion(
                    model=litellm_model,
                    messages=tool_messages,
                    tools=tools,
                    temperature=inf["temperature"],
                    max_tokens=inf["max_tokens"],
                    seed=inf["seed"],
                    **extra_kwargs,
                )
                agent_answer = follow_up.choices[0].message.content or ""
        else:
            if use_raw:
                agent_answer = resp_json["choices"][0].get("message", {}).get("content") or ""
            else:
                agent_answer = choice.message.content or ""

        agent_trace = {"tool_calls": tool_calls}
        classification = classify_failure_mode(task, agent_trace, agent_answer)

        return {
            "task": task,
            "model_id": model_cfg["id"],
            "agent_trace": agent_trace,
            "agent_answer": agent_answer,
            "classification": classification,
        }

    except Exception as e:
        return {
            "task": task,
            "model_id": model_cfg["id"],
            "agent_trace": {"tool_calls": []},
            "agent_answer": f"ERROR: {str(e)}",
            "classification": "other_error",
        }


# ---------------------------------------------------------------------------
# W&B logging
# ---------------------------------------------------------------------------

def init_wandb(model_cfg: dict, config: dict):
    import wandb

    wb_cfg = config.get("wandb", {})
    run = wandb.init(
        project=wb_cfg.get("project") or os.getenv("WANDB_PROJECT", "toolfailbench"),
        entity=wb_cfg.get("entity") or os.getenv("WANDB_ENTITY") or None,
        name=f"{model_cfg['id']}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model_id": model_cfg["id"],
            "family": model_cfg["family"],
            "size": model_cfg["size"],
            "tier": model_cfg["tier"],
            "category": model_cfg["category"],
            "inference_backend": model_cfg["inference_backend"],
            **config.get("inference", {}),
        },
    )
    return run


def log_to_wandb(run, results: list[dict], config: dict):
    import wandb

    wb_cfg = config.get("wandb", {})

    if wb_cfg.get("log_metrics", True):
        metrics = compute_all_metrics(results)
        run.log({
            "tsr": metrics["tsr"],
            "rir": metrics["rir"],
            "ofr": metrics["ofr"],
            "ctur": metrics["ctur"],
            "utr": metrics["utr"],
            "ctrl_accuracy": metrics["ctrl_accuracy"],
            "total_tasks": metrics["total_tasks"],
            "tool_required_tasks": metrics["tool_required_tasks"],
            "ctrl_tasks": metrics["ctrl_tasks"],
        })

    if wb_cfg.get("log_predictions", True):
        table = wandb.Table(columns=[
            "task_id", "domain", "target_mode", "conflict_type",
            "classification", "agent_answer", "model_id",
        ])
        for r in results:
            task = r["task"]
            table.add_data(
                task["task_id"],
                task["domain"],
                task["target_failure_mode"],
                task["conflict_type"],
                r["classification"],
                r["agent_answer"][:500],  # truncate for display
                r["model_id"],
            )
        run.log({"predictions": table})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_model(model_cfg: dict, tasks: list[dict], config: dict, output_dir: str, use_wandb: bool):
    print(f"\n{'='*60}")
    print(f"  Model: {model_cfg['id']} (Tier {model_cfg['tier']}, {model_cfg['category']})")
    print(f"{'='*60}")

    wb_run = None
    if use_wandb:
        try:
            wb_run = init_wandb(model_cfg, config)
        except Exception as e:
            print(f"  [W&B] Failed to init: {e}. Continuing without logging.")

    results = []
    for task in tqdm(tasks, desc=model_cfg["id"]):
        result = run_single_task(task, model_cfg, config)
        results.append(result)

    print("\n" + generate_summary_table(results, model_cfg["id"]))

    if wb_run:
        try:
            log_to_wandb(wb_run, results, config)
            wb_run.finish()
            print(f"  [W&B] Logged to {wb_run.url}")
        except Exception as e:
            print(f"  [W&B] Logging failed: {e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{output_dir}/{model_cfg['id']}_{timestamp}.json"
    save_results_json(results, out_path)
    print(f"  Results saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run ToolFailBench evaluation")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Model id from registry (e.g. qwen2.5-7b)")
    group.add_argument("--tier", nargs="+", type=int, help="Run all models in tier(s) (e.g. --tier 1 2)")

    parser.add_argument("--domains", nargs="+", default=ALL_DOMAINS)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to YAML config")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    # Resolve which models to run
    if args.model:
        models_to_run = [get_model_config(args.model)]
    else:
        models_to_run = []
        for tier in args.tier:
            tier_models = get_models_for_tier(tier)
            if not tier_models:
                print(f"  Warning: no models found for tier {tier}")
            models_to_run.extend(tier_models)

    print(f"Models to run: {[m['id'] for m in models_to_run]}")
    print(f"Loading tasks for domains: {args.domains}")
    tasks = load_tasks(args.domains)
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    print(f"Total tasks: {len(tasks)}")

    use_wandb = not args.no_wandb
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    all_results = {}
    for model_cfg in models_to_run:
        results = run_model(model_cfg, tasks, config, args.output_dir, use_wandb)
        all_results[model_cfg["id"]] = results

    if len(models_to_run) > 1:
        print(f"\n{'='*60}")
        print(f"  Completed {len(models_to_run)} models")
        for model_id, results in all_results.items():
            metrics = compute_all_metrics(results)
            print(f"  {model_id}: CTUR={metrics['ctur']:.2%} TSR={metrics['tsr']:.2%} RIR={metrics['rir']:.2%} OFR={metrics['ofr']:.2%}")


if __name__ == "__main__":
    main()
