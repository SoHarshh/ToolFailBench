"""
Mock tool server for ToolFailBench.
Returns predetermined values from task definitions.
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any

app = FastAPI(title="ToolFailBench Mock Server", version="0.1.0")

TASK_DIR = Path(__file__).parent.parent / "tasks"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Lookup table: (tool_name, args_hash) -> mock_tool_return
_tool_return_lookup: dict[str, Any] = {}
_tasks_loaded: list[dict] = []


def _hash_args(args: dict) -> str:
    return hashlib.md5(json.dumps(args, sort_keys=True).encode()).hexdigest()


def _load_tasks():
    global _tool_return_lookup, _tasks_loaded
    _tool_return_lookup = {}
    _tasks_loaded = []
    for domain_dir in TASK_DIR.iterdir():
        task_file = domain_dir / "tasks.json"
        if task_file.exists() and domain_dir.name != "control":
            with open(task_file) as f:
                tasks = json.load(f)
                _tasks_loaded.extend(tasks)
                for task in tasks:
                    tool_name = task["expected_tool_call"]["name"]
                    args = task["expected_tool_call"]["arguments"]
                    key = f"{tool_name}:{_hash_args(args)}"
                    _tool_return_lookup[key] = task["mock_tool_return"]
    print(f"Loaded {len(_tasks_loaded)} tasks, {len(_tool_return_lookup)} tool mappings")


class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: dict


class ToolCallResponse(BaseModel):
    result: Any
    matched: bool


@app.on_event("startup")
async def startup():
    _load_tasks()


@app.post("/call_tool", response_model=ToolCallResponse)
async def call_tool(request: ToolCallRequest):
    key = f"{request.tool_name}:{_hash_args(request.arguments)}"

    # Log the call
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "tool_name": request.tool_name,
        "arguments": request.arguments,
        "matched": key in _tool_return_lookup
    }
    log_path = RESULTS_DIR / "tool_call_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    if key in _tool_return_lookup:
        return ToolCallResponse(result=_tool_return_lookup[key], matched=True)

    # Try fuzzy match: same tool name, any args
    fuzzy_matches = {k: v for k, v in _tool_return_lookup.items() if k.startswith(f"{request.tool_name}:")}
    if len(fuzzy_matches) == 1:
        return ToolCallResponse(result=list(fuzzy_matches.values())[0], matched=True)

    raise HTTPException(status_code=404, detail=f"No mock return found for {request.tool_name} with given arguments")


@app.get("/health")
async def health():
    return {"status": "ok", "tasks_loaded": len(_tasks_loaded), "mappings": len(_tool_return_lookup)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8042)
