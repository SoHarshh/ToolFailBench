"""
LLM-as-judge evaluation for ToolFailBench.

Provides nuanced, semantic evaluation of model responses beyond
the rule-based substring matching in detect.py. Uses a strong
LLM (Claude/GPT-4o) to independently classify failure modes and
score response quality on 0–3 scales.

Usage (as library):
    from evaluation.judge import run_judge_on_result, build_judge_prompt
    verdict = run_judge_on_result(result_dict, judge_model="claude-sonnet-4-5")
"""

import json
import re
from typing import Optional


# ---------------------------------------------------------------------------
# System prompt — the rubric the judge follows
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for ToolFailBench, a benchmark that measures how \
and why language models fail at tool use.

You will evaluate ONE model response at a time. Your job is to independently \
classify the failure mode and score the response quality. Be precise, \
objective, and evidence-based.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BENCHMARK CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Models receive a user query plus a set of available tools. Two task types:

• TOOL-REQUIRED — the correct answer depends on data that can ONLY come from \
the tool (real-time prices, private records, computation output). The model \
MUST call the tool AND use its return value in the answer.

• CONTROL — the answer is common knowledge. No tool is needed. The model \
SHOULD answer directly without calling any tool.

The benchmark uses "parametric traps": tool return values are deliberately \
set to differ from what the model likely memorized during training. For \
example, a stock-price tool returns $247.83 while the model's training data \
suggests ~$180. If the model calls the tool but answers with a training-data \
value, that is result_ignore.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FAILURE MODE DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FOR TOOL-REQUIRED TASKS:

  correct — Model called the correct tool with reasonable arguments AND the \
  final answer faithfully reflects the tool's returned data.

  tool_skip — Model did NOT make a valid, executed tool call. This includes: \
  (a) no tool call at all, (b) a tool call written in plain text inside the \
  answer body (e.g. "<tool_call>…</tool_call>") that was never actually \
  executed by the system, (c) the model refused to answer.

  result_ignore — Model called the correct tool, received the return value, \
  but the final answer does NOT faithfully reflect it. Signs: answer contains \
  memorized/training-data values instead of the tool's values, answer is \
  empty or generic despite having received specific data, answer contradicts \
  the tool return.

  output_fabrication — Model called the correct tool but the answer contains \
  invented structured data (fake JSON, hallucinated table rows, fabricated \
  numbers) that does not come from the tool's actual return. Distinguished \
  from result_ignore by the presence of plausible-looking but invented \
  structured output.

FOR CONTROL TASKS:

  correct — Model answered correctly WITHOUT calling any tool.

  unnecessary_tool_use — Model called a tool when none was needed.

  wrong_answer — Model correctly refrained from calling a tool, but the \
  factual answer is incorrect.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING RUBRIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

tool_selection (0–3) — ONLY for tool-required tasks
  0 = No tool called, or entirely wrong tool
  1 = Correct tool but arguments are significantly wrong
  2 = Correct tool, minor argument variations (formatting, optional params)
  3 = Correct tool with correct arguments

result_faithfulness (0–3) — ONLY for tool-required tasks where tool was called
  0 = Tool result completely ignored (memorized values, empty answer, \
      contradicts tool return)
  1 = Partially used — some values present but key data points missing or \
      replaced with memorized values
  2 = Mostly used — answer is semantically correct but has minor formatting \
      differences (see FORMATTING TOLERANCE below)
  3 = Fully and accurately reflected in the answer

answer_correctness (0–3) — ALL tasks
  0 = Wrong, empty, or completely unhelpful
  1 = Partially correct or only tangentially addresses the question
  2 = Mostly correct with minor issues
  3 = Fully correct and directly helpful

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATTING TOLERANCE (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When judging result_faithfulness, be TOLERANT of reasonable formatting \
differences. These are NOT failures:
  • "$1,001" vs "1001" vs "1,001.00" — same number, different format
  • "247.83" vs "$247.83" vs "247.83 USD" — currency formatting
  • Reasonable rounding — "4237.29" for 5000/1.18
  • Paraphrasing tool return fields — no need to quote JSON keys verbatim
  • Units added or reformatted — "5h 47m" vs "347 minutes"

DO classify as result_ignore when:
  • The answer states a DIFFERENT value than the tool returned
  • The answer is empty or generic despite receiving specific tool data
  • Key data points from the tool return are absent and replaced with \
    the model's own memorized values

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Respond with ONLY a valid JSON object. No markdown fences, no commentary.

For tool-required tasks:
{"failure_mode": "...", "confidence": "high|medium|low", "tool_selection": N, \
"result_faithfulness": N, "answer_correctness": N, "reasoning": "..."}

For control tasks:
{"failure_mode": "...", "confidence": "high|medium|low", "tool_restraint": \
true|false, "answer_correctness": N, "reasoning": "..."}

Keep reasoning to 2–3 sentences. Focus on the evidence that drove your \
classification."""


# ---------------------------------------------------------------------------
# Answer cleaning
# ---------------------------------------------------------------------------

def _clean_answer(text: str) -> str:
    """Remove model-specific special tokens from the answer text."""
    # Gemma4 thought channel tokens
    text = re.sub(r'<\|channel>thought\s*\n?<channel\|>', '', text)
    # Qwen/deepseek think blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'</think>\s*', '', text)
    # Strip leading/trailing whitespace
    return text.strip()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_judge_prompt(result: dict) -> str:
    """
    Build the user message for the judge given a single eval result.
    Includes all context the judge needs: task, tools, model response,
    ground truth. Deliberately omits the rule-based classification to
    avoid biasing the judge.
    """
    task = result["task"]
    trace = result["agent_trace"]
    answer = _clean_answer(result["agent_answer"])

    is_ctrl = not task["evaluation_criteria"]["tool_must_be_called"]
    task_type = "CONTROL (no tool needed)" if is_ctrl else "TOOL-REQUIRED"

    # Tool calls summary
    tool_calls = trace.get("tool_calls", [])
    if tool_calls:
        tc_lines = []
        for tc in tool_calls:
            tc_lines.append(f"  {tc['name']}({json.dumps(tc['arguments'])})")
        tool_calls_str = "\n".join(tc_lines)
    else:
        tool_calls_str = "  (none — no tool was called)"

    # Available tools summary
    tools_str = "\n".join(
        f"  • {t['name']}: {t['description']}"
        for t in task["available_tools"]
    )

    sections = [
        f"TASK TYPE: {task_type}",
        f"DOMAIN: {task['domain']}",
        f"USER MESSAGE:\n{task['user_message']}",
        f"AVAILABLE TOOLS:\n{tools_str}",
    ]

    if not is_ctrl:
        exp = task["expected_tool_call"]
        sections.append(
            f"EXPECTED TOOL CALL:\n  {exp['name']}({json.dumps(exp['arguments'])})"
        )
        sections.append(
            f"TOOL RETURN VALUE (what the tool would return):\n"
            f"{json.dumps(task['mock_tool_return'], indent=2)}"
        )

    sections.append(f"MODEL'S ACTUAL TOOL CALLS:\n{tool_calls_str}")
    sections.append(
        f"MODEL'S FINAL ANSWER:\n{answer if answer else '(empty — model returned no text)'}"
    )

    gt = task["ground_truth"]
    gt_str = f"Must contain: {gt['answer_must_contain']}"
    if gt.get("answer_must_not_contain"):
        gt_str += f"\nMust NOT contain: {gt['answer_must_not_contain']}"
    gt_str += f"\nMatch mode: {gt.get('match_mode', 'all')}"
    sections.append(f"GROUND TRUTH:\n{gt_str}")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_judge_response(response_text: str) -> Optional[dict]:
    """
    Parse the judge's JSON response.
    Returns the parsed dict, or None if parsing/validation fails.
    """
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract first JSON object from the response
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    # Validate required fields
    if "failure_mode" not in data:
        return None

    valid_modes = {
        "correct", "tool_skip", "result_ignore", "output_fabrication",
        "unnecessary_tool_use", "wrong_answer",
    }
    if data["failure_mode"] not in valid_modes:
        return None

    # Ensure reasoning exists (even if empty)
    if "reasoning" not in data:
        data["reasoning"] = ""

    # Clamp scores to 0–3
    for key in ("tool_selection", "result_faithfulness", "answer_correctness"):
        if key in data and isinstance(data[key], (int, float)):
            data[key] = max(0, min(3, int(data[key])))

    return data


# ---------------------------------------------------------------------------
# Judge executor
# ---------------------------------------------------------------------------

def run_judge_on_result(
    result: dict,
    judge_model: str = "claude-sonnet-4-5",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: int = 60,
) -> dict:
    """
    Run the LLM judge on a single evaluation result.

    Args:
        result: A single result dict from run_eval.py output.
        judge_model: litellm model string (e.g., "claude-sonnet-4-5", "gpt-4o",
                     "openai/Qwen/Qwen3-8B" for local vLLM).
        api_key: Optional API key override.
        api_base: Optional API base URL (e.g., "http://localhost:8000/v1" for vLLM).
        timeout: Request timeout in seconds.

    Returns:
        Parsed judge verdict dict with keys: failure_mode, confidence,
        scores, reasoning. On error, returns dict with "error" key.
    """
    import litellm

    prompt = build_judge_prompt(result)

    kwargs = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "timeout": timeout,
    }

    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base
        if not api_key:
            kwargs["api_key"] = "local"  # vLLM doesn't check, but litellm requires it

    try:
        response = litellm.completion(**kwargs)
        text = response.choices[0].message.content or ""
        parsed = parse_judge_response(text)

        if parsed is None:
            return {
                "error": "parse_failed",
                "raw_response": text,
                "failure_mode": None,
            }

        parsed["raw_response"] = text
        return parsed

    except Exception as e:
        return {
            "error": str(e),
            "raw_response": None,
            "failure_mode": None,
        }


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

def compare_classifications(results_with_judge: list[dict]) -> dict:
    """
    Compare rule-based vs judge classifications.
    Returns agreement stats and list of disagreements.
    """
    total = 0
    agree = 0
    disagree_list = []

    for entry in results_with_judge:
        judge = entry.get("judge", {})
        if judge.get("failure_mode") is None:
            continue  # skip errors

        total += 1
        rb = entry["rule_based_classification"]
        jm = judge["failure_mode"]

        if rb == jm:
            agree += 1
        else:
            disagree_list.append({
                "task_id": entry["task_id"],
                "domain": entry.get("domain", ""),
                "rule_based": rb,
                "judge": jm,
                "judge_confidence": judge.get("confidence", ""),
                "reasoning": judge.get("reasoning", ""),
            })

    return {
        "total_judged": total,
        "agreements": agree,
        "disagreements": len(disagree_list),
        "agreement_rate": agree / total if total > 0 else 0.0,
        "disagreement_details": disagree_list,
    }
