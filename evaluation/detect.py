"""
Detection functions for ToolFailBench failure modes.

Five outcomes:
  correct               — agent behaved correctly
  tool_skip             — should have called a tool, didn't
  result_ignore         — called the tool, ignored the return value
  output_fabrication    — called the tool, fabricated structured output
  unnecessary_tool_use  — CTRL task: called a tool when none was needed
  wrong_answer          — CTRL task: no tool called but answer is incorrect
"""


def detect_tool_skip(task: dict, agent_trace: dict) -> bool:
    """
    Returns True if agent was required to call a tool but didn't.
    Always False for CTRL tasks (not calling a tool is correct for them).
    """
    if not task["evaluation_criteria"]["tool_must_be_called"]:
        return False
    expected_tool = task["expected_tool_call"]["name"]
    called_tools = [call["name"] for call in agent_trace.get("tool_calls", [])]
    return expected_tool not in called_tools


def detect_unnecessary_tool_use(task: dict, agent_trace: dict) -> bool:
    """
    Returns True if agent called a tool when the task didn't require one (CTRL tasks).
    Always False for non-CTRL tasks.
    """
    if task["evaluation_criteria"]["tool_must_be_called"]:
        return False
    called_tools = [call["name"] for call in agent_trace.get("tool_calls", [])]
    return len(called_tools) > 0


def detect_result_ignore(task: dict, agent_trace: dict, agent_answer: str) -> bool:
    """
    Returns True if agent called the tool but answer is inconsistent with the return.
    Only applies to non-CTRL tasks where tool was actually called.
    """
    if detect_tool_skip(task, agent_trace):
        return False

    ground_truth = task["ground_truth"]
    must_contain = ground_truth["answer_must_contain"]
    match_mode = ground_truth.get("match_mode", "all")

    if match_mode == "all":
        return not all(str(val) in agent_answer for val in must_contain)
    else:
        return not any(str(val) in agent_answer for val in must_contain)


def detect_output_fabrication(task: dict, agent_trace: dict, agent_answer: str) -> bool:
    """
    Returns True if agent called the tool but fabricated structured output
    instead of using the actual return value.
    Only applies to non-CTRL tasks where tool was actually called.
    """
    if detect_tool_skip(task, agent_trace):
        return False

    mock_return = task["mock_tool_return"]
    mock_values = _extract_leaf_values(mock_return)
    mock_values_in_answer = sum(1 for v in mock_values if str(v) in agent_answer)

    if mock_values_in_answer < len(mock_values) * 0.3:
        if _contains_structured_data(agent_answer):
            return True
    return False


def _answer_correct(task: dict, agent_answer: str) -> bool:
    """Check if answer contains the required ground truth values."""
    ground_truth = task["ground_truth"]
    must_contain = ground_truth["answer_must_contain"]
    match_mode = ground_truth.get("match_mode", "all")
    if match_mode == "all":
        return all(str(v) in agent_answer for v in must_contain)
    else:
        return any(str(v) in agent_answer for v in must_contain)


def classify_failure_mode(task: dict, agent_trace: dict, agent_answer: str) -> str:
    """
    Classifies the agent's response into one of:
      correct | tool_skip | result_ignore | output_fabrication |
      unnecessary_tool_use | wrong_answer
    """
    tool_must_be_called = task["evaluation_criteria"]["tool_must_be_called"]

    # --- CTRL tasks (tool not required) ---
    if not tool_must_be_called:
        if detect_unnecessary_tool_use(task, agent_trace):
            return "unnecessary_tool_use"
        return "correct" if _answer_correct(task, agent_answer) else "wrong_answer"

    # --- Tool-required tasks ---
    if detect_tool_skip(task, agent_trace):
        return "tool_skip"

    if detect_output_fabrication(task, agent_trace, agent_answer):
        return "output_fabrication"

    if detect_result_ignore(task, agent_trace, agent_answer):
        return "result_ignore"

    return "correct"


def _extract_leaf_values(obj, values=None):
    if values is None:
        values = []
    if isinstance(obj, dict):
        for v in obj.values():
            _extract_leaf_values(v, values)
    elif isinstance(obj, list):
        for v in obj:
            _extract_leaf_values(v, values)
    else:
        values.append(obj)
    return values


def _contains_structured_data(text: str) -> bool:
    indicators = ["{", "}", '":', '["', "patient_id", "balance", "stdout", "rows"]
    return sum(1 for i in indicators if i in text) >= 2
