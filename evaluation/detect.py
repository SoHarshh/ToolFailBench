"""
Detection functions for ToolFailBench failure modes.
"""
import json
from typing import Optional


def detect_tool_skip(task: dict, agent_trace: dict) -> bool:
    """Returns True if agent should have called a tool but didn't."""
    expected_tool = task["expected_tool_call"]["name"]
    called_tools = [call["name"] for call in agent_trace.get("tool_calls", [])]
    return expected_tool not in called_tools


def detect_result_ignore(task: dict, agent_trace: dict, agent_answer: str) -> bool:
    """Returns True if agent called the tool but answer is inconsistent with return."""
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
    """Returns True if agent fabricated tool-like output not matching real return."""
    if detect_tool_skip(task, agent_trace):
        return False

    mock_return = task["mock_tool_return"]
    mock_values = _extract_leaf_values(mock_return)
    mock_values_in_answer = sum(1 for v in mock_values if str(v) in agent_answer)

    if mock_values_in_answer < len(mock_values) * 0.3:
        if _contains_structured_data(agent_answer):
            return True
    return False


def classify_failure_mode(task: dict, agent_trace: dict, agent_answer: str) -> str:
    """
    Returns one of: 'correct', 'tool_skip', 'result_ignore',
    'output_fabrication', 'other_error'
    """
    if detect_tool_skip(task, agent_trace):
        return "tool_skip"

    expected_tool = task["expected_tool_call"]["name"]
    called_tools = [call["name"] for call in agent_trace.get("tool_calls", [])]
    if expected_tool not in called_tools:
        return "other_error"

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
