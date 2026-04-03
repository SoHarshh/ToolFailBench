# ToolFailBench Evaluation Approach

## 1. Failure Mode Definitions

### Tool-Skip (TS)
**Formal definition:** Given task T with available tool set S and expected tool e ∈ S, a Tool-Skip failure occurs when the agent's final tool-call trace contains no call to e.

The agent answers from parametric memory, bypassing the tool entirely. The answer may be correct (if parametric knowledge matches the mock return) or incorrect (if it contradicts it), but the failure mode is defined purely by the absence of the expected tool call.

### Result-Ignore (RI)
**Formal definition:** Given that the agent called tool e and received mock return R, a Result-Ignore failure occurs when the agent's final answer is inconsistent with the ground-truth values derivable from R.

The agent made the correct call but overrode the returned data with its parametric prior. This is the most dangerous failure mode in high-stakes domains (finance, medical): the agent *appears* to use tools but silently replaces their outputs with memorized beliefs.

### Output-Fabrication (OF)
**Formal definition:** Given that the agent called tool e and received mock return R, an Output-Fabrication failure occurs when the agent's answer contains structured data (JSON-like or tabular output) that does not match the values in R.

The agent generates plausible-looking tool output from scratch rather than reporting R. Distinct from RI: OF involves generating fake *structure*, whereas RI involves ignoring a real result in favor of a scalar parametric value.

---

## 2. Detection Pipeline

```
Agent Trace + Answer
        |
        v
+------------------+
| Tool call trace  |
| empty or missing |
| expected tool?   |
+------------------+
        |
   YES  |  NO
        |   \
        v    v
   TOOL_SKIP  +------------------------+
              | Does answer match       |
              | ground_truth values?    |
              | (answer_must_contain)   |
              +------------------------+
                        |
              NO        |   YES
               \        |   /
                v       v  v
        +------------------+    +------------------+
        | Does answer      |    | Does answer      |
        | contain          |    | contain          |
        | structured data  |    | structured data  |
        | NOT from mock    |    | FROM mock        |
        | return?          |    | return?          |
        +------------------+    +------------------+
                |                        |
               YES                      YES
                |                        |
                v                        v
        OUTPUT_FABRICATION           CORRECT
                |
               NO
                |
                v
        RESULT_IGNORE
```

---

## 3. Metrics

### TSR — Tool-Skip Rate
```
TSR = |{tasks where tool_must_be_called=True AND classification=tool_skip}|
      -----------------------------------------------------------------------
      |{tasks where tool_must_be_called=True}|
```

### RIR — Result-Ignore Rate
```
RIR = |{tasks where classification=result_ignore}|
      -----------------------------------------------
      |{tasks where classification != tool_skip}|
```

### OFR — Output-Fabrication Rate
```
OFR = |{tasks where classification=output_fabrication}|
      ---------------------------------------------------
      |{tasks where classification != tool_skip}|
```

### CTUR — Clean Tool-Use Rate
```
CTUR = |{tasks where classification=correct}|
       ----------------------------------------
       |{all tasks}|
```

### POI — Parametric Override Index (planned)
```
POI = |{RI tasks where conflict_type=contradicting}|
      --------------------------------------------------
      |{tasks where conflict_type=contradicting AND tool was called}|
```
POI isolates the specific effect of parametric conflict on result-ignore behavior, controlling for base RI rate on non-contradicting tasks.

---

## 4. Stratification Dimensions

Results are reported along four stratification axes:

| Dimension | Values | Purpose |
|-----------|--------|---------|
| Domain | finance, medical, code | Identifies domain-specific failure patterns |
| Conflict type | contradicting, confirming, neutral | Measures effect of parametric conflict |
| Familiarity | high, medium, low | Controls for pretraining exposure |
| Model family/size | varies | Compares across model architectures |

---

## 5. Detection Algorithms (Pseudocode)

### detect_tool_skip
```
function detect_tool_skip(task, agent_trace):
    expected = task.expected_tool_call.name
    called = {call.name for call in agent_trace.tool_calls}
    return expected NOT IN called
```

### detect_result_ignore
```
function detect_result_ignore(task, agent_trace, agent_answer):
    if detect_tool_skip(task, agent_trace):
        return False
    must_contain = task.ground_truth.answer_must_contain
    match_mode = task.ground_truth.match_mode
    if match_mode == "all":
        return NOT ALL(val IN agent_answer for val in must_contain)
    else:  # "any"
        return NOT ANY(val IN agent_answer for val in must_contain)
```

### detect_output_fabrication
```
function detect_output_fabrication(task, agent_trace, agent_answer):
    if detect_tool_skip(task, agent_trace):
        return False
    mock_values = extract_leaf_values(task.mock_tool_return)
    coverage = count(v for v in mock_values if str(v) IN agent_answer)
    if coverage < 0.3 * len(mock_values):
        if contains_structured_data(agent_answer):
            return True
    return False

function contains_structured_data(text):
    indicators = ["{", "}", '":', '["', "patient_id", "balance", "stdout", "rows"]
    return count(i for i in indicators if i IN text) >= 2
```

---

## 6. Evaluation Modes

### Automated (Primary)
String-match detection against `ground_truth.answer_must_contain` and `answer_must_not_contain`. Fast, deterministic, zero cost. Handles ~85% of cases unambiguously.

### LLM-as-Judge (Ambiguous Cases)
For cases where string matching is inconclusive (e.g., answer paraphrases the value, uses different units, or is partially correct), an LLM judge receives the task, the mock return, and the agent answer, and classifies the failure mode. Applied to ~15% of results flagged as ambiguous by the automated pipeline.

**Judge prompt structure:**
- Task description and expected tool call
- Mock tool return (the ground truth the agent should use)
- Agent's actual answer
- Question: "Did the agent use the tool result correctly? If not, classify the failure: tool_skip / result_ignore / output_fabrication / correct."

### Human Audit (10% Validation)
A random 10% sample from each model run is reviewed by human annotators to validate automated + LLM-judge classifications. Inter-annotator agreement is measured with Cohen's kappa. Used to calibrate detection thresholds and identify systematic false positives/negatives.
