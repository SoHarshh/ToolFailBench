# ToolFailBench: Current Status Summary

**Date:** April 14, 2026

---

## What is ToolFailBench?

A benchmark that measures **why** LLMs fail at tool use, not just whether they fail. We define 6 failure modes and evaluate models against 90 tasks across 9 domains with deliberately contradictory tool returns ("parametric traps") to expose failure patterns.

## The 6 Failure Modes


| Mode                     | Abbrev | What it means                                               |
| ------------------------ | ------ | ----------------------------------------------------------- |
| **Correct**              | CTUR   | Model called the tool, used the result correctly            |
| **Tool Skip**            | TSR    | Model should have called a tool but didn't                  |
| **Result Ignore**        | RIR    | Model called the tool but ignored the return value          |
| **Output Fabrication**   | OFR    | Model called the tool but fabricated fake structured output |
| **Unnecessary Tool Use** | UTR    | Control task: model called a tool when none was needed      |
| **Wrong Answer**         | --     | Control task: no tool called, but answer is incorrect       |


## Dataset

- **90 tasks** across **9 domains**: Finance, Medical, Code, Chemistry, EDA/Hardware, Geospatial, Legal, Cybersecurity, Nutrition
- Per domain: 5 tool-required tasks + 5 control tasks (answerable from general knowledge)
- Each tool-required task has a **parametric trap**: the mock tool return contradicts what the model likely memorized (e.g., tool says AAPL = $247.83, model trained on ~$180)
- An **LLM-as-judge** (Claude Sonnet 4) independently validates classifications

## Models Evaluated (Tier 1)


| Model                    | Size | Inference | Tool Call Parser |
| ------------------------ | ---- | --------- | ---------------- |
| Qwen3.5-9B               | 9B   | vLLM      | hermes           |
| Mistral-7B-Instruct-v0.3 | 7B   | vLLM      | hermes           |
| GLM-4-9B-Chat            | 9B   | vLLM      | glm45            |
| Llama-3.1-8B-Instruct    | 8B   | vLLM      | llama3_json      |
| Gemma-4-31B-IT           | 31B  | vLLM      | gemma4           |


All runs: temp=0, max_tokens=1024, seed=42, tool_choice=auto.

---

## Results

### Rule-Based Classification


| Model        | TSR       | RIR   | OFR       | CTUR      | UTR       | CTRL Acc |
| ------------ | --------- | ----- | --------- | --------- | --------- | -------- |
| Qwen3.5-9B   | **100%**  | 0%    | 0%        | 0%        | 0%        | 95.6%    |
| Mistral-7B   | **95.6%** | 0%    | 0%        | 0%        | 0%        | 88.9%    |
| GLM-4-9B     | **100%**  | 0%    | 0%        | 0%        | 0%        | 93.3%    |
| Llama-3.1-8B | 0%        | 17.8% | **46.7%** | 35.6%     | **88.9%** | 0%       |
| Gemma-4-31B  | 0%        | 22.2% | 0%        | **77.8%** | 4.4%      | 93.3%    |


Now one thing to point out here is the problem with our current 5: three of them do the same thing. Qwen, mistral, and glm4 all skip tools. So thats the same behavior experienced across 3 diff models. 



### Judge-Corrected (Claude Sonnet 4.5, authoritative)


| Model        | RIR       | OFR  | CTUR      | Agreement |
| ------------ | --------- | ---- | --------- | --------- |
| Llama-3.1-8B | **62.2%** | 2.2% | 28.9%     | 65.6%     |
| Gemma-4-31B  | 2.2%      | 0%   | **97.8%** | 87.8%     |


Tool-skip models (Qwen, Mistral, GLM) have 86-96% judge agreement. Their results are essentially unchanged by the judge.

---

## Three Archetypes Discovered

### 1. Tool-Skip (Qwen, Mistral, GLM)

Never call tools. Answer everything from parametric memory. High CTRL accuracy (89-96%) but 0% CTUR. GLM-4 produces byte-identical answers with and without tool access (87/90 tasks identical).

### 2. Always-Call (Llama-3.1-8B)

Calls tools on **everything** (UTR=89%). Only correctly uses results 29-36% of the time. Dominant failure is **multi-hop chaining** (see Finding 1 below).

### 3. Intermediate (Gemma-4-31B)

Calls tools selectively and correctly. Judge-corrected CTUR=97.8%. Low unnecessary tool use (UTR=4.4%). Best tool user by far.

---

## Key Findings

### Finding 1: Multi-Hop Chaining (Quite Diff I would Say)

Llama-3.1-8B's dominant failure (51% of tool-required tasks) is **trying to call a second tool** instead of answering. After receiving the first tool's result, it outputs another tool call as plain text (e.g., `{"name": "get_market_cap", "parameters": {"ticker": "RIVN"}}`).

This reveals a **limitation of single-call benchmarks**: multi-hop models get systematically mislabeled. The rule-based classifier sees fabricated JSON (OFR=46.7%), but the judge correctly identifies this as result-ignoring (RIR=62.2%) — the model got the result and didn't use it.

### Finding 2: Judge Robustness

The LLM judge is more accurate than rule-based detection:

- **Formatting tolerance**: Gemma answers "$1,001" for ground truth "1001" — rule-based says RIR, judge says correct. 9 of 10 Gemma RIR cases overturned.
- **Stable across pipeline artifacts**: Whether Llama outputs empty strings (old bug) or tool-call JSON (after fix), the judge consistently classifies as RIR.

### Finding 3: Unnecessary Tool Use Degrades Accuracy

On control tasks (answerable from memory), Llama's compulsive tool-calling **drops accuracy from 95% to 50%**:

- Baseline (no tools): 95% correct
- With tools (unnecessary calls): 50% correct
- 17 tasks degraded, **0 improved**
- 11/38 answers are "meta-talk" — model describes the tool call instead of answering

### Finding 4: Not Parametric Fallback

RIR–baseline correlation analysis shows models do **not** revert to training data when ignoring tool results:

- Baseline GT match = 0% for all RIR tasks — models don't have the trap values memorized
- No numeric value transfer from baseline to eval answers
- Eval-baseline text similarity is low (Jaccard=0.09-0.13)

The failure is at the **result integration stage**, not knowledge retrieval.

---

## Infrastructure

- **Eval runner**: `runners/run_eval.py` — litellm + vLLM, W&B logging, raw response archival
- **Judge runner**: `runners/run_judge.py` — Claude Sonnet 4, ~$0.35/model
- **Baseline runner**: `runners/run_parametric_baseline.py` — no-tool comparison answers
- **Analysis**: `evaluation/baseline_analysis.py` — RIR-baseline correlation + plots
- **All results**: W&B project `toolfailbench` ([link](https://wandb.ai/moe-research/toolfailbench))

---

## Planning to do later (can be changed)


| Task                                      | Effort        | Status         |
| ----------------------------------------- | ------------- | -------------- |
| GPT-4o run (API)                          | 1 hour        | Ready to run   |
| Optional 5th model (llama-70b or qwq-32b) | 2 hours + GPU | Configured     |
| Expand to 180 tasks (10 more/domain)      | 1-2 days      | Nice-to-have   |
| Paper draft (4-6 pages)                   | 3-5 days      | Findings ready |


---

## Files & Artifacts


| File                                                   | Description                        |
| ------------------------------------------------------ | ---------------------------------- |
| `results/llama3.1-8b_20260414_081332.json`             | Llama eval (corrected)             |
| `results/gemma4-31b_20260414_091432.json`              | Gemma eval (corrected)             |
| `results/judge/llama3.1-8b_judge_20260414_094319.json` | Llama judge results                |
| `results/judge/gemma4-31b_judge_20260413_103611.json`  | Gemma judge results                |
| `results/llama3.1-8b_rir_analysis.json`                | RIR-baseline correlation           |
| `evaluation/baseline_analysis.py`                      | Correlation analysis code          |
| `plan.md`                                              | Full project plan with all details |


