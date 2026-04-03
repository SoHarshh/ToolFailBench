# Literature Review: Tool-Use Evaluation Benchmarks

## 1. tau-bench (Yao et al., 2024)

**What it does:** A multi-turn agent benchmark that simulates realistic customer-service interactions. A language model plays the role of a user (with hidden goals), and the agent must satisfy those goals while following policy. Tasks span retail and airline domains.

**How it evaluates:** Compares the final database state against an expected end-state. Uses a pass^k consistency metric — a task is "passed" only if the agent succeeds on k independent runs, measuring reliability rather than single-run accuracy.

**What it's missing for our purposes:** No failure mode classification. An agent that never uses a tool (Tool-Skip) and one that calls the right tool but ignores the result (Result-Ignore) both produce an incorrect database state and receive the same score. There is no controlled injection of mock tool returns, so the benchmark cannot isolate individual tool-use failure modes.

---

## 2. tau2-bench (Barres et al., 2025)

**What it does:** Extends tau-bench with a dual-control design: both the agent and the simulated user now have tool access. Adds a telecom domain and introduces fault attribution — labelling whether failures stem from the agent, the user, or the environment.

**How it evaluates:** Same pass^k state comparison, augmented with fault attribution labels assigned by an LLM judge.

**What it's missing for our purposes:** Fault attribution answers "WHO failed" (agent vs. user vs. environment), not "HOW the tool-use failed." A Result-Ignore failure (agent called the tool but ignored its output) and an Output-Fabrication failure (agent generated fake tool-like output) would both be attributed to "agent" without distinguishing the mechanistic cause.

---

## 3. tau-Knowledge (Shi et al., 2026)

**What it does:** Adds a retrieval layer to the tau-bench framework: agents must retrieve relevant information from a corpus of ~700 banking documents before acting. Introduces a banking domain.

**How it evaluates:** Same pass^k state comparison; additionally measures retrieval recall over the document corpus.

**What it's missing for our purposes:** Evaluates retrieval faithfulness — whether the agent correctly locates and applies knowledge documents — not tool-use faithfulness. The failure modes of interest (Tool-Skip, Result-Ignore, Output-Fabrication) involve structured tool calls returning discrete values, not retrieval over text corpora.

---

## 4. ToolBench (Qin et al., 2023)

**What it does:** A large-scale benchmark covering multi-step API orchestration across 16,000+ real-world APIs from RapidAPI. Agents must decompose complex instructions into sequences of API calls and combine results.

**How it evaluates:** Binary pass/fail judged by a "ToolEval" LLM evaluator comparing agent outputs against reference solutions.

**What it's missing for our purposes:** Binary pass/fail provides no failure taxonomy. There is no distinction between agents that skip tool calls, agents that call tools but fabricate the returned results, or agents that call tools correctly. Controlled tool return injection is impossible because ToolBench uses real APIs.

---

## 5. BFCL (Patil et al., 2024) — Berkeley Function-Calling Leaderboard

**What it does:** Evaluates the accuracy of a model's function call syntax — whether the model selects the correct function name and passes the correct arguments. Covers simple, nested, parallel, and multi-turn scenarios.

**How it evaluates:** AST-based comparison of the generated function call against a ground-truth call. No execution; only the call structure is checked.

**What it's missing for our purposes:** BFCL evaluates whether the model *makes* the correct call, not what it does *after* the call returns a result. Result-Ignore and Output-Fabrication failures are invisible to BFCL because they occur after the tool call is issued. A model that calls the tool perfectly but ignores the return would score 100% on BFCL while completely failing in practice.

---

## 6. ToolBeHonest (Wang et al., 2024)

**What it does:** Tests two complementary capabilities: (1) solvability detection — whether the agent correctly identifies that a task can or cannot be solved with available tools; (2) planning hallucination — whether agents fabricate non-existent tool calls.

**How it evaluates:** Separate accuracy metrics for solvability judgements and for tool-call hallucination detection.

**What it's missing for our purposes:** ToolBeHonest covers the decision of *whether to use tools* and whether agents invent tools that don't exist. It does not cover what happens after a real tool is invoked: the Result-Ignore and Output-Fabrication modes occur downstream of a correct invocation, outside ToolBeHonest's scope.

---

## 7. The Reasoning Trap (2025)

**What it does:** Mechanistic analysis showing that reinforcement learning fine-tuning on reasoning benchmarks degrades tool-reliability representations in smaller LLMs. RL-tuned models develop stronger internal reasoning circuits that compete with and suppress tool-use pathways.

**How it evaluates:** Probing experiments and attention analysis on 7-8B models; behavioral comparisons between base and RL-tuned variants.

**What it's missing for our purposes:** Provides mechanistic insight into *why* tool-use degrades under RL, but does not provide a benchmark to measure the three failure modes independently. There is no controlled experimental setup to quantify Tool-Skip, Result-Ignore, and Output-Fabrication rates across model families.

---

## Gaps Summary

The existing landscape leaves four critical gaps that ToolFailBench addresses:

1. **No benchmark separates TS/RI/OF.** Every existing benchmark collapses all tool-use failures into a single "incorrect" category. A model that never calls tools and one that calls tools but ignores results are indistinguishable.

2. **No controlled tool-return injection.** Real-API benchmarks (ToolBench) cannot inject predetermined return values. Without controlled returns, it is impossible to test whether a model correctly uses a specific result versus ignoring it.

3. **No parametric conflict measurement.** No benchmark deliberately pits tool returns against the model's parametric priors to measure how strongly parametric memory overrides fresh tool evidence. This is the core condition for eliciting Result-Ignore failures.

4. **No difficulty stratification by familiarity.** Existing benchmarks do not stratify tasks by how familiar the question-answer pair is from pretraining. Familiarity is a key predictor of Tool-Skip risk: models skip tools more readily when they "know" the answer.
