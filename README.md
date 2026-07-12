<h1 align="center">ToolFailBench</h1>

<p align="center"><b>A Diagnostic Benchmark for Tool-Use Failures in LLM Agents</b></p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.04686"><img alt="Paper (arXiv)" src="https://img.shields.io/badge/arXiv-2607.04686-b31b1b?logo=arxiv&logoColor=white"></a>
  <a href="https://openreview.net/forum?id=JhaxRN8QDV"><img alt="Paper (OpenReview)" src="https://img.shields.io/badge/OpenReview-Paper-8c1b13?logo=openreview&logoColor=white"></a>
  <a href="https://github.com/SoHarshh/ToolFailBench"><img alt="Code (GitHub)" src="https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white"></a>
</p>

<h3 align="center"><a href="https://soharshh.com">Harsh Soni</a></h3>

<p align="center">University of California, Berkeley</p>

---

## Abstract

Tool calling is central to modern language model agents, but aggregate benchmark scores often hide where tool use fails. A model that never calls a needed tool and a model that calls the tool but ignores the result can look similar under final task accuracy. We introduce *ToolFailBench*, a diagnostic benchmark for measuring tool-use failures across 1,000 tasks in finance, medicine, law, cybersecurity, and real estate. Tool-required tasks return values the model can't guess, forcing it to trust the tool. Control tasks attach the same tools but should be answered directly. We label each trace with a failure-mode taxonomy covering Tool-Skip, Result-Ignore, Output-Fabrication, and Unnecessary-Tool-Use, using a deterministic rule classifier and two independent LLM judges aggregated by majority vote. Across 19 headline models, the best reaches **86.33% Clean Tool-Use Rate**, showing that faithful tool use is not saturated. More importantly, models with similar aggregate scores fail in different ways: most stay disciplined on no-tool controls, while Llama-3.1 models show an Always-Call pattern, and at the same parameter scale Llama-3.1-70B and Qwen2.5-72B differ by **89 percentage points** on control-task accuracy. Tool-use evaluation should measure not only whether agents call tools, but whether they use tool outputs correctly and avoid tools when none is needed.

---

## The four failure modes

Each task in the probe set is engineered to elicit exactly one of these. The mode is a property of *what the model did*, not of the question.

| Mode | Setting | Failure signature |
|---|---|---|
| **Tool-Skip (TS)** | Tool needed; model never makes a valid executed tool call | Answers from memorized prior; refuses; or emits tool-call-shaped text that wasn't actually executed |
| **Result-Ignore (RI)** | Tool needed; model called the tool and got the answer back | Final answer contradicts the returned value (typically falls back to training-data memory) |
| **Output-Fabrication (OF)** | Tool needed; model called the tool | Final answer cites structured data fields, dates, identifiers, or numbers that were *not* in the tool return |
| **Unnecessary-Tool-Use (UTU)** | No tool needed (control task) | Model calls a tool anyway — the Always-Call pathology |

A real model hits these in different mixtures. The mixture is the *archetype*: e.g., **Disciplined** (low UTR, high CTUR), **Always-Call** (high UTR, low CTRL accuracy), **Phantom-Call Hallucinator** (100% TSR, fabricates tool-result-style answers without emitting tool calls), **Tokenization-Broken** (output is raw-token salad).

---

## Key findings

- **Faithful tool use is not saturated.** The strongest model, Grok-4.3, reaches 86.33% Clean Tool-Use Rate (CTUR) — well below ceiling. The next two are Grok-4-1-Fast-Reasoning (84.11%) and Qwen2.5-32B-Instruct (82.68%).
- **High CTUR ≠ calling tools more often.** The top models combine moderate Tool-Skip rates with low Result-Ignore and Output-Fabrication. They aren't the most aggressive callers; they are the most accurate ones.
- **Aggregate scores hide failure profiles.** Most models cluster in a *Disciplined* low-UTR region (UTR ≤ 1.61%, CTRL-Acc ≥ 95%). Llama-3.1-8B and Llama-3.1-70B form *Always-Call* outliers (UTR 98.39% and 77.73%; CTRL-Acc 0.00% and 8.91%).
- **Scale doesn't fix tool discipline.** Llama-3.1 8B → 70B improves CTUR and RIR but barely changes UTR (still 77.73% on no-tool tasks). Same-scale Qwen2.5-72B and Llama-3.1-70B differ by **89 percentage points** on CTRL-Acc on identical inputs.
- **Two-judge ensemble is robust.** Mean Cohen's κ rule↔Qwen = 0.649; rule↔GLM = 0.664; Qwen↔GLM = 0.773; three-way Fleiss' = 0.693. Judge–judge agreement exceeds either rule–judge agreement.
- **Result-faithfulness varies by domain.** Median rule RIR is **12.24%** in Finance vs **0.68%** in Cybersecurity (across the 17 disciplined models) — supporting the five-domain design.

---

## Leaderboard

19 headline models with valid traces, sorted by ensemble Clean Tool-Use Rate. `Ens.` = majority vote across the rule classifier and the two LLM judges (Qwen3.5-397B-A17B-FP8 + GLM-4.7-FP8); ties → rule. `Rule CTUR` uses the rule classifier only. Higher is better for CTUR and CTRL-Acc; lower is better for TSR, RIR, OFR, UTR. All values are percentages.

| # | Model | Ens. CTUR | Rule CTUR | Ens. TSR | Ens. RIR | Ens. OFR | Ens. UTR | Ens. CTRL-Acc |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | **Grok-4.3** | **86.33** | 80.67 | 11.80 | 1.74 | 0.13 | 0.81 | 97.18 |
| 2 | Grok-4-1-Fast-Reasoning | 84.11 | 80.67 | 14.15 | 1.60 | 0.13 | 0.40 | 99.60 |
| 3 | Qwen2.5-32B-Instruct | 82.68 | 78.27 | 12.08 | 4.43 | 0.81 | 0.80 | 95.58 |
| 4 | Qwen3.6-27B | 79.33 | 76.67 | 19.87 | 0.67 | 0.13 | 0.00 | 98.79 |
| 5 | Claude-Sonnet-4.5 | 79.28 | 73.07 | 15.64 | 4.41 | 0.67 | 0.00 | 100.00 |
| 6 | GPT-5.4-Mini | 79.14 | 72.27 | 14.17 | 5.48 | 1.20 | 0.00 | 97.20 |
| 7 | QwQ-32B | 79.04 | 75.47 | 16.02 | 3.07 | 1.87 | 1.61 | 95.98 |
| 8 | Qwen2.5-72B-Instruct | 79.00 | 75.20 | 18.57 | 2.02 | 0.40 | 0.00 | 98.00 |
| 9 | Qwen3.6-35B-A3B | 78.47 | 75.47 | 18.98 | 2.15 | 0.40 | 0.00 | 99.60 |
| 10 | Gemma-4-31B | 78.12 | 70.80 | 19.06 | 2.68 | 0.13 | 0.00 | 98.39 |
| 11 | Qwen3.5-27B | 77.38 | 75.47 | 21.55 | 0.67 | 0.40 | 0.00 | 99.60 |
| 12 | Claude-Haiku-4.5 | 76.47 | 68.93 | 18.85 | 2.81 | 1.87 | 0.00 | 100.00 |
| 13 | DeepSeek-V4-Flash | 75.84 | 71.33 | 17.45 | 4.30 | 2.42 | 1.20 | 98.39 |
| 14 | Gemma-4-27B-A4B | 73.49 | 68.40 | 24.10 | 1.74 | 0.67 | 0.00 | 99.20 |
| 15 | GLM-4.7-Flash | 71.49 | 68.80 | 24.90 | 3.21 | 0.40 | 0.00 | 99.19 |
| 16 | Qwen3.5-9B | 70.03 | 70.40 | 26.48 | 2.82 | 0.67 | 0.00 | 99.60 |
| 17 | Qwen2.5-7B-Instruct | 65.28 | 60.80 | 28.53 | 5.11 | 1.08 | 0.00 | 95.16 |
| 18 | Llama-3.1-70B | 62.58 | 59.73 | 24.23 | 11.17 | 2.02 | 77.73 | 8.91 |
| 19 | Llama-3.1-8B | 47.32 | 47.60 | 20.64 | 30.43 | 1.61 | 98.39 | 0.00 |

Three additional runs are documented as non-headline in the paper (Appendix D): `glm-4-9b` and `mistral-7b` produced zero clean executed tool calls (Tool-Skip cases), and `deepseek-r1-distill-llama-8b` is excluded because its output was invalid raw-token text from a chat-template/detokenizer mismatch.

---

## Setup

```bash
git clone https://github.com/SoHarshh/ToolFailBench.git
cd ToolFailBench
./setup.sh
source .venv/bin/activate
```

> To **serve open-weight models locally**, install `vllm` separately (`pip install vllm`, Linux/CUDA only). It is kept out of `requirements.txt` so the core install works on any machine; closed-API evaluation does not need it.

Provider API keys (for closed-API models and the two LLM judges) are read from the shell environment. Export whichever you need:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export XAI_API_KEY=...
export HF_TOKEN=...                               # for gated open-weight models (e.g. Llama)
export VLLM_BASE_URL=http://localhost:8000/v1    # if serving open-weight models locally
export JUDGE_QWEN35_API_BASE=...                 # judge 1 endpoint
export JUDGE_GLM47_API_BASE=...                  # judge 2 endpoint
```

> **New here?** [`quickstart.ipynb`](quickstart.ipynb) runs the benchmark on one model end-to-end with no GPU — the fastest way to see the four failure modes.

### Run an eval

```bash
# Closed-API model (no GPU)
python scripts/run_eval.py --model grok-4.3
# Open-weight model (requires a running vLLM server — see scripts/serve_model.py)
python scripts/run_eval.py --model qwen2.5-32b-instruct
```

Outputs land in `results/v5/<model>_<timestamp>.json`. The runner attaches the per-domain tools, executes the two-call protocol, and produces both the rule-classifier label and the full trace.

### Run the dual-judge ensemble

```bash
make judge EVAL=results/v5/<model>_<timestamp>.json
```

This runs both LLM judges (Qwen3.5-397B-A17B-FP8 + GLM-4.7-FP8) over the trace and writes per-judge JSONs plus the rule + 3-rater ensemble (Cohen's pairwise κ + Fleiss' κ + majority-vote label) to `results/v5/judge_ensemble/<model>_ensemble.json`.

### Result traces

The full per-model evaluation and judge traces are released as the HuggingFace dataset **[SoHarshh/toolfailbench-traces](https://huggingface.co/datasets/SoHarshh/toolfailbench-traces)**. To regenerate the leaderboard from them:

```bash
pip install huggingface_hub
hf download SoHarshh/toolfailbench-traces --repo-type dataset --local-dir results/v5
python evaluation/validate_results.py
```

---

## Repository structure

```
.
├── README.md
├── LICENSE                         Apache 2.0
├── setup.sh                        one-shot install
├── Makefile                        common workflows (eval, judge, serve)
├── pyproject.toml / requirements.txt
├── configs/default.yaml            inference defaults (temperature=0, seed=42, max_tokens=1024)
│
├── tasks_v5/                       1,000 tasks × 5 domains (200 each)
│   ├── finance/                    tasks.json + tools.json
│   ├── medical/  legal/  cybersecurity/  real_estate/
│
├── system_prompts/v5/              one shared system prompt per domain
│
├── models/
│   ├── configs/                    one YAML per model (drop a file → it's registered)
│   └── registry.py                 loads + validates configs at runtime
│
├── evaluation/
│   ├── detect.py                   rule classifier (4 failure modes)
│   ├── metrics.py                  TSR / CTUR / RIR / OFR / UTR / CTRL-Acc
│   ├── data.py                     task loader
│   ├── report.py                   summary + JSON writer
│   ├── validate_results.py         decimal-reproducible leaderboard from result JSONs
│   └── judges/                     LLM-judge code, prompts, and configs
│       ├── judge.py                judge-call wrapper
│       ├── aggregate.py            Cohen's pairwise κ + Fleiss' κ + ensemble label
│       ├── configs/                per-judge YAML (model, endpoint, sampling, overlay)
│       └── prompts/                base rubric + variant overlays (decision-tree, evidence)
│
├── modal_apps/                     Modal serving definitions (the two judges + open-weight serving)
│
└── scripts/
    ├── run_eval.py, run_judge.py     entry points
    ├── serve_model.py, preflight.py  serving + pre-run checks
    └── run_eval.sh, run_judge.sh     serve→preflight→eval and judge→aggregate orchestration
```

---

## Roadmap

- [ ] **Diagnostic harness**: plug in any model, get the four-mode breakdown back
- [ ] **Multi-turn variants** (tool chaining, recovery, state)
- [ ] **API-based agent setup** (OpenAI Agents SDK, Anthropic Claude Agent SDK)
- [ ] **Model list expansion** to current SOTA closed-source models
- [ ] **Domain expansion** beyond the current five
- [ ] **More failure taxonomies** beyond Tool-Skip / Result-Ignore / Output-Fabrication / Unnecessary-Tool-Use

---

## Citation

```bibtex
@inproceedings{soni2026toolfailbench,
  title         = {ToolFailBench: Diagnosing Tool-Use Failures in {LLM} Agents},
  author        = {Harsh Soni},
  booktitle     = {Second Workshop on Agents in the Wild: Safety, Security, and Beyond},
  year          = {2026},
  eprint        = {2607.04686},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2607.04686}
}
```

---

## License

Released under the Apache License 2.0. See [`LICENSE`](LICENSE).
