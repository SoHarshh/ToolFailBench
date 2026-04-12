# ToolFailBench: Benchmarking Mechanistic Tool-Use Failure Modes in LLM Agents

## Overview

Current tool-use benchmarks (ToolBench, BFCL, tau-bench) evaluate whether agents completed tasks but don't distinguish WHY they failed. A model that never calls a calculator (Tool-Skip) and one that calls it but ignores the result (Result-Ignore) both get the same "incorrect" label. We identify three mechanistically distinct failure modes and build a benchmark with controlled mock tool returns to measure each independently across open-source LLM families.

## Failure Modes

### Tool-Skip (TS)
Agent answers from parametric memory without calling the available tool.

### Result-Ignore (RI)
Agent calls the tool correctly but generates output inconsistent with the returned result.

### Output-Fabrication (OF)
Agent calls the tool but fabricates a plausible-looking response instead of using the actual return value.

### Control (CTRL)
Paired tasks where no tool call is needed. Tests whether the model correctly avoids unnecessary tool use.

## Domains

- **Finance:** Stock prices, exchange rates, cryptocurrency prices, market capitalization, bank transactions
- **Medical:** Drug dosages, drug interactions, patient records, lab results
- **Code Execution:** Arithmetic, SQL queries, hash computation
- **Chemistry:** Molecular properties, reaction compatibility, safety data sheets, titration calculations
- **EDA / Hardware Design:** Logic synthesis, timing analysis, Verilog simulation, FPGA power estimation
- **Geospatial:** Distances, GPS coordinates, elevation data, travel time estimation
- **Legal / Case Law:** Court rulings, statute text, regulation status, case law search
- **Cybersecurity:** CVE lookups, vulnerability status, exploit details, dependency scanning
- **Nutrition:** Calorie counts, nutritional data, allergen databases, meal macro calculations

## Design Principles

1. **Mode Isolation:** Each task targets one failure mode
2. **Controlled Returns:** Mock server returns predetermined values
3. **Parametric Conflict:** Tool returns deliberately contradict model priors
4. **Detection by Design:** Unusual values (rare names, non-round numbers) make fabrication detection trivial

## Project Structure

```
ToolFailBench/
├── README.md
├── .gitignore
├── .env.example
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── default.yaml          # inference + W&B + vLLM settings
├── models/
│   ├── registry.py            # loads all model configs
│   ├── README.md
│   └── configs/
│       ├── qwen/              # qwen3.5-7b, qwen3.5-32b, qwen3.5-72b, qwq-32b
│       ├── llama/             # llama3.1-8b, llama3.1-70b
│       ├── mistral/           # mistral-7b
│       ├── gemma/             # gemma4-31b, gemma4-27b-a4b
│       ├── glm/               # glm4-9b
│       ├── deepseek/          # deepseek-r1-7b
│       ├── moonshot/          # kimi-k2
│       ├── openai/            # gpt-4o
│       └── anthropic/         # claude-sonnet-4
├── tasks/
│   ├── schema.json
│   ├── finance/tasks.json        # 10 tasks (2 TS, 2 RI, 1 OF, 5 CTRL)
│   ├── medical/tasks.json        # 10 tasks (2 TS, 1 RI, 2 OF, 5 CTRL)
│   ├── code/tasks.json           # 10 tasks (2 TS, 1 RI, 2 OF, 5 CTRL)
│   ├── chemistry/tasks.json      # 10 tasks (2 TS, 1 RI, 2 OF, 5 CTRL)
│   ├── eda/tasks.json            # 10 tasks (2 TS, 1 RI, 2 OF, 5 CTRL)
│   ├── geospatial/tasks.json     # 10 tasks (2 TS, 1 RI, 2 OF, 5 CTRL)
│   ├── legal/tasks.json          # 10 tasks (2 TS, 1 RI, 2 OF, 5 CTRL)
│   ├── cybersecurity/tasks.json  # 10 tasks (2 TS, 1 RI, 2 OF, 5 CTRL)
│   └── nutrition/tasks.json      # 10 tasks (2 TS, 1 RI, 2 OF, 5 CTRL)
├── tools/
│   └── mock_server.py         # FastAPI mock tool server
├── evaluation/
│   ├── data.py                # shared task loader
│   ├── detect.py              # failure mode classification
│   ├── metrics.py             # TSR, RIR, OFR, CTUR, UTR, CTRL accuracy
│   └── report.py              # summary tables, JSON export
├── runners/
│   ├── run_eval.py            # main eval runner (with tools)
│   └── run_parametric_baseline.py  # baseline runner (no tools)
└── results/
    └── .gitkeep
```

## Setup

```bash
cp .env.example .env   # fill in your API keys
uv pip install -r requirements.txt
```

## Usage

```bash
# Run a single model by registry id
python runners/run_eval.py --model qwen3.5-7b

# Run all models in a tier
python runners/run_eval.py --tier 1

# Run multiple tiers
python runners/run_eval.py --tier 1 2 3 4

# Collect parametric baselines (no tools)
python runners/run_parametric_baseline.py --model qwen3.5-7b
python runners/run_parametric_baseline.py --tier 1
```

See `models/README.md` for the full model registry and how to add new models.

## Metrics

| Metric | Applies to | Definition |
|--------|-----------|------------|
| TSR | Tool-required tasks | Fraction where agent skipped the tool |
| RIR | Tool-required tasks | Fraction where tool was called but result ignored |
| OFR | Tool-required tasks | Fraction where tool was called but output fabricated |
| CTUR | Tool-required tasks | Fraction fully correct |
| UTR | CTRL tasks | Fraction where agent called a tool unnecessarily |
| CTRL Acc | CTRL tasks | Fraction answered correctly without tool use |

## Current Status

90 tasks across 9 domains (10 each). Distribution: 18 TS, 10 RI, 17 OF, 45 CTRL. 14 models across 5 tiers.

## References

- tau-bench (Yao et al., 2024)
- tau2-bench (Barres et al., 2025)
- tau-Knowledge (Shi et al., 2026)
- ToolBench (Qin et al., 2023)
- BFCL (Patil et al., 2024)
- ToolBeHonest (Wang et al., 2024)
- The Reasoning Trap (2025)
