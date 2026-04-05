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

## Domains

- **Finance:** Stock prices, exchange rates, cryptocurrency prices, market capitalization, bank transactions
- **Medical:** Drug dosages, drug interactions, patient records, lab results
- **Code Execution:** Arithmetic, SQL queries, hash computation

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
├── requirements.txt
├── tasks/
│   ├── schema.json
│   ├── finance/
│   │   └── tasks.json
│   ├── medical/
│   │   └── tasks.json
│   ├── code/
│   │   └── tasks.json
│   └── control/
│       └── tasks.json
├── tools/
│   ├── tool_definitions.json
│   └── mock_server.py
├── evaluation/
│   ├── __init__.py
│   ├── detect.py
│   ├── metrics.py
│   └── report.py
├── runners/
│   ├── __init__.py
│   ├── run_eval.py
│   └── run_parametric_baseline.py
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
python runners/run_eval.py --model qwen2.5-7b --domains finance medical code

# Run all models in a tier
python runners/run_eval.py --tier 1

# Run multiple tiers
python runners/run_eval.py --tier 1 2 3 4

# Collect parametric baselines (no tools)
python runners/run_parametric_baseline.py --model qwen2.5-7b
```

See `models/README.md` for the full model registry and how to add new models.

## Current Status

30 tasks across 3 domains (10 each). Distribution: 6 Tool-Skip, 4 Result-Ignore, 5 Output-Fabrication, 15 Control (paired no-tool-needed tasks).

## References

- tau-bench (Yao et al., 2024)
- tau2-bench (Barres et al., 2025)
- tau-Knowledge (Shi et al., 2026)
- ToolBench (Qin et al., 2023)
- BFCL (Patil et al., 2024)
- ToolBeHonest (Wang et al., 2024)
- The Reasoning Trap (2025)
