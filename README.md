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
uv pip install -r requirements.txt
```

## Usage

```bash
# Run evaluation on a model
python runners/run_eval.py --model <model_name> --domains finance medical code

# Collect parametric baselines (no tools)
python runners/run_parametric_baseline.py --model <model_name>
```

## Current Status

15 seed tasks across 3 domains (5 each). Distribution: 6 Tool-Skip, 4 Result-Ignore, 5 Output-Fabrication.

## References

- tau-bench (Yao et al., 2024)
- tau2-bench (Barres et al., 2025)
- tau-Knowledge (Shi et al., 2026)
- ToolBench (Qin et al., 2023)
- BFCL (Patil et al., 2024)
- ToolBeHonest (Wang et al., 2024)
- The Reasoning Trap (2025)
