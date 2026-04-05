# Model Registry

`registry.json` is the single source of truth for all models used in ToolFailBench experiments.

## Schema

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Short identifier used in CLI args and result filenames (e.g. `qwen3.5-7b`) |
| `hf_model_id` | string \| null | Full Hugging Face model ID. `null` for closed API models. |
| `api_provider` | string | `openai` or `anthropic`. Only present for closed models. |
| `family` | string | Model family: `qwen`, `llama`, `mistral`, `gemma`, `glm`, `deepseek`, `moonshot`, `openai`, `anthropic` |
| `size` | string | Parameter count (e.g. `7B`, `72B`). MoE models show active params. `unknown` for closed. |
| `tier` | int | Experiment tier (1–5, see below) |
| `category` | string | `base` \| `reasoning` \| `closed` \| `experimental` |
| `inference_backend` | string | `vllm` \| `api` |
| `recommended_gpu` | string | Minimum GPU recommendation for serving (vllm models only) |
| `no_think` | bool | If true, append `/no_think` to system prompt (Qwen3.5 base models in Tier 1/2) |

## Tiers

| Tier | Purpose | Models |
|------|---------|--------|
| 1 | Small cross-family comparison (7–31B) | qwen3.5-7b, llama3.1-8b, mistral-7b, gemma4-31b, glm4-9b |
| 2 | Scaling comparison (70B+) | qwen3.5-72b, llama3.1-70b |
| 3 | Reasoning vs agent-optimized | qwq-32b, qwen3.5-32b, deepseek-r1-7b, gemma4-27b-a4b |
| 4 | Closed API baselines | gpt-4o, claude-sonnet-4 |
| 5 | Experimental | kimi-k2 |

## Key Model Pairs (most important comparisons)

**QwQ-32B vs Qwen3.5-32B (Tier 3)** — Same size, same family, but QwQ is RL-heavy reasoning (tends to overthink tool calls) while Qwen3.5-32B is agent-optimized (strong tool use, stable in loops). This is the primary pair for testing whether reasoning training helps or hurts tool-use faithfulness.

**Gemma4-31B vs Gemma4-27B-A4B (Tier 1 vs Tier 3)** — Same family, but Gemma4-31B is dense/non-thinking and Gemma4-27B-A4B is a MoE with thinking mode active. Controls for thinking vs non-thinking within the same family.

## Model Notes

- **Qwen3.5 vs Qwen3**: Qwen3.5 is used throughout because it has significantly better tool use and agent capabilities than Qwen3.
- **Qwen3.5 `/no_think` mode**: Qwen3.5 models support a non-thinking mode activated by appending `/no_think` to the system prompt. Registry entries with `"no_think": true` have this applied automatically by the runner. Used for Tier 1/2 base runs. QwQ-32B naturally thinks and does not use this flag.
- **Kimi-K2 (Tier 5)**: Moonshot's 1T MoE model with 32B active params. Experimental — may require custom vLLM setup.

## Inference Backends

**`vllm`** — Served via a local vLLM OpenAI-compatible server. Start before running eval:
```bash
vllm serve <hf_model_id> --port 8000 --tensor-parallel-size <n>
```
Set `VLLM_BASE_URL=http://localhost:8000/v1` in your `.env`.

**`api`** — Calls the provider's API via litellm. Requires `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `.env`.

## Adding a New Model

Add one JSON object to `registry.json`. That's it — no other files change.

```json
{
  "id": "your-short-id",
  "hf_model_id": "org/model-name-on-hf",
  "family": "family-name",
  "size": "13B",
  "tier": 1,
  "category": "base",
  "inference_backend": "vllm",
  "recommended_gpu": "1xA10G"
}
```

## Running Models

```bash
# Single model by id
python runners/run_eval.py --model qwen3.5-7b

# All models in a tier
python runners/run_eval.py --tier 1

# Multiple tiers
python runners/run_eval.py --tier 1 2 3

# All tiers
python runners/run_eval.py --tier 1 2 3 4 5
```
