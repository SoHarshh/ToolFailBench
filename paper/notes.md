# ToolFailBench Paper Notes

## Target venues
- Primary: NeurIPS 2026 (deadline ~May 22)
- Fallback: ICML 2026 workshop

## Key narrative
Not "here's a benchmark" but "here's what we found about how tool-augmented LLMs fail differently."

## Hypotheses to test
- H1: TS/RI/OF rates are independent across models
- H3: RI spikes when tool return contradicts parametric knowledge
- H5: Reasoning-enhanced models show higher TS but lower OF

## TODO
- [ ] Find one surprising result from initial evals
- [ ] Mech interp if time (probing, attention on 7-8B models)
- [ ] Mitigation experiment: targeted fix for one mode worsens another
