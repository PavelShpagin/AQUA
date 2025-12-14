### Agent-as-a-Judge for SpanishFPs

This experiment benchmarks an agent-as-a-judge against the strong `baseline/edit` judge on the SpanishFPs dataset. It targets reliable TP/FP1/FP2/FP3 discrimination (correct, hallucination/critical, ungrammatical/medium, stylistic/minor) for production and publication.

#### Keys
- Final backbone: `gpt-4.1-nano` (deterministic temperature=0)
- Benchmarks: `gpt-4.1-nano`, `o4-mini`, `o3`
- Minimal, reproducible prompts; strict JSON parsing; price tracking

#### Method summary
- Alignment: ERRANT-based multilingual alignment reused from `utils.errant_align` and fusion diff for compact prompts.
- Baseline/edit: Single-shot per-sentence edit classification with improved TN/FN rules.
- Agent-as-a-judge: Lightweight multi-signal agent (Spanish-focused heuristics + optional LLM refinement) exposed via `judges/edit/agent.py`.
  - Linguistic signals: accent fixes, agreement/conjugation patterns, synonym vs grammar discrimination, structural integrity (quotes/brackets/negation), optional LanguageTool and semantic similarity when available.
  - Aggregation: priority FP1 > FP2 > FP3 > TP with confidence gating; escalate to LLM only if low-confidence.

#### Run
```bash
python -m _experiments.run_spanishfps \
  --data /abs/path/to/SpanishFPs.csv \
  --backends gpt-4.1-nano,o4-mini,o3 \
  --methods baseline,agent \
  --output_dir data/results/_experiments
```

Outputs are saved to `data/results/_experiments/spanishfps_{method}_{backend}.csv` and include pricing and distribution stats. If a gold column is present (any of `gold_label|gold|gold_specialized|tp_fp_label_gold`), a simple accuracy is reported.

#### Production notes
- Temperature=0 for non-reasoning models; no silent fallbacks; deterministic prompts.
- Use `.env`-loaded keys; pricing reported via `utils.pricing`.
- spaCy tokenization is used by alignment utilities; avoid Stanza per project policy.



