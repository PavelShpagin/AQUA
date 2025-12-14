### Router + Two Binary Classifiers (gpt-4.1-nano)

This proof-of-concept explores a hierarchical classification strategy:

- Router: TP/FP3 vs FP2/FP1
- GROUP A: TP vs FP3
- GROUP B: FP2 vs FP1

Prompts are high-quality English variants inspired by the current baseline prompt and BEA-2019 guidance, and are defined in `judges/feedback/prompts.py` per project guidelines.

#### Files
- `judges/feedback/prompts.py`: `ROUTER_TPFP3_VS_FP2FP1_PROMPT_EN`, `TP_VS_FP3_PROMPT_EN`, `FP2_VS_FP1_PROMPT_EN`
- `experiments/router_binary_classifiers.py`: experiment runner

#### How to run
```bash
export OPENAI_API_KEY=...
/path/to/venv/bin/python experiments/router_binary_classifiers.py \
  --dataset data/eval/gold_tp_fp3_fp2_fp1_en.csv \
  --backend gpt-4.1-nano \
  --workers 96
```

#### Results (full English gold set)
- Accuracy: 51.95%
- Macro F1: 25.66%
- Binary F1 (TP vs non-TP): 67.25%

Confusion matrix (rows=true, cols=pred): TP, FP1, FP2, FP3
```
[[116, 1, 82, 25],
 [  2, 0,  4,  1],
 [  1, 0, 15,  3],
 [  2, 0,  2,  2]]
```

#### Notes
- Backend: `gpt-4.1-nano` via the project’s optimized call path with concurrency.
- Alignment: `utils.errant_align.get_alignment_for_language` (spaCy-based) for English.

#### Next steps
- Add a cheap rule-based pre-router (digits, proper nouns, prepositions) before LLM routing.
- Tighten FP2/FP1 binary with a short checklist of failure modes and 2–3 micro-examples.
- Optionally ensemble 2–3 nano passes (different seeds) with majority vote for stability.












