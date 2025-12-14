# Feedback Judges: Methods, Algorithms, and Benchmarks

AQUA is a modular LLM-powered framework to evaluate single-edit Grammatical Error Correction (GEC) suggestions across languages. It assigns one of {TP, FP3, FP2, FP1} relative to the edit.

---

Problem

- Build a reliable judge on a fixed dataset by assigning TP/FP3/FP2/FP1 labels.
- Single-edit scope; future work extends to sentence-level and data curation for MultiGEC.

Label Taxonomy

- TP: Edit fixes a real error; relative improvement even if other errors remain.
- FP3: Optional/preferential; correct→correct or style-only.
- FP2: Introduces a grammatical/punctuation error or minor meaning change.
- FP1: Major meaning change, high sensitivity, or nonsensical output.

---

Methods

- feedback/legacy (baseline-v0)
  - Early prompt; weak relative-improvement framing; high FP3 bias.

- feedback/baseline (clean relative-improvement)
  - Language-agnostic, concise rules; correct Aligned/Edit injection; strict JSON.

- feedback/modular (specialized categorical fusion)
  - Sub-judges (gpt-4o-mini): Meaning-change (0..4), Reward (-3..+3), Pair Correctness (source_correct, target_correct).
  - Fusion (deterministic):
    - If meaning ≥ 3 → FP1
    - Else if not target_correct and reward ≤ 0 → FP2
    - Else if source_correct and target_correct and meaning ≤ 1 and reward ∈ {0,1} → FP3
    - Else if target_correct and meaning ≤ 1 and reward ≥ 1 → TP
    - Else → FP3

- ensembles/escalation (deterministic router by default; 2–N experts)
  - Default deterministic router: 1 small classification call + simple language-agnostic risk flags from `Aligned` spans (e.g., number/proper-noun/long rewrite changes). Escalates to next expert only when risky/low-trust; otherwise accepts small decision.
  - Optional LLM router (`--router llm`): legacy model-driven routing retained behind a flag (combined classify+route JSON; or two-call fallback).
  - Finalizer only triggers when small vs expert disagree.
  - Works for feedback and edit.

---

How to run (English gold)

- Legacy
```
bash shell/run_judge.sh --judge feedback --method legacy --backends "gpt-4o-mini" --lang en --gold data/eval/gold_tp_fp3_fp2_fp1_en.csv --ensemble weighted --optimization on
```
- Baseline
```
bash shell/run_judge.sh --judge feedback --method baseline --backends "gpt-4o-mini" --lang en --gold data/eval/gold_tp_fp3_fp2_fp1_en.csv --ensemble weighted --optimization on
```
- Escalation
```
bash shell/run_judge.sh --judge feedback --method baseline --backends "gpt-4o-mini gpt-4o o3" --lang en --gold data/eval/gold_tp_fp3_fp2_fp1_en.csv --ensemble escalation --optimization on
# Optional router controls (deterministic is default):
#   --router deterministic|llm
#   --escalate_labels "FP1,FP2,Error"
#   --escalate_flags "number_change,proper_noun_change,long_rewrite_risk"
```
- Modular (predict + evaluate)
```
venv/bin/python -m judges.feedback.modular --input data/eval/gold_tp_fp3_fp2_fp1_en.csv --output data/results/feedback_modular_tp_fp3_nano_en/pred.csv --llm_backend gpt-4o-mini --lang en
venv/bin/python -m gold_eval.run_gold data/eval/gold_tp_fp3_fp2_fp1_en.csv data/results/feedback_modular_tp_fp3_nano_en/pred.csv --llm_backend gpt-4o-mini --judge feedback --method modular_tp_fp3 --lang en
```

---

Benchmarks (English gold; 256 examples)

- Metrics: 4-class Accuracy, Macro F1; Binary Accuracy/F1; Cost per 10K (extrapolated from usage).

| Method | 4-class Acc | Macro F1 | Binary Acc | Binary F1 | Cost/10K |
|---|---:|---:|---:|---:|---:|
| feedback/legacy (gpt-4o-mini) | 0.242 | 0.154 | 0.293 | 0.288 | $1.71 |
| feedback/modular (gpt-4o-mini) | 0.676 | 0.348 | 0.719 | 0.608 | $1.71* |
| feedback/baseline (gpt-4o-mini) | 0.867 | 0.406 | 0.887 | 0.684 | $1.42 |
| feedback/escalation (mini→4o→o3) | 0.883 | 0.521 | 0.887 | 0.673 | $1.48 |

Recent sampled run (n=200) with deterministic router:
- 4-class Acc: 0.880; Macro F1: 0.564; Cost/10K: $1.77

*Modular cost computed from aggregated per-call costs in predictions (total_cost_usd=0.04375 for 256 ⇒ ~$1.71/10K).

Notes
- Escalation is the current SOTA on English gold at near-baseline cost.
- Modular excels on Spanish FP-heavy evaluations (Macro F1 ~0.823) and is a strong specialist.

---

ICLR-2026 Research-grade Methods

- Escalation (dynamic, categorical router + final judge): simple, modular, N-expert, language-agnostic; strong accuracy/F1 gains at negligible cost delta.
- Modular (categorical fusion): interpretable multi-signal judge; excellent FP separation on Spanish; complementary to escalation.

---

Reproducibility

- Use shell/run_judge.sh for consistency and logging.
- Use venv/bin/python for all Python invocations.
- Input: data/eval/gold_tp_fp3_fp2_fp1_en.csv
- Outputs: data/results/.../report.txt, labeled.csv, errors.txt, correct.txt

