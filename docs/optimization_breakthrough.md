# Optimization Breakthroughs for GEC Judge

This document summarizes the optimizations, fixes, and architecture decisions enabling fast, reproducible, and research-grade execution across all judges and ensembles.

## Scope
- Judges: `feedback` (TP/FP1/FP2/FP3), `sentence` (TP/FP1/FP2/FP3/TN/FN; fusion alignment), `edit` (per-edit TP/FP1/FP2/FP3 + missed_error → sentence label; non-fusion alignment), `tnfn` (binary, routed via same alignment infra when applicable).
 - Prompts and classes:
   - feedback: strict 4-class TP/FP1/FP2/FP3. No truncation.
   - sentence: strict 6-class TP/FP1/FP2/FP3/TN/FN. Uses ERRANT prealigned fusion text when available.
   - edit: per-edit 4-class labels + missed_error; aggregates to 6-class sentence label; non-fusion alignment.
   - tnfn: binary TN/FN using dedicated prompt; alignment infra shared when relevant.
- Ensembles: `weighted`, `consistency`, `iter_critic`, `inner_debate`.

## Major Changes
- Strict taxonomy normalization across pipeline: labels coerced to {TP, FP1, FP2, FP3, TN, FN}; variants (e.g., TFN, FP) mapped appropriately. Unknowns → Error.
- Feedback judge enforced to 4-class taxonomy end-to-end; sentence/edit are 6-class; edit computes sentence label from per-edit labels + missed_error.
- JSON parsing hardened for all judges; code-fenced JSON supported; graceful fallback extraction.
- ERRANT alignment standardized:
  - English ERRANT annotator loaded once; native spaCy tokenizers per supported language (en, de, uk/ua, es, pt, it, fr); fallback to English tokenizer if unavailable.
  - Pure ERRANT alignment (no non-ERRANT diff fallbacks). Default merge=True to join adjacent edits for stable prompts.
  - Batch pre-alignment via `nlp.pipe` and cached artifact; macOS-safe inline pre-align (no multiprocessing); Linux module supports multi-process.

## Performance Optimizations
- Global in-process judge path: `USE_IN_PROCESS_JUDGE=1` and `PROCESSING_MODE=bulk` to avoid subprocess overhead.
- Shared ThreadPoolExecutors across ensembles to amortize thread creation cost.
- Early unanimity checks (first two judges) for faster exit in multi-judge settings.
- Optimized runner `run_optimized_process_rows` used by all ensembles when `--optimization on` to shard and parallelize per DataFrame.
- Ultra-fast single-judge batch path (12 samples/call; per-backend concurrency caps) for 100+ samples/sec on commodity hardware.
- Logging: lightweight progress for pre-alignment (rows, shards, time) without affecting throughput.

## Ensemble Correctness
- `weighted`: uses normalized labels; category separation; ignores Error; supports optimized multi-judge path and batch single-judge path.
- `consistency`: majority with custom tie-break; normalized labels; robust fallbacks; optimized runner support.
- `iter_critic`: separates TP/FP* vs TN/FN buffers; iterative judging with opinions; final judge; normalized labels.
- `inner_debate`: dominance ordering (FP1>FP2>FP3>FN>TN/TP with src==tgt rule); alternating debate; final judge; normalized labels.

## Distribution and Pricing
- Distribution detection prioritizes `tp_fp_label`; sentinel rows emit `tp_fp_label` for accurate reporting.
- Pricing tracked per call and aggregated for ensembles; cost extrapolations printed.
- Cache-aware pricing fields supported when available.

## Prompts
- Feedback baseline: 4-class TP/FP1/FP2/FP3 only.
- Sentence baseline: fusion alignment + 6-class TP/FP1/FP2/FP3/TN/FN.
- Edit baseline: per-edit JSON with labels and missed_error; compute sentence label; non-fusion alignment.
- TNFN: binary classifier and 4-class system can be combined as needed; uses fusion alignment infra where required.

## Defaults and Flags
- Optimization OFF (baseline):
  - Standard pipelines only (no in-process judge, no ultimate batch path).
  - Uses large worker counts but no special env shortcuts; enables apples-to-apples accuracy comparison.
- Optimization ON:
  - Multi-judge: sharded optimized runner.
  - Single-judge: optional ultimate batch path when workers≥400 if `ULTIMATE_BATCH=1` (disabled by default for parity).
  - In-process judge path enabled via internal flags.
- Pre-alignment defaults to merge=True; sentence uses fusion for prompts; edit uses non-fusion alignment.
- Label normalization removes legacy `TFN` and other junk from outputs.

## Observed Speed/Cost (example on 3,982 rows)
- Pre-alignment: ~9–10s on macOS inline pre-align (single-process). Faster with Linux multi-process.
- Single-judge ultra-fast: ~8–12s judge phase (332 API calls @12 samples/call).
- Cost: ~0.000054 USD/sample with gpt‑4.1‑nano batch path (example run).

## Reproducibility
- `.env` for API keys; no hard-coded keys.
- Pure ERRANT (v3.0.0), pinned; spaCy models preinstalled for supported languages.
- No non-ERRANT fallbacks; batch pre-alignment cached and reusable.

## Next Steps
- Optional Linux pre-align module for ≤2–3s pre-align on 8–16 cores.
- Backend-specific concurrency caps to avoid throttling and maximize stable throughput.
