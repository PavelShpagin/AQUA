MultiGEC Judge
========

This project provides modular, reproducible GEC (Grammatical Error Correction) judges and ensembles for TP/FP3/FP2/FP1/TN/FN evaluation. It reproduces methods/results reported in the AQUA paper draft (https://coda.io/d/AQUA-Autonomous-Quality-Assurance-for-Multilingual-GEC_dGCaDU6YTLP/AQUA-Automated-Quality-Analyzer-for-Multilingual-GEC_su7V5QRB). 

## Feedback Bot (Quick Start)

Turn raw feedback CSVs into a labeled dataset and HTML report.

```bash
# 1) cd to repo root (example path)
cd /home/ray/efs/team/pavel.shpagin/multigec/gec-annotations/scripts/multilingual/gec_judge

# 2) Run end-to-end with an absolute input path (no config needed)
./feedback_bot/run.sh --feedback_input /home/ray/efs/team/pavel.shpagin/multigec/gec-annotations/scripts/multilingual/gec_judge/data/feedback/data.csv
```

What it does
- Converts alerts → judge input, runs the judge, clusters error patterns, and opens `feedback_bot/report.html`.
- Absolute paths are used as-is.

Feedback judge modes
- **baseline**: single model per example.
- **escalation**: small model routes only uncertain/risky cases to stronger backends.
- **legacy**: `feedback/legacy` old prompt path (kept for reference only).

How escalation backends work
- Provide backends in order: `small [mid] [top]` (e.g., `gpt-4.1-nano gpt-4o o4-mini`).
- The small model both labels and returns a JSON escalation decision; if escalation triggers, we call `mid` or `top` and arbitrate the final label.

Benchmarks (indicative)
- **Baseline**: Overall Acc ≈ 0.871; Simplified Binary Acc ≈ 0.910
- **Escalation**: Overall Acc ≈ 0.891; Simplified Binary Acc ≈ 0.930
- **Legacy (old prompt)**: Very low accuracy on our slice → not recommended

Use escalation when binary accuracy is the priority and modest extra cost is acceptable.

## Appendix

### Setup

**Run all commands from `scripts/multilingual/gec_judge/`.**

```bash
cd scripts/multilingual/gec_judge/
```

1) Python environment

```bash
python -m venv .venv && source .venv/bin/activate
./install_deps.sh
```

2) Data folder

```bash
mkdir -p data/eval
# Example gold datasets are provided under data/eval. You can also place your own CSVs here.
```

3) API keys (.env in this folder)

```bash
# Required for transparent API calls
API_TOKEN=your_api_token

# Optional direct OpenAI calls (fallback)
OPENAI_API_KEY=your_openai_key
```

### Quick Start

Run:

```bash
./shell/run_judge.sh --config config.yaml --input data/eval/gold_de.csv --pref exp1
```

Outputs go to `data/results/{judge}_{method}_{backends}_{lang}[_pref]/`:
- `preprocessed_with_aligned.csv` – ERRANT pre-aligned input (for non-legacy methods)
- `<input_stem>_labeled.csv` – final classifications
- `<input_stem>_labeled_filtered.csv` – same, without `Error/API_FAILED`

### Usage examples

Run without a config file (CLI-only):

```bash
# Sentence judge, baseline method, German gold set, 3 parallel judges
./shell/run_judge.sh \
  --judge sentence --method baseline --ensemble consistency \
  --backends "gpt-4.1-nano" --lang de --n_judges 3 --workers 50

# Feedback judge (TP/FP3/FP2/FP1) with weighted ensemble
./shell/run_judge.sh \
  --judge feedback --method baseline --ensemble weighted \
  --backends "gpt-4o-mini"
```

Notes:
- `--input` is required unless `--gold` is set (then GOLD is used as input).
- `lang` is used for alignment/reporting; prompts use automatic language detection.

### Parameters (CLI/config)

- `judge`: feedback | sentence | edit | tnfn
- `method`: legacy | baseline | modular | agent
- `ensemble`: weighted | consistency | iter_critic | inner_debate
- `backends`: space-separated model ids (e.g., `gpt-4o`, `gpt-4o-mini`, `gpt-4.1-nano`)
- `lang`: target language for alignment
- `input`: CSV to classify
- `tgt`: target column name (default: `tgt`)
- `filter`: space-separated target columns; evaluate in order and keep first TP/TN
- `n_judges`: number of parallel judges per row
- `workers`: row-level parallelism (baseline caps to 50; optimized uses request)
- `moderation`: on | off (OpenAI Moderation on `src + tgt`)
- `optimization`: on | off (enables faster path and batch-pricing)
- `batch`: on | off (use batch API path where available)
- `samples`: int; process only first N rows
- `shard_size`: int; auto-shard large inputs for stability
- `pref`: string appended to results dir name
- `gold`: path to gold labels (enables benchmarking; also used as input if `--input` is absent)
- `debug`: on (extra env diagnostics)

### Classification & Filtration

Input CSV must contain at least `src` and `tgt`. For multi-candidate filtering, provide multiple target columns and pass them in priority order via `--filter`.

```bash
# Basic classification
./shell/run_judge.sh \
  --judge feedback --method baseline --ensemble weighted \
  --backends "gpt-4o-mini" --lang en \
  --input data/eval/gold_en.csv --n_judges 1 --workers 200

# Filtration over multiple candidates (keep first TP/TN)
./shell/run_judge.sh \
  --judge feedback --method baseline --ensemble weighted \
  --backends "gpt-4o-mini" --lang en \
  --input data/eval/my_candidates.csv \
  --filter "gen_rewrite_1 gen_rewrite_2 rerank_selected_1" \
  --tgt gen_rewrite_1 --n_judges 1
```

Outputs: see Quick Start. Use `_labeled_filtered.csv` for downstream analysis.

### Pre-alignment behavior

- `aligned`: produced for the column used as `tgt` at pre-align time. If `tgt` is absent, the first column listed in `filter` that exists in the CSV is used as `tgt` and `aligned` reflects that column.
- `filter_aligned`: You can request per-candidate alignments for specific filter columns. Add in your config:

```yaml
filter_aligned:
  - gen_rewrite_1_gpt5_gec_conservative
  - rerank_selected_1
```

During pre-align, for each listed column present in the CSV, an additional column is produced with the name `{col}_aligned`, containing the ERRANT alignment between `src` and that column's value. Missing columns are ignored.

This lets you run filtering/annotation flows while having explicit per-candidate alignments available, instead of only the single `aligned` column.

### Feedback Bot (details)

End-to-end TP/FP labeling, pattern clustering, and HTML report. See Quick Start above for the one-line command.

Key YAML fields (`feedback_bot/config.yaml`):
- `feedback_input`: raw feedback CSV (with `alert` or sanitized fields). Looked up relative to `data/feedback/` if not absolute.
- `feedback_processed`: optional pre-labeled CSV to skip judge (copied to `data/feedback/processed.csv`).
- `feedback_backend`: LLM for pattern summaries and overview (e.g., `gpt-4o-mini`).
- `feedback_cluster_method`: HDBSCAN | DBSCAN | KMEANS.
- `feedback_embedding`: bert | tfidf (bert uses `sentence-transformers`; falls back to tfidf if unavailable).
- `feedback_patterns`: target number of patterns per category in the report.
- `feedback_lang`: restrict conversion to a language code (e.g., `en`, `de`).
- `feedback_lang_col`: column with human-readable language labels (if present, used instead of detection).
- `feedback_sample`: limit N examples before running judge.
- `errant`: on | off (feedback: default off; uses alert/aligned text).
- Judge keys reused: `judge`, `method`, `backends`, `lang`, `ensemble`, `n_judges`, `workers`, `optimization`, `batch`.

Artifacts:
- `data/feedback/processed.csv` – final judged data (copied from latest run)
- `feedback_bot/*_clusters.json` – clustering results
- `feedback_bot/report.html` – opens automatically

#### Quick CD path example

If you are on an environment with a path like the screenshot suggests, run:

```bash
cd /home/ray/efs/team/pavel.shpagin/multigec/gec-annotations/scripts/multilingual/gec_judge
```

Then execute the Feedback Bot commands from there.

#### Minimal CLI override

The script now accepts `--feedback_input` to override the `feedback_input` configured in `feedback_bot/config.yaml`. Absolute paths are used as-is; if you pass a relative path, it is resolved relative to the current working directory. If `--feedback_input` is provided, any `feedback_processed` value in config is ignored for that run.

### Algorithm overview (high level)

#### Baseline (feedback)
- **What it does**: Single LLM call per row using the baseline TP/FP prompt to assign one of TP/FP1/FP2/FP3; optional ERRANT alignment. See `judges/feedback/baseline.py`.
- **Why it works**: Strong prompting + explicit alignment token to anchor the decision on the main edit improves stability and parsing.
- **Cost/Speed**: One model call per sample; highly parallel; supports pricing tracking.

#### Dynamic Escalation Ensemble
- **What it does**: Start with a small model; ask the same small model to decide whether to escalate; if needed, call a stronger expert backend; optionally perform a final arbitration over collected opinions. See `ensembles/escalation.py`.
- **Why it works**: Selective use of expensive models only on uncertain or risky cases (e.g., FP1/FP2) drives accuracy gains with modest cost.
- **When to use**: Production-like settings seeking 90%+ accuracy at controlled spend.

#### Legacy Adversarial Consensus (reference)
- **What it does**: Probe the initial decision with an adversarial counter-argument; escalate to an expert panel only when the probe reveals genuine uncertainty. See `_legacy/ensembles/adversarial_consensus.py`.
- **Why it works**: Targets the edge cases that simple confidence thresholds might miss by actively challenging the decision.
- **Status**: Useful reference; baseline and escalation paths are the recommended defaults.
