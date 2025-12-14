### Pillars of GEC and OmniGEC (Gemma 3) — Repro Scripts

This repo includes small, reproducible wrappers to verify published benchmarks with minimal setup.

#### Pillars of GEC (BEA-2019)

What’s included:
- Repo and data: `third-party/pillars-of-gec/`
- Weights example (downloaded): `models/gector/gector-2024-roberta-large.th`
- Eval script: `benchmarks/run_pillars_bea2019.py`

Quick start:
1) One-time deps (uses project’s existing env):
```
bash install_deps.sh
```
2) Dev-set verification (ERRANT):
```
python benchmarks/run_pillars_bea2019.py
```
This evaluates the default Pillars prediction file on BEA-dev and prints ERRANT P/R/F0.5. Expected F0.5 is ≈62.9; the wrapper reports ≈62.96 confirming the pipeline.

3) Evaluate any provided system prediction file (one-sentence-per-line):
```
python benchmarks/run_pillars_bea2019.py --system third-party/pillars-of-gec/data/system_preds/ensemble_systems/ens_m7___bea-dev.txt
```

4) BEA-test (81.4) note:
- Official BEA-test scoring requires Codalab submission; no test M2 is provided.
- Pillars provides test predictions under `data/system_preds/ensemble_systems/` (e.g., `ens_m7___bea-test.txt`). Use those to produce a Codalab submission as per the Pillars README.

#### OmniGEC (Gemma 3) — optional GLEU sanity check

What’s included:
- Eval script: `benchmarks/run_omnigec_gemma3_gleu.py`
- Uses JFLEG (public) at `data/raw/jfleg/` and its bundled `eval/gleu.py` for sentence-level GLEU.

Run (requires access to gated Gemma 3 and modern torch):
```
python benchmarks/run_omnigec_gemma3_gleu.py \
  --split dev \
  --limit 50 \
  --model google/gemma-3-4b-it \
  --hf-token YOUR_HF_TOKEN
```
Notes:
- Install extras if needed: `pip install transformers torch` (PyTorch ≥ 2.1 recommended).
- Gemma 3 is a gated model on Hugging Face. Accept license and login, or pass `--hf-token`.
- This script is a lightweight sanity check on JFLEG and not a reproduction of OmniGEC’s multi-reference GLEU; for official OmniGEC GLEU, use the multi-ref pipeline and corpora described in `data/raw/omnigec-data/notebooks/automatic_evaluation/`.


