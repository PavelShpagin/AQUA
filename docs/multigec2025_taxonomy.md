### MultiGEC‑2025-oriented taxonomy and judge settings

This note summarizes what is available in the repo for MultiGEC‑2025, reports quick empirical label distributions with sentence judges on dev splits, and proposes a taxonomy aligned to the “minimal correction” objective.

### Data availability in this repo
- Dev/train: both `*-orig-dev.md` and `*-ref1-dev.md` are present for many languages under `data/raw/multigec-2025/` (e.g., DE: `german/Merlin`, UA: `ukrainian/ua-gec`, ET: `estonian/*`, CS: `czech/*`, etc.).
- Test: `*-orig-test.md` exists; `*-ref*-test.md` is not included here. So we benchmark on dev.

### Empirical distributions (dev, sentence-level judges)
Judge: sentence/legacy (minimal-correction friendly). Full dev splits:
- DE (Merlin dev, n=103): TP 57.3%, FP1 17.5%, FN 25.2%
- UA (ua‑gec dev, n=87): TP 42.5%, FP1 47.1%, FN 10.3%

Judge: sentence/baseline (fusion alignment):
- DE: TP 50.5%, FP1 21.4%, FP3 1.0%, FN 27.2%
- UA: TP 42.5%, FP1 47.1%, FN 10.3%

Notes:
- Sentence judges produce substantially higher TP than strict edit judges, better matching MultiGEC’s minimal-correction refs.
- Test references are absent; dev is the reliable split here.

### Proposed taxonomy aligned to MultiGEC‑2025
Keep the six labels but tailor definitions to minimal correction:
- TP (required correction): The edit fixes a linguistic error or norm violation. Includes grammar (agreement, morphology, function words), orthography (spelling, capitalization, diacritics), and normative punctuation (comma rules, quotes, spacing) when required by the language’s conventions. Minor word-order fixes that restore grammaticality also fall here.
- FP3 (style-only improvement): Purely stylistic/paraphrastic changes that are not required for grammaticality or norm compliance (lexical substitutions, rhetorical rephrasing, optional punctuation for tone). Meaning preserved; original is acceptable.
- FP2 (unnecessary/incorrect norm change): Changes that are not required and partly deviate from norms (e.g., debatable grammar “fixes”, forced hyphenation/orthography variants that are not standard, overly aggressive normalization when original is fine). Meaning preserved but the change is not justified by norms.
- FP1 (meaning change or error): The edit introduces an error, alters meaning, adds hallucinated content, or removes required content.
- TN (no edit needed): Source already meets norms; target equals source (or is trivial metadata change).
- FN (missed error): The source contains a clear error that is not corrected (or a TP edit is required but absent).

Priority and sentence aggregation:
- Per-edit priority: FP1 > FP2 > FP3 > TP.
- Sentence label = highest-priority label present among its edits; if src==tgt, use TN/FN.

Mapping guidance (common MultiGEC cases):
- Spelling/case/diacritics normalization: TP.
- Required punctuation per language norms (e.g., German comma rules; Ukrainian spacing/quotes): TP.
- Morphology/agreement/preposition/determiner errors: TP.
- Optional synonym or style swap without norm pressure: FP3.
- Debatable grammar change or nonstandard normalization: FP2.
- Any meaning shift, hallucination, content insertion/removal that changes semantics: FP1.

### Recommended judge settings for MultiGEC‑style evaluation
- Primary metric for minimal corrections: sentence/legacy on edited pairs (dev). This yields TP rates aligned with the reference rationale.
- Secondary analysis: sentence/baseline (fusion) for robustness; edit/baseline for stricter over‑editing detection (expect lower TP, higher FP2/FP3).
- Binary variants to report alongside 4/6‑class: Acceptable vs Problematic, with Acceptable = TP (optionally TP+FP3) and Problematic = FP1/FP2/FN (TN excluded for edited-only sets).

### Repro (building dev CSVs and running sentence judges)
- Build paired CSVs from dev markdown files:
  - UA: `uk-ua_gec-orig-dev.md` + `uk-ua_gec-ref1-dev.md` → `data/eval/multigec_dev_ua.csv`
  - DE: `de-merlin-orig-dev.md` + `de-merlin-ref1-dev.md` → `data/eval/multigec_dev_de.csv`
- Run sentence judges:
  - `./shell/run_judge.sh --input data/eval/multigec_dev_de.csv --lang de --judge sentence --method legacy --backends "o3"`
  - `./shell/run_judge.sh --input data/eval/multigec_dev_ua.csv --lang ua --judge sentence --method legacy --backends "o3"`

These settings and definitions are designed to better reflect MultiGEC‑2025’s “minimal correction” intent while remaining compatible with the broader TP/FP taxonomy used elsewhere in this repo.


