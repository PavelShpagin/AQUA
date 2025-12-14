# GEC Dataset Processing

Minimal, reproducible pipeline to produce detokenized, sentence-level GEC datasets with ERRANT alignments. Adjacent edits are merged by default.

### Prerequisites
- W&I+LOCNESS JSON (train A/B/C): `data/raw/BEA/wi+locness/json/{A,B,C}.train.json`  
  See `data/raw/BEA/wi+locness/readme.txt` and the BEA-2019 Shared Task page: [BEA-2019 Shared Task](https://www.cl.cam.ac.uk/research/nl/bea2019st/)
- FCE JSON (train): `data/raw/BEA/fce/json/fce.train.json`  
  See `data/raw/BEA/fce/readme.txt`
- UA-GEC sentence-level dirs: `data/raw/ua-gec/data/gec-only/train/source-sentences` and `data/raw/ua-gec/data/gec-only/train/target-sentences`  
  See `data/raw/ua-gec/README.md` and the project page: [UA-GEC](https://github.com/grammarly/ua-gec)
- FalkoMerlin TSV: `data/raw/multiged-2023/german/fm-train.tsv` (MultiGED-2023 release)  
  See `data/raw/multiged-2023/german/README.md`

### Quick start
Build EN/UA/DE datasets and run the judge in one step:

```bash
./shell/process_pipe.sh --mode gold
```

Silver mode builds from OmniGEC only:

```bash
./shell/process_pipe.sh --mode silver
```

The flag `--no-merge` is supported in all processors and the pipeline to disable adjacent edit merging when needed.

Judge outputs are saved to `data/results/processed_edits/`.

### Per-language runs

```bash
python processing/en.py
python processing/ua.py
python processing/de.py
```

Full datasets:

```bash
python processing/en.py --max
python processing/ua.py --max
python processing/de.py --max
```

Mixture controls:
- EN: `--wi N --fce M --omni K`
- UA: `--ua-gec N --omni K`
- DE: `--merlin N --omni K`

### Outputs
Files are written to `data/processed/` with schema `idx,src,tgt,aligned`.

### Process Annotations

To convert raw annotations to gold data that can be used for benchmarking judges, run:

```bash
python processing/gold_annotations.py --output-dir data/eval
```

