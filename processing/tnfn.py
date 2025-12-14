#!/usr/bin/env python3
"""
TN/FN benchmark dataset builder for EN/DE/UA.

Rule per pair (src, tgt):
  - if src == tgt → add (src, tgt, TN)
  - if src != tgt → add (src, src, FN) and (tgt, tgt, TN)

Samples 500 TN and 500 FN and writes to data/eval/tnfn_{lang}.csv
Schema: idx,src,tgt,tp_fp_label

Inputs (same locations as processing scripts):
  - EN: BEA W&I+LOCNESS JSON, FCE JSON
  - DE: FalkoMerlin fm-train.tsv (MultiGED-2023)
  - UA: UA-GEC source/target sentence dirs
"""

import sys
import csv
import random
from pathlib import Path
from typing import List, Tuple, Dict

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.processing.sentence_splitter import process_text_pair
from utils.processing.filters import is_length_ok, should_keep
from utils.processing.bea19_json_loader import load_wi_locness_train, load_fce_train
from utils.processing.fm_tsv_loader import load_fm_tsv_sentence_pairs
from utils.processing.ua_gec_loader import load_ua_gec_sentences


def build_pairs_en() -> List[Tuple[str, str, str]]:
    pairs: List[Tuple[str, str, str]] = []  # (dataset, src, tgt)
    # Load full pools (sampling later)
    wi_pairs = load_wi_locness_train(None)
    fce_pairs = load_fce_train(None)

    for i, item in enumerate(wi_pairs):
        for src, tgt in process_text_pair(item['src_text'], item['tgt_text']):
            # Use shared filters (length + multiline). No aligned text yet.
            if should_keep(src, tgt, aligned_text=""):
                pairs.append(("W&I", src, tgt))
    for i, item in enumerate(fce_pairs):
        for src, tgt in process_text_pair(item['src_text'], item['tgt_text']):
            if should_keep(src, tgt, aligned_text=""):
                pairs.append(("FCE", src, tgt))
    return pairs


def build_pairs_de() -> List[Tuple[str, str, str]]:
    pairs: List[Tuple[str, str, str]] = []
    fm_pairs = load_fm_tsv_sentence_pairs('data/raw/multiged-2023/german/fm-train.tsv', None)
    for item in fm_pairs:
        src, tgt = item['src_text'], item['tgt_text']
        if should_keep(src, tgt, aligned_text=""):
            pairs.append(("FalkoMerlin", src, tgt))
    return pairs


def build_pairs_ua() -> List[Tuple[str, str, str]]:
    pairs: List[Tuple[str, str, str]] = []
    ua_pairs = load_ua_gec_sentences('data/raw/ua-gec/data/gec-only/train', None)
    for item in ua_pairs:
        src, tgt = item['src_text'], item['tgt_text']
        if should_keep(src, tgt, aligned_text=""):
            pairs.append(("UA_GEC", src, tgt))
    return pairs


def expand_to_tnfn(pairs: List[Tuple[str, str, str]]) -> Tuple[List[Dict], List[Dict]]:
    tn: List[Dict] = []
    fn: List[Dict] = []
    for dataset, src, tgt in pairs:
        if src == tgt:
            tn.append({'src': src, 'tgt': tgt, 'tp_fp_label': 'TN'})
        else:
            fn.append({'src': src, 'tgt': src, 'tp_fp_label': 'FN'})
            tn.append({'src': tgt, 'tgt': tgt, 'tp_fp_label': 'TN'})
    return tn, fn


def write_tnfn(lang: str, tn: List[Dict], fn: List[Dict], num_each: int = 500) -> str:
    random.seed(42)
    tn_sample = tn if len(tn) <= num_each else random.sample(tn, num_each)
    fn_sample = fn if len(fn) <= num_each else random.sample(fn, num_each)
    rows = tn_sample + fn_sample
    random.seed(42)
    random.shuffle(rows)
    for i, row in enumerate(rows):
        row['idx'] = i
    out_dir = Path('data/eval')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'tnfn_{lang}.csv'
    with out_file.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['idx', 'src', 'tgt', 'tp_fp_label'])
        writer.writeheader()
        writer.writerows(rows)
    return str(out_file)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Build TN/FN benchmark from existing datasets')
    parser.add_argument('--lang', type=str, required=True, choices=['en', 'de', 'ua'])
    parser.add_argument('--each', type=int, default=500, help='Number of TN and FN examples to sample')
    args = parser.parse_args()

    if args.lang == 'en':
        pairs = build_pairs_en()
    elif args.lang == 'de':
        pairs = build_pairs_de()
    else:
        pairs = build_pairs_ua()

    print(f'Collected {len(pairs)} base pairs for {args.lang.upper()}')
    tn, fn = expand_to_tnfn(pairs)
    print(f'TN candidates: {len(tn)}; FN candidates: {len(fn)}')
    out = write_tnfn(args.lang, tn, fn, args.each)
    print(f'Wrote TN/FN benchmark: {out}')


if __name__ == '__main__':
    main()


