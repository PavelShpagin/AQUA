#!/usr/bin/env python3
"""
Benchmark Clean RAG Agent (Spanish) on SpanishFPs.csv

- Compares with and without RAG for a single backbone
"""

import os
import sys
import time
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from judges.edit.rag_agent import CleanRAGAgent


def run_eval(backbone: str, use_rag: bool) -> dict:
    os.environ['AGENT_USE_RAG'] = '1' if use_rag else '0'
    agent = CleanRAGAgent(backbone=backbone, language='es')
    df = pd.read_csv(ROOT / 'data' / 'eval' / 'SpanishFPs.csv')

    correct = 0
    total = 0
    per_class = { 'TP': [0,0], 'FP1':[0,0], 'FP2':[0,0], 'FP3':[0,0] }
    start = time.time()

    for _, row in df.iterrows():
        src = str(row.get('src',''))
        tgt = str(row.get('tgt',''))
        gold = str(row.get('tp_fp_label','')).upper()
        aligned = str(row.get('aligned','')) if 'aligned' in row else ''
        try:
            out = agent.classify(src, tgt, aligned)
            pred = out['label']
        except Exception as e:
            pred = 'Error'

        if gold in per_class:
            per_class[gold][1] += 1
            if pred == gold:
                per_class[gold][0] += 1
        if pred == gold:
            correct += 1
        total += 1

    elapsed = time.time() - start
    acc = correct / total if total else 0.0
    per_class_acc = {k: (v[0]/v[1] if v[1] else 0.0) for k,v in per_class.items()}
    return {
        'backbone': backbone,
        'use_rag': use_rag,
        'accuracy': acc,
        'per_class': per_class_acc,
        'samples': total,
        'elapsed_s': elapsed,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', required=True, help='LLM backbone, e.g., gpt-4o-mini')
    args = ap.parse_args()

    res_no = run_eval(args.backend, use_rag=False)
    res_yes = run_eval(args.backend, use_rag=True)

    print("\nRESULTS (SpanishFPs)")
    for res in [res_no, res_yes]:
        tag = 'WITH RAG' if res['use_rag'] else 'NO RAG'
        print(f"- {res['backbone']} [{tag}]: acc={res['accuracy']:.3f} (n={res['samples']}) time={res['elapsed_s']:.1f}s")
        print(f"  Per-class: TP={res['per_class']['TP']:.2f} FP1={res['per_class']['FP1']:.2f} FP2={res['per_class']['FP2']:.2f} FP3={res['per_class']['FP3']:.2f}")


if __name__ == '__main__':
    main()






