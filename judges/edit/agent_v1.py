#!/usr/bin/env python3
"""
Thin wrapper for edit/agent_v1 to support subprocess calls.
Delegates to judges.edit._legacy.agent_v1 implementation.
"""

import argparse
import pandas as pd

from judges.edit._legacy.agent_v1 import call_single_judge_for_row_detailed as _call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--llm_backend', required=True)
    ap.add_argument('--lang', default='es')
    ap.add_argument('--workers', type=int, default=50)
    ap.add_argument('--moderation', default='off', choices=['on','off'])
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    rows = []
    for i, r in df.iterrows():
        out = _call(judge='edit', method='agent_v1', backend=args.llm_backend, lang=args.lang, row=r, moderation=args.moderation)
        rows.append({
            'idx': r.get('idx', i),
            'src': r.get('src',''),
            'tgt': r.get('tgt',''),
            'tp_fp_label': out.get('label','Error'),
            'reasoning': out.get('reasoning',''),
            'writing_type': out.get('writing_type',''),
        })
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == '__main__':
    main()





Thin wrapper for edit/agent_v1 to support subprocess calls.
Delegates to judges.edit._legacy.agent_v1 implementation.
"""

import argparse
import pandas as pd

from judges.edit._legacy.agent_v1 import call_single_judge_for_row_detailed as _call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--llm_backend', required=True)
    ap.add_argument('--lang', default='es')
    ap.add_argument('--workers', type=int, default=50)
    ap.add_argument('--moderation', default='off', choices=['on','off'])
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    rows = []
    for i, r in df.iterrows():
        out = _call(judge='edit', method='agent_v1', backend=args.llm_backend, lang=args.lang, row=r, moderation=args.moderation)
        rows.append({
            'idx': r.get('idx', i),
            'src': r.get('src',''),
            'tgt': r.get('tgt',''),
            'tp_fp_label': out.get('label','Error'),
            'reasoning': out.get('reasoning',''),
            'writing_type': out.get('writing_type',''),
        })
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == '__main__':
    main()







