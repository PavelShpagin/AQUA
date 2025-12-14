#!/usr/bin/env python3
"""
Edit/baseline (clean):
- Single LLM call per row with EDIT_LEVEL_JUDGE_PROMPT
- Model outputs only per-edit labels and missed_error (no sentence classification)
- Filter empty/no-op edits; aggregate per spec
"""

import os
import re
import json
import argparse
import pandas as pd
from typing import Dict, List

from utils.errant_align import get_alignment_for_language
from judges.edit.prompts import EDIT_LEVEL_JUDGE_PROMPT
from utils.judge import (
    detect_language_from_text,
    get_language_label,
    get_language_code,
    build_numbered_prompt,
    call_model_with_pricing,
    add_pricing_to_result_dict,
    print_judge_distribution,
    process_rows_with_progress,
)
from utils.llm.moderation import check_openai_moderation

PRIORITY = {"FP1": 4, "FP2": 3, "FP3": 2, "TP": 1}


def compute_sentence_label(src: str, tgt: str, missed_error: bool, edit_labels: List[str]) -> str:
    """Aggregation per spec.
    If Source == Corrected: return FN if missed_error else TN
    Else: worst among FP1<FP2<FP3<TP; if worst==TP and missed_error==true -> FN else worst
    """
    src_s = str(src)
    tgt_s = str(tgt)
    if src_s == tgt_s:
        return 'FN' if missed_error else 'TN'
    if not edit_labels:
        return 'Error'
    worst = max(edit_labels, key=lambda x: PRIORITY.get(x, 0))
    if worst == 'TP' and missed_error:
        return 'FN'
    return worst


def parse_edit_json(text: str) -> Dict:
    try:
        s = (text or '').strip()
        if s.startswith('```'):
            parts = s.split('```')
            if len(parts) >= 3:
                s = parts[1].strip()
            else:
                s = s.replace('```', '').strip()
            if s.lower().startswith('json'):
                s = s[4:].lstrip()
        try:
            data = json.loads(s)
        except Exception:
            start = s.find('{'); end = s.rfind('}')
            if start != -1 and end != -1 and end > start:
                data = json.loads(s[start:end+1])
            else:
                raise
        edits_raw = data.get('edits', {})
        edits: Dict[str, str] = {}
        if isinstance(edits_raw, dict):
            edits = edits_raw
        elif isinstance(edits_raw, list):
            for item in edits_raw:
                try:
                    span = item.get('span') or item.get('edit') or item.get('key')
                    lab = item.get('label') or item.get('class') or item.get('classification')
                    if span and lab:
                        edits[str(span)] = str(lab)
                except Exception:
                    continue
        labels: List[str] = []
        for _, lab in edits.items():
            up = str(lab).upper()
            if up in PRIORITY:
                labels.append(up)
        return {
            'edits': edits,
            'labels': labels,
            'missed_error': bool(data.get('missed_error', False)),
            'reasoning': data.get('reasoning', ''),
            'writing_type': data.get('writing_type', ''),
        }
    except Exception:
        return {'edits': {}, 'labels': [], 'missed_error': False, 'reasoning': '', 'writing_type': ''}


def main():
    # Load environment variables
    try:
        from dotenv import load_dotenv, find_dotenv
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path=dotenv_path)
        else:
            load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--llm_backend', required=True)
    parser.add_argument('--lang', default='')
    parser.add_argument('--workers', type=int, default=50)
    parser.add_argument('--moderation', default='off', choices=['on','off'])
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    def process_row(row: pd.Series) -> Dict:
        src = str(row.get('src', '')); tgt = str(row.get('tgt', ''))
        if args.moderation == 'on' and check_openai_moderation(f"{src}\n{tgt}"):
            return {'idx': row.get('idx',''), 'src': src, 'tgt': tgt, 'tp_fp_label': 'Error', 'reasoning': 'Error'}

        language_label = get_language_label(args.lang) if args.lang else detect_language_from_text(src)
        lang_code = get_language_code(args.lang if args.lang else language_label)

        raw_align = str(row.get('aligned','') or row.get('aligned_sentence','') or row.get('alert',''))
        if not raw_align:
            raw_align = get_alignment_for_language(src, tgt, language=lang_code)

        # Filter no-op spans; build edits list for the prompt
        try:
            raw_spans = re.findall(r"\{[^}]+=>[^}]+\}", raw_align)
            spans = []
            for s in raw_spans:
                body = s[1:-1]
                o, n = body.split('=>', 1)
                if o == n:
                    continue
                spans.append(s)
            edits_field = ('"' + '" , "'.join(spans) + '"') if spans else ""
        except Exception:
            edits_field = ""

        prompt = build_numbered_prompt(EDIT_LEVEL_JUDGE_PROMPT, language_label, src, tgt, raw_align, edits_field)
        ok, content, _tokens, pricing = call_model_with_pricing(
            prompt,
            args.llm_backend,
            api_token=os.getenv('API_TOKEN',''),
            moderation=False
        )
        if not ok:
            rec = {'idx': row.get('idx',''), 'src': src, 'tgt': tgt, 'tp_fp_label': 'Error', 'reasoning': 'Error'}
            return add_pricing_to_result_dict(rec, pricing)

        parsed = parse_edit_json(content)
        labels = parsed.get('labels', [])
        sent_label = compute_sentence_label(src, tgt, parsed.get('missed_error', False), labels)

        out = {
            'idx': row.get('idx',''), 'src': src, 'tgt': tgt,
            'aligned': raw_align, 'tp_fp_label': sent_label,
            'reasoning': parsed.get('reasoning',''), 'writing_type': parsed.get('writing_type','')
        }
        return add_pricing_to_result_dict(out, pricing)

    results = process_rows_with_progress(
        df,
        process_row,
        desc=f"Edit Baseline ({args.llm_backend})",
        workers=args.workers,
        optimization=False,
    )

    # Save and print distribution
    print_judge_distribution(results, f"Edit Baseline ({args.llm_backend})")
    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()


