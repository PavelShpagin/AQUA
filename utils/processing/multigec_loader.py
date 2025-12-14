#!/usr/bin/env python3
"""
MultiGEC-2025 gold data loader.

Parses orig/ref markdown files from data/raw/multigec-2025/<language>/<corpus>/
and returns src→tgt pairs for training. We use ref1 as the primary correction
when multiple references exist; extend if needed.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional


LANG_DIRS = {
    'cs': 'czech',
    'en': 'english',
    'et': 'estonian',
    'de': 'german',
    'el': 'greek',
    'is': 'icelandic',
    'it': 'italian',
    'lv': 'latvian',
    'sl': 'slovene',
    'sv': 'swedish',
    'uk': 'ukrainian',
    'ua': 'ukrainian',
    'ru': 'russian',
}


def _parse_md_blocks(md_path: Path) -> List[Tuple[str, str]]:
    blocks: List[Tuple[str, str]] = []
    if not md_path.exists():
        return blocks
    essay_id: Optional[str] = None
    acc: List[str] = []
    for raw in md_path.read_text(encoding='utf-8').splitlines():
        line = raw.rstrip('\n')
        if line.startswith('### essay_id = '):
            # flush previous
            if essay_id is not None:
                text = '\n'.join(acc).strip()
                blocks.append((essay_id, text))
                acc = []
            essay_id = line.split('=', 1)[1].strip()
        else:
            acc.append(line)
    if essay_id is not None:
        text = '\n'.join(acc).strip()
        blocks.append((essay_id, text))
    return blocks


def load_multigec_train(lang_code: str, base_dir: str, max_samples: Optional[int] = None) -> Dict[str, List[Dict]]:
    """
    Load MultiGEC-2025 gold training pairs for a language.

    Returns a mapping: dataset_name -> list of {src_text, tgt_text, id}
    """
    result: Dict[str, List[Dict]] = {}
    lang_dir_name = LANG_DIRS.get(lang_code, '')
    if not lang_dir_name:
        return result
    base = Path(base_dir) / lang_dir_name
    if not base.exists():
        return result

    # Iterate corpora directories under the language
    for corpus_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        # Expect files like <prefix>-orig-train.md and <prefix>-ref1-train.md
        try:
            orig_candidates = sorted(corpus_dir.glob('*-orig-train.md'))
            ref_candidates = sorted(corpus_dir.glob('*-ref1-train.md'))
            if not orig_candidates or not ref_candidates:
                continue
            orig_path = orig_candidates[0]
            ref_path = ref_candidates[0]

            orig_blocks = _parse_md_blocks(orig_path)
            ref_blocks = _parse_md_blocks(ref_path)
            if not orig_blocks or not ref_blocks:
                continue

            # Align by order; essay_id sequence should match
            pairs: List[Dict] = []
            take = min(len(orig_blocks), len(ref_blocks))
            limit = take if max_samples is None else min(take, max_samples)
            for i in range(limit):
                oid, otext = orig_blocks[i]
                rid, rtext = ref_blocks[i]
                # If IDs mismatch, still pair by position but keep id informative
                pid = oid if oid == rid else f"{oid}|{rid}"
                if otext and rtext and otext != rtext:
                    pairs.append({'src_text': otext, 'tgt_text': rtext, 'id': pid})

            if pairs:
                dataset_name = f"MULTIGEC_{corpus_dir.name.upper()}"
                result[dataset_name] = pairs
        except Exception:
            continue

    return result


