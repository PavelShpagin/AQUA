#!/usr/bin/env python3
"""
OmniGEC training loader (isolated from legacy processing).

Loads OmniGEC Reddit (and optionally Wiki) for a fixed set of languages,
producing sentence-level src/tgt pairs for training JSONL export.
"""

from typing import List, Dict, Optional


OMNIGEC_LANGS = [
    'cs',  # Czech
    'en',  # English
    'et',  # Estonian
    'de',  # German
    'el',  # Greek
    'it',  # Italian
    'lv',  # Latvian
    'sl',  # Slovenian
    'sv',  # Swedish
    'uk',  # Ukrainian
    'is',  # Icelandic
]


HF_DATASETS = {
    'reddit': 'lang-uk/Reddit-MultiGEC',
    'wiki': 'lang-uk/WikiEdits-MultiGEC',
}


def _matches_lang(sample_lang: str, lang_code: str) -> bool:
    code_to_name = {
        'cs': 'czech',
        'en': 'english',
        'et': 'estonian',
        'de': 'german',
        'el': 'greek',
        'it': 'italian',
        'lv': 'latvian',
        'sl': 'slovenian',
        'sv': 'swedish',
        'uk': 'ukrainian',
        'is': 'icelandic',
    }
    return sample_lang.lower() == code_to_name.get(lang_code, '')


def load_omnigec_reddit(lang_code: str, max_samples: Optional[int], simple: bool = False) -> List[Dict]:
    from datasets import load_dataset
    from utils.processing.sentence_splitter import process_text_pair

    ds = load_dataset(HF_DATASETS['reddit'], split='train', streaming=True)
    out: List[Dict] = []
    limit = None if max_samples is None else max_samples * 3
    for idx, sample in enumerate(ds):
        if limit is not None and len(out) >= limit:
            break
        if not _matches_lang(sample.get('language', ''), lang_code):
            continue
        src_text = (sample.get('text') or sample.get('src') or '').strip()
        tgt_text = (sample.get('correction') or sample.get('tgt') or '').strip()
        if not src_text or not tgt_text:
            continue
        if simple:
            out.append({'src_text': src_text, 'tgt_text': tgt_text})
        else:
            for s, t in process_text_pair(src_text, tgt_text):
                out.append({'src_text': s, 'tgt_text': t})
    return out


def load_omnigec_wiki(lang_code: str, max_samples: Optional[int], simple: bool = False) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset(HF_DATASETS['wiki'], split='train', streaming=True)
    out: List[Dict] = []
    limit = None if max_samples is None else max_samples * 3
    for idx, sample in enumerate(ds):
        if limit is not None and len(out) >= limit:
            break
        if not _matches_lang(sample.get('language', ''), lang_code):
            continue
        src_text = (sample.get('text_del') or '').strip()
        tgt_text = (sample.get('text_ins') or '').strip()
        if not src_text or not tgt_text:
            continue
        out.append({'src_text': src_text, 'tgt_text': tgt_text})
    return out


