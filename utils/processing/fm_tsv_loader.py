"""
FalkoMerlin fm-*.tsv sentence-level loader (raw, detokenized).

Reconstructs source and target sentence strings from TSV with per-token edits.
No M2, no essays; purely sentence-level.
"""

from pathlib import Path
from typing import List, Dict, Optional
from utils.processing.falko_merlin_loader import cleanup_german_text


def _finalize_sentence(current: list, pairs: list, sid: int, prefix: str):
    if not current:
        return
    src_tokens = []
    tgt_tokens = []
    for token, tag, repl in current:
        # Source always uses original token
        src_tokens.append(token)
        # Target logic
        tag = (tag or '').strip()
        repl = (repl or '').strip()
        if tag == '' or tag == '-' or tag.lower() == 'c':
            tgt_tokens.append(token)
        else:
            if repl:
                for t in repl.split():
                    tgt_tokens.append(t)
            else:
                # deletion
                # skip adding anything
                pass
    src_text = cleanup_german_text(' '.join(src_tokens))
    tgt_text = cleanup_german_text(' '.join(tgt_tokens))
    if src_text and tgt_text:
        pairs.append({'src_text': src_text, 'tgt_text': tgt_text, 'id': f"{prefix}_{sid:06d}"})


def load_fm_tsv_sentence_pairs(tsv_path: str, target_samples: Optional[int] = None) -> List[Dict[str, str]]:
    path = Path(tsv_path)
    if not path.exists():
        print(f"Error: fm TSV not found: {tsv_path}")
        return []
    pairs: List[Dict[str, str]] = []
    current = []
    sid = 0
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                _finalize_sentence(current, pairs, sid, path.stem)
                sid += 1
                current = []
                if target_samples is not None and len(pairs) >= target_samples * 3:
                    break
                continue
            parts = line.split('\t')
            token = parts[0] if len(parts) > 0 else ''
            tag = parts[1] if len(parts) > 1 else ''
            repl = parts[2] if len(parts) > 2 else ''
            if token:
                current.append((token, tag, repl))
        # finalize last
        _finalize_sentence(current, pairs, sid, path.stem)
    return pairs


