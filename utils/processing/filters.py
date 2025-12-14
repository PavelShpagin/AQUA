"""
Common filtering utilities for sentence-level GEC processing.

Goals (applied uniformly across languages):
- Length filter: keep only sentences with 10–50 words (both src and tgt)
- Edit-count cap: drop items with >= 5 edits after merge
- Optional: drop multiline entries by default
"""

import re
from typing import Tuple
import re
from collections import Counter


def is_length_ok(src_text: str, tgt_text: str, min_words: int = 10, max_words: int = 50) -> bool:
    src_words = len(src_text.split())
    tgt_words = len(tgt_text.split())
    return (min_words <= src_words <= max_words) and (min_words <= tgt_words <= max_words)


def count_edits(aligned_text: str) -> int:
    if not aligned_text:
        return 0
    return len(re.findall(r"\{[^{}]*?=>[^{}]*?\}", aligned_text))


def _tokenize_lower_alnum(text: str):
    return [t for t in re.findall(r"[\w']+", text.lower())]


def _lcs_length(a: list, b: list) -> int:
    """Compute LCS length between two token lists (O(n*m), short sentences only)."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def should_keep(src_text: str, tgt_text: str, aligned_text: str,
                max_edits: int = 4, min_words: int = 10, max_words: int = 50,
                allow_multiline: bool = False) -> bool:
    if not allow_multiline and ("\n" in src_text or "\n" in tgt_text):
        return False
    if not is_length_ok(src_text, tgt_text, min_words, max_words):
        return False
    if count_edits(aligned_text) > max_edits:
        return False
    # Universal simple filter: normalized token LCS must be sufficiently high.
    # This penalizes paraphrases, truncations, and large rewrites uniformly.
    src_tokens = _tokenize_lower_alnum(src_text)
    tgt_tokens = _tokenize_lower_alnum(tgt_text)
    if not src_tokens or not tgt_tokens:
        return False
    lcs = _lcs_length(src_tokens, tgt_tokens)
    denom = max(len(src_tokens), len(tgt_tokens))
    lcs_sim = (lcs / denom) if denom else 0.0
    # Threshold tuned to keep genuine corrections while filtering obvious rewrites/truncations
    if lcs_sim < 0.55:
        return False
    return True


