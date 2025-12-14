#!/usr/bin/env python3
"""
Multilingual Lexicon Search (lightweight, reproducible)

Expect data at: data/lexicon/{lang}.jsonl
Each JSON line:
{
  "lemma": "palabra",
  "lang": "es",
  "pos": "NOUN",
  "defs": ["brief gloss 1", "gloss 2"],
  "examples": ["example sentence 1"]
}

Indexing: TF-IDF (char 3-5 grams) for robust lookup; no heavy deps.
"""

from __future__ import annotations

import os
import json
from typing import List, Dict, Any, Optional, Tuple
import pathlib

import numpy as np


class LexiconIndex:
    def __init__(self, lang: str):
        self.lang = lang
        self.records: List[Dict[str, Any]] = []
        self.vectorizer = None
        self.matrix = None  # sparse matrix

    def load(self, base_dir: str = "data/lexicon") -> bool:
        path = pathlib.Path(base_dir) / f"{self.lang}.jsonl"
        if not path.exists():
            return False
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("lemma"):
                        self.records.append(rec)
                except Exception:
                    continue
        if not self.records:
            return False
        # Fit TF-IDF on combined text: lemma + defs + examples
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        except Exception:
            return False
        texts: List[str] = []
        for r in self.records:
            lemma = str(r.get("lemma", ""))
            defs = "; ".join(r.get("defs", []) or [])
            ex = "; ".join(r.get("examples", []) or [])
            texts.append(f"{lemma} || {defs} || {ex}")
        vec = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=1, norm='l2')
        self.matrix = vec.fit_transform(texts)
        self.vectorizer = vec
        return True

    def query(self, term: str, top_k: int = 1) -> List[Dict[str, Any]]:
        if not term or self.vectorizer is None or self.matrix is None:
            return []
        q = self.vectorizer.transform([term])
        scores = (q @ self.matrix.T)
        scores = np.asarray(scores.todense()).ravel()
        idx = np.argsort(scores)[::-1][:top_k]
        out: List[Dict[str, Any]] = []
        for i in idx:
            rec = self.records[i]
            out.append({
                "lemma": rec.get("lemma", ""),
                "pos": rec.get("pos", ""),
                "def": (rec.get("defs", []) or [""])[0],
                "example": (rec.get("examples", []) or [""])[0],
                "score": float(scores[i]),
            })
        return out


_cache: Dict[str, LexiconIndex] = {}


def get_lexicon(lang: str) -> Optional[LexiconIndex]:
    idx = _cache.get(lang)
    if idx is not None:
        return idx
    lex = LexiconIndex(lang)
    if lex.load():
        _cache[lang] = lex
        return lex
    return None


def lexicon_lookup(lang: str, terms: List[str], per_term: int = 1) -> Dict[str, List[Dict[str, Any]]]:
    idx = get_lexicon(lang)
    if not idx:
        return {}
    res: Dict[str, List[Dict[str, Any]]] = {}
    for t in terms:
        t = (t or '').strip()
        if not t:
            continue
        res[t] = idx.query(t, top_k=per_term)
    return res










