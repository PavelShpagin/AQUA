#!/usr/bin/env python3
"""
Local (offline) rulebook RAG for AQUA.

Goal: high-precision, low-noise retrieval from in-repo JSON rulebooks:
- data/rag/english/comprehensive_rules.json
- data/rag/german/comprehensive_rules.json
- data/rag/ukrainian/comprehensive_rules.json
- data/rag/spanish/comprehensive_rules.json (if present)

This intentionally avoids embeddings/vector DBs to keep runs reproducible and
lightweight. Retrieval is simple token overlap with a few safety filters.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


_WORD_RE = re.compile(r"[\\w’'-]+", re.UNICODE)


def _tokens(s: str) -> List[str]:
    s = (s or "").lower()
    return [t for t in _WORD_RE.findall(s) if len(t) >= 2]


@dataclass(frozen=True)
class RuleHit:
    score: int
    rule: Dict[str, Any]


_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def _lang_to_path(lang_code: str) -> str:
    lang_code = (lang_code or "").lower()
    mapping = {
        "en": ("english", "comprehensive_rules.json"),
        "de": ("german", "comprehensive_rules.json"),
        "ua": ("ukrainian", "comprehensive_rules.json"),
        "uk": ("ukrainian", "comprehensive_rules.json"),
        "es": ("spanish", "comprehensive_rules.json"),
    }
    folder, fname = mapping.get(lang_code, ("english", "comprehensive_rules.json"))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(project_root, "data", "rag", folder, fname)


def _load_rules(lang_code: str) -> List[Dict[str, Any]]:
    key = (lang_code or "").lower()
    if key in _CACHE:
        return _CACHE[key]
    path = _lang_to_path(lang_code)
    rules: List[Dict[str, Any]] = []
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rules = data.get("rules", []) if isinstance(data, dict) else []
    except Exception:
        rules = []
    _CACHE[key] = rules
    return rules


def query_local_rulebook(query: str, *, lang_code: str, top_k: int = 3, min_score: int = 2) -> List[Dict[str, Any]]:
    """Return up to top_k rules with token-overlap score >= min_score."""
    rules = _load_rules(lang_code)
    if not rules:
        return []

    q_tokens = set(_tokens(query))
    if not q_tokens:
        return []

    hits: List[RuleHit] = []
    for r in rules:
        try:
            blob = " ".join(
                [
                    str(r.get("rule_id", "")),
                    str(r.get("rule_name", "")),
                    str(r.get("description", "")),
                    " ".join(r.get("keywords", []) or []),
                    " ".join(r.get("examples", []) or []),
                ]
            )
            t = set(_tokens(blob))
            score = len(q_tokens & t)
            if score >= min_score:
                hits.append(RuleHit(score=score, rule=r))
        except Exception:
            continue

    hits.sort(key=lambda h: (-h.score, str(h.rule.get("rule_id", ""))))
    out: List[Dict[str, Any]] = []
    for h in hits[: max(1, top_k)]:
        r = h.rule
        out.append(
            {
                "id": r.get("rule_id", ""),
                "name": r.get("rule_name", ""),
                "description": r.get("description", ""),
                "examples": (r.get("examples", []) or [])[:2],
                "score": h.score,
            }
        )
    return out


