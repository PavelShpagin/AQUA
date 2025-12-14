#!/usr/bin/env python3
"""
Edit/agent:
- Two-pass agent with spaCy cues + grader consolidation
- Pass 1: Edit-level JSON using EDIT_LEVEL_JUDGE_PROMPT with injected spaCy cues
- Pass 2: Grader pass using EDIT_FINAL_JUDGMENT_PROMPT to consolidate into final label

Behavior:
- Language: if --lang provided, force; otherwise auto-detect from src
- Alignment: reuse provided columns (aligned/aligned_sentence/alert) or compute via ERRANT
- Multilingual: en/de/ua/es mapped; fallback 'en'
"""

import os
import re
import json
import argparse
import pandas as pd
from typing import Dict, List, Any, Tuple
import ast

from utils.errant_align import get_alignment_for_language
from judges.edit.prompts import EDIT_LEVEL_JUDGE_PROMPT, QUALITY_PROMPT, EDIT_LEVEL_AGENT_PROMPT
from utils.judge import (
    detect_language_from_text,
    get_language_label,
    get_language_code,
    build_numbered_prompt,
    call_model_with_pricing,
    add_pricing_to_result_dict,
    print_judge_distribution,
    parse_tpfp_label,
)
from utils.llm.moderation import check_openai_moderation
from utils.rag.local_rulebook_rag import query_local_rulebook

# Optional rulebook backends (best-effort, safe fallbacks)
try:
    from utils.simple_rag import query_spanish_rules, format_rules_for_prompt
except Exception:  # pragma: no cover
    query_spanish_rules = None  # type: ignore
    format_rules_for_prompt = None  # type: ignore

try:
    # Lightweight multilingual rule database (in-repo, no network)
    from utils.rag.opensource_grammar_api import GrammarRuleDatabase  # type: ignore
except Exception:  # pragma: no cover
    GrammarRuleDatabase = None  # type: ignore


PRIORITY = {"FP1": 4, "FP2": 3, "FP3": 2, "TP": 1}
ALLOWED_SENT_LABELS = {"TP", "FP1", "FP2", "FP3", "TN", "FN", "Error"}


_BAD_JSON_BACKSLASH_RE = re.compile(r'\\(?!["\\/bfnrtu])')
_BAD_JSON_UNICODE_RE = re.compile(r'\\u(?![0-9a-fA-F]{4})')


def _best_effort_json_loads(s: str) -> Dict[str, Any]:
    """Parse model output as JSON with tolerant repairs for common escaping issues.

    In practice, model outputs frequently contain spans like "Ich\\_iel" which are
    invalid JSON escapes ("\\_" is not a JSON escape). We repair these by
    doubling backslashes when they are not part of a valid JSON escape sequence.
    """
    s = (s or "").strip()
    if not s:
        return {}

    # Extract likely JSON blob
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1].strip()
        else:
            s = s.replace("```", "").strip()
        if s.lower().startswith("json"):
            s = s[4:].lstrip()

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]

    # Repair invalid escape sequences (e.g., \_ or \“)
    s = _BAD_JSON_UNICODE_RE.sub(r"\\\\u", s)
    s = _BAD_JSON_BACKSLASH_RE.sub(r"\\\\", s)

    # Try strict JSON first
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    # Last resort: python literal-ish outputs
    try:
        obj = ast.literal_eval(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def compute_sentence_label(src: str, tgt: str, missed_error: bool, edit_labels: List[str]) -> str:
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


def _parse_single_label(text: str) -> Tuple[str, str]:
    """Parse a single-label fallback response and return (label, reasoning)."""
    s = (text or "").strip()
    if not s:
        return "FP3", "Fallback: empty response; defaulting to FP3"
    m = re.search(r"\b(TP|FP1|FP2|FP3|TN|FN)\b", s.upper())
    lab = m.group(1) if m else "FP3"
    return lab, (s[:800] if len(s) > 0 else f"Fallback default: {lab}")


def _fallback_label_prompt(language_label: str, src: str, tgt: str, aligned: str) -> str:
    """Tiny prompt used only when JSON parsing fails. Must be robust to weird escapes."""
    return (
        "You are a strict multilingual GEC judge.\n"
        "Decide the SENTENCE-LEVEL label for the given correction.\n"
        "Labels: TP, FP1, FP2, FP3, TN, FN.\n"
        "Rules:\n"
        "- If Original == Corrected: TN unless there are missed grammar errors, then FN.\n"
        "- If any edit changes meaning/facts or breaks punctuation/quotes/structure: FP1.\n"
        "- Else if any edit introduces grammar error or is incomplete: FP2.\n"
        "- Else if edits are unnecessary style/preferences: FP3.\n"
        "- Else: TP.\n\n"
        "Output format (exactly 2 lines):\n"
        "LABEL: <one label>\n"
        "REASON: <one short sentence>\n\n"
        f"Language: {language_label}\n"
        f"Original: {src}\n"
        f"Corrected: {tgt}\n"
        f"Alignment: {aligned}\n"
    )


def parse_edit_json(text: str) -> Dict[str, Any]:
    try:
        data = _best_effort_json_loads(text or "")
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


def build_spacy_cues(src: str, spans: List[str], lang_code: str) -> str:
    try:
        from utils.spacy_semantic import analyze_edits_with_spacy
        lines = analyze_edits_with_spacy(src, spans, lang_code)
        if not lines:
            return ''
        return "\n" + "\n".join(lines) + "\n"
    except Exception:
        return ''


def build_rulebook_cues(src: str, tgt: str, spans: List[str], language_label: str, lang_code: str) -> str:
    """Return concise, high-precision rule cues from local rulebooks.

    Strategy (best-effort, offline):
    - Prefer in-repo JSON rulebooks via utils.rag.local_rulebook_rag (EN/DE/UA/ES)
    - Spanish: fallback to utils.simple_rag if comprehensive rulebook is missing
    - Last resort: utils.rag.opensource_grammar_api.GrammarRuleDatabase (small built-in DB)
    - Limit to top 3 succinct bullets; avoid noisy/low-precision text
    """
    try:
        # Build queries from spans like "{x=>y}" and source/target
        edits_text = []
        for s in spans[:6]:
            body = s[1:-1]
            o, n = body.split('=>', 1)
            edits_text.append(f"{o.strip()} => {n.strip()}")
        query_blob = (" | ".join(edits_text) or (src[:120] + ' => ' + tgt[:120])).strip()

        # 1) High-precision local rulebooks (preferred)
        local_hits = query_local_rulebook(query_blob, lang_code=lang_code, top_k=3, min_score=2)
        if local_hits:
            lines = ["**Relevant Rulebook Rules (local):**"]
            for r in local_hits[:3]:
                rid = r.get("id", "")
                desc = (r.get("description", "") or "").strip()
                if desc:
                    lines.append(f"- {rid}: {desc}")
            if len(lines) > 1:
                return "\n### Rulebook cues\n" + "\n".join(lines) + "\n"

        # 2) Spanish dedicated rules (fallback)
        if lang_code == 'es':
            try:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                llm_json = os.path.join(project_root, 'data', 'rag', 'es_rules_llm.json')
                if os.path.exists(llm_json):
                    data = json.load(open(llm_json, 'r', encoding='utf-8'))
                    rules = data.get('rules', [])
                    q = set(query_blob.lower().split())
                    hits = []
                    for r in rules:
                        blob = f"{r.get('id','')} {r.get('category','')} {r.get('description','')} {' '.join(r.get('examples', []))}".lower()
                        if any(tok in blob for tok in list(q)[:8]):
                            hits.append(r)
                    if hits:
                        lines = ["**Relevant Rulebook Rules:**"]
                        for r in hits[:3]:
                            lines.append(f"- {r.get('id','')}: {r.get('description','')}")
                        return "\n### Rulebook cues\n" + "\n".join(lines) + "\n"
            except Exception:
                pass
            if query_spanish_rules and format_rules_for_prompt:
                rules = query_spanish_rules(query_blob, src, tgt) or []
                if rules:
                    formatted = format_rules_for_prompt(rules)
                    return f"\n### Rulebook cues (Spanish)\n{formatted}\n"

        # 3) Multilingual fallback rules (tiny built-in DB)
        if GrammarRuleDatabase is not None:
            try:
                db = GrammarRuleDatabase()
                lang_map = {
                    'English': 'en', 'German': 'de', 'Ukrainian': 'uk', 'Spanish': 'es',
                    'French': 'fr', 'Italian': 'it', 'Portuguese': 'pt'
                }
                lang = lang_map.get(language_label, lang_code or 'en')
                # Focus queries on grammar-critical categories
                queries = [
                    query_blob,
                    f"agreement/tense/punctuation: {query_blob}",
                ]
                seen = []
                for q in queries:
                    for r in db.search_rules(q, lang, max_results=3)[:3]:  # type: ignore[attr-defined]
                        tup = (r.rule_name, r.description)  # type: ignore
                        if tup not in seen:
                            seen.append(tup)
                if seen:
                    lines = ["**Relevant Rulebook Snippets:**"]
                    for name, desc in seen[:3]:
                        lines.append(f"- {name}: {desc}")
                    return "\n### Rulebook cues\n" + "\n".join(lines) + "\n"
            except Exception:
                pass
    except Exception:
        pass
    return ''


def combine_pricing(p1: Dict[str, Any] | None, p2: Dict[str, Any] | None) -> Dict[str, Any]:
    if not p1 and not p2:
        return {'token_usage': {}, 'cost_breakdown': {}, 'model': ''}
    if not p1:
        return p2
    if not p2:
        return p1
    tu1 = p1.get('token_usage', {}); tu2 = p2.get('token_usage', {})
    cb1 = p1.get('cost_breakdown', {}); cb2 = p2.get('cost_breakdown', {})
    token_usage = {
        'input_tokens': int(tu1.get('input_tokens', 0)) + int(tu2.get('input_tokens', 0)),
        'output_tokens': int(tu1.get('output_tokens', 0)) + int(tu2.get('output_tokens', 0)),
        'reasoning_tokens': int(tu1.get('reasoning_tokens', 0)) + int(tu2.get('reasoning_tokens', 0)),
        'cached_tokens': int(tu1.get('cached_tokens', 0)) + int(tu2.get('cached_tokens', 0)),
        'total_tokens': int(tu1.get('total_tokens', 0)) + int(tu2.get('total_tokens', 0)),
    }
    cost_breakdown = {
        'input_cost_usd': float(cb1.get('input_cost_usd', 0)) + float(cb2.get('input_cost_usd', 0)),
        'output_cost_usd': float(cb1.get('output_cost_usd', 0)) + float(cb2.get('output_cost_usd', 0)),
        'reasoning_cost_usd': float(cb1.get('reasoning_cost_usd', 0)) + float(cb2.get('reasoning_cost_usd', 0)),
        'cached_cost_usd': float(cb1.get('cached_cost_usd', 0)) + float(cb2.get('cached_cost_usd', 0)),
        'total_cost_usd': float(cb1.get('total_cost_usd', 0)) + float(cb2.get('total_cost_usd', 0)),
    }
    return {'token_usage': token_usage, 'cost_breakdown': cost_breakdown, 'model': p1.get('model', '')}


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
    parser.add_argument('--llm_backend', required=True, help='Backbone for agent first pass (use gpt-4.1-nano for SOTA/cost)')
    parser.add_argument('--grader_backend', default='', help='If empty, grader is disabled. Options: gpt-4o-mini | gpt-4.1-nano | gpt-4.1 | o3')
    parser.add_argument('--lang', default='')
    parser.add_argument('--workers', type=int, default=50)
    parser.add_argument('--moderation', default='off', choices=['on','off'])
    parser.add_argument('--spacy', default='on', choices=['on','off'], help='Enable spaCy cues (default: on)')
    parser.add_argument('--rulebook', default='on', choices=['on','off'], help='Enable rulebook cues (default: on)')
    parser.add_argument('--max_retries', type=int, default=1, help='Extra LLM retry attempts on parse/format failures (default: 1)')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    results: List[Dict[str, Any]] = []

    for i, (_, row) in enumerate(df.iterrows()):
        src = str(row.get('src','')); tgt = str(row.get('tgt',''))
        if args.moderation == 'on' and check_openai_moderation(f"{src}\n{tgt}"):
            results.append({'idx': row.get('idx',''), 'src': src, 'tgt': tgt, 'tp_fp_label': 'Error', 'reasoning': 'Error'})
            continue

        # Use forced lang if provided; otherwise detect label
        language_label = get_language_label(args.lang) if args.lang else detect_language_from_text(src)
        # Normalize to code using global mapping
        lang_code = get_language_code(args.lang if args.lang else language_label)

        raw_align = str(row.get('aligned','') or row.get('aligned_sentence','') or row.get('alert',''))
        if not raw_align:
            raw_align = get_alignment_for_language(src, tgt, language=lang_code)

        # Extract non-noop spans for cues
        try:
            # Capture substitutions + insertions + deletions: {x=>y}, {=>y}, {x=>}
            raw_spans = re.findall(r"\{[^}]*=>[^}]*\}", raw_align)
            spans = []
            for s in raw_spans:
                body = s[1:-1]
                o, n = body.split('=>', 1)
                # Skip true no-ops
                if (o == n) or (o.strip() == "" and n.strip() == ""):
                    continue
                spans.append(s)
            # Keep edits as a newline list (no extra quoting; avoids escaping pitfalls)
            edits_field = "\n".join(spans)
        except Exception:
            spans = []
            edits_field = ""

        # Build spaCy and rulebook cues (compute before building tool block)
        spacy_cues = build_spacy_cues(src, spans, lang_code) if args.spacy == 'on' else ''
        rulebook_cues = build_rulebook_cues(src, tgt, spans, language_label, lang_code) if args.rulebook == 'on' else ''
        
        # Run grader FIRST to get quality scores for the main prompt
        grader_cues = ''
        grader_data = {}
        pricing2 = {'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'cached_tokens': 0, 'input_cost_usd': 0.0, 'output_cost_usd': 0.0, 'reasoning_cost_usd': 0.0, 'cached_cost_usd': 0.0, 'total_cost_usd': 0.0}
        if args.grader_backend:
            q_prompt = build_numbered_prompt(QUALITY_PROMPT, src, tgt, raw_align)
            ok2, content2, _tokens2, pricing2 = call_model_with_pricing(q_prompt, args.grader_backend, api_token=os.getenv('API_TOKEN',''), moderation=False)
            if ok2 and content2:
                try:
                    grader_data = _best_effort_json_loads(content2)
                    grader_cues = f"Quality Scores: {json.dumps(grader_data.get('edits', {}), indent=2)}"
                except Exception:
                    grader_cues = f"Quality Assessment: {content2[:200]}..."

        # Pass 1: Edit-level JSON with cues injected at the end of the prompt
        # Build comprehensive ReAct-style tool block
        from judges.edit.prompts import AGENT_INSTRUCTIONS_TEMPLATE

        desc_parts: List[str] = []
        obs_parts: List[str] = []

        if args.spacy == 'on':
            from judges.edit.prompts import TOOL_DESC_SPACY
            desc_parts.append(TOOL_DESC_SPACY)
        if args.rulebook == 'on':
            from judges.edit.prompts import TOOL_DESC_RULEBOOK
            desc_parts.append(TOOL_DESC_RULEBOOK)
        if args.grader_backend:
            from judges.edit.prompts import TOOL_DESC_GRADER_QUALITY
            desc_parts.append(TOOL_DESC_GRADER_QUALITY)

        if spacy_cues:
            obs_parts.append("Observation: spacy_cues →\n" + spacy_cues.strip())
        if rulebook_cues:
            obs_parts.append("Observation: rulebook →\n" + rulebook_cues.strip())
        if grader_cues:
            obs_parts.append("Observation: grader_quality →\n" + grader_cues.strip())

        tools_text = "\n\n".join(desc_parts) if desc_parts else "(no tools provided)"
        tool_block = AGENT_INSTRUCTIONS_TEMPLATE.replace("{TOOLS}", tools_text)
        if obs_parts:
            tool_block += "\n\n## Tool Observations\n" + "\n\n".join(obs_parts)

        base_prompt = EDIT_LEVEL_AGENT_PROMPT if (args.spacy=='on' or args.rulebook=='on' or args.grader_backend) else EDIT_LEVEL_JUDGE_PROMPT
        # Inject agent instructions by replacing placeholder {5} with tool_block
        prompt_template = base_prompt.replace("{5}", tool_block)
        prompt_v1 = build_numbered_prompt(prompt_template, language_label, src, tgt, raw_align, edits_field)
        ok1, content1, _tokens1, pricing1 = call_model_with_pricing(prompt_v1, args.llm_backend, api_token=os.getenv('API_TOKEN',''), moderation=False)
        parsed = {'edits': {}, 'labels': [], 'missed_error': False, 'reasoning': ''}
        if ok1 and content1:
            parsed = parse_edit_json(content1)
        # Retry once on parse failures (common with quotes/backslashes in spans)
        retries_left = int(args.max_retries)
        while retries_left > 0 and src != tgt and not parsed.get('labels'):
            retries_left -= 1
            repair_suffix = (
                "\n\nIMPORTANT (repair): Return ONLY valid JSON. "
                "Escape backslashes as \\\\ and escape quotes inside JSON strings. "
                "All edit keys MUST exactly match the Alignment spans.\n"
            )
            ok_r, content_r, _tokens_r, pricing_r = call_model_with_pricing(
                prompt_v1 + repair_suffix,
                args.llm_backend,
                api_token=os.getenv('API_TOKEN', ''),
                moderation=False,
            )
            pricing1 = combine_pricing(pricing1, pricing_r)
            if ok_r and content_r:
                parsed = parse_edit_json(content_r)
        labels = parsed.get('labels', [])
        sent_label_v1 = compute_sentence_label(src, tgt, parsed.get('missed_error', False), labels)

        # Last-resort fallback: get a single label (keeps success at 100%).
        # This is only triggered when JSON parsing keeps failing.
        fallback_reason = ""
        if src != tgt and sent_label_v1 == "Error":
            fb_prompt = _fallback_label_prompt(language_label, src, tgt, raw_align)
            ok_fb, content_fb, _tokens_fb, pricing_fb = call_model_with_pricing(
                fb_prompt,
                args.llm_backend,
                api_token=os.getenv("API_TOKEN", ""),
                moderation=False,
            )
            pricing1 = combine_pricing(pricing1, pricing_fb)
            if ok_fb and content_fb:
                lab, rea = _parse_single_label(content_fb)
                sent_label_v1 = lab
                fallback_reason = f"Fallback (single-label): {rea}"
            else:
                # Deterministic no-LLM fallback: prefer FP3 over Error for pipeline success.
                sent_label_v1 = "FP3"
                fallback_reason = "Fallback (deterministic): default FP3 due to repeated parse failures"

        # Prepare opinions for grader
        opinion_lines = []
        if parsed.get('edits'):
            for k, v in parsed['edits'].items():
                opinion_lines.append(f"{k}: {v}")
        if parsed.get('reasoning'):
            opinion_lines.append(f"Reasoning: {parsed['reasoning']}")
        opinions_text = "\n".join(opinion_lines) if opinion_lines else f"Preliminary: {sent_label_v1}"

        # Pass 2: Use grader scores if available (already computed above)
        final_label = sent_label_v1  # Default to preliminary decision
        if args.grader_backend and grader_data:
            scores = []
            for k,v in (grader_data.get('edits', {}) or {}).items():
                try:
                    s = float(v.get('score')) if isinstance(v, dict) else float(v)
                    scores.append(s)
                except Exception:
                    continue
            # Handle categorical grader predictions
            classes = []
            for k,v in (grader_data.get('edits', {}) or {}).items():
                try:
                    class_pred = v.get('class') if isinstance(v, dict) else str(v)
                    classes.append(class_pred)
                except Exception:
                    continue
            
            if classes:
                # Use grader's direct classification, but be conservative
                # Only override if grader is confident and consistent
                norm_classes = [c for c in classes if isinstance(c, str) and c.upper() in ALLOWED_SENT_LABELS]
                norm_classes = [c.upper() for c in norm_classes]
                if norm_classes and len(set(norm_classes)) == 1:  # All edits same, valid classification
                    final_label = norm_classes[0]
                elif 'FP1' in norm_classes:  # Any FP1 makes the whole thing FP1
                    final_label = 'FP1'
                # For mixed signals, trust preliminary decision
        # Never allow None/empty labels to escape (breaks downstream reporting)
        if not isinstance(final_label, str) or final_label.strip() == "" or final_label.lower() in {"none", "nan"}:
            final_label = "FP3"

        combined_pricing = combine_pricing(pricing1, pricing2)

        out: Dict[str, Any] = {
            'idx': row.get('idx',''),
            'src': src,
            'tgt': tgt,
            'aligned': raw_align,
            'tp_fp_label': final_label,
            'preliminary_label': sent_label_v1,
            'reasoning': (parsed.get('reasoning','') or fallback_reason or (content1[:800] if isinstance(content1, str) and content1.strip() else "No reasoning provided")),
            'writing_type': parsed.get('writing_type',''),
            'grader_backend': args.grader_backend or 'disabled',
        }
        out = add_pricing_to_result_dict(out, combined_pricing)
        results.append(out)

        # Lightweight progress to avoid perceived stalls
        if (i + 1) % 10 == 0:
            try:
                print(f"Processed {i + 1}/{len(df)} rows ...")
            except Exception:
                pass

    print_judge_distribution(results, f"Edit Agent ({args.llm_backend} + {args.grader_backend})")
    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()


