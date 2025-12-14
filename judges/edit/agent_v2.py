#!/usr/bin/env python3
"""
Edit-level Agent v2 (incremental, publication-oriented).

Design goals:
- Start clean: ReAct-style reasoning without external tools.
- Keep cost low: one call by default; optional second pass only on uncertainty.
- Output stable sentence-level label: TP/FP1/FP2/FP3/TN/FN (no Error).

This file intentionally avoids the complex tool stack in judges/edit/agent.py.
We will iterate improvements here in small, measurable steps.
"""

import argparse
import os
import re
from typing import Any, Dict, Tuple, Optional, List

import pandas as pd

from utils.errant_align import get_alignment_for_language
from utils.judge import (
    detect_language_from_text,
    get_language_label,
    get_language_code,
    call_model_with_pricing,
    add_pricing_to_result_dict,
)


ALLOWED = {"TP", "FP1", "FP2", "FP3", "TN", "FN"}


SYSTEM_PROMPT = """You are an expert multilingual GEC judge.

Task: decide the SENTENCE-LEVEL label for a proposed correction.

Labels:
- TN: Original==Corrected and no clear grammar errors.
- FN: Original==Corrected but there are clear missed grammar errors.
- TP: Corrects objective grammar/spelling/punctuation errors with minimal meaning change.
- FP3: Unnecessary optional preference/style change (original was already correct).
- FP2: Introduces a new grammar error OR makes correction incomplete (would need more edits).
- FP1: Meaning/fact change OR breaks structure (quotes/brackets/markup) OR introduces nonsense.

Be conservative: do not call something FP1 unless it's clearly meaning/structure damaging.
"""

GATED_USER_TEMPLATE = """Language: {language}
Original: {src}
Corrected: {tgt}
Alignment: {aligned}
Hints (computed, may be imperfect):
- src==tgt: {same}
- edit_spans: {n_spans}
- digits_changed: {digits_changed}
- quote_balance_changed: {quote_balance_changed}
- backslash_present: {backslash_present}

Decision protocol (do this internally, but output only in the format below):
1) Decide BINARY: TP if the system's assessment is correct (good correction or correctly did nothing).
   Decide BINARY: FP if the assessment is incorrect (bad correction / unnecessary correction / missed errors).
2) Then choose LABEL:
   - If src==tgt: TN or FN.
   - Else if BINARY==TP: TP.
   - Else (BINARY==FP): FP1/FP2/FP3 based on severity definitions above.

Return exactly 4 lines:
BINARY: <TP|FP>
LABEL: <TP|FP1|FP2|FP3|TN|FN>
CONF: <0-100 integer>
REASON: <one short sentence>
"""


USER_TEMPLATE = """Language: {language}
Original: {src}
Corrected: {tgt}
Alignment: {aligned}

Return exactly 2 lines:
LABEL: <TP|FP1|FP2|FP3|TN|FN>
REASON: <one short sentence>
"""


CRITIC_TEMPLATE = """You are a strict reviewer of a GEC-judge decision.

Given the same example and a proposed label, either CONFIRM it or CHANGE it.
Only CHANGE if there is a clear mistake.

Return exactly 3 lines:
DECISION: <CONFIRM|CHANGE>
LABEL: <TP|FP1|FP2|FP3|TN|FN>
REASON: <one short sentence>

Example:
Language: {language}
Original: {src}
Corrected: {tgt}
Alignment: {aligned}
Proposed LABEL: {proposed}
Proposed REASON: {proposed_reason}
"""

VERIFY_BINARY_TEMPLATE = """You are a strict multilingual GEC judge.

Binary task only:
Return TP if the correction decision is correct (good correction or correctly did nothing).
Return FP if the decision is incorrect (unnecessary/bad correction or missed errors).

Return exactly 2 lines:
BINARY: <TP|FP>
REASON: <one short sentence>

Language: {language}
Original: {src}
Corrected: {tgt}
Alignment: {aligned}
"""


def _parse_two_line(text: str) -> Tuple[str, str]:
    s = (text or "").strip()
    lab = ""
    reason = ""
    for line in s.splitlines():
        if line.upper().startswith("LABEL:"):
            lab = line.split(":", 1)[1].strip().upper()
        if line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    if lab not in ALLOWED:
        # fallback: search token
        m = re.search(r"\b(TP|FP1|FP2|FP3|TN|FN)\b", s.upper())
        lab = m.group(1) if m else "FP3"
    if not reason:
        reason = (s[:240] if s else "No reason provided")
    return lab, reason


def _parse_gated(text: str) -> Tuple[str, str, int, str]:
    s = (text or "").strip()
    binary = ""
    lab = ""
    conf = -1
    reason = ""
    for line in s.splitlines():
        u = line.upper()
        if u.startswith("BINARY:"):
            binary = line.split(":", 1)[1].strip().upper()
        elif u.startswith("LABEL:"):
            lab = line.split(":", 1)[1].strip().upper()
        elif u.startswith("CONF:"):
            try:
                conf = int(re.findall(r"-?\\d+", line.split(":", 1)[1])[0])
            except Exception:
                conf = -1
        elif u.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    if binary not in {"TP", "FP"}:
        m = re.search(r"\\b(TP|FP)\\b", s.upper())
        binary = m.group(1) if m else "FP"
    if lab not in (ALLOWED | {"FP"}):
        m = re.search(r"\\b(TP|FP1|FP2|FP3|TN|FN)\\b", s.upper())
        lab = m.group(1) if m else "FP3"
    if lab == "FP":
        lab = "FP2"
    if conf < 0 or conf > 100:
        conf = 50
    if not reason:
        reason = (s[:240] if s else "No reason provided")
    return binary, lab, conf, reason


def _parse_critic(text: str) -> Tuple[str, str, str]:
    s = (text or "").strip()
    decision = ""
    label = ""
    reason = ""
    for line in s.splitlines():
        u = line.upper()
        if u.startswith("DECISION:"):
            decision = line.split(":", 1)[1].strip().upper()
        elif u.startswith("LABEL:"):
            label = line.split(":", 1)[1].strip().upper()
        elif u.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    if decision not in {"CONFIRM", "CHANGE"}:
        decision = "CONFIRM"
    if label not in ALLOWED:
        label = ""
    if not reason:
        reason = (s[:240] if s else "No critic reason")
    return decision, label, reason


def _parse_binary_only(text: str) -> Tuple[str, str]:
    s = (text or "").strip()
    binary = ""
    reason = ""
    for line in s.splitlines():
        u = line.upper()
        if u.startswith("BINARY:"):
            binary = line.split(":", 1)[1].strip().upper()
        elif u.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    if binary not in {"TP", "FP"}:
        m = re.search(r"\b(TP|FP)\b", s.upper())
        binary = m.group(1) if m else "FP"
    if not reason:
        reason = (s[:240] if s else "No reason")
    return binary, reason


def _is_uncertain(label: str, reason: str) -> bool:
    # Cheap heuristic: only re-ask when model expresses uncertainty or FP2/FP1 borderline language.
    r = (reason or "").lower()
    if any(w in r for w in ["unclear", "unsure", "hard", "ambiguous", "maybe", "not sure", "uncertain"]):
        return True
    if label in {"FP2"}:
        return True
    # FP1 is expensive mistake; double-check if reason is short/weak.
    if label == "FP1" and len(r) < 40:
        return True
    return False


def _count_edit_spans(aligned: str) -> int:
    try:
        return len(re.findall(r"\\{[^}]*=>[^}]*\\}", aligned or ""))
    except Exception:
        return 0


def _digits_changed(src: str, tgt: str) -> bool:
    try:
        a = re.findall(r"\\d+(?:[\\.,]\\d+)?", src or "")
        b = re.findall(r"\\d+(?:[\\.,]\\d+)?", tgt or "")
        return a != b
    except Exception:
        return False


def _quote_balance_changed(src: str, tgt: str) -> bool:
    # extremely lightweight: compare counts of common quote characters
    qs = ['"', "“", "”", "„", "«", "»", "’", "‘", "'"]
    try:
        return sum((src or "").count(q) for q in qs) != sum((tgt or "").count(q) for q in qs)
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--llm_backend", required=True)
    parser.add_argument("--lang", default="")
    parser.add_argument("--workers", type=int, default=3)  # kept for interface parity; loop is serial by design
    parser.add_argument("--critic", default="off", choices=["on", "off"], help="Enable cheap second-pass critique on uncertain cases")
    parser.add_argument("--critic_backend", default="", help="Backend for critique (default: same as llm_backend)")
    parser.add_argument("--format", default="gated", choices=["gated", "simple"], help="Output format: gated (binary+label) or simple (label only)")
    parser.add_argument("--binary_verify", default="off", choices=["on", "off"], help="If on, run a cheap binary-only verifier on low-confidence cases")
    parser.add_argument("--verify_backend", default="", help="Backend for binary verifier (default: same as llm_backend)")
    parser.add_argument("--verify_conf_lt", type=int, default=70, help="Trigger verifier when CONF < this threshold (default: 70)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        src = str(row.get("src", ""))
        tgt = str(row.get("tgt", ""))

        language_label = get_language_label(args.lang) if args.lang else detect_language_from_text(src)
        lang_code = get_language_code(args.lang if args.lang else language_label)

        aligned = str(row.get("aligned", "") or row.get("aligned_sentence", "") or row.get("alert", ""))
        if not aligned:
            aligned = get_alignment_for_language(src, tgt, language=lang_code)

        same = (src == tgt)
        n_spans = _count_edit_spans(aligned)
        digits_changed = _digits_changed(src, tgt)
        quote_balance_changed = _quote_balance_changed(src, tgt)
        backslash_present = ("\\" in (aligned or "")) or ("\\" in (tgt or "")) or ("\\" in (src or ""))

        if args.format == "gated":
            user_block = GATED_USER_TEMPLATE.format(
                language=language_label,
                src=src,
                tgt=tgt,
                aligned=aligned,
                same=same,
                n_spans=n_spans,
                digits_changed=digits_changed,
                quote_balance_changed=quote_balance_changed,
                backslash_present=backslash_present,
            )
        else:
            user_block = USER_TEMPLATE.format(language=language_label, src=src, tgt=tgt, aligned=aligned)

        prompt = SYSTEM_PROMPT + "\n\n" + user_block
        ok, content, _tokens, pricing = call_model_with_pricing(
            prompt, args.llm_backend, api_token=os.getenv("API_TOKEN", ""), moderation=False
        )
        raw_binary: Optional[str] = None
        raw_conf: Optional[int] = None
        if not ok:
            label, reason = "FP3", f"Call failed: {content}"
        else:
            if args.format == "gated":
                binary, label, conf, reason = _parse_gated(content)
                raw_binary, raw_conf = binary, conf
                # enforce internal consistency with src==tgt
                if same and label not in {"TN", "FN"}:
                    label = "TN"
                if (not same) and label in {"TN", "FN"}:
                    label = "FP3"
            else:
                label, reason = _parse_two_line(content)

        final_label = label
        final_reason = reason

        # Optional: binary-only verifier on low confidence (aim: improve binary accuracy cheaply)
        if args.binary_verify == "on" and args.format == "gated" and raw_conf is not None and raw_conf < int(args.verify_conf_lt):
            vb = args.verify_backend or args.llm_backend
            v_prompt = VERIFY_BINARY_TEMPLATE.format(
                language=language_label, src=src, tgt=tgt, aligned=aligned
            )
            okv, v_content, _tv, v_pricing = call_model_with_pricing(
                v_prompt, vb, api_token=os.getenv("API_TOKEN", ""), moderation=False
            )
            # merge pricing (lightweight)
            pricing = {
                "token_usage": {
                    "input_tokens": int(pricing.get("token_usage", {}).get("input_tokens", 0))
                    + int(v_pricing.get("token_usage", {}).get("input_tokens", 0)),
                    "output_tokens": int(pricing.get("token_usage", {}).get("output_tokens", 0))
                    + int(v_pricing.get("token_usage", {}).get("output_tokens", 0)),
                    "reasoning_tokens": int(pricing.get("token_usage", {}).get("reasoning_tokens", 0))
                    + int(v_pricing.get("token_usage", {}).get("reasoning_tokens", 0)),
                    "cached_tokens": int(pricing.get("token_usage", {}).get("cached_tokens", 0))
                    + int(v_pricing.get("token_usage", {}).get("cached_tokens", 0)),
                    "total_tokens": int(pricing.get("token_usage", {}).get("total_tokens", 0))
                    + int(v_pricing.get("token_usage", {}).get("total_tokens", 0)),
                },
                "cost_breakdown": {
                    "input_cost_usd": float(pricing.get("cost_breakdown", {}).get("input_cost_usd", 0))
                    + float(v_pricing.get("cost_breakdown", {}).get("input_cost_usd", 0)),
                    "output_cost_usd": float(pricing.get("cost_breakdown", {}).get("output_cost_usd", 0))
                    + float(v_pricing.get("cost_breakdown", {}).get("output_cost_usd", 0)),
                    "reasoning_cost_usd": float(pricing.get("cost_breakdown", {}).get("reasoning_cost_usd", 0))
                    + float(v_pricing.get("cost_breakdown", {}).get("reasoning_cost_usd", 0)),
                    "cached_cost_usd": float(pricing.get("cost_breakdown", {}).get("cached_cost_usd", 0))
                    + float(v_pricing.get("cost_breakdown", {}).get("cached_cost_usd", 0)),
                    "total_cost_usd": float(pricing.get("cost_breakdown", {}).get("total_cost_usd", 0))
                    + float(v_pricing.get("cost_breakdown", {}).get("total_cost_usd", 0)),
                },
                "model": pricing.get("model", vb),
            }
            if okv and v_content:
                vbinary, vreason = _parse_binary_only(v_content)
                # reconcile label with binary decision (keep TN/FN if src==tgt)
                if not same:
                    if vbinary == "TP" and final_label in {"FP1", "FP2", "FP3"}:
                        final_label = "TP"
                        final_reason = f"{final_reason} | verifier→TP: {vreason}"
                    if vbinary == "FP" and final_label == "TP":
                        final_label = "FP3"
                        final_reason = f"{final_reason} | verifier→FP: {vreason}"

        # Optional: second pass only when uncertain (minimal cost increase).
        if args.critic == "on" and _is_uncertain(label, reason):
            cb = args.critic_backend or args.llm_backend
            c_prompt = CRITIC_TEMPLATE.format(
                language=language_label,
                src=src,
                tgt=tgt,
                aligned=aligned,
                proposed=label,
                proposed_reason=reason,
            )
            ok2, content2, _t2, pricing2 = call_model_with_pricing(
                c_prompt, cb, api_token=os.getenv("API_TOKEN", ""), moderation=False
            )
            # Merge pricing
            pricing = {
                "token_usage": {
                    "input_tokens": int(pricing.get("token_usage", {}).get("input_tokens", 0))
                    + int(pricing2.get("token_usage", {}).get("input_tokens", 0)),
                    "output_tokens": int(pricing.get("token_usage", {}).get("output_tokens", 0))
                    + int(pricing2.get("token_usage", {}).get("output_tokens", 0)),
                    "reasoning_tokens": int(pricing.get("token_usage", {}).get("reasoning_tokens", 0))
                    + int(pricing2.get("token_usage", {}).get("reasoning_tokens", 0)),
                    "cached_tokens": int(pricing.get("token_usage", {}).get("cached_tokens", 0))
                    + int(pricing2.get("token_usage", {}).get("cached_tokens", 0)),
                    "total_tokens": int(pricing.get("token_usage", {}).get("total_tokens", 0))
                    + int(pricing2.get("token_usage", {}).get("total_tokens", 0)),
                },
                "cost_breakdown": {
                    "input_cost_usd": float(pricing.get("cost_breakdown", {}).get("input_cost_usd", 0))
                    + float(pricing2.get("cost_breakdown", {}).get("input_cost_usd", 0)),
                    "output_cost_usd": float(pricing.get("cost_breakdown", {}).get("output_cost_usd", 0))
                    + float(pricing2.get("cost_breakdown", {}).get("output_cost_usd", 0)),
                    "reasoning_cost_usd": float(pricing.get("cost_breakdown", {}).get("reasoning_cost_usd", 0))
                    + float(pricing2.get("cost_breakdown", {}).get("reasoning_cost_usd", 0)),
                    "cached_cost_usd": float(pricing.get("cost_breakdown", {}).get("cached_cost_usd", 0))
                    + float(pricing2.get("cost_breakdown", {}).get("cached_cost_usd", 0)),
                    "total_cost_usd": float(pricing.get("cost_breakdown", {}).get("total_cost_usd", 0))
                    + float(pricing2.get("cost_breakdown", {}).get("total_cost_usd", 0)),
                },
                "model": pricing.get("model", cb),
            }
            if ok2 and content2:
                decision, new_label, new_reason = _parse_critic(content2)
                if decision == "CHANGE" and new_label in ALLOWED:
                    final_label = new_label
                    final_reason = f"{new_reason} (critic changed from {label})"

        # Ensure never emit Error
        if final_label not in ALLOWED:
            final_label = "FP3"
        out = {
            "idx": row.get("idx", ""),
            "src": src,
            "tgt": tgt,
            "aligned": aligned,
            "tp_fp_label": final_label,
            "reasoning": final_reason,
            "writing_type": "",
            "agent_variant": f"react_v2(format={args.format}, critic={args.critic}, verify={args.binary_verify})",
        }
        out = add_pricing_to_result_dict(out, pricing)
        out_rows.append(out)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pd.DataFrame(out_rows).to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


