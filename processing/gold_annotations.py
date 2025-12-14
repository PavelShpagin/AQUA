import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


LANG_TO_FILE = {
    "EN": "data/annotations/f3120198.csv",
    "DE": "data/annotations/f3120001.csv",
    "UA": "data/annotations/f3119729.csv",
}


LABEL_MAP: Dict[str, str] = {
    # Expanded mapping to standard labels
    "good_edit": "TP",
    "optional_edit": "FP3",
    "incorrect_grammar_edit": "FP2",
    "hallucination_or_meaning_change": "FP1",
    # Already standardized
    "TP": "TP",
    "FP3": "FP3",
    "FP2": "FP2",
    "FP1": "FP1",
}


def coerce_label(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.lower() == "nan":
        return None
    return LABEL_MAP.get(s, None)


def parse_aligned_edits(aligned_text: str) -> List[Dict[str, str]]:
    """Parse aligned text to extract edits in the format {a=>b} or {=>b} or {a=>}."""
    if pd.isna(aligned_text) or not aligned_text:
        return []
    
    # Pattern to match edits like {old=>new} including insertions {=>new} and deletions {old=>}
    edit_pattern = r'\{([^}]*?)=>([^}]*?)\}'
    edits = []
    
    for match in re.finditer(edit_pattern, aligned_text):
        old_text = match.group(1).strip()
        new_text = match.group(2).strip()
        edits.append({
            'old': old_text,
            'new': new_text,
            'full_match': match.group(0)
        })
    
    return edits


def aggregate_edit_labels(row: pd.Series) -> List[Optional[str]]:
    """Aggregate edit labels for each edit using majority vote with tie-breaking."""
    # Parse the aligned text to get edits
    aligned = row.get('aligned', '')
    edits = parse_aligned_edits(aligned)
    
    if not edits:
        return []
    
    aggregated_labels: List[Optional[str]] = []

    # Pre-parse annotator edits once for matching
    annotator_parsed: List[Tuple[Optional[str], Optional[Tuple[str, str]]]] = []
    for annotator_idx in range(1, 6):
        label_col = f'id1_label{annotator_idx}'
        annotation_col = f'annotation_{annotator_idx}'
        raw_label = row.get(label_col)
        mapped_label = coerce_label(raw_label)
        annotation_val = row.get(annotation_col, '')
        matched_pair: Optional[Tuple[str, str]] = None
        if isinstance(annotation_val, str) and annotation_val:
            m = re.search(r'\{([^}]*?)=>([^}]*?)\}', annotation_val)
            if m:
                old_text = m.group(1).strip()
                new_text = m.group(2).strip()
                matched_pair = (old_text, new_text)
        annotator_parsed.append((mapped_label, matched_pair))

    # For each edit, collect labels that specifically map to this edit
    for edit_idx, edit in enumerate(edits):
        edit_labels: List[str] = []

        # If there's only one edit, any provided label applies
        if len(edits) == 1:
            for mapped_label, _ in annotator_parsed:
                if mapped_label:
                    edit_labels.append(mapped_label)
        else:
            # Multi-edit: attribute a label only if the annotator's annotation
            # matches this particular edit by old/new text
            for mapped_label, matched_pair in annotator_parsed:
                if not mapped_label or not matched_pair:
                    continue
                old_text, new_text = matched_pair
                if old_text == edit['old'] and new_text == edit['new']:
                    edit_labels.append(mapped_label)

        # Aggregate labels for this edit using majority vote with tie-breaking
        if edit_labels:
            counts = pd.Series(edit_labels).value_counts()
            max_count = int(counts.max())
            candidates = [lab for lab, cnt in counts.items() if cnt == max_count]

            if len(candidates) == 1:
                aggregated_label = candidates[0]
            else:
                # Tie-break: FP1 > FP2 > FP3 > TP (FP1 highest priority)
                priority = {"FP1": 4, "FP2": 3, "FP3": 2, "TP": 1}
                aggregated_label = max(candidates, key=lambda x: priority.get(x, 0))

            aggregated_labels.append(aggregated_label)
        else:
            aggregated_labels.append(None)

    return aggregated_labels


def _aggregate_edit_labels_across_group(row: pd.Series, all_data: Optional[pd.DataFrame]) -> List[Optional[str]]:
    """Aggregate labels per edit across all rows sharing the same src.

    For each canonical edit in row.aligned, collect votes from all rows with the same src,
    where a vote is counted only when an annotation_k matches that specific {old=>new} edit
    and contributes its corresponding id1_labelk (coerced to TP/FP*).
    """
    aligned = row.get('aligned', '')
    edits = parse_aligned_edits(aligned)
    if not edits or all_data is None or 'src' not in all_data.columns:
        return aggregate_edit_labels(row)

    src = str(row.get('src', ''))
    group = all_data[all_data['src'] == src]

    # Prepare containers for each edit index
    per_edit_votes: List[List[str]] = [[] for _ in range(len(edits))]

    # Build quick lookup of canonical edits for matching
    canonical = [(e['old'], e['new']) for e in edits]

    # Iterate over group rows and collect votes
    for _, r in group.iterrows():
        # Extract up to 5 annotations and their labels
        for k in range(1, 6):
            ann = r.get(f'annotation_{k}', '')
            raw_label = r.get(f'id1_label{k}', None)
            mapped = coerce_label(raw_label)
            if not mapped or not isinstance(ann, str) or not ann:
                continue
            m = re.search(r'\{([^}]*?)=>([^}]*?)\}', ann)
            if not m:
                continue
            old_text = m.group(1).strip()
            new_text = m.group(2).strip()
            # Find which canonical edit this matches
            for idx, (c_old, c_new) in enumerate(canonical):
                if old_text == c_old and new_text == c_new:
                    per_edit_votes[idx].append(mapped)
                    break

    # Majority with FP priority per edit
    aggregated: List[Optional[str]] = []
    priority = {"FP1": 4, "FP2": 3, "FP3": 2, "TP": 1}
    for votes in per_edit_votes:
        if votes:
            counts = pd.Series(votes).value_counts()
            max_count = int(counts.max())
            candidates = [lab for lab, cnt in counts.items() if cnt == max_count]
            if len(candidates) == 1:
                aggregated.append(candidates[0])
            else:
                aggregated.append(max(candidates, key=lambda x: priority.get(x, 0)))
        else:
            aggregated.append(None)

    return aggregated


def create_alignment_labels(row: pd.Series, all_data: Optional[pd.DataFrame] = None) -> str:
    """Create the alignment_labels string with aggregated labels for each edit.

    Uses group-level aggregation across all rows with the same src when all_data is provided.
    Falls back to row-level mapping when grouping information is absent.
    """
    aligned = row.get('aligned', '')
    if pd.isna(aligned) or not aligned:
        return aligned

    # Parse edits
    edits = parse_aligned_edits(aligned)
    if not edits:
        return aligned

    # Get aggregated labels (group-level if possible)
    labels = _aggregate_edit_labels_across_group(row, all_data)

    # Build result by replacing each edit with its labeled version
    result = aligned
    # Process edits in reverse order to maintain correct positions
    for edit, label in reversed(list(zip(edits, labels))):
        if label:
            start_pos = result.find(edit['full_match'])
            if start_pos != -1:
                labeled_edit = edit['full_match'][:-1] + f":::{label}" + "}"
                result = result[:start_pos] + labeled_edit + result[start_pos + len(edit['full_match']):]

    return result


def compute_edit_vote_stats(
    row: pd.Series,
    all_data: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """Compute detailed per-edit vote stats across all rows with the same src.

    Returns keys:
      - labels_by_edit: List[List[str]] including 'TP','FP1','FP2','FP3','cant_judge' as seen
      - dominant_per_edit: List[Optional[str]] dominant labels (valid labels only), tie-broken
      - dominant_votes_per_edit: List[int] number of votes for dominant label (0 if none)
      - has_tie: bool if any edit has a tie among valid labels
      - cj_counts_per_edit: List[int] counts of 'cant_judge' per edit
      - alignment_labels_detailed: str constructed by appending :::label_i for each label in labels_by_edit
      - missed_error_detailed: int number of rows with missed_error == 'missed_error'
      - num_annotators: int number of rows in group
    """
    aligned = row.get("aligned", "")
    edits = parse_aligned_edits(aligned)
    if not edits or all_data is None or "src" not in all_data.columns:
        return {
            "labels_by_edit": [[] for _ in range(len(edits))],
            "dominant_per_edit": [None for _ in range(len(edits))],
            "dominant_votes_per_edit": [0 for _ in range(len(edits))],
            "has_tie": False,
            "cj_counts_per_edit": [0 for _ in range(len(edits))],
            "alignment_labels_detailed": aligned,
            "missed_error_detailed": 0,
            "num_annotators": 0,
        }

    src = str(row.get("src", ""))
    group = all_data[all_data["src"] == src]
    num_annotators = int(len(group))

    # Prepare containers per edit
    labels_by_edit: List[List[str]] = [[] for _ in range(len(edits))]
    dominant_per_edit: List[Optional[str]] = [None for _ in range(len(edits))]
    dominant_votes_per_edit: List[int] = [0 for _ in range(len(edits))]
    cj_counts_per_edit: List[int] = [0 for _ in range(len(edits))]

    canonical = [(e["old"], e["new"]) for e in edits]

    # Collect votes across all rows
    for _, r in group.iterrows():
        for k in range(1, 6):
            ann = r.get(f"annotation_{k}", "")
            raw_label = (r.get(f"id1_label{k}", None))
            if not isinstance(ann, str) or not ann:
                continue
            m = re.search(r"\{([^}]*?)=>([^}]*?)\}", ann)
            if not m:
                continue
            old_text = m.group(1).strip()
            new_text = m.group(2).strip()
            # Determine mapped label for detailed view
            label_str: Optional[str]
            if raw_label is None or str(raw_label).strip() == "":
                label_str = None
            else:
                s = str(raw_label).strip()
                if s == "cant_judge":
                    label_str = "cant_judge"
                else:
                    label_str = LABEL_MAP.get(s, s)

            if label_str is None:
                continue

            for idx, (c_old, c_new) in enumerate(canonical):
                if old_text == c_old and new_text == c_new:
                    labels_by_edit[idx].append(label_str)
                    if label_str == "cant_judge":
                        cj_counts_per_edit[idx] += 1
                    break

    # Compute dominant labels, ties, and build detailed alignment
    valid_labels = {"TP", "FP1", "FP2", "FP3"}
    tie_any = False
    priority = {"FP1": 4, "FP2": 3, "FP3": 2, "TP": 1}
    for idx, labels in enumerate(labels_by_edit):
        # Count only valid labels for dominance
        counts = pd.Series([l for l in labels if l in valid_labels]).value_counts() if labels else pd.Series(dtype=int)
        if not counts.empty:
            max_count = int(counts.max())
            candidates = [lab for lab, cnt in counts.items() if cnt == max_count]
            dominant_votes_per_edit[idx] = max_count
            if len(candidates) == 1:
                dominant_per_edit[idx] = candidates[0]
            else:
                tie_any = True
                dominant_per_edit[idx] = max(candidates, key=lambda x: priority.get(x, 0))
        else:
            dominant_per_edit[idx] = None
            dominant_votes_per_edit[idx] = 0

    # Build alignment_labels_detailed by appending all labels for each edit
    result_detailed = aligned
    for edit, label_list in reversed(list(zip(edits, labels_by_edit))):
        suffix = ":::" + ":::".join(label_list) if label_list else ":::"
        start_pos = result_detailed.find(edit["full_match"])
        if start_pos != -1:
            replaced = edit["full_match"][:-1] + suffix + "}"
            result_detailed = (
                result_detailed[:start_pos]
                + replaced
                + result_detailed[start_pos + len(edit["full_match"]):]
            )

    # Count missed_error voters in group
    me_count = 0
    if "missed_error" in group.columns:
        me_count = int((group["missed_error"] == "missed_error").sum())

    return {
        "labels_by_edit": labels_by_edit,
        "dominant_per_edit": dominant_per_edit,
        "dominant_votes_per_edit": dominant_votes_per_edit,
        "has_tie": tie_any,
        "cj_counts_per_edit": cj_counts_per_edit,
        "alignment_labels_detailed": result_detailed,
        "missed_error_detailed": me_count,
        "num_annotators": num_annotators,
    }


def compute_sentence_level_label_all(row: pd.Series, all_data: pd.DataFrame = None) -> Tuple[Optional[str], bool]:
    """ALL strategy mirroring annotations/analysis_utils.py

    Steps:
    1) missing_error = majority choice (across same src), ties prefer True
    2) sentence edit label = majority across id1_label1..5 with tie-break FP1>FP2>FP3>TP
    3) if src==tgt: return FN if missing_error else TN
    4) else: if missing_error and label==TP → FN; else label
    
    Returns:
        Tuple of (label, missed_error)
    """
    src = str(row.get("src", ""))
    tgt = str(row.get("tgt", ""))

    # 1) Aggregate missing_error across occurrences of the same src
    missing_error = False
    if all_data is not None and "src" in all_data.columns:
        group = all_data[all_data["src"] == src]
        true_count = 0
        false_count = 0
        if "missed_error" in group.columns:
            true_count = (group["missed_error"] == "missed_error").sum()
            false_count = len(group) - true_count
        else:
            # Fallback to per-row booleans if present
            for col in ["missed_error", "missed_error_gold", "llm_missed_error"]:
                if col in row.index and pd.notna(row[col]):
                    if row[col] is True or str(row[col]) in {"True", "missed_error"}:
                        true_count = 1
                        false_count = 0
                    else:
                        true_count = 0
                        false_count = 1
                    break

        if true_count > false_count:
            missing_error = True
        elif false_count > true_count:
            missing_error = False
        else:
            missing_error = True  # tie → prefer True

    # 3) If src == tgt, handle TN/FN
    if src == tgt:
        return ("FN" if missing_error else "TN", missing_error)

    # 2) Majority across id1_label1..5 with FP priority tie-break
    labels: List[str] = []
    for i in range(1, 6):
        mapped = coerce_label(row.get(f"id1_label{i}"))
        if mapped is not None:
            labels.append(mapped)

    if len(labels) == 0:
        return (None, missing_error)

    counts = pd.Series(labels).value_counts()
    max_count = int(counts.max())
    candidates = [lab for lab, cnt in counts.items() if cnt == max_count]
    if len(candidates) == 1:
        sentence_label = candidates[0]
    else:
        # Tie-break FP1 > FP2 > FP3 > TP (FP1 highest priority)
        priority = {"FP1": 4, "FP2": 3, "FP3": 2, "TP": 1}
        sentence_label = max(candidates, key=lambda x: priority.get(x, 0))

    # 4) Final override with missing_error
    if missing_error and sentence_label == "TP":
        return ("FN", missing_error)
    return (sentence_label, missing_error)


def load_language_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    # Ensure required columns exist
    required_cols = {"src", "tgt", "id1_label1", "id1_label2", "id1_label3", "id1_label4", "id1_label5", "aligned"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        # aligned column might be missing, but we can continue without it
        if missing == ["aligned"]:
            print(f"Warning: 'aligned' column missing in {csv_path}, alignment_labels will be empty")
            df["aligned"] = ""
        else:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    return df


def build_gold_for_language(
    lang: str,
    csv_path: Path,
    drop_cant: Optional[int] = None,
    drop_ties: bool = False,
    consistency: Optional[int] = None,
) -> pd.DataFrame:
    df = load_language_df(csv_path)

    # Compute sentence-level label and missed_error with ALL strategy
    results = df.apply(lambda r: compute_sentence_level_label_all(r, df), axis=1)
    df["label"] = results.apply(lambda x: x[0] if x else None)
    df["missed_error"] = results.apply(lambda x: x[1] if x else False)

    # Keep only rows with a valid consensus label
    df = df[df["label"].notna()].copy()
    
    # Create alignment_labels column (group-aware aggregation)
    df["alignment_labels"] = df.apply(lambda r: create_alignment_labels(r, df), axis=1)

    # Compute detailed edit vote stats for filtering and detailed columns
    stats = df.apply(lambda r: compute_edit_vote_stats(r, df), axis=1)
    df["alignment_labels_detailed"] = stats.apply(lambda s: s.get("alignment_labels_detailed", ""))
    df["missed_error_detailed"] = stats.apply(lambda s: s.get("missed_error_detailed", 0))
    df["num_annotators"] = stats.apply(lambda s: s.get("num_annotators", 0))

    # Internal helper columns for filtering
    df["_has_tie_any"] = stats.apply(lambda s: bool(s.get("has_tie", False)))
    df["_cj_max"] = stats.apply(lambda s: max(s.get("cj_counts_per_edit", []) or [0]))
    df["_dom_min_votes"] = stats.apply(lambda s: min(s.get("dominant_votes_per_edit", []) or [0]))

    # Apply per-edit filters if requested (do not affect missed_error aggregation)
    if drop_cant in {1, 2, 3}:
        df = df[df["_cj_max"] < int(drop_cant)]
    if drop_ties:
        df = df[~df["_has_tie_any"]]
    if consistency is not None:
        if consistency not in {1, 2, 3, 4, 5}:
            raise ValueError("--consistency must be one of 1,2,3,4,5")
        df = df[df["_dom_min_votes"] >= int(consistency)]

    # Keep necessary columns including aligned, alignment labels, and detailed info
    columns_to_keep = [
        "src",
        "tgt",
        "label",
        "aligned",
        "alignment_labels",
        "alignment_labels_detailed",
        "missed_error",
        "missed_error_detailed",
        "num_annotators",
    ]
    # Only keep columns that exist
    columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    df = df[columns_to_keep].drop_duplicates().reset_index(drop=True)

    # Enforce unique sentence-edit pairs (unique src,tgt); keep first occurrence
    df = df.drop_duplicates(subset=["src", "tgt"], keep="first").reset_index(drop=True)

    # Assign idx per language
    df.insert(0, "idx", range(len(df)))
    return df


def main(
    output_dir: Path,
    drop_cant: Optional[int] = None,
    drop_ties: bool = False,
    consistency: Optional[int] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[Tuple[str, str]] = [
        ("EN", "gold_en.csv"),
        ("DE", "gold_de.csv"),
        ("UA", "gold_ua.csv"),
    ]

    for lang, out_name in outputs:
        csv_path = Path(LANG_TO_FILE[lang])
        gold_df = build_gold_for_language(
            lang,
            csv_path,
            drop_cant=drop_cant,
            drop_ties=drop_ties,
            consistency=consistency,
        )
        out_path = output_dir / out_name
        gold_df.to_csv(out_path, index=False)
        print(f"Wrote {lang} gold to {out_path} with {len(gold_df)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process annotations into ALL-strategy gold datasets (EN/DE/UA)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/eval",
        help="Directory to write gold CSVs (default: data/eval)",
    )
    parser.add_argument(
        "--drop_cant",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Drop sentences where any edit has at least N cant_judge votes (N in {1,2,3})",
    )
    parser.add_argument(
        "--drop_ties",
        action="store_true",
        help="Drop sentences where any edit has a tie for dominant label (valid labels only)",
    )
    parser.add_argument(
        "--consistency",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Drop sentences where any edit's dominant label has fewer than N votes",
    )
    args = parser.parse_args()

    main(
        Path(args.output_dir),
        drop_cant=args.drop_cant,
        drop_ties=args.drop_ties,
        consistency=args.consistency,
    )


