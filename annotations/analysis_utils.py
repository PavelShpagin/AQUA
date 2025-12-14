import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from collections import Counter

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data' / 'annotations'

STANDARD_LABELS: List[str] = ['TP', 'FP1', 'FP2', 'FP3', 'TN', 'FN']

LANG_FILES: Dict[str, str] = {
    'UA': 'f3119729.csv',
    'DE': 'f3120001.csv',
    'EN': 'f3120198.csv',
}

LABEL_MAP = {
    'good_edit': 'TP',
    'optional_edit': 'FP3',
    'incorrect_grammar_edit': 'FP2',
    'hallucination_or_meaning_change': 'FP1',
}


def load_annotation_data() -> pd.DataFrame:
    """Load all annotation data. Each row = one annotator's judgment of one sentence."""
    frames: List[pd.DataFrame] = []
    for lang, fname in LANG_FILES.items():
        df = pd.read_csv(DATA_DIR / fname)
        df['language'] = lang
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_processed_data() -> pd.DataFrame:
    """
    UNIFIED: Load processed data from processed.csv if available, 
    otherwise fall back to analysis_utils aggregation.
    """
    processed_path = DATA_DIR / 'processed.csv'
    
    if processed_path.exists():
        print("Loading from processed.csv...")
        return pd.read_csv(processed_path)
    else:
        print("processed.csv not found, using analysis_utils aggregation...")
        all_data = load_annotation_data()
        return aggregate_sentence_annotations(all_data)


def aggregate_sentence_annotations(all_data: pd.DataFrame) -> pd.DataFrame:
    """
    CORRECTED: Aggregate multiple annotator judgments per sentence.
    Each row in input = one annotator's judgment. Group by sentence to get multiple annotators.
    """
    aggregated_sentences = []
    
    # Group by sentence (language + src)
    sentence_groups = all_data.groupby(['language', 'src'])
    
    for (language, src), group in sentence_groups:
        # Get basic sentence info from first row (should be same across annotators)
        first_row = group.iloc[0]
        
        sentence_data = {
            'language': language,
            'src': src,
            'tgt': first_row.get('tgt', ''),
            'llm_label': first_row.get('llm_label', ''),
            'llm_reasoning': first_row.get('llm_reasoning', ''),
            'num_annotators': len(group)
        }
        
        # Aggregate edit-level labels correctly
        for edit_num in range(1, 6):
            col = f'id1_label{edit_num}'
            if col in group.columns:
                edit_labels = []
                for _, row in group.iterrows():
                    if pd.notna(row[col]) and str(row[col]).strip() != '':
                        raw_label = str(row[col]).strip()
                        mapped_label = LABEL_MAP.get(raw_label, raw_label)
                        edit_labels.append(mapped_label)
                
                sentence_data[f'edit{edit_num}_labels'] = edit_labels
                sentence_data[f'edit{edit_num}_label_counts'] = dict(Counter(edit_labels))
        
        # Aggregate missing_error judgments
        missing_error_judgments = []
        for _, row in group.iterrows():
            missed_error = row.get('missed_error', None)
            if pd.notna(missed_error):
                if str(missed_error).strip() == 'missed_error':
                    missing_error_judgments.append(True)
                else:
                    missing_error_judgments.append(False)
        
        sentence_data['missing_error_judgments'] = missing_error_judgments
        sentence_data['missing_error_counts'] = dict(Counter(missing_error_judgments))
        
        # Compute sentence-level human label
        sentence_data['human_label'] = compute_sentence_level_human_label(sentence_data)
        
        aggregated_sentences.append(sentence_data)
    
    return pd.DataFrame(aggregated_sentences)


def compute_sentence_level_human_label(sentence_data: Dict) -> str:
    """Compute sentence-level human label using corrected hybrid algorithm."""
    src = str(sentence_data.get('src', ''))
    tgt = str(sentence_data.get('tgt', ''))
    
    # Step 1: Majority vote for missing_error (ties prefer True)
    missing_error_judgments = sentence_data.get('missing_error_judgments', [])
    if missing_error_judgments:
        true_count = sum(missing_error_judgments)
        false_count = len(missing_error_judgments) - true_count
        if true_count > false_count:
            missing_error = True
        elif false_count > true_count:
            missing_error = False
        else:
            missing_error = True  # Tie -> prefer missing_error=True
    else:
        missing_error = False
    
    # Step 3: Handle src == tgt cases first
    if src == tgt:
        return 'FN' if missing_error else 'TN'
    
    # Step 2: Get all edit labels and compute majority vote
    all_edit_labels = []
    for edit_num in range(1, 6):
        edit_labels = sentence_data.get(f'edit{edit_num}_labels', [])
        all_edit_labels.extend(edit_labels)
    
    if not all_edit_labels:
        return 'TN'
    
    # Count labels and find majority
    label_counts = Counter(all_edit_labels)
    max_count = max(label_counts.values())
    most_common = [lab for lab, count in label_counts.items() if count == max_count]
    
    if len(most_common) == 1:
        sentence_label = most_common[0]
    else:
        # Tie-breaking: FP1 > FP2 > FP3 > TP priority (FP1 is worst)
        tie_priority = {'FP1': 4, 'FP2': 3, 'FP3': 2, 'TP': 1}
        sentence_label = max(most_common, key=lambda x: tie_priority.get(x, 0))
    
    # Step 4: Apply final logic
    if missing_error and sentence_label == 'TP':
        return 'FN'
    else:
        return sentence_label


# Add missing functions for complete functionality
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa


def build_kappa_matrix_correct(aggregated_df: pd.DataFrame) -> np.ndarray:
    """
    Build Fleiss' kappa matrix CORRECTLY using properly aggregated edit-level data.
    Each row = one sentence, columns = [TP, FP3, FP2, FP1] counts from all annotators.
    """
    cats = ['TP', 'FP3', 'FP2', 'FP1']
    matrices = []
    row_totals = []
    
    for _, sentence in aggregated_df.iterrows():
        # Count all edit-level labels across all edits for this sentence
        total_counts = Counter()
        
        for edit_num in range(1, 6):
            label_counts = sentence.get(f'edit{edit_num}_label_counts', {})
            for label, count in label_counts.items():
                if label in cats:
                    total_counts[label] += count
        
        if not total_counts:
            continue
            
        # Convert to matrix row
        counts = [total_counts.get(cat, 0) for cat in cats]
        total = sum(counts)
        
        if total == 0:
            continue
            
        matrices.append(counts)
        row_totals.append(total)
    
    if not matrices:
        return np.zeros((0, 4), dtype=float)
    
    # For Fleiss' kappa, use modal number of total ratings per sentence
    mode_total = Counter(row_totals).most_common(1)[0][0]
    filtered = [m for m, t in zip(matrices, row_totals) if t == mode_total]
    
    return np.asarray(filtered, dtype=float) if filtered else np.zeros((0, 4), dtype=float)


def fleiss_kappa_overall_and_by_lang(aggregated_df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """Calculate CORRECTED Fleiss' kappa overall and by language."""
    overall_mat = build_kappa_matrix_correct(aggregated_df)
    overall = float(fleiss_kappa(overall_mat)) if overall_mat.size else float('nan')
    
    per_lang = {}
    for lang in ['EN', 'DE', 'UA']:
        lang_data = aggregated_df[aggregated_df['language'] == lang]
        mat = build_kappa_matrix_correct(lang_data)
        per_lang[lang] = float(fleiss_kappa(mat)) if mat.size else float('nan')
    
    return overall, per_lang


def aggregate_sentences(all_data: pd.DataFrame, filter_method: str = 'all') -> pd.DataFrame:
    """Main aggregation function with filtering."""
    return aggregate_sentence_annotations(all_data)


def overall_agreement(aggregated_df: pd.DataFrame) -> float:
    """Calculate overall LLM-Human agreement."""
    valid_rows = aggregated_df.dropna(subset=['llm_label', 'human_label'])
    if len(valid_rows) == 0:
        return 0.0
    return float((valid_rows['llm_label'] == valid_rows['human_label']).mean() * 100)


def agreement_by_language(aggregated_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate LLM-Human agreement by language."""
    agreements = {}
    for lang in ['EN', 'DE', 'UA']:
        lang_data = aggregated_df[aggregated_df['language'] == lang]
        agreements[lang] = overall_agreement(lang_data)
    return agreements


def build_kappa_matrix_sentence_level(aggregated_df: pd.DataFrame) -> np.ndarray:
    """
    Build CORRECT Fleiss' kappa matrix using sentence-level labels.
    Each annotator provides ONE sentence-level judgment per sentence.
    Matrix: rows = sentences, columns = [TP, FP3, FP2, FP1, TN, FN] counts
    """
    cats = ['TP', 'FP3', 'FP2', 'FP1', 'TN', 'FN']
    matrices = []
    row_totals = []
    
    for _, sentence in aggregated_df.iterrows():
        num_annotators = sentence['num_annotators']
        
        # For sentence-level Kappa, we need to derive what each annotator's 
        # sentence-level judgment would be based on their edit-level judgments
        
        # This is complex because we need to go back to raw data to get
        # individual annotator's sentence-level judgments
        # For now, skip this sentence if we can't determine individual judgments
        continue
    
    return np.zeros((0, 6), dtype=float)


def build_kappa_matrix_edit_level(aggregated_df: pd.DataFrame, edit_num: int) -> np.ndarray:
    """
    Build Fleiss' kappa matrix for a specific edit number.
    This is probably the most meaningful approach - measure agreement per edit.
    """
    cats = ['TP', 'FP3', 'FP2', 'FP1']
    matrices = []
    row_totals = []
    
    for _, sentence in aggregated_df.iterrows():
        label_counts = sentence.get(f'edit{edit_num}_label_counts', {})
        
        if not label_counts:
            continue
            
        # Convert to matrix row
        counts = [label_counts.get(cat, 0) for cat in cats]
        total = sum(counts)
        
        if total == 0:
            continue
            
        matrices.append(counts)
        row_totals.append(total)
    
    if not matrices:
        return np.zeros((0, 4), dtype=float)
    
    # Use modal number of raters for this edit
    mode_total = Counter(row_totals).most_common(1)[0][0]
    filtered = [m for m, t in zip(matrices, row_totals) if t == mode_total]
    
    return np.asarray(filtered, dtype=float) if filtered else np.zeros((0, 4), dtype=float)


def build_kappa_matrix_all_edits_correct(aggregated_df: pd.DataFrame) -> np.ndarray:
    """
    CORRECT Fleiss Kappa matrix construction:
    - Each ROW = one edit instance (from any sentence, any edit position)  
    - Each COLUMN = [TP, FP3, FP2, FP1]
    - Each CELL = count of annotators who gave that label for this edit
    
    This creates a single matrix with ALL edits, not averaged across edit positions.
    """
    cats = ['TP', 'FP3', 'FP2', 'FP1']
    all_edit_rows = []
    row_totals = []
    
    for _, sentence in aggregated_df.iterrows():
        # Go through each edit position in this sentence
        for edit_num in range(1, 6):
            label_counts = sentence.get(f'edit{edit_num}_label_counts', {})
            
            if not label_counts:
                continue
                
            # Convert to matrix row: [TP_count, FP3_count, FP2_count, FP1_count]
            counts = [label_counts.get(cat, 0) for cat in cats]
            total = sum(counts)
            
            if total == 0:
                continue
                
            all_edit_rows.append(counts)
            row_totals.append(total)
    
    if not all_edit_rows:
        return np.zeros((0, 4), dtype=float)
    
    # For Fleiss Kappa, need consistent rater count per item
    mode_total = Counter(row_totals).most_common(1)[0][0]
    filtered = [row for row, total in zip(all_edit_rows, row_totals) if total == mode_total]
    
    return np.asarray(filtered, dtype=float) if filtered else np.zeros((0, 4), dtype=float)


def fleiss_kappa_overall_and_by_lang_correct(aggregated_df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """CORRECT Fleiss Kappa calculation - single matrix per language with all edits."""
    overall_mat = build_kappa_matrix_all_edits_correct(aggregated_df)
    overall = float(fleiss_kappa(overall_mat)) if overall_mat.size else float('nan')
    
    per_lang = {}
    for lang in ['EN', 'DE', 'UA']:
        lang_data = aggregated_df[aggregated_df['language'] == lang]
        mat = build_kappa_matrix_all_edits_correct(lang_data)
        per_lang[lang] = float(fleiss_kappa(mat)) if mat.size else float('nan')
    
    return overall, per_lang


def plot_llm_vs_human_on_axis(ax, aggregated_df: pd.DataFrame, title: str = '') -> None:
    """Render LLM vs Human bar chart on a provided axis using project style."""
    import numpy as np
    # Get label counts
    llm_counts = aggregated_df['llm_label'].value_counts().reindex(STANDARD_LABELS, fill_value=0)
    human_counts = aggregated_df['human_label'].value_counts().reindex(STANDARD_LABELS, fill_value=0)
    # Create bar plot
    x = np.arange(len(STANDARD_LABELS))
    width = 0.35
    bars1 = ax.bar(x - width/2, llm_counts.values, width, label='LLM', color='lightcoral', alpha=0.8)
    bars2 = ax.bar(x + width/2, human_counts.values, width, label='Human', color='steelblue', alpha=0.8)
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{int(h)}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{int(h)}', ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Labels')
    ax.set_ylabel('Count')
    if title:
        ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(STANDARD_LABELS)
    ax.grid(axis='y', alpha=0.3)


def plot_overall_llm_vs_human(aggregated_df: pd.DataFrame, save_path: str = '') -> None:
    """Plot overall LLM vs Human bar chart; save if save_path provided."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_llm_vs_human_on_axis(ax, aggregated_df, 'LLM vs Human Labels (Overall)')
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_by_language_llm_vs_human(aggregated_df: pd.DataFrame, save_path: str = '') -> None:
    """Plot per-language LLM vs Human bar charts; save if save_path provided."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    languages = ['EN', 'DE', 'UA']
    for idx, lang in enumerate(languages):
        lang_data = aggregated_df[aggregated_df['language'] == lang]
        plot_llm_vs_human_on_axis(axes[idx], lang_data, f'{lang} (n={len(lang_data)})')
        if idx == 0:
            axes[idx].legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def sample_english_aligned_examples(aggregated_df: pd.DataFrame, n_per_label: int = 3) -> Dict[str, List[str]]:
    """
    Sample English examples for each label in the exact report format.
    Format: aligned_sentence [LLM: X, Human: Y] [LLM_reasoning: ...] (for LLM TN/FN only)
    """
    english_data = aggregated_df[aggregated_df['language'] == 'EN']
    labels_to_sample = ['TP', 'FP1', 'FP2', 'FP3', 'TN', 'FN']
    
    examples = {}
    for label in labels_to_sample:
        label_examples = english_data[english_data['human_label'] == label]
        if len(label_examples) > 0:
            sample_examples = []
            for _, row in label_examples.head(n_per_label).iterrows():
                # Use the aligned_sentence from processed.csv
                if 'aligned_sentence' in row and pd.notna(row['aligned_sentence']):
                    aligned_text = str(row['aligned_sentence']).strip()
                    
                    # Check if LLM reasoning is already in aligned_sentence
                    if '[llm_reasoning:' in aligned_text:
                        # Extract the main text and reasoning separately
                        parts = aligned_text.split(' [llm_reasoning:')
                        main_text = parts[0]
                        reasoning_part = '[llm_reasoning:' + parts[1]
                        
                        # Format: main_text [LLM: X, Human: Y] reasoning_part
                        example = f"{main_text} [LLM: {row['llm_label']}, Human: {row['human_label']}] {reasoning_part}"
                    else:
                        # No LLM reasoning - just add the labels
                        example = f"{aligned_text} [LLM: {row['llm_label']}, Human: {row['human_label']}]"
                else:
                    # Fallback for raw data (shouldn't happen with processed.csv)
                    example = f"{row['src']} [LLM: {row['llm_label']}, Human: {row['human_label']}]"
                
                sample_examples.append(example)
            examples[label] = sample_examples
        else:
            examples[label] = []
    
    return examples
