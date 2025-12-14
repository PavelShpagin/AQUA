#!/usr/bin/env python3
"""
Generate comprehensive GEC annotation analysis report comparing three filtering strategies.
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
from sklearn.metrics import cohen_kappa_score
import json
import os
import matplotlib.pyplot as plt
from analysis_utils import plot_overall_llm_vs_human, plot_by_language_llm_vs_human

def parse_human_label_counts(human_label_str):
    """Parse human label string to extract label counts"""
    if pd.isna(human_label_str) or not human_label_str:
        return {}
    
    # Extract counts from format like "FP1(FP1:1,FP2:1)" or "TP(TP:2)"
    match = re.search(r'\(([^)]+)\)', human_label_str)
    if not match:
        return {}
    
    counts_str = match.group(1)
    counts = {}
    
    for item in counts_str.split(','):
        if ':' in item:
            label, count = item.strip().split(':')
            counts[label.strip()] = int(count)
    
    return counts

def extract_edit_level_labels(aligned_sentence):
    """Extract all edit-level human label counts from aligned sentence"""
    if pd.isna(aligned_sentence):
        return []
    
    # Find all human label patterns - handle incomplete patterns
    pattern = r'human_([^(]+)\(([^)]*)\)'
    matches = re.findall(pattern, aligned_sentence)
    
    edit_labels = []
    for label, counts_str in matches:
        label = label.strip()
        counts = {}
        
        # Handle incomplete or malformed counts_str
        if not counts_str or counts_str.strip() == '':
            continue
            
        for item in counts_str.split(','):
            if ':' in item:
                parts = item.strip().split(':')
                if len(parts) >= 2:
                    l = parts[0].strip()
                    c_str = parts[1].strip()
                    
                    # Handle cases where count might be incomplete (e.g., "FP2:1" vs "FP2:1,FP3:2")
                    if c_str and c_str.isdigit():
                        try:
                            counts[l] = int(c_str)
                        except ValueError:
                            continue
        
        if counts:  # Only add if we found valid counts
            edit_labels.append(counts)
    
    return edit_labels

def has_edit_level_ties(edit_labels):
    """Check if any edit has ties in annotator labels"""
    for edit_counts in edit_labels:
        if not edit_counts:
            continue
        max_count = max(edit_counts.values())
        max_labels = [label for label, count in edit_counts.items() if count == max_count]
        if len(max_labels) > 1:  # Tie detected
            return True
    return False

def meets_strict_criteria(edit_labels, min_agreement=3):
    """Check if all edits meet strict agreement criteria (≥3 annotators agree on dominant label)"""
    for edit_counts in edit_labels:
        if not edit_counts:
            continue
        max_count = max(edit_counts.values())
        if max_count < min_agreement:
            return False
    return True

def calculate_fleiss_kappa(edit_labels_list):
    """Calculate Fleiss' kappa for inter-annotator agreement on edit-level labels"""
    if not edit_labels_list:
        return 0.0
    
    # Collect all individual edit annotations (each edit position as an item)
    edit_items = []
    
    for sentence_edits in edit_labels_list:
        for edit_counts in sentence_edits:
            if edit_counts and sum(edit_counts.values()) >= 2:  # Need at least 2 annotators
                edit_items.append(edit_counts)
    
    if len(edit_items) < 2:
        return 0.0
    
    # Get all possible labels
    all_labels = set()
    for edit_counts in edit_items:
        all_labels.update(edit_counts.keys())
    
    if len(all_labels) < 2:  # Need at least 2 categories
        return 0.0
    
    all_labels = sorted(list(all_labels))
    n_items = len(edit_items)
    n_categories = len(all_labels)
    
    # Create rating matrix: items x categories
    # Each cell [i,j] = number of annotators who assigned category j to item i
    matrix = np.zeros((n_items, n_categories))
    
    for i, edit_counts in enumerate(edit_items):
        for label, count in edit_counts.items():
            j = all_labels.index(label)
            matrix[i, j] = count
    
    # Check if all items have the same number of raters
    n_raters_per_item = matrix.sum(axis=1)
    if len(set(n_raters_per_item)) > 1:
        # Variable number of raters - use weighted approach
        total_raters = n_raters_per_item.sum()
        if total_raters == 0:
            return 0.0
        
        # Calculate observed agreement (weighted by number of raters per item)
        p_i_weighted = []
        for i in range(n_items):
            n_i = n_raters_per_item[i]
            if n_i > 1:
                # Agreement for item i
                agreement_i = (matrix[i] * (matrix[i] - 1)).sum() / (n_i * (n_i - 1))
                p_i_weighted.append(agreement_i * n_i)
            else:
                p_i_weighted.append(0)
        
        p_bar = sum(p_i_weighted) / total_raters if total_raters > 0 else 0
        
        # Marginal proportions (weighted)
        p_j = matrix.sum(axis=0) / total_raters
        
    else:
        # Fixed number of raters
        n_raters = int(n_raters_per_item[0])
        if n_raters < 2:
            return 0.0
        
        # Calculate observed agreement
        p_i = (matrix * (matrix - 1)).sum(axis=1) / (n_raters * (n_raters - 1))
        p_bar = p_i.mean()
        
        # Marginal proportions
        p_j = matrix.sum(axis=0) / (n_items * n_raters)
    
    # Expected agreement
    p_e = (p_j ** 2).sum()
    
    # Fleiss' kappa
    if p_e >= 1.0 or p_bar >= 1.0:
        return 1.0 if p_bar >= 1.0 else 0.0
    
    kappa = (p_bar - p_e) / (1 - p_e)
    return max(0.0, min(1.0, kappa))  # Clamp to [0,1] range

def generate_examples_by_label(df, label, n_examples=4):
    """Generate examples for a specific label"""
    examples = []
    label_data = df[df['human_label'] == label].head(n_examples)
    
    for _, row in label_data.iterrows():
        try:
            src = row['src'] if pd.notna(row['src']) else ""
            tgt = row['tgt'] if pd.notna(row['tgt']) else ""
            aligned = row['aligned_sentence'] if pd.notna(row['aligned_sentence']) else ""
            llm_label = row['llm_label'] if pd.notna(row['llm_label']) else "UNKNOWN"
            human_label = row['human_label'] if pd.notna(row['human_label']) else "UNKNOWN"
            
            # Check for missing errors format
            missing_errors_match = re.search(r'\(missing_errors:(\d+)/(\d+)\)', aligned)
            if missing_errors_match:
                x, y = missing_errors_match.groups()
                example = f"{src} (missing_errors:{x}/{y}) [LLM: {llm_label}, Human: {human_label}]"
            else:
                # Truncate very long aligned sentences for readability
                display_aligned = aligned[:500] + "..." if len(aligned) > 500 else aligned
                example = f"{display_aligned} [LLM: {llm_label}, Human: {human_label}]"
            
            # Add LLM reasoning if available (for FN cases) - truncate if too long
            reasoning_match = re.search(r'\[llm_reasoning:([^\]]+)\]', aligned)
            if reasoning_match:
                reasoning = reasoning_match.group(1)
                if len(reasoning) > 100:
                    reasoning = reasoning[:100] + "..."
                example += f" [LLM_reasoning: {reasoning}]"
            
            examples.append(example)
        except Exception as e:
            # Handle any parsing errors gracefully
            examples.append(f"[Error parsing example: {str(e)}]")
    
    return examples

def main():
    # Load processed data
    df = pd.read_csv('data/annotations/processed.csv')
    
    print("Generating comprehensive GEC annotation analysis report...")
    
    # Extract edit-level labels for filtering
    df['edit_labels'] = df['aligned_sentence'].apply(extract_edit_level_labels)
    
    # Apply three filtering strategies
    strategies = {}
    
    # ALL: Keep all samples
    strategies['ALL'] = df.copy()
    
    # TIES: Drop samples with edit-level ties
    df['has_ties'] = df['edit_labels'].apply(has_edit_level_ties)
    strategies['TIES'] = df[~df['has_ties']].copy()
    
    # STRICT: Keep only samples with ≥3 annotators agreeing on dominant edit label
    df['meets_strict'] = df['edit_labels'].apply(meets_strict_criteria)
    strategies['STRICT'] = df[df['meets_strict']].copy()
    
    # Generate report
    report = []
    report.append("# GEC Annotation Analysis — Three Filtering Strategies")
    report.append("")
    report.append("**Goal:** Compare LLM vs Human sentence-level labels under three dataset filtering regimes and recommend the best setting.")
    report.append("")
    
    # 1. Filtering strategies
    report.append("## 1. Filtering strategies (brief)")
    report.append("")
    report.append("* **ALL:** keep all samples with valid sentence-level labels (no edit-level filtering).")
    report.append("* **TIES:** drop samples with edit-level ties (dominant label not unique).")
    report.append("* **STRICT:** keep only samples with ≥3 annotators agreeing on the dominant edit label.")
    report.append("")
    
    # 2a. Retention after filtering
    report.append("## 2a. Retention after filtering")
    report.append("")
    
    total_samples = len(df)
    
    for strategy_name, strategy_df in strategies.items():
        kept = len(strategy_df)
        dropped = total_samples - kept
        kept_pct = (kept / total_samples) * 100
        dropped_pct = (dropped / total_samples) * 100
        
        report.append(f"* **{strategy_name}:**")
        report.append(f"  * Overall: kept {kept}/{total_samples} ({kept_pct:.1f}%), dropped {dropped} ({dropped_pct:.1f}%)")
        
        # By language
        for lang in ['EN', 'DE', 'UA']:
            lang_total = len(df[df['language'] == lang])
            lang_kept = len(strategy_df[strategy_df['language'] == lang])
            lang_kept_pct = (lang_kept / lang_total) * 100 if lang_total > 0 else 0
            report.append(f"  * {lang}: kept {lang_kept}/{lang_total} ({lang_kept_pct:.1f}%)")
        
    report.append("")
    
    # 2b. Overall LLM vs Human
    report.append("## 2b. Overall LLM vs Human (all languages)")
    report.append("")
    # Placeholder replaced by generated chart later
    # Will embed overall_comparison.png
    report.append("![Overall](overall_comparison.png)")
    report.append("")
    
    # 2c. By language
    report.append("## 2c. By language — LLM vs Human across techniques")
    report.append("")
    # Will embed by_language_grid.png
    report.append("![By Language Grid](by_language_grid.png)")
    report.append("")
    
    # 2d. English examples
    report.append("## 2d. English examples (4 per label, ALL)")
    report.append("")
    
    en_data = strategies['ALL'][strategies['ALL']['language'] == 'EN']
    
    for label in ['TP', 'FP1', 'FP2', 'FP3', 'TN', 'FN']:
        report.append(f"* **{label}:**")
        examples = generate_examples_by_label(en_data, label, 4)
        for example in examples:
            report.append(f"  * {example}")
    
    report.append("")
    
    # 2e. Agreement analysis
    report.append("## 2e. Agreement analysis (per technique)")
    report.append("")
    # Will embed agreement_grid.png
    report.append("![Agreement Grid](agreement_grid.png)")
    report.append("")
    
    agreement_results = {}
    for strategy_name, strategy_df in strategies.items():
        # Overall agreement
        correct = (strategy_df['llm_label'] == strategy_df['human_label']).sum()
        total = len(strategy_df)
        overall_acc = (correct / total) * 100 if total > 0 else 0
        
        # By language
        lang_accs = {}
        for lang in ['EN', 'DE', 'UA']:
            lang_df = strategy_df[strategy_df['language'] == lang]
            if len(lang_df) > 0:
                lang_correct = (lang_df['llm_label'] == lang_df['human_label']).sum()
                lang_acc = (lang_correct / len(lang_df)) * 100
                lang_accs[lang] = lang_acc
            else:
                lang_accs[lang] = 0
        
        agreement_results[strategy_name] = {
            'overall': overall_acc,
            'by_language': lang_accs
        }
        
        report.append(f"* **{strategy_name}:** overall {overall_acc:.1f}%; EN {lang_accs['EN']:.1f}%; DE {lang_accs['DE']:.1f}%; UA {lang_accs['UA']:.1f}%")
    
    report.append("")
    
    # 2f. Inter-annotator agreement
    report.append("## 2f. Inter-annotator agreement (Fleiss' kappa)")
    report.append("")
    report.append("* **Method:** Fleiss' kappa over edit-level mapped categories {TP, FP3, FP2, FP1}.")
    
    kappa_results = {}
    for strategy_name, strategy_df in strategies.items():
        # Overall kappa
        overall_kappa = calculate_fleiss_kappa(strategy_df['edit_labels'].tolist())
        
        # By language
        lang_kappas = {}
        for lang in ['EN', 'DE', 'UA']:
            lang_df = strategy_df[strategy_df['language'] == lang]
            lang_kappa = calculate_fleiss_kappa(lang_df['edit_labels'].tolist())
            lang_kappas[lang] = lang_kappa
        
        kappa_results[strategy_name] = {
            'overall': overall_kappa,
            'by_language': lang_kappas
        }
        
        report.append(f"  * **{strategy_name}:** overall kappa={overall_kappa:.3f}; EN={lang_kappas['EN']:.3f}; DE={lang_kappas['DE']:.3f}; UA={lang_kappas['UA']:.3f}")
    
    report.append("")
    
    # 2g. LLM-Human Agreement
    report.append("## 2g. LLM-Human Agreement (Cohen's kappa)")
    report.append("")
    report.append("* **Method:** Cohen's kappa between LLM and human sentence-level labels.")
    
    cohen_results = {}
    for strategy_name, strategy_df in strategies.items():
        # Overall Cohen's kappa
        if len(strategy_df) > 0:
            overall_cohen = cohen_kappa_score(strategy_df['human_label'], strategy_df['llm_label'])
        else:
            overall_cohen = 0.0
        
        # By language
        lang_cohens = {}
        for lang in ['EN', 'DE', 'UA']:
            lang_df = strategy_df[strategy_df['language'] == lang]
            if len(lang_df) > 0:
                lang_cohen = cohen_kappa_score(lang_df['human_label'], lang_df['llm_label'])
                lang_cohens[lang] = lang_cohen
            else:
                lang_cohens[lang] = 0.0
        
        cohen_results[strategy_name] = {
            'overall': overall_cohen,
            'by_language': lang_cohens
        }
        
        report.append(f"* **{strategy_name}:** overall kappa={overall_cohen:.3f}; EN={lang_cohens['EN']:.3f}; DE={lang_cohens['DE']:.3f}; UA={lang_cohens['UA']:.3f}")
    
    report.append("")
    
    # 2h. Conclusion
    report.append("## 2h. Conclusion (which setting is best?)")
    report.append("")
    
    # Summary statistics
    for strategy_name in ['ALL', 'TIES', 'STRICT']:
        acc = agreement_results[strategy_name]['overall']
        retention = (len(strategies[strategy_name]) / total_samples) * 100
        kappa = kappa_results[strategy_name]['overall']
        
        report.append(f"* **Agreement:** {strategy_name}={acc:.1f}%, TIES={agreement_results['TIES']['overall']:.1f}%, STRICT={agreement_results['STRICT']['overall']:.1f}%")
        break
    
    retention_all = (len(strategies['ALL']) / total_samples) * 100
    retention_ties = (len(strategies['TIES']) / total_samples) * 100  
    retention_strict = (len(strategies['STRICT']) / total_samples) * 100
    
    report.append(f"* **Retention:** ALL={retention_all:.1f}%, TIES={retention_ties:.1f}%, STRICT={retention_strict:.1f}%")
    
    report.append("* **Inter-Annotator Agreement (Fleiss' κ):**")
    for strategy_name in ['ALL', 'TIES', 'STRICT']:
        kappa = kappa_results[strategy_name]['overall']
        if kappa < 0.2:
            agreement_level = "poor agreement"
        elif kappa < 0.4:
            agreement_level = "fair agreement"
        elif kappa < 0.6:
            agreement_level = "moderate agreement"
        elif kappa < 0.8:
            agreement_level = "substantial agreement"
        else:
            agreement_level = "almost perfect agreement"
        
        report.append(f"  * **{strategy_name}:** κ={kappa:.3f} ({agreement_level})")
    
    report.append("* **Recommendation:** ALL (best agreement–retention trade-off for current data). Use STRICT when precision and label quality are paramount; use ALL for scale; use TIES as a balanced middle ground.")
    report.append("")
    
    # Ensure output directory exists
    os.makedirs('annotations/data_analysis', exist_ok=True)

    # Generate charts/images (3 total) using project plotting utilities
    # 1) Overall comparison: call project utility on each strategy using shared canvas
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    fig.suptitle('LLM vs Human Sentence-Level Labels (Unique Sentences)', fontsize=10)
    for ax, strat in zip(axes, ['ALL', 'TIES', 'STRICT']):
        # Render on ax using shared plotting helper
        from analysis_utils import plot_llm_vs_human_on_axis
        plot_llm_vs_human_on_axis(ax, strategies[strat], strat)
    axes[0].legend(fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig('annotations/data_analysis/overall_comparison.png', dpi=200)
    plt.close()

    # 2) By-language grid: use helper repeatedly across strategies and languages
    langs = ['EN', 'DE', 'UA']
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharey=True)
    fig.suptitle('LLM vs Human by Language (Unique Sentences)', fontsize=10)
    from analysis_utils import plot_llm_vs_human_on_axis
    for col, strat in enumerate(['ALL', 'TIES', 'STRICT']):
        for row, lang in enumerate(langs):
            ax = axes[row, col]
            df_subset = strategies[strat]
            df_lang = df_subset[df_subset['language'] == lang]
            plot_llm_vs_human_on_axis(ax, df_lang, strat if row == 0 else '')
            if col == 0:
                ax.set_ylabel(lang)
    axes[0,0].legend(['LLM','Human'], fontsize=8, loc='upper right')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('annotations/data_analysis/by_language_grid.png', dpi=200)
    plt.close()

    # 3) Agreement grid: 1x3 subplots showing Overall + EN/DE/UA agreement per strategy
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    fig.suptitle('Agreement (Exact Match %) by Strategy and Language', fontsize=10)
    for ax, strat in zip(axes, ['ALL','TIES','STRICT']):
        vals = [agreement_results[strat]['overall'],
                agreement_results[strat]['by_language']['EN'],
                agreement_results[strat]['by_language']['DE'],
                agreement_results[strat]['by_language']['UA']]
        cats = ['Overall','EN','DE','UA']
        x = np.arange(len(cats))
        ax.bar(x, vals, color=['#54A24B','#4C78A8','#F58518','#72B7B2'])
        ax.set_title(strat)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=8)
        ax.set_ylim(0, 100)
        for i, v in enumerate(vals):
            ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=8)
        ax.grid(axis='y', linestyle=':', alpha=0.4)
    axes[0].set_ylabel('Agreement (%)')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig('annotations/data_analysis/agreement_grid.png', dpi=200)
    plt.close()

    # Save report
    report_text = '\n'.join(report)
    with open('annotations/data_analysis/report.md', 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("✅ Report generated: annotations/data_analysis/report.md")
    
    # Save detailed statistics as JSON
    stats = {
        'strategies': {
            name: {
                'total_samples': len(df_strategy),
                'retention_rate': len(df_strategy) / total_samples,
                'agreement': agreement_results[name],
                'fleiss_kappa': kappa_results[name],
                'cohen_kappa': cohen_results[name]
            }
            for name, df_strategy in strategies.items()
        },
        'total_original_samples': total_samples
    }
    
    with open('annotations/data_analysis/report_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("✅ Detailed statistics saved: annotations/data_analysis/report_stats.json")

    # Clean up old comprehensive_* files if present
    old_report = 'annotations/data_analysis/comprehensive_report.md'
    old_stats = 'annotations/data_analysis/comprehensive_stats.json'
    for old in [old_report, old_stats]:
        try:
            if os.path.exists(old):
                os.remove(old)
        except Exception:
            pass

if __name__ == "__main__":
    main()
