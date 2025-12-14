#!/usr/bin/env python3
"""
Correct annotation processing to match the report format exactly
"""

import csv
import sys
from collections import defaultdict, Counter
import re

def get_language_from_country(country):
    """Map country codes to language"""
    country_to_lang = {
        'UA': 'UA',  # Ukrainian
        'DE': 'DE',  # German  
        'US': 'EN',  # English (US)
        'MX': 'EN'   # English (Mexico)
    }
    return country_to_lang.get(country, 'UNKNOWN')

def parse_llm_edits(llm_edits_text):
    """Parse LLM edits to extract edit spans with labels"""
    if not llm_edits_text:
        return []
    
    # First try contextual pattern: word{old=>new:::label} (handle 3+ colons)
    contextual_pattern = r'(\w+)\{([^}]*?)=>([^}]*?):::+([^}]*?)\}'
    contextual_matches = re.findall(contextual_pattern, llm_edits_text)
    
    # Then try standalone pattern: {old=>new:::label} (handle 3+ colons)
    standalone_pattern = r'\{([^}]*?)=>([^}]*?):::+([^}]*?)\}'
    standalone_matches = re.findall(standalone_pattern, llm_edits_text)
    
    edits = []
    
    # Process contextual matches first (these handle cases like "Chinese{=> speakers who:::TP}")
    for context_word, old, new, label in contextual_matches:
        old_text = old  # Don't strip! Preserve original spacing
        new_text = new  # Don't strip! Preserve original spacing
        
        if not old_text:
            # Pure insertion after context word: "Chinese{=> speakers who}" 
            # This should create span: {=> speakers who:::...} (not include "Chinese")
            edits.append({
                'old': '',
                'new': new_text,
                'llm_label': label.strip(),
                'type': 'insertion',
                'context_word': context_word
            })
        else:
            # Regular replacement within context
            edits.append({
                'old': old_text,
                'new': new_text,
                'llm_label': label.strip(),
                'type': 'replacement'
            })
    
    # Process standalone matches (these are regular {old=>new:::label} patterns)
    # But skip any that are part of contextual matches
    processed_patterns = set()
    for context_word, old, new, label in contextual_matches:
        # Mark the exact pattern that was processed contextually
        processed_patterns.add(f"{{{old}=>{new}:::{label}}}")
    
    for match in re.finditer(standalone_pattern, llm_edits_text):
        old, new, label = match.groups()
        pattern = f"{{{old}=>{new}:::{label}}}"
        
        # Skip if this exact pattern was already processed as contextual
        if pattern in processed_patterns:
            continue
            
        old_text = old.strip()
        new_text = new.strip()
        
        edits.append({
            'old': old_text,
            'new': new_text,
            'llm_label': label.strip(),
            'type': 'replacement'
        })
    
    return edits

def parse_human_annotations(sentence_data):
    """Parse human annotations from multiple annotators"""
    annotations = []
    
    for row in sentence_data:
        # Get edit-level labels from id1_label columns
        for i in range(1, 16):  # Check id1_label1 through id1_label15
            label_col = f'id1_label{i}'
            annotation_col = f'annotation_{i}'
            
            if label_col in row and row[label_col] and row[label_col].strip():
                label = row[label_col].strip()
                annotation = row.get(annotation_col, '').strip()
                
                # Map labels to standard format
                label_map = {
                    'good_edit': 'TP',
                    'optional_edit': 'FP3', 
                    'incorrect_grammar_edit': 'FP2',
                    'hallucination_or_meaning_change': 'FP1'
                }
                
                mapped_label = label_map.get(label, label)
                if mapped_label in ['TP', 'FP1', 'FP2', 'FP3']:
                    annotations.append({
                        'annotation': annotation,
                        'label': mapped_label,
                        'worker_id': row.get('_worker_id', '')
                    })
    
    return annotations

def compute_sentence_level_human_label_simple(sentence_data):
    """Compute sentence-level human label using CORRECT algorithm:
    1. Per-edit majority vote with tie-breaking (FP1 > FP2 > FP3 > TP)
    2. Worst-case policy across all edit majority labels
    """
    
    # Step 1: Determine missing_error using majority choice (ties prefer True)
    missing_error_judgments = []
    for row in sentence_data:
        missed_error = row.get('missed_error', '').strip()
        if missed_error == 'missed_error':
            missing_error_judgments.append(True)
        else:
            missing_error_judgments.append(False)
    
    # Majority vote for missing_error (ties prefer True)
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
    
    # Check src == tgt case
    src = sentence_data[0].get('src', '').strip()
    tgt = sentence_data[0].get('tgt', '').strip()
    
    if src == tgt:
        return 'FN' if missing_error else 'TN'
    
    # Step 2: For each edit position, compute majority choice with tie-breaking
    edit_majority_labels = []
    
    # Group labels by edit position (id1_label1, id1_label2, etc.)
    for edit_pos in range(1, 16):  # Check edit positions 1-15
        label_col = f'id1_label{edit_pos}'
        edit_labels = []
        
        # Collect all annotator labels for this edit position
        for row in sentence_data:
            if label_col in row and row[label_col] and row[label_col].strip():
                label = row[label_col].strip()
                
                # Map to standard labels
                label_map = {
                    'good_edit': 'TP',
                    'optional_edit': 'FP3',
                    'incorrect_grammar_edit': 'FP2',
                    'hallucination_or_meaning_change': 'FP1'
                }
                
                mapped_label = label_map.get(label, label)
                if mapped_label in ['TP', 'FP1', 'FP2', 'FP3']:
                    edit_labels.append(mapped_label)
        
        # If this edit position has labels, compute majority with tie-breaking
        if edit_labels:
            label_counts = Counter(edit_labels)
            max_count = max(label_counts.values())
            most_common = [lab for lab, count in label_counts.items() if count == max_count]
            
            if len(most_common) == 1:
                edit_majority = most_common[0]
            else:
                # Tie-breaking: FP1 > FP2 > FP3 > TP priority
                tie_priority = {'FP1': 4, 'FP2': 3, 'FP3': 2, 'TP': 1}
                edit_majority = max(most_common, key=lambda x: tie_priority.get(x, 0))
            
            edit_majority_labels.append(edit_majority)
    
    # Step 3: Apply worst-case policy across all edit majority labels
    if not edit_majority_labels:
        return 'TN'
    
    # Find worst label (FP1 > FP2 > FP3 > TP)
    worst_priority = {'FP1': 4, 'FP2': 3, 'FP3': 2, 'TP': 1}
    worst_label = max(edit_majority_labels, key=lambda x: worst_priority.get(x, 0))
    
    # Step 4: Apply final missing_error logic
    if missing_error and worst_label == 'TP':
        return 'FN'
    else:
        return worst_label

def create_aligned_sentence_format(sentence_data, llm_label, human_label):
    """Create the aligned sentence in the report format"""
    
    first_row = sentence_data[0]
    src = first_row.get('src', '').strip()
    tgt = first_row.get('tgt', '').strip()
    llm_edits_text = first_row.get('llm_edits', '')
    llm_reasoning = first_row.get('llm_reasoning', '').strip()
    
    # Start with source text
    aligned_sentence = src
    
    # Check if there are LLM edits
    if llm_edits_text and '{' in llm_edits_text and '=>' in llm_edits_text:
        # Parse LLM edits from the llm_edits field
        llm_edits = parse_llm_edits(llm_edits_text)
        
        if llm_edits:
            # Collect human annotations for each edit position
            edit_human_labels = defaultdict(list)
            
            for row in sentence_data:
                for i in range(1, 16):
                    annotation_col = f'annotation_{i}'
                    label_col = f'id1_label{i}'
                    
                    if (annotation_col in row and row[annotation_col] and 
                        label_col in row and row[label_col]):
                        
                        annotation = row[annotation_col].strip()
                        label = row[label_col].strip()
                        
                        # Map to standard labels
                        label_map = {
                            'good_edit': 'TP',
                            'optional_edit': 'FP3',
                            'incorrect_grammar_edit': 'FP2',
                            'hallucination_or_meaning_change': 'FP1'
                        }
                        
                        mapped_label = label_map.get(label, label)
                        if mapped_label in ['TP', 'FP1', 'FP2', 'FP3']:
                            edit_human_labels[annotation].append(mapped_label)
            
            # Replace edits in the text with formatted spans
            # Sort edits by position to avoid conflicts
            sorted_edits = sorted(llm_edits, key=lambda x: len(x['old']), reverse=True)
            
            for edit in sorted_edits:
                old_text = edit['old']
                new_text = edit['new'] 
                llm_edit_label = edit['llm_label']
                edit_type = edit.get('type', 'replacement')
                context_word = edit.get('context_word', '')
                
                # Find matching human labels for this edit
                human_counts = Counter()
                
                for annotation, labels in edit_human_labels.items():
                    # Match the edit pattern - handle different spacing formats
                    if edit_type == 'insertion':
                        # For insertions, look for {=> new_text} pattern
                        patterns_to_check = [
                            f"{{=> {new_text}}}",
                            f"{{ => {new_text}}}",
                            f"{{=>{new_text}}}",
                            f"{{ =>{new_text}}}"
                        ]
                    else:
                        # For replacements, look for {old_text=>new_text} pattern
                        patterns_to_check = [
                            f"{{ {old_text}=> {new_text}}}",
                            f"{{{old_text}=> {new_text}}}",
                            f"{{ {old_text}=>{new_text}}}",
                            f"{{{old_text}=>{new_text}}}"
                        ]
                    
                    if any(pattern in annotation for pattern in patterns_to_check):
                        for label in labels:
                            human_counts[label] += 1
                
                # Compute edit-level human label with tie-breaking
                edit_human_label = human_label  # Default to sentence-level
                if human_counts:
                    max_count = max(human_counts.values())
                    most_common = [lab for lab, count in human_counts.items() if count == max_count]
                    
                    if len(most_common) == 1:
                        edit_human_label = most_common[0]
                    else:
                        # Tie-breaking: FP1 > FP2 > FP3 > TP priority (FP1 is worst/highest priority)
                        tie_priority = {'FP1': 4, 'FP2': 3, 'FP3': 2, 'TP': 1}
                        edit_human_label = max(most_common, key=lambda x: tie_priority.get(x, 0))
                
                # Format human label counts
                human_counts_str = ""
                if human_counts:
                    count_parts = []
                    for label in ['TP', 'FP1', 'FP2', 'FP3']:
                        if human_counts[label] > 0:
                            count_parts.append(f"{label}:{human_counts[label]}")
                    if count_parts:
                        human_counts_str = f"({','.join(count_parts)})"
                
                # Create the formatted span
                span = f"{{{old_text}=>{new_text}:::llm_{llm_edit_label}:::human_{edit_human_label}{human_counts_str}}}"
                
                # Handle replacement based on edit type
                if edit_type == 'insertion':
                    # For insertions, we need to insert after the context word
                    if context_word and context_word in aligned_sentence:
                        # Insert the span after the context word
                        aligned_sentence = aligned_sentence.replace(context_word, context_word + span, 1)
                elif old_text and old_text in aligned_sentence:
                    # For contextual edits, replace only in the specific context
                    if context_word and edit_type != 'insertion':
                        # Look for the specific pattern: context_word + old_text (old_text already includes spacing)
                        contextual_pattern = context_word + old_text
                        if contextual_pattern in aligned_sentence:
                            # Replace the contextual pattern with context_word + span
                            aligned_sentence = aligned_sentence.replace(contextual_pattern, context_word + span, 1)
                    else:
                        # Regular replacement - but avoid replacing inside existing spans
                        # Split by existing spans to avoid nested replacements
                        parts = []
                        current_pos = 0
                        
                        # Find all existing spans to avoid replacing inside them
                        import re
                        span_pattern = r'\{[^}]*?=>[^}]*?:::llm_[^}]*?:::human_[^}]*?\}'
                        
                        for match in re.finditer(span_pattern, aligned_sentence):
                            # Add text before the span
                            before_span = aligned_sentence[current_pos:match.start()]
                            if old_text in before_span:
                                before_span = before_span.replace(old_text, span, 1)
                            parts.append(before_span)
                            
                            # Add the span as-is (don't modify)
                            parts.append(aligned_sentence[match.start():match.end()])
                            current_pos = match.end()
                        
                        # Add remaining text after last span
                        remaining = aligned_sentence[current_pos:]
                        if old_text in remaining:
                            remaining = remaining.replace(old_text, span, 1)
                        parts.append(remaining)
                        
                        aligned_sentence = ''.join(parts)
    
    # Add missing errors info for TN/FN cases
    if human_label in ['TN', 'FN'] or llm_label in ['TN', 'FN']:
        missing_error_judgments = []
        for row in sentence_data:
            missed_error = row.get('missed_error', '').strip()
            missing_error_judgments.append(missed_error == 'missed_error')
        
        true_count = sum(missing_error_judgments)
        total_count = len(missing_error_judgments)
        aligned_sentence += f" (missing_errors:{true_count}/{total_count})"
    
    # Add LLM reasoning for LLM TN/FN cases
    if llm_label in ['TN', 'FN'] and llm_reasoning:
        reasoning_text = llm_reasoning[:100] + "..." if len(llm_reasoning) > 100 else llm_reasoning
        aligned_sentence += f" [llm_reasoning:{reasoning_text}]"
    
    return aligned_sentence

def main():
    print("Processing annotations with correct aligned spans format...")
    
    # Load data from CSV files
    csv_files = [
        ('../data/annotations/f3119729.csv', 'UA'),
        ('../data/annotations/f3120001.csv', 'DE'),
        ('../data/annotations/f3120198.csv', 'EN')
    ]
    
    all_data = []
    for file_path, language in csv_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row['language'] = language
                    all_data.append(row)
        except FileNotFoundError:
            print(f"Warning: Could not find {file_path}")
    
    print(f"Loaded {len(all_data)} raw annotation rows")
    
    # Group by _unit_id
    unit_groups = defaultdict(list)
    for row in all_data:
        unit_id = row.get('_unit_id', '')
        if unit_id:
            unit_groups[unit_id].append(row)
    
    print(f"Grouped into {len(unit_groups)} unique sentences by _unit_id")
    
    # Process each sentence
    results = []
    for unit_id, sentence_data in unit_groups.items():
        if not sentence_data:
            continue
            
        first_row = sentence_data[0]
        src = first_row.get('src', '')
        tgt = first_row.get('tgt', '')
        language = first_row.get('language', 'UNKNOWN')
        
        # Get LLM label
        llm_label = first_row.get('llm_label', '').strip()
        if llm_label not in ['TP', 'FP1', 'FP2', 'FP3', 'TN', 'FN']:
            llm_label = 'UNKNOWN'
        
        # Compute human label
        human_label = compute_sentence_level_human_label_simple(sentence_data)
        
        # Create aligned sentence with proper format
        aligned_sentence = create_aligned_sentence_format(sentence_data, llm_label, human_label)
        
        results.append({
            'idx': len(results) + 1,
            'src': src,
            'tgt': tgt,
            'aligned_sentence': aligned_sentence,
            'llm_label': llm_label,
            'human_label': human_label,
            'language': language
        })
        
        if len(results) % 100 == 0:
            print(f"Processed {len(results)} sentences...")
    
    # Save to CSV
    output_path = '../data/annotations/processed.csv'
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['idx', 'src', 'tgt', 'aligned_sentence', 'llm_label', 'human_label', 'language']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\\nSaved {len(results)} processed sentences to: {output_path}")
    
    # Calculate accuracy
    total_sentences = len(results)
    correct_predictions = sum(1 for r in results if r['llm_label'] == r['human_label'])
    overall_accuracy = correct_predictions / total_sentences * 100 if total_sentences > 0 else 0
    
    print(f"\\n=== ACCURACY REPORT ===")
    print(f"Overall: {correct_predictions}/{total_sentences} ({overall_accuracy:.1f}%)")
    
    # Per-language accuracy
    language_counts = defaultdict(lambda: {'total': 0, 'correct': 0})
    for result in results:
        lang = result['language']
        language_counts[lang]['total'] += 1
        if result['llm_label'] == result['human_label']:
            language_counts[lang]['correct'] += 1
    
    print("Per-language accuracy:")
    for language in sorted(language_counts.keys()):
        lang_data = language_counts[language]
        lang_accuracy = lang_data['correct'] / lang_data['total'] * 100 if lang_data['total'] > 0 else 0
        print(f"  {language}: {lang_data['correct']}/{lang_data['total']} ({lang_accuracy:.1f}%)")
    
    print("\\n✅ Processing complete with correct aligned spans format!")

if __name__ == "__main__":
    main()
