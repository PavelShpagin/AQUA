#!/usr/bin/env python3
"""
Distribution analysis utilities for GEC judges.

This module provides utilities for analyzing and displaying distribution
of TP/FP/TN/FN results across different judges and evaluation scenarios.

Usage:
    from utils.distribution import analyze_distribution, print_distribution_summary
    
    results = [{'pred_specialized': 'TP', 'gold_specialized': 'TP'}, ...]
    distribution = analyze_distribution(results)
    print_distribution_summary(distribution, "Judge Results")
"""

import pandas as pd
from collections import Counter
from typing import Dict, List, Any, Optional

def analyze_distribution(results: List[Dict[str, Any]], 
                        pred_col: str = 'pred_specialized', 
                        gold_col: str = 'gold_specialized') -> Dict[str, Any]:
    """
    Analyze distribution of TP/FP/TN/FN predictions vs gold labels.
    
    Args:
        results: List of result dictionaries containing predictions and gold labels
        pred_col: Column name for predictions
        gold_col: Column name for gold labels
    
    Returns:
        Dictionary containing distribution analysis
    """
    if not results:
        return {
            'total_count': 0,
            'predictions': {},
            'gold_labels': {},
            'confusion_matrix': {},
            'accuracy_metrics': {}
        }
    
    # Extract predictions and gold labels
    predictions = [r.get(pred_col, 'Unknown') for r in results]
    gold_labels = [r.get(gold_col, 'Unknown') for r in results]
    
    # Count distributions
    pred_counts = Counter(predictions)
    gold_counts = Counter(gold_labels)
    
    # Build confusion matrix
    confusion_matrix = {}
    for pred, gold in zip(predictions, gold_labels):
        key = f"{gold}->{pred}"
        confusion_matrix[key] = confusion_matrix.get(key, 0) + 1
    
    # Calculate accuracy metrics
    total_count = len(results)
    correct_predictions = sum(1 for p, g in zip(predictions, gold_labels) if p == g)
    accuracy = correct_predictions / total_count if total_count > 0 else 0.0
    
    # Calculate per-class metrics
    all_labels = set(predictions + gold_labels)
    per_class_metrics = {}
    
    for label in all_labels:
        if label == 'Unknown':
            continue
            
        tp = confusion_matrix.get(f"{label}->{label}", 0)
        fp = sum(confusion_matrix.get(f"{other}->{label}", 0) 
                for other in all_labels if other != label)
        fn = sum(confusion_matrix.get(f"{label}->{other}", 0) 
                for other in all_labels if other != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': gold_counts.get(label, 0)
        }
    
    return {
        'total_count': total_count,
        'predictions': dict(pred_counts),
        'gold_labels': dict(gold_counts),
        'confusion_matrix': confusion_matrix,
        'accuracy_metrics': {
            'overall_accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'per_class': per_class_metrics
        }
    }

def print_distribution_summary(distribution: Dict[str, Any], 
                             title: str = "Distribution Analysis",
                             show_confusion: bool = True) -> str:
    """
    Print formatted distribution summary.
    
    Args:
        distribution: Distribution analysis from analyze_distribution()
        title: Title for the summary
        show_confusion: Whether to show confusion matrix details
    
    Returns:
        Formatted string summary
    """
    if distribution['total_count'] == 0:
        return f"\n{title}\n{'='*50}\nNo data available for analysis\n"
    
    lines = []
    lines.append(f"\n{title}")
    lines.append("=" * 50)
    lines.append(f"Total samples: {distribution['total_count']}")
    
    # Prediction distribution
    lines.append(f"\nPrediction Distribution:")
    for label, count in sorted(distribution['predictions'].items()):
        percentage = (count / distribution['total_count']) * 100
        lines.append(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Gold label distribution
    lines.append(f"\nGold Label Distribution:")
    for label, count in sorted(distribution['gold_labels'].items()):
        percentage = (count / distribution['total_count']) * 100
        lines.append(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Accuracy metrics
    accuracy_metrics = distribution['accuracy_metrics']
    lines.append(f"\nOverall Accuracy: {accuracy_metrics['overall_accuracy']:.3f}")
    lines.append(f"Correct Predictions: {accuracy_metrics['correct_predictions']}/{distribution['total_count']}")
    
    # Per-class metrics
    if accuracy_metrics['per_class']:
        lines.append(f"\nPer-Class Metrics:")
        for label, metrics in sorted(accuracy_metrics['per_class'].items()):
            if label != 'Unknown':
                acc = metrics.get('accuracy', metrics['f1_score'])  # Use F1 as fallback
                lines.append(f"  {label}: F1={metrics['f1_score']:.3f}, Acc={acc:.3f}, Support={metrics['support']}")
    
    # Confusion matrix (if requested)
    if show_confusion and distribution['confusion_matrix']:
        lines.append(f"\nConfusion Matrix (Gold->Pred):")
        for transition, count in sorted(distribution['confusion_matrix'].items()):
            percentage = (count / distribution['total_count']) * 100
            lines.append(f"  {transition}: {count} ({percentage:.1f}%)")
    
    return "\n".join(lines) + "\n"

def add_distribution_to_result_dict(result: Dict[str, Any], 
                                   results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Add distribution analysis to a result dictionary.
    Used by judges to include distribution info in their output.
    
    Args:
        result: Individual result dictionary to update
        results_list: Complete list of results for analysis
    
    Returns:
        Updated result dictionary with distribution info
    """
    if len(results_list) > 10:  # Only analyze if we have meaningful data
        distribution = analyze_distribution(results_list)
        result['distribution_summary'] = print_distribution_summary(distribution)
        result['distribution_metrics'] = distribution['accuracy_metrics']
    
    return result

def write_distribution_report(results: List[Dict[str, Any]], 
                            output_file: str,
                            title: str = "Distribution Analysis Report") -> None:
    """
    Write distribution analysis to a file.
    
    Args:
        results: List of result dictionaries
        output_file: Path to output file
        title: Report title
    """
    distribution = analyze_distribution(results)
    summary = print_distribution_summary(distribution, title, show_confusion=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)
        
        # Add detailed breakdown if data available
        if distribution['total_count'] > 0:
            f.write("\nDetailed Confusion Matrix Analysis:\n")
            f.write("-" * 40 + "\n")
            
            # Sort by frequency
            sorted_transitions = sorted(
                distribution['confusion_matrix'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for transition, count in sorted_transitions:
                percentage = (count / distribution['total_count']) * 100
                f.write(f"{transition:15} | {count:5} samples ({percentage:5.1f}%)\n")
