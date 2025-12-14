#!/usr/bin/env python3
"""
Clean pricing reporting utility with separation of concerns.

This module handles all pricing calculations and reporting after processing,
using the pricing table to compute accurate costs from token usage.
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from utils.pricing import calculate_cost


@dataclass
class PricingReport:
    """Clean pricing report data structure."""
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    total_reasoning_tokens: int
    total_cached_tokens: int
    total_samples: int
    cost_per_sample: float
    cost_for_10k: float
    cached_savings_usd: float = 0.0
    batch_savings_usd: float = 0.0
    total_savings_usd: float = 0.0
    cost_without_optimizations: float = 0.0


class PricingReporter:
    """Handles all pricing calculations and reporting."""
    
    def __init__(self):
        self.reports = []
    
    def calculate_pricing_from_results(self, results: List[Dict[str, Any]], backend: str, use_batch_pricing: bool = False) -> PricingReport:
        """
        Calculate pricing from processed results using the pricing table.
        
        Args:
            results: List of processed results with judge_outputs
            backend: Backend model name for pricing lookup
            
        Returns:
            PricingReport with all pricing details
        """
        total_input = 0
        total_output = 0
        total_reasoning = 0
        total_cached = 0
        
        # Extract token usage primarily from judge outputs; fallback to top-level tokens
        for result in results:
            judge_outputs = result.get('judge_outputs', [])
            if judge_outputs:
                for judge_output in judge_outputs:
                    total_input += int(judge_output.get('input_tokens', 0))
                    total_output += int(judge_output.get('output_tokens', 0))
                    total_reasoning += int(judge_output.get('reasoning_tokens', 0))
                    total_cached += int(judge_output.get('cached_tokens', 0))
            else:
                # Fallback path (e.g., when results were created via utils.ensemble.create_result_dict)
                total_input += int(result.get('input_tokens', 0))
                total_output += int(result.get('output_tokens', 0))
                total_reasoning += int(result.get('reasoning_tokens', 0))
                total_cached += int(result.get('cached_tokens', 0))
        
        # Calculate costs using pricing table
        cost_breakdown = calculate_cost(
            backend,
            input_tokens=total_input,
            output_tokens=total_output,
            reasoning_tokens=total_reasoning,
            cached_tokens=total_cached
        )
        
        # Calculate cached token savings (what we would have paid without caching)
        cost_without_caching = calculate_cost(
            backend,
            input_tokens=total_input,  # All input tokens at regular price
            output_tokens=total_output,
            reasoning_tokens=total_reasoning,
            cached_tokens=0  # No caching
        )
        cached_savings = cost_without_caching.total_cost - cost_breakdown.total_cost
        
        # Apply batch pricing discount (50% off) if using real batch API
        final_cost = cost_breakdown.total_cost
        batch_savings = 0.0
        if use_batch_pricing:
            batch_savings = final_cost * 0.5  # 50% savings
            final_cost = final_cost * 0.5
        
        # Calculate total savings (cached + batch)
        total_savings = cached_savings + batch_savings
        
        # Calculate derived metrics
        total_samples = len(results)
        cost_per_sample = final_cost / total_samples if total_samples > 0 else 0.0
        cost_for_10k = cost_per_sample * 10000
        
        return PricingReport(
            total_cost_usd=final_cost,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_reasoning_tokens=total_reasoning,
            total_cached_tokens=total_cached,
            total_samples=total_samples,
            cost_per_sample=cost_per_sample,
            cost_for_10k=cost_for_10k,
            cached_savings_usd=cached_savings,
            batch_savings_usd=batch_savings,
            total_savings_usd=total_savings,
            cost_without_optimizations=cost_without_caching.total_cost
        )
    
    def print_pricing_report(self, report: PricingReport) -> None:
        """Print comprehensive pricing report with savings breakdown."""
        print(f"\nToken Usage:")
        print(f"  Total tokens: {report.total_input_tokens + report.total_output_tokens + report.total_reasoning_tokens:,}")
        print(f"  Average tokens per sample: {(report.total_input_tokens + report.total_output_tokens + report.total_reasoning_tokens) / report.total_samples:.1f}")
        print(f"  Input tokens: {report.total_input_tokens:,} | Output tokens: {report.total_output_tokens:,}")
        print(f"  Cached tokens: {report.total_cached_tokens:,} | Reasoning tokens: {report.total_reasoning_tokens:,}")
        
        # Calculate cache hit rate
        if report.total_input_tokens > 0:
            cache_hit_rate = (report.total_cached_tokens / report.total_input_tokens) * 100
            print(f"  Cache hit rate: {cache_hit_rate:.1f}%")
        
        print(f"\nCost Analysis:")
        print(f"  Final cost (with optimizations): ${report.total_cost_usd:.4f}")
        print(f"  Average cost per sample: ${report.cost_per_sample:.6f}")
        
        # Show savings breakdown
        if report.total_savings_usd > 0:
            print(f"\nSavings Breakdown:")
            if report.cached_savings_usd > 0:
                print(f"  Cached token savings: ${report.cached_savings_usd:.4f}")
            if report.batch_savings_usd > 0:
                print(f"  Batch API savings (50% off): ${report.batch_savings_usd:.4f}")
            print(f"  Total savings: ${report.total_savings_usd:.4f}")
            
            if report.cost_without_optimizations > 0:
                print(f"  Cost without optimizations: ${report.cost_without_optimizations:.4f}")
                savings_percentage = (report.total_savings_usd / report.cost_without_optimizations) * 100
                print(f"  Savings percentage: {savings_percentage:.1f}%")
        
        print(f"\nProjections:")
        print(f"  Estimated cost for 10K samples: ${report.cost_for_10k:.2f}")
        if report.total_savings_usd > 0:
            cost_10k_without_opt = (report.cost_without_optimizations / report.total_samples) * 10000
            print(f"  Cost for 10K without optimizations: ${cost_10k_without_opt:.2f}")
            print(f"  Projected savings for 10K: ${cost_10k_without_opt - report.cost_for_10k:.2f}")
    
    def get_pricing_summary_dict(self, report: PricingReport) -> Dict[str, Any]:
        """Get pricing data as dictionary for logging/saving."""
        summary = {
            'total_cost_usd': report.total_cost_usd,
            'cost_per_sample': report.cost_per_sample,
            'total_input_tokens': report.total_input_tokens,
            'total_output_tokens': report.total_output_tokens,
            'total_reasoning_tokens': report.total_reasoning_tokens,
            'total_cached_tokens': report.total_cached_tokens,
            'total_samples': report.total_samples,
            'cost_for_10k': report.cost_for_10k,
            'cached_savings_usd': report.cached_savings_usd,
            'batch_savings_usd': report.batch_savings_usd,
            'total_savings_usd': report.total_savings_usd,
            'cost_without_optimizations': report.cost_without_optimizations
        }
        
        # Add cache hit rate
        if report.total_input_tokens > 0:
            summary['cache_hit_rate'] = (report.total_cached_tokens / report.total_input_tokens) * 100
        else:
            summary['cache_hit_rate'] = 0.0
            
        # Add savings percentage
        if report.cost_without_optimizations > 0:
            summary['savings_percentage'] = (report.total_savings_usd / report.cost_without_optimizations) * 100
        else:
            summary['savings_percentage'] = 0.0
            
        return summary


def calculate_and_report_pricing(results: List[Dict[str, Any]], backend: str, use_batch_pricing: bool = False) -> Tuple[PricingReport, Dict[str, Any]]:
    """
    Convenience function to calculate and report pricing.
    
    Returns:
        Tuple of (PricingReport, summary_dict)
    """
    reporter = PricingReporter()
    report = reporter.calculate_pricing_from_results(results, backend, use_batch_pricing)
    reporter.print_pricing_report(report)
    summary = reporter.get_pricing_summary_dict(report)
    
    return report, summary
