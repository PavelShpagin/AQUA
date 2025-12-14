#!/usr/bin/env python3
"""
Pricing calculations for LLM API usage.

This module provides accurate pricing data for various LLM models and utilities
for calculating and tracking costs across API calls.

Pricing sources:
- OpenAI: https://openai.com/pricing (January 2025)
- Google: https://ai.google.dev/pricing (January 2025)
- Anthropic: https://www.anthropic.com/pricing
- AWS Bedrock: https://aws.amazon.com/bedrock/pricing/
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import pandas as pd


# Model pricing data (cost per million tokens)
# Sources verified as of January 2025
MODEL_PRICING = {
    # ============== GOOGLE GEMINI MODELS ==============
    
    # Gemini 2.5 Series
    "gemini-2.5-pro": {
        "input": 1.25, "output": 10.00, "reasoning": 0.0, "cached": 0.31,
        "source": "Google AI official (prompts <= 200k tokens)"
    },
    "gas_gemini25_pro": {
        "input": 1.25, "output": 10.00, "reasoning": 0.0, "cached": 0.31,
        "source": "Google AI official"
    },
    "gemini-2.5-flash": {
        "input": 0.30, "output": 2.50, "reasoning": 0.0, "cached": 0.075,
        "source": "Google AI official"
    },
    "gas_gemini25_flash": {
        "input": 0.30, "output": 2.50, "reasoning": 0.0, "cached": 0.075,
        "source": "Google AI official"
    },
    "gemini-2.5-flash-lite": {
        "input": 0.10, "output": 0.40, "reasoning": 0.0, "cached": 0.025,
        "source": "Google AI official"
    },
    "gas_gemini25_flash_lite": {
        "input": 0.10, "output": 0.40, "reasoning": 0.0, "cached": 0.025,
        "source": "Google AI official"
    },
    
    # Gemini 2.0 Series
    "gemini-2.0-flash": {
        "input": 0.10, "output": 0.40, "reasoning": 0.0, "cached": 0.025,
        "source": "Google AI official"
    },
    "gas_gemini20_flash": {
        "input": 0.10, "output": 0.40, "reasoning": 0.0, "cached": 0.025,
        "source": "Google AI official"
    },
    "gemini-2.0-flash-lite": {
        "input": 0.075, "output": 0.30, "reasoning": 0.0, "cached": 0.0,
        "source": "Google AI official"
    },
    "gas_gemini20_flash_lite": {
        "input": 0.075, "output": 0.30, "reasoning": 0.0, "cached": 0.0,
        "source": "Google AI official"
    },
    
    # Gemini 1.5 Series
    "gemini-1.5-pro": {
        "input": 1.25, "output": 5.00, "reasoning": 0.0, "cached": 0.3125,
        "source": "Google AI official (prompts <= 128k tokens)"
    },
    "gemini-1.5-flash": {
        "input": 0.075, "output": 0.30, "reasoning": 0.0, "cached": 0.01875,
        "source": "Google AI official (prompts <= 128k tokens)"
    },
    "gemini-1.5-flash-8b": {
        "input": 0.0375, "output": 0.15, "reasoning": 0.0, "cached": 0.01,
        "source": "Google AI official (prompts <= 128k tokens)"
    },
    
    # ============== OPENAI MODELS ==============
    
    # GPT-5 Series (latest - January 2025)
    "gpt-5": {
        "input": 1.25, "output": 10.00, "reasoning": 0.0, "cached": 0.125,
        "source": "OpenAI official (January 2025)"
    },
    "openai_direct_chat_gpt5": {
        "input": 1.25, "output": 10.00, "reasoning": 0.0, "cached": 0.125,
        "source": "OpenAI official (January 2025)"
    },
    "gpt-5-mini": {
        "input": 0.25, "output": 2.00, "reasoning": 0.0, "cached": 0.025,
        "source": "OpenAI official (January 2025)"
    },
    "openai_direct_chat_gpt5_mini": {
        "input": 0.25, "output": 2.00, "reasoning": 0.0, "cached": 0.025,
        "source": "OpenAI official (January 2025)"
    },
    "gpt-5-nano": {
        "input": 0.05, "output": 0.40, "reasoning": 0.0, "cached": 0.005,
        "source": "OpenAI official (January 2025)"
    },
    "openai_direct_chat_gpt5_nano": {
        "input": 0.05, "output": 0.40, "reasoning": 0.0, "cached": 0.005,
        "source": "OpenAI official (January 2025)"
    },
    
    # GPT-4.1 Series
    "gpt-4.1": {
        "input": 2.00, "output": 8.00, "reasoning": 0.0, "cached": 0.50,
        "source": "OpenAI official"
    },
    "openai_direct_gpt41": {
        "input": 2.00, "output": 8.00, "reasoning": 0.0, "cached": 0.50,
        "source": "OpenAI official"
    },
    "gpt-4.1-mini": {
        "input": 0.40, "output": 1.60, "reasoning": 0.0, "cached": 0.10,
        "source": "OpenAI official"
    },
    "openai_direct_gpt41_mini": {
        "input": 0.40, "output": 1.60, "reasoning": 0.0, "cached": 0.10,
        "source": "OpenAI official"
    },
    "gpt-4.1-nano": {
        "input": 0.10, "output": 0.40, "reasoning": 0.0, "cached": 0.025,
        "source": "OpenAI official"
    },
    "oai_chat_gpt41_nano": {
        "input": 0.10, "output": 0.40, "reasoning": 0.0, "cached": 0.025,
        "source": "OpenAI official"
    },
    "openai_direct_gpt41_nano": {
        "input": 0.10, "output": 0.40, "reasoning": 0.0, "cached": 0.025,
        "source": "OpenAI official"
    },
    
    # GPT-4o Series
    "gpt-4o": {
        "input": 2.50, "output": 10.00, "reasoning": 0.0, "cached": 1.25,
        "source": "OpenAI official"
    },
    "gpt-4o-2024-08-06": {
        "input": 2.50, "output": 10.00, "reasoning": 0.0, "cached": 1.25,
        "source": "OpenAI official"
    },
    "oai_chat_gpt4o_2024_08_06": {
        "input": 2.50, "output": 10.00, "reasoning": 0.0, "cached": 1.25,
        "source": "OpenAI official"
    },
    "gpt-4o-2024-05-13": {
        "input": 5.00, "output": 15.00, "reasoning": 0.0, "cached": 0.0,
        "source": "OpenAI official"
    },
    "gpt-4o-mini": {
        "input": 0.15, "output": 0.60, "reasoning": 0.0, "cached": 0.075,
        "source": "OpenAI official"
    },
    "oai_chat_gpt4o_mini_2024_07_18": {
        "input": 0.15, "output": 0.60, "reasoning": 0.0, "cached": 0.075,
        "source": "OpenAI official"
    },
    "openai_direct_chat_gpt4o_mini": {
        "input": 0.15, "output": 0.60, "reasoning": 0.0, "cached": 0.075,
        "source": "OpenAI official"
    },
    "gpt-4o-mini-2024-07-18": {
        "input": 0.15, "output": 0.60, "reasoning": 0.0, "cached": 0.075,
        "source": "OpenAI official"
    },
    
    # O-series reasoning models
    "o1": {
        "input": 15.00, "output": 60.00, "reasoning": 60.00, "cached": 7.50,
        "source": "OpenAI official"
    },
    "o1-preview": {
        "input": 15.00, "output": 60.00, "reasoning": 60.00, "cached": 7.50,
        "source": "OpenAI official"
    },
    "o1-mini": {
        "input": 1.10, "output": 4.40, "reasoning": 4.40, "cached": 0.55,
        "source": "OpenAI official"
    },
    "o1-pro": {
        "input": 150.00, "output": 600.00, "reasoning": 600.00, "cached": 0.0,
        "source": "OpenAI official"
    },
    "o3": {
        "input": 2.00, "output": 8.00, "reasoning": 8.00, "cached": 0.50,
        "source": "OpenAI official"
    },
    "o3-pro": {
        "input": 20.00, "output": 80.00, "reasoning": 80.00, "cached": 0.0,
        "source": "OpenAI official"
    },
    "o3-mini": {
        "input": 1.10, "output": 4.40, "reasoning": 4.40, "cached": 0.55,
        "source": "OpenAI official"
    },
    "o3-2025-04-16": {
        "input": 2.00, "output": 8.00, "reasoning": 8.00, "cached": 0.50,
        "source": "OpenAI official"
    },
    "o3-mini-2025-01-31": {
        "input": 1.10, "output": 4.40, "reasoning": 4.40, "cached": 0.55,
        "source": "OpenAI official"
    },
    "o4-mini": {
        "input": 1.10, "output": 4.40, "reasoning": 4.40, "cached": 0.275,
        "source": "OpenAI official"
    },
    "o4-mini-2025-04-16": {
        "input": 1.10, "output": 4.40, "reasoning": 4.40, "cached": 0.275,
        "source": "OpenAI official"
    },
    "openai_direct_o3": {
        "input": 2.00, "output": 8.00, "reasoning": 8.00, "cached": 0.50,
        "source": "OpenAI official"
    },
    "openai_direct_o4_mini": {
        "input": 1.10, "output": 4.40, "reasoning": 4.40, "cached": 0.275,
        "source": "OpenAI official"
    },
    
    # GPT-4 Legacy models
    "gpt-4": {
        "input": 30.00, "output": 60.00, "reasoning": 0.0, "cached": 0.0,
        "source": "OpenAI official (gpt-4-0613)"
    },
    "gpt-4-turbo": {
        "input": 10.00, "output": 30.00, "reasoning": 0.0, "cached": 0.0,
        "source": "OpenAI official"
    },
    "gpt-4-turbo-2024-04-09": {
        "input": 10.00, "output": 30.00, "reasoning": 0.0, "cached": 0.0,
        "source": "OpenAI official"
    },
    
    # GPT-3.5 Legacy
    "gpt-3.5-turbo": {
        "input": 0.50, "output": 1.50, "reasoning": 0.0, "cached": 0.0,
        "source": "OpenAI official"
    },
    "oai_chat_gpt35_we": {
        "input": 0.50, "output": 1.50, "reasoning": 0.0, "cached": 0.0,
        "source": "OpenAI official"
    },
    
    # ============== ANTHROPIC CLAUDE MODELS ==============
    
    # Claude 3.5 Series
    "claude-3.5-sonnet": {
        "input": 3.00, "output": 15.00, "reasoning": 0.0, "cached": 0.30,
        "source": "Anthropic official"
    },
    "aws_bedrock_claude35_sonnet": {
        "input": 3.00, "output": 15.00, "reasoning": 0.0, "cached": 0.30,
        "source": "AWS Bedrock"
    },
    "aws_bedrock_claude35_sonnet_v2": {
        "input": 3.00, "output": 15.00, "reasoning": 0.0, "cached": 0.30,
        "source": "AWS Bedrock"
    },
    "claude-3.5-haiku": {
        "input": 0.80, "output": 4.00, "reasoning": 0.0, "cached": 0.08,
        "source": "Anthropic official"
    },
    "aws_bedrock_claude35_haiku": {
        "input": 0.80, "output": 4.00, "reasoning": 0.0, "cached": 0.08,
        "source": "AWS Bedrock"
    },
    
    # Claude 3 Series
    "claude-3-opus": {
        "input": 15.00, "output": 75.00, "reasoning": 0.0, "cached": 1.50,
        "source": "Anthropic official"
    },
    "aws_bedrock_claude3_opus": {
        "input": 15.00, "output": 75.00, "reasoning": 0.0, "cached": 1.50,
        "source": "AWS Bedrock"
    },
    "claude-3-sonnet": {
        "input": 3.00, "output": 15.00, "reasoning": 0.0, "cached": 0.30,
        "source": "Anthropic official"
    },
    "claude-3.7-sonnet": {
        "input": 3.00, "output": 15.00, "reasoning": 0.0, "cached": 0.30,
        "source": "Anthropic official"
    },
    "aws_bedrock_claude37_sonnet": {
        "input": 3.00, "output": 15.00, "reasoning": 0.0, "cached": 0.30,
        "source": "AWS Bedrock"
    },
    "claude-3-haiku": {
        "input": 0.25, "output": 1.25, "reasoning": 0.0, "cached": 0.03,
        "source": "Anthropic official"
    },
    "aws_bedrock_claude3_haiku": {
        "input": 0.25, "output": 1.25, "reasoning": 0.0, "cached": 0.03,
        "source": "AWS Bedrock"
    },
    
    # ============== AWS BEDROCK MODELS ==============
    
    # AWS Bedrock Llama models
    "aws_bedrock_llama32_1b": {
        "input": 0.10, "output": 0.10, "reasoning": 0.0, "cached": 0.0,
        "source": "AWS Bedrock pricing"
    },
    "aws_bedrock_llama32_3b": {
        "input": 0.15, "output": 0.15, "reasoning": 0.0, "cached": 0.0,
        "source": "AWS Bedrock pricing"
    },
    "aws_bedrock_llama31_8b": {
        "input": 0.22, "output": 0.22, "reasoning": 0.0, "cached": 0.0,
        "source": "AWS Bedrock pricing"
    },
    "aws_bedrock_llama31_70b": {
        "input": 0.99, "output": 0.99, "reasoning": 0.0, "cached": 0.0,
        "source": "AWS Bedrock"
    },
    "aws_bedrock_llama33_70b": {
        "input": 0.99, "output": 0.99, "reasoning": 0.0, "cached": 0.0,
        "source": "AWS Bedrock"
    },
    "aws_bedrock_llama31_405b": {
        "input": 2.65, "output": 3.50, "reasoning": 0.0, "cached": 0.0,
        "source": "AWS Bedrock"
    },
    
    # AWS Nova models
    "aws_bedrock_nova_lite": {
        "input": 0.06, "output": 0.24, "reasoning": 0.0, "cached": 0.0,
        "source": "AWS Bedrock pricing"
    },
    "aws_bedrock_nova_micro": {
        "input": 0.035, "output": 0.14, "reasoning": 0.0, "cached": 0.0,
        "source": "AWS Bedrock pricing"
    },
    "aws_bedrock_nova_pro": {
        "input": 0.80, "output": 3.20, "reasoning": 0.0, "cached": 0.0,
        "source": "AWS Bedrock pricing"
    },
    
    # ============== OTHER MODELS ==============
    
    # Mistral models
    "mistral-7b": {
        "input": 0.18, "output": 0.18, "reasoning": 0.0, "cached": 0.0,
        "source": "Mistral AI"
    },
    "mistral_7b_trt_llm": {
        "input": 0.18, "output": 0.18, "reasoning": 0.0, "cached": 0.0,
        "source": "Internal deployment"
    },
    
    # Hermes models (internal fine-tuned)
    "hermes_3b_semisynt1m_tn20_2025_03_07": {
        "input": 0.20, "output": 0.20, "reasoning": 0.0, "cached": 0.0,
        "source": "Internal deployment estimate"
    },
    "hermes-llama-3b-sft-enhanced11-jul2-2025": {
        "input": 0.20, "output": 0.20, "reasoning": 0.0, "cached": 0.0,
        "source": "Internal deployment estimate"
    },
    
    # DeepSeek models
    "deepseek_r1_azure_alpha": {
        "input": 0.55, "output": 2.19, "reasoning": 2.19, "cached": 0.14,
        "source": "DeepSeek pricing"
    },
    
    # ============== ALIASES FOR COMPATIBILITY ==============
    "llama3.2-1b": {"input": 0.10, "output": 0.10, "reasoning": 0.0, "cached": 0.0, "source": "Alias"},
    "llama3.2-3b": {"input": 0.15, "output": 0.15, "reasoning": 0.0, "cached": 0.0, "source": "Alias"},
    "nova-lite": {"input": 0.06, "output": 0.24, "reasoning": 0.0, "cached": 0.0, "source": "Alias"},
    "hermes-3b": {"input": 0.20, "output": 0.20, "reasoning": 0.0, "cached": 0.0, "source": "Alias"},
}


@dataclass
class TokenUsage:
    """Container for token usage data"""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    
    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens + self.reasoning_tokens


@dataclass 
class CostBreakdown:
    """Container for cost breakdown data"""
    input_cost: float = 0.0
    output_cost: float = 0.0
    reasoning_cost: float = 0.0
    cached_cost: float = 0.0
    total_cost: float = 0.0
    model: str = ""
    currency: str = "USD"
    
    def __post_init__(self):
        if self.total_cost == 0.0:
            self.total_cost = self.input_cost + self.output_cost + self.reasoning_cost + self.cached_cost


def get_model_pricing(model: str) -> Optional[Dict[str, Any]]:
    """Get pricing data for a specific model with robust alias/prefix matching."""
    if not model:
        return None
    m = model.lower().strip().replace(' ', '_')
    # Exact
    if m in MODEL_PRICING:
        return MODEL_PRICING[m]
    # Common internal aliases
    aliases = [
        m.replace('openai_direct_', ''),
        m.replace('oai_chat_', ''),
        m.replace('gas_', ''),
    ]
    for a in aliases:
        if a in MODEL_PRICING:
            return MODEL_PRICING[a]
    # Longest prefix/substring match
    best = None
    best_len = 0
    for k in MODEL_PRICING.keys():
        if m.startswith(k) or k in m:
            if len(k) > best_len:
                best = k
                best_len = len(k)
    return MODEL_PRICING.get(best) if best else None


def calculate_cost(model: str, input_tokens: int = 0, output_tokens: int = 0, 
                  reasoning_tokens: int = 0, cached_tokens: int = 0) -> CostBreakdown:
    """
    Calculate costs for different token types.
    
    Args:
        model: Model name
        input_tokens: Number of input tokens (total, including cached)
        output_tokens: Number of output tokens  
        reasoning_tokens: Number of reasoning tokens (for o1/o3 models)
        cached_tokens: Number of cached tokens (subset of input_tokens)
        
    Returns:
        CostBreakdown object with cost details
    """
    pricing = get_model_pricing(model)
    
    if not pricing:
        # Return zero cost if model not found (with warning in model name)
        return CostBreakdown(
            input_cost=0.0,
            output_cost=0.0,
            reasoning_cost=0.0,
            cached_cost=0.0,
            total_cost=0.0,
            model=f"{model} (UNKNOWN PRICING)"
        )
    
    # Calculate costs per token type (pricing is per million tokens)
    # IMPORTANT: cached_tokens is a subset of input_tokens, not additional
    uncached_input_tokens = max(0, input_tokens - cached_tokens)
    
        # Uncached input uses regular input pricing
    input_cost = (uncached_input_tokens / 1_000_000) * pricing["input"]
    
    # Cached tokens use discounted pricing
    cached_cost = (cached_tokens / 1_000_000) * pricing["cached"]
    
    # Calculate savings from caching (no per-sample logging)
    # Cache savings will be reported in final summary only
    
    # Output and reasoning tokens use their respective pricing
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    reasoning_cost = (reasoning_tokens / 1_000_000) * pricing["reasoning"]
    
    return CostBreakdown(
        input_cost=input_cost,
        output_cost=output_cost,
        reasoning_cost=reasoning_cost,
        cached_cost=cached_cost,
        total_cost=input_cost + output_cost + reasoning_cost + cached_cost,
        model=model
    )


class PricingTracker:
    """Tracks cumulative pricing across multiple API calls"""
    
    def __init__(self):
        self.usage_by_model: Dict[str, TokenUsage] = {}
        self.cost_by_model: Dict[str, CostBreakdown] = {}
        self.total_requests = 0
    
    def add_usage(self, model: str, input_tokens: int = 0, output_tokens: int = 0,
                  reasoning_tokens: int = 0, cached_tokens: int = 0):
        """Add token usage for a model"""
        self.total_requests += 1
        
        if model not in self.usage_by_model:
            self.usage_by_model[model] = TokenUsage()
            self.cost_by_model[model] = CostBreakdown(model=model)
        
        # Update token usage
        usage = self.usage_by_model[model]
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens  
        usage.reasoning_tokens += reasoning_tokens
        usage.cached_tokens += cached_tokens
        usage.total_tokens += input_tokens + output_tokens + reasoning_tokens
        
        # Update costs
        cost_data = calculate_cost(model, input_tokens, output_tokens, reasoning_tokens, cached_tokens)
        cost = self.cost_by_model[model]
        cost.input_cost += cost_data.input_cost
        cost.output_cost += cost_data.output_cost
        cost.reasoning_cost += cost_data.reasoning_cost
        cost.cached_cost += cost_data.cached_cost
        cost.total_cost += cost_data.total_cost
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked usage and costs"""
        total_tokens = sum(u.total_tokens for u in self.usage_by_model.values())
        total_cost = sum(c.total_cost for c in self.cost_by_model.values())
        
        return {
            'total_requests': self.total_requests,
            'total_tokens': total_tokens,
            'total_cost_usd': total_cost,
            'models_used': list(self.usage_by_model.keys()),
            'by_model': {
                model: {
                    'tokens': self.usage_by_model[model].total_tokens,
                    'cost_usd': self.cost_by_model[model].total_cost
                }
                for model in self.usage_by_model
            }
        }
    
    def get_total_cost(self) -> float:
        """Get total cost across all models"""
        return sum(c.total_cost for c in self.cost_by_model.values())


def extrapolate_costs(current_cost: float, current_samples: int, target_samples: int) -> Dict[str, Any]:
    """
    Extrapolate costs from current usage to target usage.
    
    Args:
        current_cost: Current total cost in USD
        current_samples: Number of samples processed
        target_samples: Target number of samples to extrapolate to
        
    Returns:
        Dictionary with extrapolation data
    """
    if current_samples == 0:
        return {
            'extrapolated_cost': 0.0,
            'cost_per_sample': 0.0,
            'multiplier': 0.0
        }
    
    cost_per_sample = current_cost / current_samples
    extrapolated_cost = cost_per_sample * target_samples
    
    return {
        'extrapolated_cost': extrapolated_cost,
        'cost_per_sample': cost_per_sample,
        'multiplier': target_samples / current_samples
    }


def generate_cost_report(tracker: PricingTracker, detailed: bool = True) -> List[str]:
    """
    Generate a human-readable cost report.
    
    Args:
        tracker: PricingTracker instance with usage data
        detailed: Whether to include detailed breakdown
        
    Returns:
        List of report lines
    """
    summary = tracker.get_summary()
    report = []
    
    report.append("=" * 60)
    report.append("COST ANALYSIS REPORT")
    report.append("=" * 60)
    report.append(f"Total Requests: {summary['total_requests']:,}")
    report.append(f"Total Tokens: {summary['total_tokens']:,}")
    
    # Calculate and report cache savings if applicable
    total_cached = sum(u.cached_tokens for u in tracker.usage_by_model.values())
    total_input = sum(u.input_tokens for u in tracker.usage_by_model.values())
    
    if total_cached > 0 and total_input > 0:
        cache_rate = (total_cached / total_input) * 100
        # Calculate savings from caching for the primary model
        if tracker.usage_by_model:
            # Get the first/primary model
            primary_model = list(tracker.usage_by_model.keys())[0]
            pricing = get_model_pricing(primary_model)
            if pricing and pricing["cached"] < pricing["input"]:
                hypothetical_cost = (total_cached / 1_000_000) * pricing["input"]
                actual_cached_cost = (total_cached / 1_000_000) * pricing["cached"]
                cache_savings = hypothetical_cost - actual_cached_cost
                
                report.append(f"Cache Hit Rate: {cache_rate:.1f}% ({total_cached:,}/{total_input:,} tokens)")
                report.append(f"Cache Savings: ${cache_savings:.4f}")
                report.append(f"Total Cost (with caching): ${summary['total_cost_usd']:.6f}")
                report.append(f"Would have cost (no cache): ${summary['total_cost_usd'] + cache_savings:.6f}")
            else:
                report.append(f"Total Cost: ${summary['total_cost_usd']:.6f}")
        else:
            report.append(f"Total Cost: ${summary['total_cost_usd']:.6f}")
    else:
        report.append(f"Total Cost: ${summary['total_cost_usd']:.6f}")
    
    if summary['total_requests'] > 0:
        avg_cost = summary['total_cost_usd'] / summary['total_requests']
        report.append(f"Average Cost per Request: ${avg_cost:.6f}")
        
        # Extrapolations
        for target in [1000, 10000, 100000, 1000000]:
            ext = extrapolate_costs(summary['total_cost_usd'], summary['total_requests'], target)
            report.append(f"Projected cost for {target:,} requests: ${ext['extrapolated_cost']:.2f}")
    
    if detailed and summary['by_model']:
        report.append("\nBreakdown by Model:")
        report.append("-" * 40)
        for model, data in summary['by_model'].items():
            report.append(f"{model}:")
            report.append(f"  Tokens: {data['tokens']:,}")
            report.append(f"  Cost: ${data['cost_usd']:.6f}")
    
    # Cost optimization suggestions
    if summary['total_cost_usd'] > 0:
        report.append("\nCost Optimization Suggestions:")
        report.append("-" * 40)
        
        # Check if using expensive models
        expensive_models = ['gpt-4', 'o3', 'claude-3-opus', 'gpt-4.1']
        using_expensive = any(m in str(summary['models_used']) for m in expensive_models)
        
        if using_expensive:
            report.append("⚠️ You're using expensive models. Consider:")
            report.append("  - gemini-2.0-flash-lite: $0.075/$0.30 per 1M tokens")
            report.append("  - gpt-4o-mini: $0.15/$0.60 per 1M tokens")
            report.append("  - aws_bedrock_llama32_1b: $0.10/$0.10 per 1M tokens")
        
        # Suggest batching
        if summary['total_requests'] < 100:
            report.append("💡 Tip: Batch requests to reduce overhead")
        
        # Suggest caching
        report.append("💡 Tip: Enable prefix caching for repeated prompts")
    
    report.append("=" * 60)
    
    return report


def compare_model_costs(models: List[str], input_tokens: int = 1000, 
                        output_tokens: int = 100) -> pd.DataFrame:
    """
    Compare costs across multiple models.
    
    Args:
        models: List of model names to compare
        input_tokens: Number of input tokens for comparison
        output_tokens: Number of output tokens for comparison
        
    Returns:
        DataFrame with cost comparison
    """
    data = []
    
    for model in models:
        cost_data = calculate_cost(model, input_tokens, output_tokens)
        data.append({
            'Model': model,
            'Input Cost': f"${cost_data.input_cost:.6f}",
            'Output Cost': f"${cost_data.output_cost:.6f}",
            'Total Cost': f"${cost_data.total_cost:.6f}",
            '10K Requests': f"${cost_data.total_cost * 10000:.2f}",
            '1M Requests': f"${cost_data.total_cost * 1000000:.2f}"
        })
    
    df = pd.DataFrame(data)
    return df.sort_values('Total Cost')


if __name__ == "__main__":
    # Test pricing calculations
    print("Testing pricing calculations with official rates...")
    
    test_models = [
        'gas_gemini20_flash_lite',  # $0.075/$0.30
        'gemini-2.0-flash',  # $0.10/$0.40
        'gemini-2.5-flash-lite',  # $0.10/$0.40
        'gemini-1.5-flash',  # $0.075/$0.30
        'gpt-4.1-nano',  # $0.10/$0.40
        'gpt-4o-mini',  # $0.15/$0.60
        'gpt-4.1',  # $2.00/$8.00
        'gpt-4o',  # $2.50/$10.00
        'claude-3.5-haiku',  # $0.80/$4.00
        'aws_bedrock_llama32_1b'  # $0.10/$0.10
    ]
    
    print("\nCost comparison (1000 input tokens, 100 output tokens):")
    comparison = compare_model_costs(test_models)
    print(comparison.to_string(index=False))
    
    print("\n✅ Pricing module ready with official rates!")