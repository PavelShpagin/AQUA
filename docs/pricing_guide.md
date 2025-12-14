# LLM Pricing Guide

## Overview
This document provides accurate pricing information for all supported LLM models in the GEC Judge system.

## Quick Summary

### Why the 3x Difference Was Wrong
You noticed that GPT-4.1 showed only 3x the cost of Gemini Flash Lite. The issue was:
1. **Missing Pricing Data**: GPT-4.1 (full model) had no pricing entry, defaulting to $0
2. **Model Confusion**: The system was likely using GPT-4.1-nano pricing instead
3. **Now Fixed**: GPT-4.1 is properly priced at ~30x more than GPT-4.1-nano

## Current Pricing Tiers (per 10K requests, 300 input + 50 output tokens)

### 🆓 Free Tier
- **gas_gemini20_flash_lite**: $0.00 (free during preview)
- **gemini-2.0-flash-lite**: $0.00 (free during preview)

### 💰 Ultra-Budget Tier (<$1)
- **aws_bedrock_llama32_1b**: $0.35
- **aws_bedrock_nova_lite**: $0.30
- **gpt-4.1-nano**: $0.75
- **gpt-4o-mini**: $0.75

### 💵 Budget Tier ($1-10)
- **aws_bedrock_llama32_3b**: $0.53
- **mistral-7b**: $0.63
- **aws_bedrock_llama31_8b**: $0.77
- **claude-3-haiku**: $1.31
- **claude-3.5-haiku**: $4.40

### 💸 Standard Tier ($10-50)
- **gpt-4o**: $12.50
- **gpt-4.1**: $22.50
- **claude-3.5-sonnet**: $16.50
- **aws_bedrock_llama31_70b**: $3.47

### 💎 Premium Tier ($50+)
- **o3-mini**: $25.00
- **o3**: $120.00
- **gpt-4**: $115.50
- **claude-3-opus**: $57.75

## Price Comparisons

### Key Ratios (for typical GEC usage)
- **GPT-4.1 vs GPT-4.1-nano**: 30x more expensive
- **GPT-4.1 vs Gemini Flash Lite**: ∞ (Gemini is free)
- **GPT-4o vs GPT-4o-mini**: 16.7x more expensive
- **Claude Sonnet vs Claude Haiku**: 12.6x more expensive

### Cost per 10K Requests (300/50 tokens)
| Model | Cost | Notes |
|-------|------|-------|
| gas_gemini20_flash_lite | $0.00 | Free during preview |
| aws_bedrock_llama32_1b | $0.35 | Cheapest paid option |
| gpt-4.1-nano | $0.75 | Good quality/cost balance |
| gpt-4o-mini | $0.75 | OpenAI's budget option |
| claude-3.5-haiku | $4.40 | Anthropic's budget option |
| gpt-4o | $12.50 | High quality |
| gpt-4.1 | $22.50 | Very expensive |
| claude-3.5-sonnet | $16.50 | Premium quality |

## Recommendations

### For Different Use Cases

#### Development & Testing
- **Best**: `gas_gemini20_flash_lite` (free)
- **Alternative**: `aws_bedrock_llama32_1b` ($0.35/10K)

#### Production - Cost Optimized
- **Best**: `gpt-4.1-nano` or `gpt-4o-mini` ($0.75/10K)
- **Alternative**: `aws_bedrock_llama32_3b` ($0.53/10K)

#### Production - Quality Optimized
- **Best**: `gpt-4o` ($12.50/10K)
- **Alternative**: `claude-3.5-sonnet` ($16.50/10K)

#### Research & Benchmarking
- **Start with**: `gas_gemini20_flash_lite` (free)
- **Scale to**: `gpt-4o-mini` for consistency
- **Premium**: `gpt-4o` or `claude-3.5-sonnet` for best results

### Cost Optimization Tips

1. **Use Free Models During Development**
   - Gemini Flash Lite is free and supports 50+ languages
   - Perfect for initial testing and development

2. **Batch Processing**
   - Reduces overhead and can qualify for volume discounts
   - Some providers offer batch pricing (not yet implemented)

3. **Implement Caching**
   - Many models offer reduced rates for cached/repeated content
   - Can reduce costs by 50-80% for repetitive tasks

4. **Choose the Right Model**
   - Don't use GPT-4.1 when GPT-4.1-nano would suffice
   - Test cheaper models first - they're often good enough

5. **Monitor Usage**
   - Use the pricing tracker to monitor costs
   - Set up alerts for unexpected usage spikes

## Pricing Data Sources

All pricing data is sourced from official documentation:
- **OpenAI**: https://openai.com/pricing
- **Google**: https://ai.google.dev/pricing  
- **Anthropic**: https://www.anthropic.com/pricing
- **AWS Bedrock**: https://aws.amazon.com/bedrock/pricing/

Last updated: January 2025

## Implementation Details

### How Pricing Works
1. Token counts are extracted from API responses
2. Costs calculated per million tokens based on model
3. Different rates for input/output/reasoning/cached tokens
4. Total cost aggregated across all token types

### Adding New Models
To add pricing for a new model:
1. Edit `utils/pricing.py`
2. Add entry to `MODEL_PRICING` dictionary
3. Include source reference
4. Run `python test/verify_pricing.py` to verify

### Troubleshooting

**Issue**: Model shows $0.00 cost
- Check if model has pricing entry in `utils/pricing.py`
- Some models (Gemini Flash Lite) are genuinely free

**Issue**: Unexpected high costs
- Verify token counts in API response
- Check for reasoning tokens (o1/o3 models)
- Ensure correct model name mapping

**Issue**: Price differences seem wrong
- Run `python test/test_pricing_comparison.py`
- Compare with official pricing pages
- Check for promotional/preview pricing
