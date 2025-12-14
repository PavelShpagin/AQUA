# LLM Pricing Guide (January 2025)

## Overview
This document contains the official paid pricing for all supported LLM models as of January 2025. All prices are in USD per million tokens.

## Quick Reference Table

### Ultra-Budget Tier (<$1/10K requests)
| Model | Input ($/1M) | Output ($/1M) | 10K Requests* | Notes |
|-------|--------------|---------------|---------------|-------|
| **Llama 3.2 1B** | $0.10 | $0.10 | $0.35 | AWS Bedrock, cheapest overall |
| **Gemini 1.5 Flash** | $0.075 | $0.30 | $0.37 | Older but very cheap |
| **Gemini 2.5 Flash-Lite** | $0.10 | $0.40 | $0.50 | Currently free preview |
| **GPT-4.1 Nano** | $0.10 | $0.40 | $0.50 | OpenAI's ultra-budget |
| **GPT-4o Mini** | $0.15 | $0.60 | $0.75 | Good quality/price |

### Budget Tier ($1-5/10K requests)
| Model | Input ($/1M) | Output ($/1M) | 10K Requests* | Notes |
|-------|--------------|---------------|---------------|-------|
| **Llama 3.2 3B** | $0.15 | $0.15 | $0.53 | AWS Bedrock |
| **Mistral 7B** | $0.18 | $0.18 | $0.63 | Open source |
| **Claude 3 Haiku** | $0.25 | $1.25 | $1.31 | Anthropic budget |
| **GPT-4.1 Mini** | $0.40 | $1.60 | $2.00 | Mid-tier GPT |
| **Gemini 2.5 Flash** | $0.30 | $2.50 | $2.15 | Latest Flash model |
| **Claude 3.5 Haiku** | $0.80 | $4.00 | $4.40 | Better Haiku |

### Standard Tier ($5-20/10K requests)
| Model | Input ($/1M) | Output ($/1M) | 10K Requests* | Notes |
|-------|--------------|---------------|---------------|-------|
| **Gemini 1.5 Pro** | $1.25 | $5.00 | $5.63 | Pro version |
| **GPT-4.1** | $2.00 | $8.00 | $10.00 | Latest GPT-4.1 |
| **Gemini 2.5 Pro** | $1.25 | $10.00 | $8.75 | Latest Pro |
| **GPT-4o** | $2.50 | $10.00 | $12.50 | Optimized GPT-4 |
| **Claude 3.5 Sonnet** | $3.00 | $15.00 | $16.50 | Anthropic standard |

### Premium Tier ($20+/10K requests)
| Model | Input ($/1M) | Output ($/1M) | 10K Requests* | Notes |
|-------|--------------|---------------|---------------|-------|
| **O1 Mini** | $3.00 | $12.00 | $15.00** | With reasoning tokens |
| **O3 Mini** | $5.00 | $20.00 | $25.00** | Next-gen reasoning |
| **GPT-4** | $30.00 | $60.00 | $115.50 | Original GPT-4 |
| **Claude 3 Opus** | $15.00 | $75.00 | $57.75 | Anthropic premium |
| **O1 Preview** | $15.00 | $60.00 | $75.00** | Advanced reasoning |
| **O3** | $20.00 | $100.00 | $120.00** | Top reasoning model |

*Based on 300 input + 50 output tokens per request (typical GEC usage)
**O-series models also charge for reasoning tokens at output rates

## Key Insights

### Best Value Models
1. **Llama 3.2 1B** ($0.35/10K) - Absolute cheapest
2. **Gemini 1.5 Flash** ($0.37/10K) - Very cheap, good quality
3. **GPT-4.1 Nano** ($0.50/10K) - OpenAI quality at low cost
4. **GPT-4o Mini** ($0.75/10K) - Best balance of quality/cost

### Price Comparisons
- **Gemini 2.5 Flash** is 5.7x more expensive than Gemini 1.5 Flash
- **GPT-4.1** is 20x more expensive than GPT-4.1 Nano
- **GPT-4o** is 1.25x more expensive than GPT-4.1
- **Claude 3.5 Sonnet** is 12.6x more expensive than Claude 3 Haiku

### Model Evolution Pricing
- **Gemini**: 1.5 Flash ($0.37) → 2.5 Flash-Lite ($0.50) → 2.5 Flash ($2.15)
- **GPT-4.1**: Nano ($0.50) → Mini ($2.00) → Standard ($10.00)
- **Claude**: Haiku ($1.31) → Haiku 3.5 ($4.40) → Sonnet 3.5 ($16.50)

## Special Considerations

### Free Preview Models
Some models are currently free during their preview period:
- **Gemini 2.5 Flash-Lite**: Free now, will be $0.10/$0.40 per 1M tokens

### Cached Input Pricing
Most models offer 50-75% discount for cached/repeated inputs:
- OpenAI: 50% discount on cached inputs
- Google: 75% discount on cached inputs  
- Anthropic: 90% discount on cached inputs

### Reasoning Token Pricing
O-series models (O1, O3) charge separately for reasoning tokens:
- Reasoning tokens are charged at the same rate as output tokens
- Can significantly increase costs for complex reasoning tasks

## Recommendations by Use Case

### High Volume, Cost-Sensitive
- **Primary**: Llama 3.2 1B or Gemini 1.5 Flash
- **Backup**: GPT-4.1 Nano

### Balanced Quality/Cost
- **Primary**: GPT-4o Mini
- **Backup**: Claude 3 Haiku

### High Quality Requirements
- **Primary**: GPT-4.1 or GPT-4o
- **Backup**: Claude 3.5 Sonnet

### Maximum Quality
- **Primary**: O3 or GPT-4
- **Backup**: Claude 3 Opus

## Data Sources
- OpenAI: https://openai.com/pricing
- Google AI: https://ai.google.dev/pricing
- Anthropic: https://www.anthropic.com/pricing
- AWS Bedrock: https://aws.amazon.com/bedrock/pricing/

Last updated: January 2025
