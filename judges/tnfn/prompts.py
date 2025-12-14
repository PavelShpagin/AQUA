#!/usr/bin/env python3
"""
TN/FN prompts with language placeholder {0}.
"""

TNFN_PROMPT = """
Check if the text below has grammatical errors.

IMPORTANT: Analyze ONLY the specific text provided. Do not imagine or reference any other text.

## Classification
- **TN (True Negative)**: Text is grammatically correct - can be used for training
- **FN (False Negative)**: Text has grammatical errors - cannot be used for training

Language: {0}

Text to evaluate: {1}

{2}

Output JSON only:
{
  "reason": "Brief explanation of grammatical status",
  "errors_found": ["list any errors found, or empty if none"],
  "confidence": "High/Medium/Low",
  "classification": "TN or FN"
}
"""

