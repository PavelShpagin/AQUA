# LLM Backbone and Prompts Documentation

## LLM Backbone

### Primary Testing Backend: `gas_gemini20_flash_lite`
- **Model**: Gemini 2.0 Flash Lite
- **Provider**: Google (via internal GAS service)
- **Cost**: Very low (~$0.70-1.45 per 10K requests)
- **Speed**: Fast inference
- **Quality**: Good for structured outputs

### Available Backends
```python
# High-end models (best quality)
- o3-2025-04-16          # OpenAI O3 (best accuracy)
- gpt-4.1                # GPT-4 Turbo
- gpt-4o                 # GPT-4 Optimized

# Mid-tier models (balanced)
- gas_gemini20_flash     # Gemini 2.0 Flash (full)
- gpt-4o-mini            # GPT-4 Mini
- gpt-4.1-nano          # GPT-4 Nano

# Fast/cheap models (for testing)
- gas_gemini20_flash_lite # Gemini 2.0 Flash Lite (current)
- gemini-2.0-flash-lite   # Direct Gemini API
```

## Modular Component Prompts

### 1. Nonsense Detection Prompt (Same for Edit & Feedback)

```prompt
You are a nonsense detector for GEC evaluation. Rate how much the text introduces 
nonsense, structural incoherence, or makes the text uninterpretable.

SCORE SCALE:
-1: Reduced nonsense (text becomes clearer/more coherent)
 0: Neutral (no change in nonsense level)
 1: Slight nonsense (minor confusion or awkwardness)
 2: Medium nonsense (noticeable incoherence or confusion)
 3: Major nonsense (loss of information, syntax breaking, uninterpretable)

Text to evaluate:
{text}

Return format:
SCORE: [-1 to 3]
REASON: [brief explanation]
```

### 2. Meaning Change Detection Prompt (Same for Edit & Feedback)

```prompt
You are a meaning-change detector for GEC evaluation. Compare the original and 
suggested sentences and rate the meaning change severity.

SEVERITY LEVELS:
0 - No meaning change (grammatical corrections only)
1 - Minor clarification or style change (same core meaning)
2 - Noticeable but acceptable change (slight shift in emphasis/nuance)
3 - Significant meaning alteration (important information changed)
4 - Major meaning change or contradiction (completely different meaning)

ORIGINAL:
{original}

SUGGESTED:
{suggested}

Return format:
SEVERITY: [0-4]
REASONING: [explain the meaning change level]
```

### 3. Quality/Reward Assessment Prompt

#### Edit Version
```prompt
You are a text quality assessor for GEC evaluation. Compare the original and 
suggested sentences and rate the relative improvement.

ORIGINAL:
{original}

SUGGESTED:
{suggested}

Rate the relative improvement on a scale from -3 to +3:
-3: Much worse (introduces major errors, significantly degrades quality)
-2: Worse (introduces noticeable errors, degrades quality)
-1: Slightly worse (minor degradation)
 0: No change in quality (preferential/stylistic)
+1: Slightly better (minor improvement)
+2: Better (fixes clear errors, improves clarity)
+3: Much better (significant improvement, major error fixes)

Return format:
IMPROVEMENT: [-3 to +3]
REASONING: [explain the relative improvement assessment]
```

#### Feedback Version (Nearly identical, minor wording difference)
```prompt
Rate the relative improvement on a scale from -3 to +3:
-3: Much worse (introduces major errors, significantly degrades quality)
-2: Worse (introduces noticeable errors, degrades quality)
-1: Slightly worse (minor degradation)
 0: No change in quality
+1: Slightly better (minor improvement)
+2: Better (fixes errors, improves clarity)
+3: Much better (significant improvement, major error fixes)
```

## Agent Judge Prompts

### Edit Agent Prompt
```prompt
You are a GEC edit judge agent that uses tools to evaluate edits.

## AVAILABLE TOOLS
To use a tool, write exactly: Action: tool_name(arg1, arg2)

Tools:
- nonsense_detector(text) - Check if text is nonsensical
- meaning_change(original, suggested) - Get meaning change severity (0-4)
- reward_quality(original, suggested) - Get quality improvement (-3 to +3)
- grammar_rag(text, language) - Find relevant grammar rules
- comprehensive_analysis(original, suggested, language) - Full analysis

## CLASSIFICATION FRAMEWORK
**Per-Edit Classifications:**
- TP (True Positive): Edit correctly fixes a grammatical error or improves clarity
- FP1 (Critical): Edit creates severe issues (nonsense, major meaning change ≥3, sensitivity)
- FP2 (Medium): Edit degrades grammar or creates moderate issues (meaning change ≥2)
- FP3 (Minor): Edit is preferential without clear benefit (improvement ≤ 0)
- TN (True Negative): No edit needed and none applied
- FN (False Negative): Edit was needed but not applied

## EVALUATION CONTEXT
**Language:** {language}
**Original Text:** "{original}"
**Suggested Text:** "{suggested}"

## TASK
Provide both per-edit assessments and a final sentence-level classification.

Thought: I will examine the edit operations and assess their validity.

When ready to conclude, output:
Final Answer: [TP|FP1|FP2|FP3] - [brief reason]
```

### Feedback Agent Prompt (Simplified)
```prompt
You are a GEC judge agent that uses tools to evaluate grammatical corrections.

## AVAILABLE TOOLS
To use a tool, write exactly: Action: tool_name(arg1, arg2)

Tools:
- nonsense_detector(text) - Check if text is nonsensical
- meaning_change(original, suggested) - Get meaning change severity (0-4)
- reward_quality(original, suggested) - Get quality improvement (-3 to +3)
- grammar_rag(text, language) - Find relevant grammar rules
- comprehensive_analysis(original, suggested, language) - Full analysis

## CLASSIFICATION
**TP** - Fixes real errors without problems
**FP1** - Critical: Major meaning change or nonsense
**FP2** - Medium: Introduces grammar errors
**FP3** - Minor: Stylistic preference only

## YOUR TASK
Language: {language}
Original: "{original}"
Suggested: "{suggested}"

Use tools to analyze this edit, then provide your final answer.

Example tool usage:
Thought: I should check if this introduces nonsense.
Action: nonsense_detector(suggested text here)

After using tools, provide:
Final Answer: [TP/FP1/FP2/FP3] - [brief reason]

Begin:
```

## Key Differences Between Edit and Feedback

### Edit Judge
- More detailed prompt with full 6-class taxonomy (TP/FP1/FP2/FP3/TN/FN)
- Mentions edit operations and fused spans
- More formal classification framework
- Designed for edit-level evaluation

### Feedback Judge  
- Simplified 4-class taxonomy (TP/FP1/FP2/FP3)
- More concise prompts
- Focus on sentence-level feedback
- Cleaner, more straightforward instructions

## Modular Decision Logic (Both Judges)

```python
# Cascading classification based on scores:
if nonsense_score >= 2:
    → FP1 (Critical: Major nonsense)
elif meaning_change >= 2:
    → FP1 (Critical: Major meaning change)
elif quality_score < 0:
    → FP2 (Medium: Quality degradation)
elif 0 <= quality_score < 1:
    → FP3 (Minor: Minimal improvement)
else:  # quality_score >= 1
    → TP (Valid correction)
```

## Performance with Current Setup

Using `gas_gemini20_flash_lite` backbone:

| Judge | Method | Binary F1 | Notes |
|-------|--------|-----------|-------|
| Edit | Baseline | 79.5% | Best overall |
| Feedback | Baseline | 70.0% | Reliable |
| Feedback | Agent | 68.3% | After fixes |
| Edit | Agent | 64.1% | Good |
| Feedback | Modular | 62.5% | After prompt fix |
| Edit | Modular | 62.5% | Consistent |

### Why This Backbone?
1. **Cost-effective**: ~$0.70-1.45 per 10K requests
2. **Fast**: Quick inference for rapid iteration
3. **Sufficient quality**: 60-80% accuracy is good for structured tasks
4. **Scalable**: Can process large datasets without budget concerns

### For Production/Publication
Consider using higher-end models for final results:
- `o3-2025-04-16` for maximum accuracy (but expensive)
- `gpt-4.1` for good balance of quality and cost
- `gas_gemini20_flash` (full version) for better quality at reasonable cost




























