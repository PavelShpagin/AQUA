#!/usr/bin/env python3
"""
Ensemble-specific prompts for final judgment in iter_critic.

These prompts are used by the iter_critic ensemble to make final decisions
based on previous judges' opinions.
"""

# Final judgment prompt for 4-class feedback evaluation (TP/FP1/FP2/FP3)
FEEDBACK_FINAL_JUDGMENT_PROMPT = """
You are a final judgment classifier for feedback evaluation. Based on the previous judges' opinions below, make the final classification decision.

Classify into one of these categories:
- **TP**: True Positive (correction is valid and beneficial)
- **FP1**: Critical False Positive (major meaning change, sensitivity risk, or nonsensical)
- **FP2**: Medium False Positive (introduces grammatical error or minor meaning change)  
- **FP3**: Minor False Positive (stylistic preference, both versions valid)

Previous judges' opinions:
{0}

Original Text: {1}
Suggested Text: {2}

Provide your classification in JSON format:
{{
  "classification": "TP|FP1|FP2|FP3",
  "reasoning": "Detailed explanation of your final decision based on the judges' opinions."
}}"""

# Final judgment prompt for 6-class sentence/edit evaluation (TP/FP1/FP2/FP3/TN/FN)
SENTENCE_FINAL_JUDGMENT_PROMPT = """
You are a final judgment classifier for sentence-level evaluation. Based on the previous judges' opinions below, make the final classification decision.

Classify into one of these categories:
- **TP**: True Positive (correction is valid and beneficial)
- **FP1**: Critical False Positive (major meaning change, sensitivity risk, or nonsensical)
- **FP2**: Medium False Positive (introduces grammatical error or minor meaning change)  
- **FP3**: Minor False Positive (stylistic preference, both versions valid)
- **TN**: True Negative (no correction needed, correctly unchanged)
- **FN**: False Negative (correction was needed but not applied)

Previous judges' opinions:
{0}

Original Text: {1}
Suggested Text: {2}

Provide your classification in JSON format:
{{
  "classification": "TP|FP1|FP2|FP3|TN|FN",
  "reasoning": "Detailed explanation of your final decision based on the judges' opinions."
}}"""

# Final judgment prompt for edit-level evaluation (TP/FP1/FP2/FP3/TN/FN)
EDIT_FINAL_JUDGMENT_PROMPT = """
You are a final judgment classifier for edit-level evaluation. Based on the previous judges' opinions below, make the final classification decision.

Classify into one of these categories:
- **TP**: True Positive (edit is valid and beneficial)
- **FP1**: Critical False Positive (major meaning change, sensitivity risk, or nonsensical)
- **FP2**: Medium False Positive (introduces grammatical error or minor meaning change)  
- **FP3**: Minor False Positive (stylistic preference, both versions valid)
- **TN**: True Negative (no edit needed, correctly unchanged)
- **FN**: False Negative (edit was needed but not applied)

Previous judges' opinions:
{0}

Original Text: {1}
Suggested Text: {2}

Provide your classification in JSON format:
{{
  "classification": "TP|FP1|FP2|FP3|TN|FN",
  "reasoning": "Detailed explanation of your final decision based on the judges' opinions."
}}"""

# Inner Debate prompts for debate-style ensemble judgments
FEEDBACK_DEBATE_PROMPT = """
You are a final judgment classifier for feedback evaluation. You will see a debate between the two most dominant opinion classes, then make the final decision.

Classify into one of these categories:
- **TP**: True Positive (correction is valid and beneficial)
- **FP1**: Critical False Positive (major meaning change, sensitivity risk, or nonsensical)
- **FP2**: Medium False Positive (introduces grammatical error or minor meaning change)  
- **FP3**: Minor False Positive (stylistic preference, both versions valid)

Debate between judges:
{0}

Original Text: {1}
Suggested Text: {2}

After carefully considering both sides of the debate, provide your final classification in JSON format:
{{
  "classification": "TP|FP1|FP2|FP3",
  "reasoning": "Your final decision after weighing the debate arguments. Explain which side was more convincing and why."
}}"""

SENTENCE_DEBATE_PROMPT = """
You are a final judgment classifier for sentence-level evaluation. You will see a debate between the two most dominant opinion classes, then make the final decision.

Classify into one of these categories:
- **TP**: True Positive (correction is valid and beneficial)
- **FP1**: Critical False Positive (major meaning change, sensitivity risk, or nonsensical)
- **FP2**: Medium False Positive (introduces grammatical error or minor meaning change)  
- **FP3**: Minor False Positive (stylistic preference, both versions valid)
- **TN**: True Negative (no correction needed, correctly unchanged)
- **FN**: False Negative (correction was needed but not applied)

Debate between judges:
{0}

Original Text: {1}
Suggested Text: {2}

After carefully considering both sides of the debate, provide your final classification in JSON format:
{{
  "classification": "TP|FP1|FP2|FP3|TN|FN",
  "reasoning": "Your final decision after weighing the debate arguments. Explain which side was more convincing and why."
}}"""

EDIT_DEBATE_PROMPT = """
You are a final judgment classifier for edit-level evaluation. You will see a debate between the two most dominant opinion classes, then make the final decision.

Classify into one of these categories:
- **TP**: True Positive (edit is valid and beneficial)
- **FP1**: Critical False Positive (major meaning change, sensitivity risk, or nonsensical)
- **FP2**: Medium False Positive (introduces grammatical error or minor meaning change)  
- **FP3**: Minor False Positive (stylistic preference, both versions valid)
- **TN**: True Negative (no edit needed, correctly unchanged)
- **FN**: False Negative (edit was needed but not applied)

Debate between judges:
{0}

Original Text: {1}
Suggested Text: {2}

After carefully considering both sides of the debate, provide your final classification in JSON format:
{{
  "classification": "TP|FP1|FP2|FP3|TN|FN",
  "reasoning": "Your final decision after weighing the debate arguments. Explain which side was more convincing and why."
}}"""

# Language-agnostic escalation router (model-driven)
ESCALATION_ROUTER_PROMPT_GENERIC = """
You are an escalation router for a severity classifier (TP/FP1/FP2/FP3). Decide whether to escalate this case to a stronger expert.

Available experts (in increasing capability): {experts_list}

Return strict JSON only:
{{
  "escalate": true|false,
  "expert": "none" | {experts_json},
  "confidence_bucket": "vlow" | "low" | "medium" | "high" | "vhigh",
  "reason": "brief"
}}

Guidance (language-agnostic):
- Escalate when classification is uncertain/ambiguous or potentially harmful if wrong.
- Buckets: vlow=very uncertain; low=uncertain; medium=borderline; high=confident; vhigh=very confident.
- Prefer higher-tier experts for potentially critical/hard cases; otherwise choose the lowest capable expert.
- If the case is clear and non-risky, set escalate=false and expert="none".

Context:
Small model label: {small_label}
Small model rationale: {small_reason}
Original: {src}
Suggested: {tgt}
Aligned: {aligned}
"""

# Combined classify + route for small model (single call)
COMBINED_CLASSIFY_ROUTE_PROMPT = """
You are a severity classifier and router. Classify the edit and decide whether to escalate to an expert.

Return strict JSON only:
{{
  "classification": "TP" | "FP1" | "FP2" | "FP3",
  "escalate": true | false,
  "expert": "none" | "gpt-4o" | "o3",
  "confidence_bucket": "vlow" | "low" | "medium" | "high" | "vhigh",
  "reason": "brief"
}}

Guidance:
- Classification is relative to the edit.
- Escalate when uncertain (vlow/low/medium) or when label is FP1/FP2.
- Choose the lowest capable expert (gpt-4o) unless the case is very hard.

If you decide to escalate, do NOT output the final classification yet; defer to the expert.

Original: {src}
Suggested: {tgt}
Aligned: {aligned}
"""

# Small-model K-votes prompt for consensus (iter_escalation)
FEEDBACK_VOTES_PROMPT = """
You are a severity classifier (TP/FP1/FP2/FP3). Produce N votes (labels) for the case.

Return strict JSON only:
{{
  "votes": ["TP"|"FP1"|"FP2"|"FP3", ...],
  "reason": "brief"
}}

Guidance:
- Vote independently per slot; it's OK if votes repeat.
- If the edit clearly fixes an error → TP.
- If major meaning change/sensitivity → FP1.
- If introduces an error/minor meaning change → FP2.
- If optional preference/correct→correct → FP3.

Context:
Original: {src}
Suggested: {tgt}
Aligned: {aligned}
N: {n_votes}
"""

# Lightweight pair correctness (ensemble gating)
ENSEMBLE_PAIR_CORRECTNESS_PROMPT = """
You are a categorical correctness judge for a single edit. Decide grammatical correctness of ORIGINAL and SUGGESTED independently (ignore style).

Return strict JSON only:
{{
  "source_correct": true|false,
  "target_correct": true|false,
  "reason": "brief"
}}

Original: {src}
Suggested: {tgt}
Aligned: {aligned}
"""

# Mixture-of-Experts router (chooses one expert only)
MOE_ROUTER_PROMPT = """
You are a routing controller for a severity classifier (TP/FP1/FP2/FP3). Choose exactly one target expert to run.

Available experts (in increasing capability): {experts_list}

Return strict JSON only:
{{
  "expert": {experts_json},
  "confidence_bucket": "vlow" | "low" | "medium" | "high" | "vhigh",
  "reason": "brief"
}}

Guidance:
- Pick the lowest capable expert likely to classify correctly; escalate only when needed.
- Prefer "Expert" (mid) over "Senior"/"Principal" for borderline cases.

Context:
Original: {src}
Suggested: {tgt}
Aligned: {aligned}
"""

# Lightweight relative-improvement (ensemble gating)
ENSEMBLE_REWARD_PROMPT = """
You are a categorical relative-improvement judge. Rate how the edit changes grammatical quality.

Return strict JSON only:
{{
  "improvement": -1 | 0 | 1,
  "reason": "brief"
}}

Guidance:
- -1: worse or breaks grammar; 0: neutral/style-only; 1: fixes a real error.

Original: {src}
Suggested: {tgt}
Aligned: {aligned}
"""