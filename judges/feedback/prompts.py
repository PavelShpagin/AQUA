#!/usr/bin/env python3
"""
Feedback judge prompts (copied from legacy, no imports from legacy files).

These are full, verbatim prompts to ensure reproducibility.
"""

# Legacy TP/FP prompt (previous baseline). Used for --method legacy
TPFP_PROMPT_LEGACY = """
You are an Error Severity classifier tasked with classifying false positives (FPs) into specific error categories based on their severity. Follow the guidelines below to determine the appropriate category for each case.

Consider these error categories:

1. **Critical (FP1):**
   - These errors must be addressed immediately due to their severity.
   - Key questions to consider:
     - Does the suggestion create a risky sensitivity issue?
     - Does it cause a significant change in meaning?
     - Is the suggestion nonsensical?
   - **Examples**:
     - Sensitivity issues: "recognizing the annexation of {{crimea=>crime}}"
     - Major meaning change: "headaches on the scale {{2-3/10=>3/10}}"
     - Nonsensical: "{{Lisa leads=>led}} everyone in prayer."

2. **Medium (FP2):**
   - Errors are common but not as severe.
   - Key questions to consider:
     - Does the suggestion create a minor sensitivity issue?
     - Does it make the sentence less grammatical?
     - Does it cause a minor to medium change in meaning?
     - Is both the suggestion and the original incorrect?
   - **Examples**:
     - Minor sensitivity: "Sticking feathers up your {{ars=>arms}}"
     - Major grammar issues: "get {{genetic=>genetically}} testing"
     - Medium meaning change: "being rude {{didn't=>would not}}"
     - Both incorrect: "add this {{-=>--}}"

3. **Minor (FP3):**
   - Optional suggestions rather than errors.
   - Key questions to consider:
     - Is the suggestion as valid as the original?
     - Is the suggestion a minor improvement?
     - Is it a matter of personal preference?
   - **Examples**:
     - As valid: "autonomy {{principle=>principles}}"
     - Slight improvement: "iron content in {{foods=>food}}"
     - Preference: "He is {{not reachable=>unreachable}}"

4. **Not an error (TP):**
    - The suggestion is correct and it doesn't contain any errors.

# Steps

1. Analyze the false positive (FP) suggestion provided.
2. Use the key questions for each severity category to assess the FP.
3. Compare the impact of the suggestion with both the original text and the example scenarios.
4. Classify the FP into one of the four categories: Critical (FP1), Medium (FP2), Minor (FP3), Not an error (TP).

# Output Format

Provide the output in JSON format:
{
  "classification": "FP1/FP2/FP3/TP",
  "reason": "Detailed explanation of the classification decision.",
  "tags": ["lists of relevant tags related to the classification."],
  "type_of_writing": "The type of writing where the error happened."
}

Example 1:

Original Text: However, can I bring Colombian empanadas from the Colombian restaurant Mi Terra to the carne asada if I do make it?

Suggested Text: However, can I bring Colombian empanadas from the Mexican restaurant Mi Terra to the carne asada if I do make it?

Output:
{
  "classification": "FP1",
  "reason": "The suggestion changes the origin of the restaurant from 'Colombian' too 'Mexican', which is a significant change in meaning. This could lead to confusion or misrepresentation of the restaurant's cultural identity, making it a critical error.",
  "tags": ["significant meaning change", "cultural misrepresentation"],
  "type_of_writing": "Informal communication"
}

Example 2:

Original Text: I loves to eat pizza.

Suggested Text: I love to eat pizza.

Output:
{
  "classification": "TP",
  "reason": "The suggestion corrects a grammatical error in the original text. 'I loves' is incorrect because 'loves' is not the correct verb form for the first person singular. The correct form is 'I love.'",
  "tags": ["grammar", "verb agreement"],
  "type_of_writing": "Informal writing"
}

Example 3:

Original Text: "echo "doi:10.1146/annurev-virology-123456-890123" | g gandalf -v"

Suggested Text: "echo "doi:10.1146/Andrew-virology-123456-890123" | g gandalf -v"

Output:
{
  "classification": "FP1",
  "reason": "The suggestion changes 'annurev' to 'Andrew', which significantly alters the meaning of the text. 'annurev' is part of a DOI (Digital Object Identifier) and changing it to 'Andrew' makes the DOI invalid and nonsensical. This is a critical error as it affects the integrity of the reference.",
  "tags": ["nonsensical", "significant meaning change", "DOI"],
  "type_of_writing": "Technical/Academic Writing"
}

Example 4:

Original Text: NSWB members accompanied faculty advisor

Suggested Text: NSWB members' accompanied the faculty advisor

Output:
{
  "classification": "FP3",
  "reason": "The suggestion to add 'the' before 'faculty advisor' is a minor improvement. It is a matter of personal preference and does not significantly alter the meaning or grammatical correctness of the sentence. Both versions are valid.",
  "tags": ["minor improvement", "personal preference"],
  "type_of_writing": "Academic or organizational report"
}

Example 5:

Original Text: She decided to go to the store for groceries.

Suggested Text: She decided to go store for groceries.

Output:
{
  "classification": "FP2",
  "reason": "The suggestion 'go store' removes the preposition 'to', which is necessary for grammatical correctness in this context. This results in a major grammatical issue, as the sentence becomes ungrammatical without it. The error is common but not severe enough to cause a critical misunderstanding.",
  "tags": ["grammar", "preposition"],
   "type_of_writing": "Casual writing"
}

Note: The text being evaluated is in {0} language.

Original Text: {1}

Suggested Text: {2}

Output:
"""

# Simple output variant (laconic baseline)
TPFP_PROMPT_BASELINE = """
You are an Error Severity Classifier. Compare ORIGINAL and SUGGESTED and assign one label — FP1, FP2, FP3, or TP — based only on the specific edit (relative improvement). Ignore unrelated errors outside the edit.

Severity (relative to the edit)
- FP1 (Critical): Meaning/factual change, structural break, or nonsense introduced by the edit.
- FP2 (Medium): Edit introduces a grammatical/usage/orthographic error or removes a required function word; degrades correctness.
- FP3 (Minor): Both versions fully correct; change is stylistic/preferential; no clear clarity gain.
- TP (Not an error): Edit fixes a real error or clearly improves clarity without new errors.

Decision order: FP1 → FP2 → TP → FP3.

Explicit tests (apply quickly):
- FP1 triggers: proper noun change, number/date/unit change, factual/entity change, preposition meaning shift, unbalanced structure (quotes/brackets/tags), hallucination.
- FP2 triggers: missing/extra required function word; wrong agreement/tense/case; degraded spelling/casing mid-sentence; malformed punctuation that breaks a rule.
- FP3 triggers: both fully correct; optional style (Oxford comma, contractions, spacing/formatting), zero clarity gain.
- TP triggers: objective error fixed; clearer/more correct form without new errors.

Hard constraints:
- If ANY FP1 trigger is present → classify FP1 (do not choose TP/FP3).
- If ANY FP2 trigger is present (and not FP1) → classify FP2.
- Only classify FP3 when BOTH texts are 100% correct and the change is purely stylistic.
- Otherwise classify TP.

Punctuation decision rules (be strict):
- TP only when a clear rule is fixed (e.g., comma splice fixed with semicolon, balance of paired punctuation, mandatory comma in nonrestrictive clause explicitly marked).
- FP2 when punctuation is added/changed without a clear mandatory rule (e.g., optional commas before "but" when not joining independent clauses; stray commas in short phrases; list commas that are not required by a rule).
- FP3 for purely stylistic punctuation (Oxford comma in simple lists; contraction vs full form; colon vs period before a quote when both are acceptable).

Return strict JSON:
{
  "type_of_writing": "Academic/Research | Business/Professional | Personal/Casual | Technical/Documentation | Other",
  "reason": "Concise rationale focused on the edit",
  "classification": "FP1 / FP2 / FP3 / TP"
}

Examples

Example 1
Language: English
Original: He works with patients.
Suggested: He works on patients.
Aligned: He works {with=>on} patients.
Edit: {with=>on}
Output:
{
  "type_of_writing": "Other",
  "reason": "Preposition change shifts meaning (collaborate vs operate on).",
  "classification": "FP1"
}

Example 2
Language: Spanish
Original: Decidió ir a la biblioteca.
Suggested: Decidió ir biblioteca.
Aligned: Decidió ir {a la=>} biblioteca.
Edit: {a la=>}
Output:
{
  "type_of_writing": "Personal/Casual",
  "reason": "Elimina una preposición/determinante requerido; agramatical.",
  "classification": "FP2"
}

Example 3
Language: Ukrainian
Original: Ми купили яблука, апельсини і банани.
Suggested: Ми купили яблука, апельсини, і банани.
Aligned: Ми купили яблука, апельсини {і=>, і} банани.
Edit: {і=>, і}
Output:
{
  "type_of_writing": "Other",
  "reason": "Додає стилістичну кому; обидва варіанти прийнятні.",
  "classification": "FP3"
}

Example 4
Language: German
Original: Sie haben ein Fehler gemacht.
Suggested: Sie haben einen Fehler gemacht.
Aligned: Sie haben {ein=>einen} Fehler gemacht.
Edit: {ein=>einen}
Output:
{
  "type_of_writing": "Business/Professional",
  "reason": "Korrigiert Kasusübereinstimmung (Akkusativ).",
  "classification": "TP"
}

Example 5
Language: English
Original: We arrived late, the meeting had started.
Suggested: We arrived late; the meeting had started.
Aligned: We arrived late{,=>;} the meeting had started.
Edit: {,=>;}
Output:
{
  "type_of_writing": "Business/Professional",
  "reason": "Semicolon fixes comma splice between independent clauses.",
  "classification": "FP3"
}

Language: {0}
Original Text: {1}
Suggested Text: {2}
Aligned: {3}
Edit: {4}
Output:
"""

# Nonsense detection prompt per docs/general.md - returns YES/NO for nonsense detection
NONSENSE_PROMPT = """You are a nonsense detector for GEC evaluation. Rate how much the text introduces nonsense, structural incoherence, or makes the text uninterpretable.

SCORE SCALE:
-1: Reduced nonsense (text becomes clearer/more coherent)
 0: Neutral (no change in nonsense level)
 1: Slight nonsense (minor confusion or awkwardness)
 2: Medium nonsense (noticeable incoherence or confusion)
 3: Major nonsense (loss of information, syntax breaking, uninterpretable)

Text to evaluate:
{0}

Return format:
SCORE: [-1 to 3]
REASON: [brief explanation]"""


# Meaning change detection per docs/general.md - returns severity level 0-4
MEANING_CHANGE_PROMPT = """You are a meaning-change detector for GEC evaluation. Compare the original and suggested sentences and rate the meaning change severity.

SEVERITY LEVELS:
0 - No meaning change (grammatical corrections only)
1 - Minor clarification or style change (same core meaning)
2 - Noticeable but acceptable change (slight shift in emphasis/nuance)
3 - Significant meaning alteration (important information changed)
4 - Major meaning change or contradiction (completely different meaning)

ORIGINAL:
{0}

SUGGESTED:
{1}

Return format:
SEVERITY: [0-4]
REASONING: [explain the meaning change level]"""


# Reward model per docs/general.md - assigns relative improvement score -3 to +3
RELATIVE_REWARD_PROMPT = """You are a text quality assessor for GEC evaluation. Compare the original and suggested sentences and rate the relative improvement.

ORIGINAL:
{0}

SUGGESTED:
{1}

Rate the relative improvement on a scale from -3 to +3:
-3: Much worse (introduces major errors, significantly degrades quality)
-2: Worse (introduces noticeable errors, degrades quality)
-1: Slightly worse (minor degradation)
 0: No change in quality
+1: Slightly better (minor improvement)
+2: Better (fixes errors, improves clarity)
+3: Much better (significant improvement, major error fixes)

Return format:
IMPROVEMENT: [-3 to +3]
REASONING: [explain the relative improvement assessment]"""


AGENT_PROMPT = """You are a linguistic analyst specializing in grammatical error correction evaluation. Your task is to assess whether a suggested correction is appropriate and classify it according to established criteria.

## ANALYTICAL FRAMEWORK

**Classification System:**
- **TP (True Positive)**: Correction addresses a genuine grammatical error without introducing new issues
- **FP1 (Critical False Positive)**: Correction causes significant meaning alteration, introduces nonsense, or creates major structural problems  
- **FP2 (Medium False Positive)**: Correction introduces new grammatical errors or makes text less correct
- **FP3 (Minor False Positive)**: Correction changes already-correct text for stylistic reasons only

## AVAILABLE TOOL

**Advanced Grammar Analysis Tools:**

**Primary Tool - Enhanced Grammar RAG:**
Action: grammar_rag(query, language)

Use this for comprehensive grammar rule analysis. Query in the same language as the text for best results.

For Spanish: "concordancia de género y número", "uso de ser y estar", "acentuación", "subjuntivo vs indicativo"
For English: "subject-verb agreement", "pronoun case", "past participle forms", "comma usage"

**Specialized Tools (use when needed):**
- direct_rule_query(rule_name, language) - Get complete information about a specific rule
- category_search(category, language) - Find all rules in a category (grammar, orthography, punctuation)  
- error_focused_search(error_description, language) - Find rules based on error descriptions

**Analysis Strategy:**
1. First identify the type of error (grammar, spelling, punctuation, etc.)
2. Use grammar_rag with targeted queries
3. For complex cases, use specialized tools for deeper analysis
4. Always consider both correctness AND necessity of changes

## ANALYTICAL PROCESS

**Text Under Analysis:**
- Language: {language}
- Original: "{original}"
- Suggested: "{suggested}"
- Edit Alignment: {aligned}

**Methodology:**
1. **Parse the edit alignment** to identify specific changes made
2. **Assess necessity**: Does the original contain an actual error requiring correction?
3. **Query relevant grammar rules** using grammar_rag for specific linguistic phenomena involved
4. **Evaluate correction quality**: Does the suggested change follow established grammatical principles?
5. **Check for side effects**: Does the correction introduce new errors or unintended changes?

**Decision Framework:**
- If original has error AND correction fixes it properly → **TP**
- If correction alters meaning significantly or creates nonsense → **FP1** 
- If correction introduces new grammatical errors → **FP2**
- If original is correct but correction changes it stylistically → **FP3**

Provide your analysis using this format:
Thought: [Your reasoning about what to investigate]
Action: grammar_rag(specific_query, {language})
Observation: [Tool response]
[Additional Thought/Action cycles as needed]
Final Answer: [TP/FP1/FP2/FP3] - [Concise justification based on grammatical principles]

Begin your analysis:"""

# -------------------------------------------------------------
# Router + Binary Classifiers (High-quality English prompts)
# Inspired by TPFP_PROMPT_BASELINE and BEA-2019 guidelines
# -------------------------------------------------------------

ROUTER_TPFP3_VS_FP2FP1_PROMPT_EN = """
You are a "Routing Judge" for grammatical error correction (GEC). Output strictly JSON, no extra text.

Goal: Decide which bucket to send this edit to before detailed classification:
- GROUP_A = TP/FP3 (fixes a real error OR purely stylistic)
- GROUP_B = FP2/FP1 (introduces problems: grammar degradation OR severe meaning/factual issues)

Checklist (BEA-2019 aligned, TPFP baseline inspired):
1) FP1 tests (severe):
   - Proper noun/number/fact changes, preposition meaning shift, identity/sensitivity changes
   - Nonsense/incoherence/structure break (quotes/brackets)
2) FP2 tests (medium):
   - New grammar/usage/spelling errors introduced, wrong capitalization mid-sentence
3) TP tests:
   - Objective error in original is fixed (agreement, tense, articles, spelling, punctuation)
4) FP3 tests:
   - Both versions fully correct; change is style/preference only; no clarity gain

Routing rule:
- If ANY FP1 test is clearly triggered → GROUP_B
- Else if FP2 tests clearly triggered → GROUP_B
- Else → GROUP_A

Defaults:
- Be conservative; if unsure, choose GROUP_A (TP/FP3).

Return JSON only:
{
  "route": "TP_FP3" or "FP2_FP1",
  "confidence": 0.0-1.0,
  "type_of_writing": "Academic/Business/Personal/Technical/Other",
  "reason": "brief justification"
}

Original: {src}
Suggested: {tgt}
Aligned: {aligned}
"""

# -------------------------------------------------------------------
# Tagger prompt (features only; no classification in model output)
# -------------------------------------------------------------------
TAGGER_FEATURES_PROMPT = """
You are a multilingual GEC tagging assistant. Compare ORIGINAL and SUGGESTED focusing only on the edit span(s). Output STRICT JSON: a single object with boolean feature flags. Do not include any labels, summaries, or extra text. Only the JSON object. You MUST include ALL listed flags below as keys with true/false values (use false when not applicable). No additional keys.

Feature flags (50 keys; set true only if clearly present due to the edit):
- numbers_changed, range_collapsed, percent_changed, currency_changed, unit_changed, date_changed
- entity_changed, proper_noun_identity_changed, proper_noun_casing_only, brand_casing_standardized
- url_changed, code_fragment_changed
- preposition_semantic_shift, function_word_removed, function_word_added_required, article_usage_error
- subject_verb_agreement_error, noun_number_agreement_error, pronoun_reference_changed
- tense_shift_without_anchor, tense_shift_with_anchor, aspect_change, voice_change, polarity_flip, negation_introduced
- word_order_change, paraphrase_semantic, long_rewrite_risk, multi_edit_complexity
- spelling_error_introduced, diacritics_normalization, capitalization_mid_sentence
- punctuation_only_change, punctuation_style_only, punctuation_breaks_rule, comma_splice_fixed, run_on_fixed
- oxford_comma_added, oxford_comma_removed, list_format_changed
- quote_style_change, quote_balance_changed, bracket_balance_changed
- whitespace_only_change, hyphenation_change
- fixes_grammar, clarity_improved, both_correct, style_only, hallucination

Rules:
- Consider only the relative edit. Ignore unrelated errors outside the span(s).
- Be conservative: set a flag true only if the evidence is explicit in the edit.
- If uncertain, set false.

Return STRICT JSON only (must include ALL flags), e.g.:
{
  "numbers_changed": false,
  "entity_changed": false,
  "function_word_removed": true,
  "fixes_grammar": true,
  "clarity_improved": true
}

Examples

Example 1
Language: English
Original: working  Fintech
Suggested: working in Fintech
Aligned: working { =>in } Fintech
Edit: { =>in }
Output:
{
  "function_word_added_required": true,
  "fixes_grammar": true,
  "clarity_improved": true
}

Example 2
Language: Spanish
Original: Lo compré en ebay.
Suggested: Lo compré en eBay.
Aligned: Lo compré en {ebay=>eBay}.
Edit: {ebay=>eBay}
Output:
{
  "brand_casing_standardized": true,
  "both_correct": false,
  "fixes_grammar": true
}

Language: {0}
Original Text: {1}
Suggested Text: {2}
Aligned: {3}
Edit: {4}
Output:
"""

# -------------------------------------------------------------
# Spanish FP1 Binary Detector (Few-shot, aligned keys)
# -------------------------------------------------------------

FP1_BINARY_PROMPT_ES = """
Eres un clasificador de falsos positivos críticos (FP1) para correcciones de texto en Español.
Tu tarea: decidir si la SUGERENCIA es un **FP1** (crítico) o **NOT_FP1** (no crítico).

Definiciones (usa estas pruebas):
- FP1 (crítico):
  - Cambia hechos/propios/números (identidades, cantidades, fechas, unidades)
  - Cambia el significado central (preposiciones con sentido distinto)
  - Rompe la integridad estructural (comillas/pares, formato)
  - Alucinación: añade contenido no sustentado por el ORIGINAL
- NOT_FP1: cualquier caso que no cumpla claramente FP1 (correcciones válidas, estilo, errores menores)

Devuelve SOLO JSON con estas claves (en inglés):
{
  "classification": "FP1" o "NOT_FP1",
  "meaning_change_severity": 0-4,
  "hallucination": true/false,
  "structural_break": true/false,
  "reasoning": "breve justificación"
}

Contexto:
ORIGINAL: {src}
SUGGESTED: {tgt}
ALIGNED: {aligned}

Ejemplos:
1) FP1 (cambio factual — topónimo)
ORIGINAL: "El evento será en Córdoba."
SUGERENCIA: "El evento será en Granada."
ALIGNED: "en {{Córdoba=>Granada}}"
Respuesta JSON:
{
  "classification": "FP1",
  "meaning_change_severity": 3,
  "hallucination": false,
  "structural_break": false,
  "reasoning": "Se cambia el lugar del evento (hecho objetivo)."
}

2) FP1 (número/cantidad)
ORIGINAL: "Se vendieron 2–3 cajas."
SUGERENCIA: "Se vendieron 3 cajas."
ALIGNED: "{{2–3=>3}} cajas"
Respuesta JSON:
{
  "classification": "FP1",
  "meaning_change_severity": 3,
  "hallucination": false,
  "structural_break": false,
  "reasoning": "Corrupción numérica que altera la magnitud."
}

3) FP1 (preposición cambia sentido)
ORIGINAL: "Trabaja con niños."
SUGERENCIA: "Trabaja a niños."
ALIGNED: "Trabaja {{con=>a}} niños"
Respuesta JSON:
{
  "classification": "FP1",
  "meaning_change_severity": 4,
  "hallucination": false,
  "structural_break": false,
  "reasoning": "La preposición modifica el rol semántico del complemento."
}

4) NOT_FP1 (corrección gramatical)
ORIGINAL: "La empresa estan feliz."
SUGERENCIA: "La empresa está feliz."
ALIGNED: "La empresa {{estan=>está}} feliz"
Respuesta JSON:
{
  "classification": "NOT_FP1",
  "meaning_change_severity": 0,
  "hallucination": false,
  "structural_break": false,
  "reasoning": "Arreglo verbal; no introduce cambios de hecho ni de sentido."
}

5) NOT_FP1 (diacríticos/normalización)
ORIGINAL: "tecnica avanzada"
SUGERENCIA: "técnica avanzada"
ALIGNED: "{{tecnica=>técnica}} avanzada"
Respuesta JSON:
{
  "classification": "NOT_FP1",
  "meaning_change_severity": 0,
  "hallucination": false,
  "structural_break": false,
  "reasoning": "Mejora ortográfica; contenido semántico intacto."
}

6) FP1 (rotura estructural — comillas angulares)
ORIGINAL: "«Atención» urgente"
SUGERENCIA: "Atención» urgente"
ALIGNED: "{{«=>}}Atención» urgente"
Respuesta JSON:
{
  "classification": "FP1",
  "meaning_change_severity": 1,
  "hallucination": false,
  "structural_break": true,
  "reasoning": "Par de comillas desbalanceado; rompe integridad."
}

7) NOT_FP1 (formato)
ORIGINAL: "<em>Resumen</em> ejecutivo"
SUGERENCIA: "Resumen ejecutivo"
ALIGNED: "{{<em>=>}}Resumen ejecutivo{{</em>=>}}"
Respuesta JSON:
{
  "classification": "NOT_FP1",
  "meaning_change_severity": 0,
  "hallucination": false,
  "structural_break": false,
  "reasoning": "Elimina itálicas; no afecta el significado."
}
"""

TP_VS_FP3_PROMPT_EN = """
You are a binary classifier for GEC edits: decide between TP and FP3. Output strictly JSON.

Definitions (English, BEA-2019 aligned):
- TP (True Positive): Fixes a real error (agreement/tense/articles/prepositions/spelling/punctuation/word form); improves correctness.
- FP3 (Minor False Positive): Both original and suggestion fully correct; change is purely stylistic (Oxford comma, contractions, number format); no clarity gain.

Decision:
- DEFAULT → TP if any objective error is fixed.
- Only return FP3 when BOTH are 100% correct and the change is purely stylistic.

JSON only:
{
  "label": "TP" or "FP3",
  "confidence": 0.0-1.0,
  "type_of_writing": "Academic/Business/Personal/Technical/Other",
  "reason": "brief justification"
}

Original: {src}
Suggested: {tgt}
Aligned: {aligned}
"""

FP2_VS_FP1_PROMPT_EN = """
You are a binary classifier for GEC edits: decide between FP2 and FP1. Output strictly JSON.

Definitions (English, BEA-2019 aligned):
- FP1 (Critical False Positive): Severe issues — meaning/factual change, identity/sensitivity alteration, preposition meaning shift, nonsense/incoherence, structure break (quotes/brackets), numeric corruption.
- FP2 (Medium False Positive): Introduces grammatical/usage errors or degrades correctness without severe meaning/factual harm; wrong capitalization mid-sentence; minor meaning shift.

Decision:
- FP1 only when severe harm is clear.
- Otherwise FP2 if the sentence becomes less grammatical/correct than original.

JSON only:
{
  "label": "FP2" or "FP1",
  "confidence": 0.0-1.0,
  "type_of_writing": "Academic/Business/Personal/Technical/Other",
  "reason": "brief justification"
}

Original: {src}
Suggested: {tgt}
Aligned: {aligned}
"""


# (Removed legacy debate classifier prompt to simplify baseline usage)

# Shrinker Uncertainty Detection Prompt (enhanced based on TPFP_PROMPT_BASELINE)
SHRINKER_UNCERTAINTY_PROMPT = """
You are an **Error Severity Classifier** for grammatical error correction with uncertainty detection. Your task is to compare an original sentence and a suggested revision and provide your assessment with honest uncertainty quantification.

**CRITICAL INSTRUCTION**: Be honest about uncertainty. If you see ANY ambiguity between labels, provide ALL plausible labels as a JSON list. Only provide a single label if you are absolutely certain.

**BIAS TOWARD UNCERTAINTY**: It's better to escalate unclear cases than make wrong confident decisions.

> **Note:**
> - **TP** (Not an error) labels denote suggestions that **should** be made.  
> - **FP1/FP2/FP3** labels denote suggestions that **should not** be made.

---

## Severity Categories

1. **Not an error (TP)** - DEFAULT CLASSIFICATION
   - ANY correction that fixes a real error
   - Improves clarity, grammar, or spelling
   - Even minor improvements count as TP
   - **Examples:** "I loves"→"I love", "their are"→"there are", adding missing articles

2. **Critical (FP1)** - ONLY FOR SEVERE ISSUES
   - Changes factual content (proper nouns, numbers)
   - Alters core meaning of the sentence
   - **Examples ONLY:** "Colombian"→"Mexican", "2-3 million"→"3 million", "bye"→"buy"

3. **Medium (FP2)** - ONLY FOR NEW ERRORS
   - Introduces grammatical errors that weren't there
   - Makes sentence ungrammatical
   - **Examples ONLY:** "go to store"→"go store", "curious"→"Curious" (as adjective)

4. **Minor (FP3)** - RARE - BOTH MUST BE PERFECT
   - BOTH versions 100% grammatically correct
   - Zero improvement in clarity
   - **Example ONLY:** Oxford comma when truly optional

---

## How to Judge (Enhanced Methodology)

1. **Identify Writing Type**:
   - Academic/Research (formal, technical terms important)
   - Business/Professional (clarity critical)
   - Personal/Casual (informal ok)
   - Technical/Documentation (precision required)
   - Other

2. **Internal Debate** - Consider TWO perspectives:
   
   **Perspective A (Pro-Correction)**: 
   "This looks like it's fixing something..."
   - Is there a grammar/spelling error being corrected?
   - Does it improve clarity or formality?
   - Is the change appropriate for the writing type?
   
   **Perspective B (Anti-Correction)**: 
   "But wait, check if it causes problems..."
   - Does it change factual information?
   - Does it introduce new errors?
   - Is it unnecessary for this context?

3. **Apply Decision Tree** (after debate):

   **DEFAULT: Lean toward TP** - Most corrections are valid attempts to fix something.

   **Override to FP1 ONLY for SEVERE issues**:
   - Factual changes (proper nouns: Colombian→Mexican)
   - Statistical terms removed ("mean" deleted)
   - Meaning-changing preposition removal (drawing on→drawing)
   - Number corruption (2-3→3)
   - Word substitution that changes meaning (bye→buy)

   **Override to FP2 ONLY for NEW errors**:
   - Clear grammar mistakes introduced
   - Wrong capitalization (curious→Curious as adjective)

   **Override to FP3 ONLY when BOTH perfect**:
   - Both versions 100% correct
   - Pure style preference
   - No clarity improvement

   **If uncertain between multiple labels → Provide ALL plausible labels**

4. **UNCERTAINTY DETECTION TRIGGERS**:

   **ALWAYS provide multiple labels for these patterns:**
   - Academic citations (et al., periods) → Usually ["FP2", "FP3"] 
   - Capitalization (proper nouns vs common nouns) → Usually ["TP", "FP3"]
   - Word choice (jump in vs jump on) → Usually ["TP", "FP2", "FP3"]
   - Punctuation preferences (comma placement) → Usually ["FP2", "FP3"]
   - Style vs grammar (eBay vs ebay) → Usually ["TP", "FP3"]
   - Brand names and spellings → Usually ["TP", "FP3"]
   - Minor punctuation changes → Usually ["FP2", "FP3"]
   - Sentence structure variations → Usually ["TP", "FP2"]

5. **Output** your judgment in JSON with required fields:

{{
  "type_of_writing": "Academic/Business/Personal/Technical/Other",
  "debate": "Pro: [why correction seems valid] | Con: [any issues found]",
  "labels": ["TP"] or ["TP", "FP3"] or ["FP2", "FP3"],
  "confidence": "high/medium/low",
  "reasoning": "Final decision based on debate and uncertainty factors"
}}

## Examples (Multilingual)

**Example 1 (English, Uncertainty - Academic Citation)**

Original: LaGuardia, Z., et al., Comparison of mineral transformation...
Suggested: LaGuardia, Z., et al.., Comparison of mineral transformation...
Aligned Changes: LaGuardia, Z., et {{al.,=>al..,}} Comparison...

Output:
{{
  "type_of_writing": "Academic",
  "debate": "Pro: Could be fixing citation format | Con: Adds extra period which is incorrect",
  "labels": ["FP2", "FP3"],
  "confidence": "medium",
  "reasoning": "Academic citation pattern - uncertain between low-value error (FP2) and style preference (FP3)"
}}

**Example 2 (English, Uncertainty - Brand Name)**

Original: I bought it on ebay
Suggested: I bought it on eBay
Aligned Changes: I bought it on {{ebay=>eBay}}

Output:
{{
  "type_of_writing": "Personal",
  "debate": "Pro: Corrects proper noun capitalization | Con: Both forms widely used",
  "labels": ["TP", "FP3"],
  "confidence": "medium",
  "reasoning": "Brand name capitalization - uncertain between correction (TP) and style preference (FP3)"
}}

**Example 3 (English, Certain - Clear Grammar Fix)**

Original: I loves to eat pizza.
Suggested: I love to eat pizza.
Aligned Changes: I love{{s=>}} to eat pizza.

Output:
{{
  "type_of_writing": "Personal",
  "debate": "Pro: Fixes subject-verb agreement | Con: None",
  "labels": ["TP"],
  "confidence": "high",
  "reasoning": "Clear grammatical error correction - subject-verb agreement fix"
}}

## Important Notes

- **Be honest about uncertainty** - providing multiple labels when unsure is valuable for expert resolution
- **A suggestion can be incomplete**, although if it is not FP1, FP2, or FP3, it should be classified as TP
- Please make sure your **reasoning is fully coherent with your labels**
- Original and suggested sentences are **different**

Note: The text being evaluated is in {0} language.

Original Text: {1}

Suggested Text: {2}

Aligned Changes: {3}

Output:
"""

# Shrinker Resolution Prompt (enhanced for large model when small model is uncertain)
SHRINKER_RESOLUTION_PROMPT = """
You are an expert **Error Severity Classifier** for grammatical error correction. A smaller model is uncertain between these labels: {0}

**CRITICAL INSTRUCTION**: Pick the SINGLE most accurate label from the candidates provided by the smaller model. Use your expert knowledge to resolve the uncertainty.

**DEFAULT: Lean toward TP** - Most corrections are valid attempts to fix something.

> **Note:**
> - **TP** (Not an error) labels denote suggestions that **should** be made.  
> - **FP1/FP2/FP3** labels denote suggestions that **should not** be made.

---

## Severity Categories (Expert Level)

1. **Not an error (TP)** - DEFAULT CLASSIFICATION
   - ANY correction that fixes a real error
   - Improves clarity, grammar, or spelling
   - Even minor improvements count as TP
   - **Examples:** "I loves"→"I love", "their are"→"there are", adding missing articles

2. **Critical (FP1)** - ONLY FOR SEVERE ISSUES
   - Changes factual content (proper nouns, numbers)
   - Alters core meaning of the sentence
   - **Examples ONLY:** "Colombian"→"Mexican", "2-3 million"→"3 million", "bye"→"buy"

3. **Medium (FP2)** - ONLY FOR NEW ERRORS
   - Introduces grammatical errors that weren't there
   - Makes sentence ungrammatical
   - **Examples ONLY:** "go to store"→"go store", "curious"→"Curious" (as adjective)

4. **Minor (FP3)** - RARE - BOTH MUST BE PERFECT
   - BOTH versions 100% grammatically correct
   - Zero improvement in clarity
   - **Example ONLY:** Oxford comma when truly optional

---

## Expert Resolution Strategy

**Candidate Labels**: {0}

**Your expert analysis should consider:**

1. **Writing Type Context**:
   - Academic/Research (formal, technical terms important)
   - Business/Professional (clarity critical)
   - Personal/Casual (informal ok)
   - Technical/Documentation (precision required)

2. **Pattern-Specific Expert Knowledge**:
   - **Academic citations (et al., periods)**: Extra periods in "et al.." are incorrect → FP2
   - **Proper noun capitalization (eBay, iPhone)**: Standard brand spellings are corrections → TP
   - **Word choice variants (jump in/on, different/various)**: Context-dependent, usually FP3 if both acceptable
   - **Punctuation preferences**: TP if fixes grammar rule, FP3 if pure style
   - **Article usage**: Missing articles are usually errors → TP when added
   - **Subject-verb agreement**: Always errors when wrong → TP when fixed

3. **Expert Decision Framework**:
   - **If TP is a candidate**: Choose TP unless there's clear harm
   - **If choosing between FP2/FP3**: FP2 if introduces errors, FP3 if both correct
   - **If FP1 is a candidate**: Only choose if meaning/facts clearly changed

**Output** your expert decision in JSON:

{{
  "type_of_writing": "Academic/Business/Personal/Technical/Other",
  "expert_analysis": "Your expert reasoning for the specific pattern",
  "final_label": "TP/FP1/FP2/FP3",
  "reasoning": "Why this label is most accurate among the candidates"
}}

## Expert Examples

**Example 1: Academic Citation Resolution**
Candidates: ["FP2", "FP3"]
Original: LaGuardia, Z., et al., Comparison...
Suggested: LaGuardia, Z., et al.., Comparison...

Expert Decision:
{{
  "type_of_writing": "Academic",
  "expert_analysis": "Adding extra period after 'et al.' violates standard citation format",
  "final_label": "FP2",
  "reasoning": "Creates incorrect citation format - academic writing requires precision"
}}

**Example 2: Brand Name Resolution**
Candidates: ["TP", "FP3"]
Original: I bought it on ebay
Suggested: I bought it on eBay

Expert Decision:
{{
  "type_of_writing": "Personal",
  "expert_analysis": "eBay is the official brand spelling with specific capitalization",
  "final_label": "TP",
  "reasoning": "Corrects proper noun to standard brand spelling"
}}

Note: The text being evaluated is in {1} language.

Original Text: {2}

Suggested Text: {3}

Aligned Changes: {4}

Output:
"""
# Alias for simple prompt (same as baseline for now to fix import)
TPFP_PROMPT_BASELINE_SIMPLE = TPFP_PROMPT_BASELINE
