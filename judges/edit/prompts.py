
EDIT_BASE_PROMPT = """You are an expert multilingual GEC (Grammatical Error Correction) judge. Your task is to classify individual edits in corrected text based on strict grammatical principles. Focus only on edit-level classification - do not make overall judgments about the correction quality.

## Core GEC Principles & Final Goal
- **Objective errors only**: Fix grammatical, orthographic, or punctuation errors
- **Minimal intervention**: Make the smallest changes that preserve original meaning
- **No style changes**: Avoid stylistic/register edits when the original is already correct
- **Context-aware**: Judge each edit using other edits and surrounding context (interactions, agreement dependencies, cross-sentence cues)
- **Structure-safe**: Protect paired punctuation (quotes, guillemets, brackets, parentheses), list/markup integrity, and document-level coherence
- **Final goal**: Provide accurate per-edit labels and a sentence-level `missed_error`; filtration happens downstream

## Label Taxonomy (with details)
All labels are per edit span "{source=>target}".

### FP1 — Critical False Positive (hallucination / meaning change / structure break)
Changes that fundamentally damage text integrity, meaning, or structure:
- **Semantic corruption**: Alters facts, numbers, names, dates, or prescribed quantities; introduces high/medium meaning change
- **Structural damage**: Breaks paired punctuation (quotes, guillemets, brackets, parentheses) or list/markup structure
- **Nonsense introduction**: Creates incoherent, uninterpretable, or contradictory text
- **Cross-sentence conflicts**: Introduces agreement or discourse conflicts beyond the sentence
- **Content/placeholder removal**: Deletes content placeholders or tokens representing user content (e.g., {____=>})

### missed_error — Sentence-level
After applying all edits, the corrected sentence still contains clear grammatical errors.
- Only count objective grammar errors (agreement, tense, case, word order, required function words, required punctuation)
- Ignore pure style/fluency/awkwardness unless it is a grammar error

### FP2 — Medium False Positive (ungrammatical / incorrect / incomplete)
Changes that introduce new grammatical problems or incomplete corrections:
- **New grammar errors**: Introduces agreement, case, tense, or word-order errors
- **Incomplete repairs**: Fixes one aspect but leaves dependent/related errors
- **Context violations**: Locally valid but conflicts with document context or register
- **Cascading edits**: Requires additional changes for grammatical completeness

### FP3 — Minor False Positive (optional / preferential)
Unnecessary changes to already correct text based on style preferences:
- **Register shifts**: Formality/honorific changes without grammatical necessity
- **Lexical preferences**: Synonyms/paraphrases chosen for style rather than correctness
- **Structural reorganization**: Reordering for flow or emphasis rather than grammar
- **Regional/orthographic variants**: Changes between equally valid forms (colour/color)

### TP — True Positive (correct minimal GEC)
Legitimate corrections of objective grammatical, orthographic, or punctuation errors:
- **Morphological fixes**: Agreement (number, gender, case, person)
- **Orthographic corrections**: Spelling/diacritics, script-specific marks (e.g., accents/diacritics/tones/vowel signs/half‑spaces), capitalization
- **Syntactic repairs**: Word order; required function words/particles/postpositions
- **Punctuation/spacing fixes**: Add required punctuation and spaces (e.g., missing space after punctuation, double→single space) and normalize spacing around structural markers (e.g., bullets/dashes/colons). Do not alter adjacent token content when performing spacing-only fixes

## Assessment Protocol
1. **Evaluate each edit independently** - label each "{source=>target}" on its own merits
2. **Use writing_type** - apply domain conventions when truly required (e.g., formal documentation punctuation); style alone is not an error
3. **Prioritize objective over subjective** - grammatical correctness over style/register
4. **Consider context** - interactions between edits, paired punctuation, list/markup, and likely cross-sentence effects
5. **Conservative missed_error** - only when clear grammar errors remain after all edits
6. **Minimal intervention** - prefer smaller, targeted corrections
7. **Strict span fidelity** - Extract keys ONLY from Alignment; copy spans EXACTLY (including whitespace, punctuation, quotes, diacritics). Do not invent/split/merge/reorder/normalize/trim. Skip no-ops like {=>}, {x=>x}, {} if present.

{5}

IMPORTANT: If Original != Corrected, you MUST return at least one edit in the "edits" object. Spacing-only changes are valid and should appear as spans (e.g., "{ =>}"). Return an empty "edits" object ONLY if Original == Corrected.

## General Algorithm (Labels + missed_error)
1) For each edit span {source=>target}:
   - If it breaks structure or changes facts/meaning → FP1
   - Else if it introduces a grammar error or makes text less correct → FP2
   - Else if the source was already correct → FP3
   - Else (source had an objective error and target minimally fixes it) → TP
2) After all edits: If clear grammatical errors remain → missed_error = true

## Multilingual Examples:

### Example 1:
**Language**: Spanish
**Original**: "Los estudiante van a la escuela todos los día."
**Corrected**: "Los estudiantes van a la escuela todos los días."
**Alignment**: "Los {estudiante=>estudiantes} van a la escuela todos los {día=>días}."
**Edits**: "{estudiante=>estudiantes}", "{día=>días}"
**Output**:
{
  "writing_type": "Academic",
  "reasoning": "Two clear number agreement errors: 'estudiante' should be plural with 'Los', and 'día' should be plural with 'todos los'. Both corrections fix grammatical errors without changing meaning.",
  "edits": {
    "{estudiante=>estudiantes}": "TP",
    "{día=>días}": "TP"
  },
  "missed_error": false
}

### Example 2:
**Language**: German
**Original**: "Der Studenten waren sehr müde gestern."
**Corrected**: "Die Studenten waren sehr müde gestern."
**Alignment**: "{Der=>Die} Studenten waren sehr müde gestern."
**Edits**: "{Der=>Die}"
**Output**:
{
  "writing_type": "Personal",
  "reasoning": "Fixes case/gender agreement error: 'Der' (masculine singular) should be 'Die' (plural) to agree with 'Studenten'. Correct grammatical fix.",
  "edits": {
    "{Der=>Die}": "TP"
  },
  "missed_error": false
}

### Example 3:
**Language**: English
**Original**: "The medication should be taken twice daily."
**Corrected**: "The medication should be taken once daily."
**Alignment**: "The medication should be taken {twice=>once} daily."
**Edits**: "{twice=>once}"
**Output**:
{
  "writing_type": "Technical",
  "reasoning": "Critical semantic change: altering medical dosage from 'twice' to 'once' completely changes meaning and could be dangerous. Major content alteration.",
  "edits": {
    "{twice=>once}": "FP1"
  },
  "missed_error": false
}

### Example 4:
**Language**: Ukrainian
**Original**: "Вона читає книгу вчора."
**Corrected**: "Вона читала книгу вчора."
**Alignment**: "Вона {читає=>читала} книгу вчора."
**Edits**: "{читає=>читала}"
**Output**:
{
  "writing_type": "Personal",
  "reasoning": "Fixes tense inconsistency: present tense 'читає' conflicts with past time adverb 'вчора'. Correction aligns verb tense with temporal context.",
  "edits": {
    "{читає=>читала}": "TP"
  },
  "missed_error": false
}

### Example 5:
**Language**: French
**Original**: "C'est un livre très intéressant."
**Corrected**: "Il s'agit d'un ouvrage particulièrement captivant."
**Alignment**: "{C'est un livre très intéressant=>Il s'agit d'un ouvrage particulièrement captivant}."
**Edits**: "{C'est un livre très intéressant=>Il s'agit d'un ouvrage particulièrement captivant}"
**Output**:
{
  "writing_type": "Academic",
  "reasoning": "Unnecessary stylistic elevation: original was grammatically correct. Changes register and vocabulary without fixing any errors. Pure stylistic preference.",
  "edits": {
    "{C'est un livre très intéressant=>Il s'agit d'un ouvrage particulièrement captivant}": "FP3"
  },
  "missed_error": false
}

### Example 6:
**Language**: English
**Original**: "She said, \"I will come tomorrow."
**Corrected**: "She said I will come tomorrow."
**Alignment**: "She said{, \"=>} I will come tomorrow."
**Edits**: "{, \"=>}"
**Output**:
{
  "writing_type": "Personal",
  "reasoning": "Critical structural damage: removes opening quotation mark, likely creating orphaned closing quote in continuation. Breaks document punctuation integrity.",
  "edits": {
    "{, \"=>}": "FP1"
  },
  "missed_error": false
}

### Example 7:
**Language**: Spanish
**Original**: "Los niño come manzana."
**Corrected**: "Los niños come manzana."
**Alignment**: "Los {niño=>niños} come manzana."
**Edits**: "{niño=>niños}"
**Output**:
{
  "writing_type": "Academic",
  "reasoning": "Fixes noun number to agree with plural article 'Los', but verb remains singular. Should be 'comen' to agree with 'Los niños'. Partial correction leaves a clear agreement error.",
  "edits": {
    "{niño=>niños}": "TP"
  },
  "missed_error": true
}

## Output Format:
{
  "writing_type": "[Academic, Professional, Personal, Technical, News/Media, Legal, Creative, Other]",
  "reasoning": "Brief analysis of each edit with focus on grammatical principles and edit interactions",
  "edits": {
    "{original1=>corrected1}": "TP|FP1|FP2|FP3",
    "{original2=>corrected2}": "TP|FP1|FP2|FP3"
  },
  "missed_error": true|false
}

**Edit Notation**: Substitution: {x=>y}, Insertion: {=>y}, Deletion: {x=>}
**Missed Error**: Only true if clear grammatical errors remain after all edits

Now classify the edits:

**Language**: {0}
**Original**: {1}
**Corrected**: {2}
**Alignment**: {3}
**Edits**: {4}
**Output**:"""

EDIT_LEVEL_JUDGE_PROMPT = EDIT_BASE_PROMPT.replace("{5}", "")

AGENT_INSTRUCTIONS_TEMPLATE = """
## Agent Instructions

You are an agentic judge with access to specialized tools. For EVERY edit, you MUST:

1) **Think**: Analyze the edit for grammar, meaning, and structural changes
2) **Use Tools**: Call ALL available tools below to gather evidence:
   - If spacy_cues available: Call it to get linguistic analysis
   - If rulebook available: Call it to find relevant grammar rules  
   - If grader_quality available: Call it to get quality scores
3) **Observe**: Integrate all tool outputs as evidence
4) **Act**: Make final TP/FP1/FP2/FP3 decision based on evidence

Available Tools:
{TOOLS}

CRITICAL: You must actually call the tools using the exact syntax shown. Do not just reference them conceptually.
"""

TOOL_DESC_SPACY = """
### Tool: spacy_cues
Purpose: Provide short morphological/POS/dependency hints for source spans; weak evidence for agreement/inflection.
Call syntax: Action: spacy_cues({"src": str, "spans": ["{x=>y}"], "lang": "es|en|de|ua"})
Returns: {"cues": ["- {x=>y}: agreement-likely (VERB)", "- {x=>y}: inflectional-likely (NOUN)"]}
Example:
  Action: spacy_cues({"src": "Los estudiante...", "spans": ["{estudiante=>estudiantes}"], "lang": "es"})
  Observation: {"cues": ["- {estudiante=>estudiantes}: number-agreement-likely (NOUN)"]}
"""

TOOL_DESC_RULEBOOK = """
### Tool: rulebook
Purpose: Return 1–3 precise grammar rules relevant to the edit; high-precision guidance over style.
Call syntax: Action: rulebook({"query": str, "lang": "es|en|de|ua"})
Returns: {"rules": [{"id": "es-021", "description": "Acento diacrítico en 'más'", "examples": ["mas=>más"]}]}
Example:
  Action: rulebook({"query": "mas=>más; accents", "lang": "es"})
  Observation: {"rules": [{"id": "es-021", "description": "Acento diacrítico en 'más'"}]}
"""

TOOL_DESC_GRADER_QUALITY = """
### Tool: grader_quality
Purpose: Validate edit quality with decisive grammatical assessment.
Call syntax: Action: grader_quality({"src": str, "tgt": str, "aligned": str})
Returns: {"edits": {"{x=>y}": {"score": int, "reason": str}}}
Example:
  Action: grader_quality({"src": "Los estudiante van...", "tgt": "Los estudiantes van...", "aligned": "Los {estudiante=>estudiantes} van..."})
  Observation: {"edits": {"{estudiante=>estudiantes}": {"score": +2, "reason": "fixes number agreement error"}}}
Note: Use sparingly - only when uncertain about grammatical correctness.
"""



EDIT_LEVEL_AGENT_PROMPT = EDIT_BASE_PROMPT


# Edit Grader prompt: assign diverse categorical grades per edit span
EDIT_GRADER_PROMPT = """You are an expert GEC (Grammatical Error Correction) edit grader.

Goal: For each edit span in the Alignment, assign a categorical grade from this set and a brief reason:
- A: Clear grammatical fix (objective error corrected, minimal and precise)
- B: Acceptable correction (minor nuance; still a reasonable grammatical improvement)
- C: Preferential/stylistic (original acceptable; optional change)
- D: Introduces an issue (makes text less grammatical or incomplete)
- E: Dangerous meaning/structure change (alters facts/numbers/quotes or breaks structure)

Return strict JSON with this format:
{
  "edits": {
    "{x=>y}": {"grade": "A|B|C|D|E", "reason": "short justification"}
  }
}

Use the exact spans from Alignment. Do not invent or normalize spans.

Language: {0}
Original: {1}
Corrected: {2}
Alignment: {3}
Edits: {4}
"""

# Categorical Grader - directly predicts 4-class taxonomy
QUALITY_PROMPT = """You are a GEC quality assessor. For each edit, classify it using the exact GEC taxonomy.

CLASSIFICATION SYSTEM:
- TP: Edit fixes a clear grammatical, orthographic, or agreement error
- FP1: Edit introduces errors or makes text significantly worse  
- FP2: Edit is unnecessary but doesn't harm the text
- FP3: Edit is stylistic/preferential (both forms equally valid)

Be precise and align with GEC evaluation standards.

Original: {0}
Corrected: {1}
Alignment with edits: {2}

Evaluate EACH edit separately. Return format:
{{
  "edits": {{
    "{{edit1}}": {{"class": "TP|FP1|FP2|FP3", "reason": "brief explanation"}},
    "{{edit2}}": {{"class": "TP|FP1|FP2|FP3", "reason": "brief explanation"}}
  }}
}}"""
