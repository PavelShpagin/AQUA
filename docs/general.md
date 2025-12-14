## FP1 Binary Experiment (SpanishFPs.csv)

We added `experiments/FP1/run_fp1.py` to evaluate FP1 detection as a binary task on `data/eval/SpanishFPs.csv`.

Label mapping used:
- FP1 → good (positive = 1)
- TP/FP3/FP2/TN/FN → bad (negative = 0)

Signals combined (minimal, fast):
- ERRANT alignment (spaCy + English ERRANT over Spanish spaCy) with compact heuristics:
  - number/currency changes, proper-noun changes, large insertions
- Structural integrity check (balanced pairs) as a deterministic FP1 trigger
- Lightweight probes on meaning-change severity and hallucination (low-temperature, Gemini 2.0 Flash Lite)

Run:
```bash
python3 experiments/FP1/run_fp1.py \
  --input data/eval/SpanishFPs.csv \
  --output experiments/FP1/results_fp1.csv \
  --backend gas_gemini20_flash_lite \
  --workers 64
```

Output includes per-row predictions and pricing tokens/costs for reproducibility.

# AQUA: Automated Quality Analyzer for Multilingual GEC

AQUA is a modular LLM-powered framework to automatically evaluate, triage, and act on Grammatical Error Correction (GEC) suggestions across any language.

## Problem

Build a reliable LLM-based judge system to vet GEC suggestions on a fixed dataset. This is done by assigning TP/FP3/FP2/FP1 labels, with TP - what Grammarly should suggest, and FP3/FP2/FP1, what should be rejected. Note that GEC suggestions are single-edit level; however, future steps will extend the framework to full sentence-level GEC, and a filter-based approach to generate high-quality data for a MultiGEC system training data.

## Label Taxonomy

Labels fall into TP/FP3/FP2/FP1 categories, with FP3/FP2/FP1 listed in ascending order with respect to error severity. Specifically, for FPs, FP3 denotes minor FP, FP2 - medium FP, and FP1 - critical FP.

### Detailed Taxonomy:

Overall, the original and corrected texts differ by a single edit, which can independently improve or worsen the grammatical correctness of the text. We consider four classes for this problem:

- **TP** - The suggestion strictly improves the grammatical correctness of the text, even if some errors remain.
- **FP3** - The suggestion is optional, a matter of stylistic preference, or a correction of a correct clause.
- **FP2** - The suggestion introduces a grammatical/punctuation error, a slight meaning change, or a slight sensitivity introduction.
- **FP1** - The suggestion introduces high sensitivity, or meaning-changing GEC.

## Methods

### Ensemble Judging

**Baseline prompt:**
Given a starting TP/FP3/FP2/FP1 prompt and 3-iteration weighted label aggregation, using o3-2025-04-16 yields best results among other tested models; however, it still struggles considerably with TP/FP3 differentiation.

**Improved prompt:**
The refined prompt provides cleaner instructions, with guidance on using citations and applying rules leveraging domain dependence. Using o3-2025-04-16 gives a 5% increase in simple accuracy, and a 30-40% boost in accuracy for TP/FP and TP/FP3/FP2/FP1 classifications. Additionally, Precision, Recall, and F1 scores are consistently increased by 15-30%. Moreover, using a separate domain classification model gives an additional 2-3% accuracy boost.

**Iterative Critic ensemble:**
We invented three variations of the ensemble algorithm, specifically the IterEnsemble, CriticEnsemble, and IterCriticEnsemble, with the latter yielding the best consistent results. Specifically, IterCriticEnsemble gives a 7% accuracy boost in 4-class fine-grained classification, 6% increase in binary TP/FP, and 2% increase in the simple TP/FP scores. Pseudo-algorithm of IterCriticEnsemble is outlined below:

```
Input: n - starting number of judges, max_n - maximum number of judges, M - LLM judge model.

nonadj(label):
    // return a non adjacent label in the list [TP, FP3, FP2, FP1], e.g. nonadj(TP) = [FP2, FP1]

check_consensus([r_i, c_i, i=1, ..., k]):  // r_i - reasoning, c_i - class
    // insert c_i into hashmap H, counting its number of occurences
    // take the most frequent class c* from H
    // if both hold:
    // -- if c* frequency in the list is ≥ 2/3
    // -- for each class in nonadj(c*), if its frequency is ≤ 1/4
    // then, return c*
    // otherwise, return NO

IterCriticEnsemble(x, S):
    L ~ [M(x, S), ..., M(x, S)]  // n LLM calls
    cons = check_consensus (L)
    for i=max_n, ..., n do {
        if cons != NO:
            break
        L.append(M(x, S, L[-1], ..., L[-n]))
        cons = check_consensus (L)
    }
    // take 2 most frequent classes from L - c1 and c2
    result = M(c1, c2, x, S)  // final judgement
    return result
```

**Modular ensemble:**
This new method involves a set of separate specialized models, each dedicated to a subtask, which, in composition, will produce a final label. Models can be implemented either by defining specialized prompts or fine-tuning via QLORA on target datasets for each language. Thus, the ensemble has the following structure:

#### Models:

**Reward model:**
Assigns a score from -3 to 3 based on the relative quality of the sentence, with the higher score being corrected, the better. In the future, it can be trained using RLHF on binary sentence or edit-level quality labels

**Meaning-change model:**
Given a source and target sentence, this model assigns an integer label from 0 to 4 based on the meaning-changing degree. If an actual error was fixed but there was a slight meaning change, score it as 0 or 1 at most.

**Nonsense detector:**
Given a GEC correction, returns a score from -1 to 3 on how much the model introduces nonsense, -1 being reduced nonsense, 0 - neutral, 1 - slight nonsense, 2 - medium nonsense, 3 - major nonsense/loss of information/syntax breaking.

#### Algorithm:

**Outline and intuition:**
The algorithm takes in a user sentence, its GEC correction, and sequentially classifies it into FP1, FP2, FP3, TP using models described in the previous section and guidelines outlined in https://docs.google.com/document/d/1SXt15D8XCyBYb7P5_tqwZgH1tlHrYGwGQ9oisw8dmaM/edit?tab=t.0

Denote input text as x and GEC suggestion as S. Let x{S} denote corrected x using S. Also, we denote the full prompt as P=x⊕S for simplicity.

**Check FP1:**
- Check if similarity is significantly altered using (x, x{S}) ≥ 2
- Detect nonsense by applying (P) ≥ 2.
- Scores from each step can be aggregated to form a decision to classify as FP1 or move on to the next check.

**Check FP2:**
- Examine P for low-to-medium risk and sensitive content using the score from (P).
- Check if similarity is slightly altered using (x, x{S}) ≥ 2
- Detect nonsense by applying (P) ≥ 0.
- If the correction was incorrect (x, x{S}) < 0.

**Check FP3:**
- To cover preferential, low-value, and correct=>correct GEC, perform a simple 0 < R(x, x{S}) < 1 check.

**TP:**
- If all of the checks are passed, we classify P as TP.

**Notes:**
Due to the high degree of specialization, each model can be small enough to drastically reduce its footprint and cost as opposed to using a single large "universal" LLMs. In terms of processing speed, each "subcheck" of each FP1, FP2, FP3 check is independent and thus can be easily parallelized.

**Agent-as-a-judge:**
This method combines the previous techniques by building a agent with tools, that can be helpful in edge-cases and where other approaches fail. Specifically, the agent is equipped with the following tools:
- Language rulebooks with grammar rules (e.g. RAG over the language rulebook/convention details)
- Guidelines, notes, and a more general knowledge book
- Web search for domain knowledge
- Meaning-change tool
- Nonsense detector
- Reward/quality tool
- TNFN classifier

**Note:**

**Edit Judge:**
Has the same prompts as a sentence-level judge; however, it assesses the meaning-changing, nonsense, and reward score for each edit separately (so each model returns a score for each edit, which are then aggregated at each edit level to determine edit labels).

## Calibration & Metrics

### Scalable "gold" data generation approach:

**Raw data:** Each starting record consists of a text and multiple suggestions made by the model, each of which is rated as terrible, bad, optional, good, excellent. There are 3 annotators per record.

**Processing:**
1) First, for each record, keep it, if the language of its text is target_lang, and unannotable=false, sensitive=false. Additionally, if input text after GEC remains unchanged, it's omitted as well.
2) Then, for each sentence, for each model edit, if it has 3/3 suggestion-rating agreement among annotators, consider these cases:
   - If edit is 3/3 consistently suggestion-rating=terrible, map it to FP1. Otherwise, if all annotators agree on terrible / bad, and at least one annotator indicates meaning-changed=true, map it to FP1 as well.
   - If edit is 3/3 consistently suggestion-rating=bad, map it to FP2.
   - If edit is 3/3 consistently suggestion-rating=optional, map it to FP3.
   - If the edit is 3/3 consistently a good / excellent, while other edits in the same sentence are voted by each annotator to be terrible / bad / optional, map to TP.

**Notes:**
- Empirically, this approach leaves only ~50% of the raw ~7K data records.
- After the automatic processing step, the data correctness is curated by a human to ensure quality.

### Benchmarks on human-curated data:

**Evaluation schemes:**
- Full TP/FP3/FP2/FP1 eval, where we calculate the standard accuracy of multiclass classification, as well as averaged Precision, Recall, and F1 metrics across each class.
- TP/FP eval, for which we explicitly convert FP3→FP, FP1→FP, FP2→FP; we can evaluate a few algorithms.
- Simple TP/FP eval, for which we explicitly convert FP3→TP, FP1→FP, FP2→FP; we can evaluate a few algorithms.

Full evaluation scores are reported in the tables below:

#### TP/FP3/FP2/FP1 Judge Evaluation (Averages)

| Methods | Accuracy | Precision | Recall | F1 |
|---------|----------|-----------|--------|-----|
| Baseline prompt | 0.56 | 0.62 | 0.75 | 0.55 |
| Improved prompt | 0.86 | 0.717 | 0.694 | 0.645 |
| Iterative critic | 0.93 | 0.755 | 0.781 | 0.743 |

#### TP/FP Judge Evaluation

| Methods | Accuracy | Precision | Recall | F1 |
|---------|----------|-----------|--------|-----|
| Baseline prompt | 0.58 | 0.61 | 0.75 | 0.53 |
| Improved prompt | 0.9 | 0.783 | 0.906 | 0.826 |
| Iterative critic | 0.95 | 0.865 | 0.931 | 0.894 |

#### Simple TP/FP Judge Evaluation

| Methods | Accuracy | Precision | Recall | F1 |
|---------|----------|-----------|--------|-----|
| Baseline prompt | 0.91 | 0.55 | 0.8 | 0.65 |
| Improved prompt | 0.95 | 0.9 | 0.69 | 0.78 |
| Iterative critic | 0.96 | 0.84 | 0.808 | 0.824 |

## Final TP/FP3/FP2/FP1/TN/FN system

### Models:
- TN/FN system M1
- TP/FP3/FP2/FP1 system M2

### Algorithm(x, x{S}):
```
if x == x{S}:
    return M1(x, x{S})
else:
    if M1(x{S}, x{S}) == FN:  // incomplete GEC
        c = M2(x, x{S})
        if c == TP:
            return FN
        else:
            return c
    else:
        return M2(x, x{S})  // complete GEC
```

Evaluation scores of the TN/FN classifier with a final prompt for JFLEG and COWS-L2H are reported in the two tables below:

#### TN/FN Judge Evaluation

| Models | Accuracy |
|--------|----------|
| gpt-4o-mini | 0.84 |
| o3 | 0.84 |

#### Spanish TN/FN Judge Evaluation

| Methods | Accuracy |
|---------|----------|
| gpt-4o-mini | 0.870 |
| o3 | 0.875 |

To create a prompt for Spanish, a prompt in English is prompted to o4-mini-high to produce a Spanish version, e.g.:

"Translate to Spanish, considering Spanish-specific nuances, since the prompt above is designed for English. Make the prompt accurate, reflect TN/FN in Spanish with its specifics."

#### Simple Spanish TP/FP Judge Evaluation

| Methods | Accuracy |
|---------|----------|
| Baseline | 0,75 |
| Best Method | 0,81 |

#### Spanish TP/FP Judge Evaluation

| Methods | Accuracy |
|---------|----------|
| Baseline | 0,37 |
| Best Method | 0,69 |

Using the PAWS meaning-changing dataset, we compare a lightweight Gemini-2.0-flash-lite and Poppins in terms of accuracy. Overall, we get a strong boost of ~5-7% in F1 and accuracy, which demonstrates the promising direction of using small LLMs for multilingual meaning-changing tasks.

#### Meaning Changing Evaluation

| Name | Accuracy | F1 |
|------|----------|-----|
| Poppins | 0,73 | 0,78 |
| Gemini-2.0-flash-lite | 0,79 | 0,82 |

#### Nonsense Evaluation

| Name | Accuracy | F1 |
|------|----------|-----|
| Gemini-2.0-flash-lite | 0,8 | 0,81 |
| Gemini-Combined | 0,82 | 0,89 |