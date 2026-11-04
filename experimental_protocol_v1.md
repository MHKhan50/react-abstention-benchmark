# Experimental Protocol v1.0 — ReAct Abstention Benchmark

**Release:** RAB v0.1.0  
**Author:** Muzammal Hussain  
**Study class:** synthetic simulation / rule-based policy comparison

## 1. Motivation

Tool-using agents interleave reasoning with external actions and observations. A decision to answer should therefore depend not only on whether a tool returned something, but also on whether the returned evidence appears relevant and mutually consistent. Selective-classification research formalizes a related reject option: a system may abstain, trading answer coverage against risk.

This first release isolates that decision layer in a controlled simulator. It does not evaluate a language model's reasoning trace or the factual quality of live tool outputs.

## 2. Research question

Can evidence-feature-based abstention policies reduce unsupported answers under missing, irrelevant, or contradictory evidence while preserving answer coverage when evidence is sufficient?

## 3. Hypotheses

- **H1 — Presence-only limitation:** `presence_only` will emit answers on irrelevant and contradictory episodes because it treats any returned evidence as sufficient.
- **H2 — Threshold/risk trade-off:** increasing the evidence threshold will generally reduce unsupported answers but may also reduce answerable coverage.
- **H3 — Conservative abstention cost:** a sufficiently strict threshold will increase unnecessary abstention on some answerable episodes because the evidence-score distributions overlap.

These are hypotheses about this simulator, not universal claims about real agents.

## 4. Experimental unit and generated variables

One row is one synthetic evidence episode. Each episode contains:
- a seed and unique episode identifier;
- evidence presence and source count;
- evidence relevance and consistency scores in [0, 1];
- a hidden evidence condition;
- an answerability label;
- a simulated correctness outcome for an attempted answer.

The condition and answerability label are evaluator-only fields. Policy functions receive only evidence presence, relevance, and consistency.

## 5. Conditions and score distributions

| Condition | Answerable? | Evidence presence | Relevance | Consistency |
|---|---:|---:|---:|---:|
| sufficient | yes | yes | Uniform(0.55, 1.00) | Uniform(0.55, 1.00) |
| missing | no | no | 0 | 0 |
| irrelevant | no | yes | Uniform(0.05, 0.80) | Uniform(0.55, 1.00) |
| contradictory | no | yes | Uniform(0.55, 1.00) | Uniform(0.05, 0.80) |

Condition allocation is balanced as closely as possible within each seed. The distributions deliberately overlap so a threshold does not perfectly recover the hidden condition.

For answerable episodes, correctness is sampled using:
`p(correct) = 0.60 + 0.40 × min(relevance, consistency)`.
This is an explicit simulator assumption, not a fitted or empirical probability.

## 6. Policies

All policies use only observable evidence fields.

1. `always_answer`: always answers.
2. `presence_only`: answers if evidence is present.
3. `evidence_gate_0.50`: requires evidence presence and both scores ≥ 0.50.
4. `evidence_gate_0.70`: requires evidence presence and both scores ≥ 0.70.
5. `evidence_gate_0.85`: requires evidence presence and both scores ≥ 0.85.

No policy receives `evidence_condition`, `answerable_label`, or `answer_correct_if_attempted`.

## 7. Outcomes and denominators

- Answerable-attempt accuracy = correct answers / answered answerable episodes.
- Answerable coverage = answered answerable episodes / all answerable episodes.
- Unsupported-answer rate = answers on unanswerable episodes / all unanswerable episodes.
- Correct-abstention rate = abstentions on unanswerable episodes / all unanswerable episodes.
- Unnecessary-abstention rate = abstentions on answerable episodes / all answerable episodes.
- Overall coverage = all answers / all episodes.
- Selective risk = (unsupported answers + incorrect answers on answerable episodes) / all emitted answers.
- Supported-answer yield = correct answers / all episodes.

Conditional metrics with zero denominator are undefined and represented as null/blank.

## 8. Execution plan

Default release experiment:
- seeds: 17, 29, 43, 71, 101;
- episodes per seed: 1,000;
- total unique episodes: 5,000;
- policies: 5;
- policy-decision rows: 25,000.

Run the same seeds and episode count for all policies. Report pooled values and seed-level values. Report mean and sample standard deviation across seeds; do not select the best seed.

## 9. Analysis plan

1. Compare unsupported-answer rate and correct-abstention rate.
2. Compare answerable coverage and unnecessary-abstention rate.
3. Inspect selective risk and overall coverage.
4. Inspect condition-level policy behaviour.
5. Examine the threshold sweep for monotonicity and risk-coverage movement.
6. Treat seed-level variation descriptively. Do not claim statistical significance from this synthetic baseline.

## 10. Validity limits and next study

The simulator omits language generation, tool selection, tool execution, evidence provenance, semantic entailment, calibration, multi-turn state, adversarial prompt injection, and human review. Score distributions and correctness probabilities are chosen assumptions. A stronger follow-up should evaluate actual agents on a curated, independently annotated evidence corpus; preregister the protocol; include annotator agreement; compare calibrated confidence and semantic evidence-verification methods; and run sensitivity analyses over distribution shifts.

## 11. References

- Chow, C. K. (1970). *On optimum recognition error and reject tradeoff*. IEEE Transactions on Information Theory, 16(1), 41–46. https://doi.org/10.1109/TIT.1970.1054406
- El-Yaniv, R., & Wiener, Y. (2010). *On the foundations of noise-free selective classification*. Journal of Machine Learning Research, 11, 1605–1641. https://jmlr.csail.mit.edu/beta/papers/v11/el-yaniv10a.html
- Geifman, Y., & El-Yaniv, R. (2017). *Selective classification for deep neural networks*. NeurIPS 2017. https://papers.nips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html
- Yao, S., et al. (2023). *ReAct: Synergizing reasoning and acting in language models*. ICLR 2023. https://arxiv.org/abs/2210.03629
