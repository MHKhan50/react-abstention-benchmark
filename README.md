# ReAct Abstention Benchmark (RAB) v0.1

**Author:** Muzammal Hussain · **Python:** 3.9+ · **Dependencies:** standard library only (the notebook also uses pandas and matplotlib)

**Research question:** When a tool-using agent gets missing, irrelevant or contradictory evidence, can a simple rule on the evidence decide when it should say "I don't know", and how much useful answering does that cost when the evidence is good?

Background: the ReAct paper ([Yao et al. 2023](https://arxiv.org/abs/2210.03629)) gives agents no way to abstain, and the authors trace 23% of ReAct errors to uninformative search results. Selective classification (Chow 1970; El-Yaniv and Wiener 2010; Geifman and El-Yaniv 2017) studies the same trade-off: answering less often (lower coverage) to make fewer mistakes (lower risk).

## Scope
v0.1 is a **synthetic simulator**. It does not call a language model or run real tools. Each episode is a row of evidence features drawn from fixed ranges, and five answer-or-abstain rules are scored on them. The results describe the simulator, not real agents.

| Link | What it is |
|---|---|
| [Kaggle dataset](https://www.kaggle.com/datasets/muzammalhussain11/react-abstention-benchmark-dataset) | the generated episodes |
| [Kaggle notebook](https://www.kaggle.com/code/muzammalhussain11/react-abstention-benchmark-evidence-based-agent) | runs the benchmark on the dataset |
| [Open in Colab](https://colab.research.google.com/github/MHKhan50/react-abstention-benchmark/blob/main/notebooks/RAB_Kaggle_Colab.ipynb) | same notebook in Google Colab |

## Simulator
Four conditions, equally likely. Only the first is answerable.

| Condition | Evidence returned | Relevance | Consistency |
|---|---|---|---|
| sufficient | yes | Uniform(0.55, 1.00) | Uniform(0.55, 1.00) |
| missing | no | 0 | 0 |
| irrelevant | yes | Uniform(0.05, 0.80) | Uniform(0.55, 1.00) |
| contradictory | yes | Uniform(0.55, 1.00) | Uniform(0.05, 0.80) |

An answer to an answerable episode is correct with probability 0.60 + 0.40 × min(relevance, consistency). This is an assumption, not a fitted value.

| Policy | Answers when |
|---|---|
| `always_answer` | always |
| `presence_only` | any evidence is returned |
| `evidence_gate_0.50` / `0.70` / `0.85` | evidence is returned and relevance and consistency are both at least the threshold |

Policies see only evidence presence, relevance and consistency, never the condition or the label.

## Metrics
- **Answerable coverage:** share of answerable episodes the policy answers.
- **Unsupported-answer rate:** share of unanswerable episodes the policy answers anyway.
- **Answer accuracy:** correct answers / answers given on answerable episodes.
- **Overall coverage:** share of all episodes answered.
- **Selective risk:** (unsupported answers + wrong answers) / all answers given.

## Run
```bash
python3 run_benchmark.py --episodes-per-seed 4000 --seeds 17 29 43 71 101
python3 src/rab/expected.py      # formula vs simulation, writes outputs/expected_vs_observed.csv
python3 -m unittest discover -s tests -v
```

## Results
From my run on a MacBook M4: 5 seeds (17, 29, 43, 71, 101) × 4,000 episodes = 20,000 episodes (5,000 per condition), 100,000 policy decisions. [`outputs/policy_metrics.csv`](outputs/policy_metrics.csv) also has the mean and SD across seeds.

| Policy | Answerable coverage | Unsupported-answer rate | Answer accuracy | Overall coverage | Selective risk |
|---|---|---|---|---|---|
| always_answer | 1.000 | 1.000 | 0.875 | 1.000 | 0.781 |
| presence_only | 1.000 | 0.667 | 0.875 | 0.750 | 0.708 |
| evidence_gate_0.50 | 1.000 | 0.260 | 0.875 | 0.445 | 0.508 |
| evidence_gate_0.70 | 0.444 | 0.058 | 0.918 | 0.154 | 0.339 |
| evidence_gate_0.85 | 0.110 | 0.000 | 0.973 | 0.028 | 0.027 |

**What the numbers show:** raising the threshold cuts unsupported answers (26% to 6% to 0%) but the agent also stops answering most good questions (coverage 100% to 44% to 11%). There is no threshold that keeps both, because the score ranges overlap.

**Check against the formula.** Every value above can be worked out by hand from the four uniform ranges and the correctness formula ([`src/rab/expected.py`](src/rab/expected.py)). For example, the 0.70 gate answers a sufficient episode only if both scores are at least 0.70: (0.30 / 0.45)² = 0.444, which is the coverage in the table. Across all 5 policies and 12 sweep thresholds (85 values), the largest gap between the formula and the simulation is 0.013 ([`outputs/expected_vs_observed.csv`](outputs/expected_vs_observed.csv)). So v0.1 confirms the code is correct, but its numbers come from the ranges I chose, not from how any agent behaves. That is the main reason for v0.2.

## Limitations
- No language model, no real tool output, no text: relevance and consistency are given as numbers, while a real agent has to judge them from text.
- The score ranges and the correctness formula are my assumptions; other ranges would give other numbers.
- Seed-to-seed variation is reported but is only sampling noise of the simulator.

## Next: v0.2 with real local models
Run a ReAct agent with qwen2.5:3b and llama3.1:8b (Ollama) on clinic questions where the lookup tool returns sufficient, missing, irrelevant or contradictory text, in English and Roman Urdu, and measure how often the model answers when it should abstain. This connects to the ReAct Roman Urdu pilot in [agentic-ai-research-log](https://github.com/MHKhan50/agentic-ai-research-log/tree/main/react-roman-urdu).

## Files
- `src/rab/benchmark.py`: episode generator, policies and metrics
- `src/rab/expected.py`: closed-form expected values
- `data/rab_synthetic_episodes.csv`: the 20,000 generated episodes (see [`DATASET_CARD.md`](DATASET_CARD.md))
- `outputs/`: metrics per policy, seed, condition and threshold
- `experimental_protocol_v1.md`: the plan written before the run
- `notebooks/RAB_Kaggle_Colab.ipynb`: Kaggle and Colab notebook

## References
1. Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory*, 16(1), 41-46. https://doi.org/10.1109/TIT.1970.1054406
2. El-Yaniv, R., and Wiener, Y. (2010). On the foundations of noise-free selective classification. *JMLR*, 11, 1605-1641. https://jmlr.org/papers/v11/el-yaniv10a.html
3. Geifman, Y., and El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS 2017*. https://papers.nips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html
4. Yao, S., et al. (2023). ReAct: Synergizing reasoning and acting in language models. *ICLR 2023*. https://arxiv.org/abs/2210.03629

## Licence
MIT, see [LICENSE](LICENSE).
