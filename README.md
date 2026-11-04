# ReAct Abstention Benchmark (RAB) — v0.1.0

**Author:** Muzammal Hussain  
**Project type:** deterministic synthetic benchmark / policy evaluation  
**Python:** 3.9+  
**Core dependencies:** Python standard library only

## Research question

Can an agent abstain reliably when tool evidence is missing, irrelevant, or contradictory, while retaining useful answer coverage when evidence is sufficient?

This benchmark studies the *answer-versus-abstain decision* using observable evidence features. It is inspired by selective classification and reject-option research, where coverage is traded against risk.

## Important scope statement

RAB v0.1.0 is a **synthetic, rule-based policy benchmark**. It does not call an LLM, execute real tools, train a model, or establish real-world safety. The generated labels and evidence scores are simulator variables. Results must be described as synthetic simulation results, not empirical measurements of deployed agents.

## Included policies

All policies receive only observable fields (`evidence_present`, `evidence_relevance`, `evidence_consistency`). They do **not** receive the hidden condition or answerability label.

| Policy | Decision rule |
|---|---|
| `always_answer` | Always answer |
| `presence_only` | Answer whenever at least one evidence item is present |
| `evidence_gate_0.50` | Answer if evidence is present and relevance and consistency are both ≥ 0.50 |
| `evidence_gate_0.70` | Same rule with threshold 0.70 |
| `evidence_gate_0.85` | Same rule with threshold 0.85 |

## Conditions

- `sufficient`: evidence is intended to be relevant and mutually consistent.
- `missing`: no evidence is returned.
- `irrelevant`: evidence is present but has low-to-overlapping relevance.
- `contradictory`: evidence appears relevant but consistency is low-to-overlapping.

The score distributions intentionally overlap at the boundaries. This prevents the benchmark from becoming a trivial condition-label lookup.

## Metrics

- **Answerable-attempt accuracy:** correct answers / attempted answers on answerable episodes.
- **Answerable coverage:** answered answerable episodes / all answerable episodes.
- **Unsupported-answer rate:** answered unanswerable episodes / all unanswerable episodes. Here “unsupported” means the simulator says no answer is warranted; it is not a claim that every such answer is factually false.
- **Correct-abstention rate:** abstentions on unanswerable episodes / all unanswerable episodes.
- **Unnecessary-abstention rate:** abstentions on answerable episodes / all answerable episodes.
- **Overall coverage:** all episodes receiving an answer / all episodes.
- **Selective risk:** unsupported answers plus incorrect answers on answerable episodes, divided by all emitted answers.
- **Supported-answer yield:** correct answers / all episodes.

Undefined conditional metrics are written as `null` in JSON and blank in CSV rather than silently treated as zero.

## Quick start — Mac Terminal

From the extracted project directory:

```bash
python3 --version
python3 run_benchmark.py --episodes-per-seed 1000 --seeds 17 29 43 71 101
python3 -m unittest discover -s tests -v
```

The default run creates:
- `data/rab_synthetic_episodes.csv` — 5,000 self-generated synthetic episodes.
- `outputs/episode_results.csv` — one policy decision row per episode and policy.
- `outputs/policy_metrics.csv` — pooled metrics per policy.
- `outputs/seed_metrics.csv` — per-seed metrics.
- `outputs/condition_metrics.csv` — condition-level policy behaviour.
- `outputs/threshold_sweep.csv` — threshold sensitivity analysis.
- `outputs/summary.json` — machine-readable summary.

## Dataset

The CSV in `data/` is self-generated for this benchmark. It is not collected from people, production systems, or external websites. The `answerable_label` and `evidence_condition` columns are evaluation labels; a policy must not use them when deciding whether to answer.

See `DATASET_CARD.md` and `data/dataset-metadata.json`.

## Kaggle and Google Colab

`notebooks/RAB_Kaggle_Colab.ipynb` supports both platforms. The core experiment needs no GPU. The notebook uses pandas and matplotlib for analysis. Upload `data/rab_synthetic_episodes.csv` as a Kaggle dataset if you want a separately versioned dataset release; the included CSV is already ready for that upload.

Suggested Kaggle dataset slug after publication:
`muzammalhussain11/rab-synthetic-evidence-abstention-v1`

Do not claim the Kaggle dataset is public until you have uploaded and published it. After creating the GitHub repository, add its URL to the Kaggle dataset description and add the Kaggle dataset URL back to this README.

## Reproducibility

The default release uses five fixed seeds: `17, 29, 43, 71, 101`, with 1,000 episodes per seed. The same code version, episode count, and seed list reproduce the same dataset and results.

## Research references

1. Chow, C. K. (1970). *On optimum recognition error and reject tradeoff*. IEEE Transactions on Information Theory, 16(1), 41–46. https://doi.org/10.1109/TIT.1970.1054406
2. El-Yaniv, R., & Wiener, Y. (2010). *On the foundations of noise-free selective classification*. Journal of Machine Learning Research, 11, 1605–1641. https://jmlr.csail.mit.edu/beta/papers/v11/el-yaniv10a.html
3. Geifman, Y., & El-Yaniv, R. (2017). *Selective classification for deep neural networks*. NeurIPS 2017. https://papers.nips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html
4. Yao, S., et al. (2023). *ReAct: Synergizing reasoning and acting in language models*. ICLR 2023. https://arxiv.org/abs/2210.03629

## Version status

v0.1.0 is a reproducible synthetic baseline. It is not yet a peer-reviewed paper, external validation study, or real-agent benchmark.
