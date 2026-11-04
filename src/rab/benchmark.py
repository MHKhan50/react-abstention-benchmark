"""ReAct Abstention Benchmark: deterministic synthetic policy evaluation."""
from __future__ import annotations

import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

CONDITIONS = ("sufficient", "missing", "irrelevant", "contradictory")
POLICIES = (
    "always_answer",
    "presence_only",
    "evidence_gate_0.50",
    "evidence_gate_0.70",
    "evidence_gate_0.85",
)
THRESHOLDS = {
    "evidence_gate_0.50": 0.50,
    "evidence_gate_0.70": 0.70,
    "evidence_gate_0.85": 0.85,
}


def generate_episodes(n: int, seed: int) -> list[dict]:
    """Generate n balanced synthetic episodes for one seed."""
    if n < 1:
        raise ValueError("episodes per seed must be >= 1")

    rng = random.Random(seed)
    base, remainder = divmod(n, len(CONDITIONS))
    conditions = [c for c in CONDITIONS for _ in range(base)]
    if remainder:
        conditions.extend(rng.sample(list(CONDITIONS), remainder))
    rng.shuffle(conditions)

    episodes = []
    for i, condition in enumerate(conditions):
        if condition == "sufficient":
            present = True
            source_count = rng.randint(1, 4)
            relevance = rng.uniform(0.55, 1.00)
            consistency = rng.uniform(0.55, 1.00)
            answerable = True
        elif condition == "missing":
            present = False
            source_count = 0
            relevance = 0.0
            consistency = 0.0
            answerable = False
        elif condition == "irrelevant":
            present = True
            source_count = rng.randint(1, 4)
            relevance = rng.uniform(0.05, 0.80)
            consistency = rng.uniform(0.55, 1.00)
            answerable = False
        else:
            present = True
            source_count = rng.randint(2, 5)
            relevance = rng.uniform(0.55, 1.00)
            consistency = rng.uniform(0.05, 0.80)
            answerable = False

        # Explicit synthetic assumption; not a fitted probability.
        p_correct = 0.60 + 0.40 * min(relevance, consistency) if answerable else 0.0
        correct_if_attempted = bool(answerable and rng.random() < p_correct)

        episodes.append({
            "episode_id": f"{seed}-{i:06d}",
            "seed": int(seed),
            "evidence_condition": condition,
            "evidence_present": bool(present),
            "source_count": int(source_count),
            "evidence_relevance": round(relevance, 6),
            "evidence_consistency": round(consistency, 6),
            "answerable_label": bool(answerable),
            "answer_correct_if_attempted": bool(correct_if_attempted),
        })
    return episodes


def write_dataset(episodes: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episodes[0].keys()))
        writer.writeheader()
        writer.writerows(episodes)


def load_dataset(path: str | Path) -> list[dict]:
    """Load CSV and restore bool/int/float types."""
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "episode_id": row["episode_id"],
                "seed": int(row["seed"]),
                "evidence_condition": row["evidence_condition"],
                "evidence_present": row["evidence_present"].strip().lower() in ("true", "1", "yes"),
                "source_count": int(row["source_count"]),
                "evidence_relevance": float(row["evidence_relevance"]),
                "evidence_consistency": float(row["evidence_consistency"]),
                "answerable_label": row["answerable_label"].strip().lower() in ("true", "1", "yes"),
                "answer_correct_if_attempted": row["answer_correct_if_attempted"].strip().lower() in ("true", "1", "yes"),
            })
    if not rows:
        raise ValueError("Input dataset is empty")
    return rows


def decide(policy: str, episode: dict) -> bool:
    """Return whether to answer; deliberately ignores evaluator-only labels."""
    if policy == "always_answer":
        return True
    if policy == "presence_only":
        return bool(episode["evidence_present"])
    if policy in THRESHOLDS:
        threshold = THRESHOLDS[policy]
        return (
            bool(episode["evidence_present"])
            and float(episode["evidence_relevance"]) >= threshold
            and float(episode["evidence_consistency"]) >= threshold
        )
    raise ValueError(f"Unknown policy: {policy}")


def evaluate(episodes: list[dict]) -> list[dict]:
    results = []
    for episode in episodes:
        for policy in POLICIES:
            answered = decide(policy, episode)
            answerable = bool(episode["answerable_label"])
            correct = bool(answered and answerable and episode["answer_correct_if_attempted"])
            unsupported = bool(answered and not answerable)
            wrong_answerable = bool(answered and answerable and not episode["answer_correct_if_attempted"])
            results.append({
                **episode,
                "policy": policy,
                "answered": answered,
                "correct_answer": correct,
                "unsupported_answer": unsupported,
                "wrong_answerable_attempt": wrong_answerable,
                "correct_abstention": bool((not answered) and not answerable),
                "unnecessary_abstention": bool((not answered) and answerable),
                "unsafe_or_wrong_answer": bool(unsupported or wrong_answerable),
            })
    return results


def _ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def metric_row(rows: list[dict]) -> dict:
    total = len(rows)
    answerable_rows = [r for r in rows if r["answerable_label"]]
    unanswerable_rows = [r for r in rows if not r["answerable_label"]]
    answered_rows = [r for r in rows if r["answered"]]
    answered_answerable = [r for r in answerable_rows if r["answered"]]
    return {
        "episodes": total,
        "answerable_episodes": len(answerable_rows),
        "unanswerable_episodes": len(unanswerable_rows),
        "answerable_attempt_accuracy": _ratio(
            sum(r["correct_answer"] for r in answered_answerable), len(answered_answerable)),
        "answerable_coverage": _ratio(len(answered_answerable), len(answerable_rows)),
        "unsupported_answer_rate": _ratio(
            sum(r["unsupported_answer"] for r in unanswerable_rows), len(unanswerable_rows)),
        "correct_abstention_rate": _ratio(
            sum(r["correct_abstention"] for r in unanswerable_rows), len(unanswerable_rows)),
        "unnecessary_abstention_rate": _ratio(
            sum(r["unnecessary_abstention"] for r in answerable_rows), len(answerable_rows)),
        "overall_coverage": _ratio(len(answered_rows), total),
        "selective_risk": _ratio(
            sum(r["unsafe_or_wrong_answer"] for r in answered_rows), len(answered_rows)),
        "supported_answer_yield": _ratio(
            sum(r["correct_answer"] for r in rows), total),
    }


def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    return statistics.mean(values), statistics.stdev(values)


def summarize(results: list[dict]) -> dict:
    by_policy = defaultdict(list)
    by_seed_policy = defaultdict(list)
    by_condition_policy = defaultdict(list)
    for row in results:
        by_policy[row["policy"]].append(row)
        by_seed_policy[(row["seed"], row["policy"])].append(row)
        by_condition_policy[(row["evidence_condition"], row["policy"])].append(row)

    pooled = {policy: metric_row(rows) for policy, rows in by_policy.items()}
    per_seed = [
        {"seed": seed, "policy": policy, **metric_row(rows)}
        for (seed, policy), rows in sorted(by_seed_policy.items())
    ]
    aggregate = {}
    for policy in POLICIES:
        seed_rows = [r for r in per_seed if r["policy"] == policy]
        aggregate[policy] = {}
        for metric in (
            "answerable_attempt_accuracy", "answerable_coverage",
            "unsupported_answer_rate", "correct_abstention_rate",
            "unnecessary_abstention_rate", "overall_coverage",
            "selective_risk", "supported_answer_yield",
        ):
            values = [r[metric] for r in seed_rows if r[metric] is not None]
            mean, sd = _mean_sd(values)
            aggregate[policy][metric + "_seed_mean"] = mean
            aggregate[policy][metric + "_seed_sd"] = sd

    by_condition = [
        {"condition": condition, "policy": policy, **metric_row(rows)}
        for (condition, policy), rows in sorted(by_condition_policy.items())
    ]
    return {
        "benchmark": "ReAct Abstention Benchmark",
        "version": "0.1.0",
        "data_type": "synthetic",
        "warning": "Simulator results only; not real-world agent or LLM measurements.",
        "policies": pooled,
        "seed_aggregates": aggregate,
        "per_seed": per_seed,
        "by_condition": by_condition,
    }


def threshold_sweep(episodes: list[dict]) -> list[dict]:
    rows = []
    for step in range(40, 96, 5):
        threshold = step / 100.0
        evaluated = []
        for ep in episodes:
            answered = (
                bool(ep["evidence_present"])
                and float(ep["evidence_relevance"]) >= threshold
                and float(ep["evidence_consistency"]) >= threshold
            )
            answerable = bool(ep["answerable_label"])
            correct = bool(answered and answerable and ep["answer_correct_if_attempted"])
            unsupported = bool(answered and not answerable)
            wrong = bool(answered and answerable and not ep["answer_correct_if_attempted"])
            evaluated.append({
                **ep, "answered": answered, "correct_answer": correct,
                "unsupported_answer": unsupported,
                "wrong_answerable_attempt": wrong,
                "correct_abstention": bool((not answered) and not answerable),
                "unnecessary_abstention": bool((not answered) and answerable),
                "unsafe_or_wrong_answer": bool(unsupported or wrong),
            })
        metrics = metric_row(evaluated)
        rows.append({"threshold": threshold, **metrics})
    return rows


def write_results(results: list[dict], summary: dict, sweep: list[dict], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def write_csv(name, rows):
        if not rows:
            return
        fields = list(rows[0].keys())
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_csv("episode_results.csv", results)
    write_csv("seed_metrics.csv", summary["per_seed"])
    write_csv("condition_metrics.csv", summary["by_condition"])

    policy_rows = [{"policy": p, **m, **summary["seed_aggregates"][p]}
                   for p, m in summary["policies"].items()]
    write_csv("policy_metrics.csv", policy_rows)
    write_csv("threshold_sweep.csv", sweep)

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
