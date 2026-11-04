import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv
import tempfile
import unittest
from pathlib import Path

from rab.benchmark import (
    CONDITIONS, POLICIES, generate_episodes, decide, evaluate, metric_row,
    summarize, threshold_sweep, write_dataset, load_dataset,
)


class RABTests(unittest.TestCase):
    def test_reproducible_generation(self):
        self.assertEqual(generate_episodes(40, 17), generate_episodes(40, 17))

    def test_balanced_conditions_for_divisible_n(self):
        rows = generate_episodes(40, 29)
        counts = {c: sum(r["evidence_condition"] == c for r in rows) for c in CONDITIONS}
        self.assertEqual(set(counts.values()), {10})

    def test_remainder_balance(self):
        rows = generate_episodes(41, 29)
        counts = [sum(r["evidence_condition"] == c for r in rows) for c in CONDITIONS]
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_invalid_episode_count(self):
        with self.assertRaises(ValueError):
            generate_episodes(0, 1)

    def test_policy_does_not_need_hidden_labels(self):
        observable_only = {
            "evidence_present": True,
            "evidence_relevance": 0.91,
            "evidence_consistency": 0.92,
        }
        self.assertTrue(decide("evidence_gate_0.70", observable_only))

    def test_missing_evidence_abstention(self):
        observable_only = {
            "evidence_present": False,
            "evidence_relevance": 0.0,
            "evidence_consistency": 0.0,
        }
        self.assertTrue(decide("always_answer", observable_only))
        self.assertFalse(decide("presence_only", observable_only))
        self.assertFalse(decide("evidence_gate_0.50", observable_only))

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            decide("oracle_policy", {})

    def test_evaluation_row_count(self):
        episodes = generate_episodes(20, 43)
        results = evaluate(episodes)
        self.assertEqual(len(results), 20 * len(POLICIES))

    def test_metrics_known_fixture(self):
        fixture = [
            {"answerable_label": True, "answered": True, "correct_answer": True,
             "unsupported_answer": False, "correct_abstention": False,
             "unnecessary_abstention": False, "unsafe_or_wrong_answer": False},
            {"answerable_label": True, "answered": False, "correct_answer": False,
             "unsupported_answer": False, "correct_abstention": False,
             "unnecessary_abstention": True, "unsafe_or_wrong_answer": False},
            {"answerable_label": False, "answered": True, "correct_answer": False,
             "unsupported_answer": True, "correct_abstention": False,
             "unnecessary_abstention": False, "unsafe_or_wrong_answer": True},
            {"answerable_label": False, "answered": False, "correct_answer": False,
             "unsupported_answer": False, "correct_abstention": True,
             "unnecessary_abstention": False, "unsafe_or_wrong_answer": False},
        ]
        metrics = metric_row(fixture)
        self.assertEqual(metrics["answerable_attempt_accuracy"], 1.0)
        self.assertEqual(metrics["answerable_coverage"], 0.5)
        self.assertEqual(metrics["unsupported_answer_rate"], 0.5)
        self.assertEqual(metrics["correct_abstention_rate"], 0.5)
        self.assertEqual(metrics["unnecessary_abstention_rate"], 0.5)
        self.assertEqual(metrics["overall_coverage"], 0.5)
        self.assertEqual(metrics["selective_risk"], 0.5)

    def test_threshold_sweep_has_expected_rows(self):
        rows = threshold_sweep(generate_episodes(100, 71))
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(0.0 <= r["overall_coverage"] <= 1.0 for r in rows))

    def test_csv_round_trip(self):
        episodes = generate_episodes(40, 101)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            write_dataset(episodes, path)
            loaded = load_dataset(path)
            self.assertEqual(episodes, loaded)

    def test_summary_has_all_policies_and_seeds(self):
        episodes = generate_episodes(40, 17) + generate_episodes(40, 29)
        summary = summarize(evaluate(episodes))
        self.assertEqual(set(summary["policies"]), set(POLICIES))
        self.assertEqual(len(summary["per_seed"]), 2 * len(POLICIES))


if __name__ == "__main__":
    unittest.main()
