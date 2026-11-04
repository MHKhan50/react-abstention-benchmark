from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rab.benchmark import evaluate, generate_episodes, summarize  # noqa: E402
from rab.expected import POLICY_ARGS, expected_metrics  # noqa: E402


class ExpectedTests(unittest.TestCase):
    def test_hand_values(self) -> None:
        e = expected_metrics(0.70)
        self.assertAlmostEqual(e["answerable_coverage"], (0.30 / 0.45) ** 2, places=6)
        self.assertAlmostEqual(e["answerable_attempt_accuracy"], 0.92, places=6)
        self.assertAlmostEqual(expected_metrics(None, True)["unsupported_answer_rate"], 2 / 3, places=6)
        self.assertEqual(expected_metrics(0.85)["unsupported_answer_rate"], 0.0)

    def test_simulation_matches_formula(self) -> None:
        eps = [e for s in (1, 2, 3) for e in generate_episodes(4000, s)]
        pol = summarize(evaluate(eps))["policies"]
        for name, (t, pres) in POLICY_ARGS.items():
            exp = expected_metrics(t, pres)
            for m in ("answerable_coverage", "unsupported_answer_rate", "overall_coverage"):
                self.assertAlmostEqual(pol[name][m], exp[m], delta=0.02, msg=f"{name} {m}")


if __name__ == "__main__":
    unittest.main()
