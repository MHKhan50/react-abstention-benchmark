"""Closed-form expected metrics for the RAB v0.1 simulator.

Every policy result in RAB v0.1 follows from the simulator's own assumptions:
four equally likely conditions, uniform score ranges and
p(correct) = 0.60 + 0.40 * min(relevance, consistency). This module computes
those expected values by hand, so the simulation can be checked against them.
If the two agree, the simulation measures the assumptions, not agent behaviour.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

SUFF = (0.55, 1.00)  # sufficient relevance and consistency; also irrelevant consistency
LOW = (0.05, 0.80)   # irrelevant relevance; contradictory consistency


def p_at_least(t: float, lo: float, hi: float) -> float:
    """P(U >= t) for U ~ Uniform(lo, hi)."""
    return min(1.0, max(0.0, (hi - t) / (hi - lo)))


def answer_probs(t: Optional[float], presence_only: bool = False) -> Dict[str, float]:
    """Chance of answering in each condition. t=None with presence_only=False means always answer."""
    if t is None:
        return ({"sufficient": 1.0, "missing": 0.0, "irrelevant": 1.0, "contradictory": 1.0}
                if presence_only else dict.fromkeys(("sufficient", "missing", "irrelevant", "contradictory"), 1.0))
    s = p_at_least(t, *SUFF)
    low = p_at_least(t, *LOW)
    return {"sufficient": s * s, "missing": 0.0, "irrelevant": low * s, "contradictory": s * low}


def expected_metrics(t: Optional[float], presence_only: bool = False) -> Dict[str, float]:
    q = answer_probs(t, presence_only)
    floor = max(t or 0.0, SUFF[0])
    # mean of min of two iid Uniform(floor, 1) is floor + (1 - floor) / 3
    acc = 0.60 + 0.40 * (floor + (1.0 - floor) / 3.0) if q["sufficient"] > 0 else float("nan")
    unans = q["missing"] + q["irrelevant"] + q["contradictory"]
    answered = q["sufficient"] + unans
    wrong = q["sufficient"] * (1.0 - acc) if q["sufficient"] > 0 else 0.0
    return {
        "answerable_attempt_accuracy": acc,
        "answerable_coverage": q["sufficient"],
        "unsupported_answer_rate": unans / 3.0,
        "overall_coverage": answered / 4.0,
        "selective_risk": (unans + wrong) / answered if answered > 0 else float("nan"),
    }


POLICY_ARGS = {
    "always_answer": (None, False), "presence_only": (None, True),
    "evidence_gate_0.50": (0.50, False), "evidence_gate_0.70": (0.70, False),
    "evidence_gate_0.85": (0.85, False),
}
METRICS = ("answerable_attempt_accuracy", "answerable_coverage", "unsupported_answer_rate",
           "overall_coverage", "selective_risk")


def compare(out_dir: Path) -> List[Dict[str, object]]:
    """Expected vs observed for every policy and every sweep threshold."""
    rows: List[Dict[str, object]] = []
    sources = [("policy", out_dir / "policy_metrics.csv"), ("threshold", out_dir / "threshold_sweep.csv")]
    for key, path in sources:
        with path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                name = r[key]
                t, pres = POLICY_ARGS[name] if key == "policy" else (float(name), False)
                exp = expected_metrics(t, pres)
                for m in METRICS:
                    if r.get(m, "") == "":
                        continue
                    obs = float(r[m])
                    rows.append({"source": key, "setting": name, "metric": m,
                                 "expected": round(exp[m], 4), "observed": round(obs, 4),
                                 "abs_diff": round(abs(exp[m] - obs), 4)})
    return rows


def main(out_dir: str = "outputs") -> float:
    out = Path(out_dir)
    rows = compare(out)
    with (out / "expected_vs_observed.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    worst = max(float(r["abs_diff"]) for r in rows)
    print(f"{len(rows)} values compared; largest gap between formula and simulation: {worst:.4f}")
    print(f"Wrote {out / 'expected_vs_observed.csv'}")
    return worst


if __name__ == "__main__":
    main()
