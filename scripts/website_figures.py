"""Website figure from outputs/threshold_sweep.csv and outputs/expected_vs_observed.csv.

    python scripts/website_figures.py --out figures_web
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, YELLOW = "#1F4E9C", "#E3A800"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures_web")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    sweep = list(csv.DictReader(open(ROOT / "outputs" / "threshold_sweep.csv")))
    exp = {(r["setting"], r["metric"]): float(r["expected"])
           for r in csv.DictReader(open(ROOT / "outputs" / "expected_vs_observed.csv")) if r["source"] == "threshold"}
    t = [float(r["threshold"]) for r in sweep]
    keys = [r["threshold"] for r in sweep]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, [exp[(k, "answerable_coverage")] for k in keys], color=BLUE, alpha=0.5, lw=1.5, label="Formula")
    ax.plot(t, [exp[(k, "unsupported_answer_rate")] for k in keys], color=YELLOW, alpha=0.6, lw=1.5)
    ax.plot(t, [float(r["answerable_coverage"]) for r in sweep], "o", color=BLUE,
            label="Answerable questions answered (simulation)")
    ax.plot(t, [float(r["unsupported_answer_rate"]) for r in sweep], "s", color=YELLOW,
            label="Unanswerable questions answered (simulation)")
    ax.set_xlabel("Evidence threshold (relevance and consistency)")
    ax.set_ylabel("Share of episodes")
    ax.set_ylim(0, 1.05)
    ax.set_title("Evidence gate: fewer unsupported answers, far fewer good ones\n"
                 "20,000 synthetic episodes; lines are the closed-form formula", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "react-abstention-benchmark.png", dpi=150)
    print("wrote", out / "react-abstention-benchmark.png")


if __name__ == "__main__":
    main()
