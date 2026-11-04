#!/usr/bin/env python3
"""Run RAB and write the synthetic dataset plus evaluation outputs."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rab.benchmark import (  # noqa: E402
    generate_episodes, load_dataset, write_dataset, evaluate, summarize,
    threshold_sweep, write_results,
)


def main():
    parser = argparse.ArgumentParser(description="Run ReAct Abstention Benchmark v0.1.0")
    parser.add_argument("--episodes-per-seed", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43, 71, 101])
    parser.add_argument("--input-data", default=None, help="Use an existing CSV dataset instead of generating one")
    parser.add_argument("--data-out", default="data/rab_synthetic_episodes.csv")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    if args.input_data:
        episodes = load_dataset(args.input_data)
        print(f"Loaded {len(episodes)} episodes from {args.input_data}")
    else:
        if args.episodes_per_seed < 1:
            parser.error("--episodes-per-seed must be >= 1")
        episodes = []
        for seed in args.seeds:
            episodes.extend(generate_episodes(args.episodes_per_seed, seed))
        write_dataset(episodes, args.data_out)
        print(f"Generated {len(episodes)} synthetic episodes at {args.data_out}")

    results = evaluate(episodes)
    summary = summarize(results)
    summary["input_episode_count"] = len(episodes)
    summary["seeds"] = sorted({int(e["seed"]) for e in episodes})
    sweep = threshold_sweep(episodes)
    write_results(results, summary, sweep, args.out_dir)

    print("\nPooled policy metrics")
    for policy, metrics in summary["policies"].items():
        print(
            f"{policy:20s} "
            f"unsupported={metrics['unsupported_answer_rate']:.4f} "
            f"answerable_coverage={metrics['answerable_coverage']:.4f} "
            f"overall_coverage={metrics['overall_coverage']:.4f} "
            f"selective_risk={metrics['selective_risk']:.4f}"
        )
    print(f"\nWrote outputs to: {args.out_dir}/")
    print("Interpretation: synthetic simulator baseline only.")


if __name__ == "__main__":
    main()
