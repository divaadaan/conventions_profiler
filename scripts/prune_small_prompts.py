"""Remove prompt groups with fewer than N essays from the dataset."""

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("src/conventions_profiler/data/raw/train.csv")


def main():
    parser = argparse.ArgumentParser(description="Prune small prompt groups from dataset")
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_PATH,
        help=f"Path to input CSV (default: {DEFAULT_PATH})",
    )
    parser.add_argument(
        "--min-n", type=int, default=50,
        help="Minimum essays per prompt to keep (default: 50)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    counts = df.groupby("prompt").size()
    total_before = len(df)
    prompts_before = len(counts)

    keep = counts[counts >= args.min_n].index
    drop = counts[counts < args.min_n]

    if drop.empty:
        print(f"No prompts with fewer than {args.min_n} essays. Nothing to prune.")
        return

    print(f"Dropping {len(drop)} prompts (N < {args.min_n}):")
    for prompt, n in drop.sort_values().items():
        print(f"  {prompt}: {n}")

    df = df[df["prompt"].isin(keep)]
    print(f"\nBefore: {total_before} essays, {prompts_before} prompts")
    print(f"After:  {len(df)} essays, {len(keep)} prompts")
    print(f"Removed: {total_before - len(df)} essays")

    df.to_csv(args.input, index=False)
    print(f"\nSaved to {args.input}")


if __name__ == "__main__":
    main()
