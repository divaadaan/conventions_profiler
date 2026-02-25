"""Debug script: dump all unique LanguageTool rule_ids and categories from checkpoint data."""

import json
from collections import Counter
from pathlib import Path

import pandas as pd


def main():
    checkpoint = Path("cache/checkpoints/languagetool.parquet")
    if not checkpoint.exists():
        print(f"No checkpoint at {checkpoint}")
        return

    df = pd.read_parquet(checkpoint)
    print(f"Loaded {len(df)} essays from checkpoint")

    rule_ids = Counter()
    categories = Counter()
    sample_matches = {}

    for _, row in df.iterrows():
        data = json.loads(row["pass_results"])
        for m in data.get("matches", []):
            rid = m["rule_id"]
            cat = m["category"]
            rule_ids[rid] += 1
            categories[cat] += 1
            if rid not in sample_matches:
                sample_matches[rid] = m["message"][:100]

    print(f"\n=== Categories ({len(categories)} unique) ===")
    for cat, count in categories.most_common():
        print(f"  {cat}: {count}")

    print(f"\n=== Rule IDs ({len(rule_ids)} unique) ===")
    for rid, count in rule_ids.most_common():
        print(f"  {rid}: {count}  -- {sample_matches[rid]}")


if __name__ == "__main__":
    main()
