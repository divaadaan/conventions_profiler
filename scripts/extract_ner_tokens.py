"""Extract all unique @TOKEN anonymization labels from the ASAP++ dataset.

Reads the processed CSV and finds every @SOMETHING pattern in the essay
column, grouped by token type (e.g., CAPS, NAME, LOCATION).

Usage:
    python scripts/extract_ner_tokens.py
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd

DATA_PATH = Path("src/conventions_profiler/data/processed/asap_pp_conventions.csv")

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} essays from {DATA_PATH}\n")

# Find all @TOKEN patterns
pattern = re.compile(r"@[A-Z]+\d*")
all_tokens = []
for text in df["essay"]:
    all_tokens.extend(pattern.findall(str(text)))

counts = Counter(all_tokens)

# Group by type prefix (strip trailing digits)
type_prefix = re.compile(r"^(@[A-Z]+?)\d*$")
types: dict[str, list[tuple[str, int]]] = {}
for token, count in sorted(counts.items()):
    match = type_prefix.match(token)
    prefix = match.group(1) if match else token
    types.setdefault(prefix, []).append((token, count))

# Report
print("=" * 60)
print("ANONYMIZATION TOKENS BY TYPE")
print("=" * 60)
total = 0
for prefix in sorted(types):
    tokens = types[prefix]
    type_total = sum(c for _, c in tokens)
    total += type_total
    print(f"\n{prefix}  ({len(tokens)} unique, {type_total} occurrences)")
    for token, count in sorted(tokens, key=lambda x: x[0]):
        print(f"  {token:<20s} {count:>6d}")

print(f"\n{'=' * 60}")
print(f"TOTAL: {len(counts)} unique tokens, {total} occurrences")
print(f"TOKEN TYPES: {sorted(types.keys())}")
