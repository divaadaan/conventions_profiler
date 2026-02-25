"""Analyze whether @TOKEN numbering is per-essay or global across the dataset.

Checks if the same @TOKEN (e.g., @CAPS1) refers to different things in
different essays, which would indicate per-essay numbering.

Usage:
    python scripts/analyze_ner_scope.py
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd

DATA_PATH = Path("src/conventions_profiler/data/processed/asap_pp_conventions.csv")

df = pd.read_csv(DATA_PATH)
pattern = re.compile(r"@[A-Z]+\d*")

# Per-essay: what's the max token index per type per essay?
type_prefix_re = re.compile(r"^@([A-Z]+?)(\d+)$")

max_per_essay = []
for _, row in df.iterrows():
    tokens = pattern.findall(str(row["essay"]))
    type_maxes = {}
    for tok in tokens:
        m = type_prefix_re.match(tok)
        if m:
            typ, idx = m.group(1), int(m.group(2))
            type_maxes[typ] = max(type_maxes.get(typ, 0), idx)
    max_per_essay.append(type_maxes)

# Distribution of max CAPS index per essay
caps_maxes = [d.get("CAPS", 0) for d in max_per_essay]
caps_counter = Counter(caps_maxes)

print("=" * 60)
print("MAX @CAPS INDEX PER ESSAY (distribution)")
print("=" * 60)
for idx in sorted(caps_counter):
    print(f"  max @CAPS index = {idx:>3d}:  {caps_counter[idx]:>5d} essays")

print(f"\nHighest @CAPS index in any single essay: {max(caps_maxes)}")
print(f"Essays with zero @CAPS tokens: {caps_counter[0]}")

# Check if @CAPS1 context differs across essays (per-essay numbering test)
print("\n" + "=" * 60)
print("SAMPLE CONTEXTS FOR @CAPS1 (first 10 essays that use it)")
print("=" * 60)
shown = 0
for _, row in df.iterrows():
    text = str(row["essay"])
    if "@CAPS1" in text:
        # Show 30 chars around first occurrence
        idx = text.index("@CAPS1")
        start = max(0, idx - 30)
        end = min(len(text), idx + 36)
        context = text[start:end].replace("\n", " ")
        print(f"  essay {row['essay_id']:>5d}: ...{context}...")
        shown += 1
        if shown >= 10:
            break

# Same for @PERSON1
print("\n" + "=" * 60)
print("SAMPLE CONTEXTS FOR @PERSON1 (first 10 essays that use it)")
print("=" * 60)
shown = 0
for _, row in df.iterrows():
    text = str(row["essay"])
    if "@PERSON1" in text:
        idx = text.index("@PERSON1")
        start = max(0, idx - 30)
        end = min(len(text), idx + 40)
        context = text[start:end].replace("\n", " ")
        print(f"  essay {row['essay_id']:>5d}: ...{context}...")
        shown += 1
        if shown >= 10:
            break

# Also check @MONTH1 contexts
print("\n" + "=" * 60)
print("SAMPLE CONTEXTS FOR @MONTH1 (first 15 essays that use it)")
print("=" * 60)
shown = 0
for _, row in df.iterrows():
    text = str(row["essay"])
    if "@MONTH1" in text:
        idx = text.index("@MONTH1")
        start = max(0, idx - 30)
        end = min(len(text), idx + 37)
        context = text[start:end].replace("\n", " ")
        print(f"  essay {row['essay_id']:>5d}: ...{context}...")
        shown += 1
        if shown >= 15:
            break
