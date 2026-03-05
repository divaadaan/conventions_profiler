"""
Merge two CSV files on Essay ID.

File 1: contains 'essay_id', 'essay_set', 'essay' (plus other columns to discard)
        - Rows with essay_set in {1, 2, 7, 8} are excluded

File 2: contains 'Essay ID', 'Language' (plus other columns to discard)

Output: merged CSV with essay_id, essay_set, essay, Language
"""

import pandas as pd
import sys

# ── Configuration ──────────────────────────────────────────────────────────────
FILE1 = "train.csv"          # ← change to your actual filename
FILE2 = "asap_pp-scored.csv"        # ← change to your actual filename
OUTPUT = "merged_essays.csv"

EXCLUDE_ESSAY_SETS = {1, 2, 7, 8}
# ──────────────────────────────────────────────────────────────────────────────


def main():
    # Allow overriding filenames from the command line
    f1 = sys.argv[1] if len(sys.argv) > 1 else FILE1
    f2 = sys.argv[2] if len(sys.argv) > 2 else FILE2
    out = sys.argv[3] if len(sys.argv) > 3 else OUTPUT

    print(f"Reading '{f1}' ...")
    df1 = pd.read_csv(f1, sep=None, engine="python", encoding="utf-8-sig")

    print(f"Reading '{f2}' ...")
    df2 = pd.read_csv(f2, sep=None, engine="python")

    # ── Normalise column names (strip whitespace, lowercase for matching) ──────
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    # ── Keep only the columns we need ─────────────────────────────────────────
    df1 = df1[["essay_id", "essay_set", "essay"]].copy()
    df2 = df2[["essay_id", "Language"]].copy()
    print(df2["Language"].unique())

    # ── Filter out unwanted essay sets ────────────────────────────────────────
    before = len(df1)
    df1 = df1[~df1["essay_set"].isin(EXCLUDE_ESSAY_SETS)]
    print(f"Rows after excluding essay_set {EXCLUDE_ESSAY_SETS}: "
          f"{len(df1):,} (dropped {before - len(df1):,})")

    # ── Ensure join key types match ────────────────────────────────────────────
    df1["essay_id"] = df1["essay_id"].astype(str).str.strip()
    df2["essay_id"] = df2["essay_id"].astype(str).str.strip()

    # ── Merge ─────────────────────────────────────────────────────────────────
    merged = pd.merge(df1, df2, on="essay_id", how="inner")
    print(f"Merged rows: {len(merged):,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    merged.to_csv(out, index=False)
    print(f"Saved → '{out}'")
    print(merged.head())


if __name__ == "__main__":
    main()