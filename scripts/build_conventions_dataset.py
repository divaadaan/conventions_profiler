from pathlib import Path

import pandas as pd
import seaborn as sns

RAW = Path("src/conventions_profiler/data/raw")
OUT = Path("src/conventions_profiler/data/processed")
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(RAW / "asap_pp_train.tsv", sep="\t")
text = train[["essay_id", "essay_set", "essay"]].copy()

#join with ASAP++ extension data
# Prompt 1
p1_scores = pd.read_csv(RAW / "asap_pp-prompt1-scores.csv")
p1_scores = p1_scores.rename(columns={"EssayID": "essay_id"})[["essay_id", "Conventions"]]

p1 = text[text["essay_set"] == 1].merge(p1_scores, on="essay_id", how="inner")
p1 = p1.rename(columns={"essay_set": "prompt_id", "Conventions": "conventions_score"})
p1 = p1[["essay_id", "prompt_id", "essay", "conventions_score"]]

# Prompt 2
p2_scores = pd.read_csv(RAW / "asap_pp-prompt2-scores.csv")
p2_scores = p2_scores.rename(columns={"Essay ID": "essay_id"})[["essay_id", "Conventions"]] # note this column has space whereas 1 did not

p2 = text[text["essay_set"] == 2].merge(p2_scores, on="essay_id", how="inner")
p2 = p2.rename(columns={"essay_set": "prompt_id", "Conventions": "conventions_score"})
p2 = p2[["essay_id", "prompt_id", "essay", "conventions_score"]]

# Prompt 7 — trait 4 = Conventions
p7_raw = train[train["essay_set"] == 7].copy()
p7_raw["conventions_score"] = p7_raw[["rater1_trait4", "rater2_trait4"]].mean(axis=1)
p7 = p7_raw[["essay_id", "essay_set", "essay", "conventions_score"]].rename(
    columns={"essay_set": "prompt_id"}
)

# Prompt 8 — trait 6 = Conventions
p8_raw = train[train["essay_set"] == 8].copy()
p8_raw["conventions_score"] = p8_raw[["rater1_trait6", "rater2_trait6"]].mean(axis=1)
p8 = p8_raw[["essay_id", "essay_set", "essay", "conventions_score"]].rename(
    columns={"essay_set": "prompt_id"}
)

# Combine
combined = pd.concat([p1, p2, p7, p8], ignore_index=True)
combined["essay_id"] = combined["essay_id"].astype(int)
combined["prompt_id"] = combined["prompt_id"].astype(int)

# Normalise to [0, 1] using known rubric bounds
scale_min = combined["prompt_id"].map({1: 1, 2: 1, 7: 0, 8: 1})
scale_range = combined["prompt_id"].map({1: 5, 2: 5, 7: 3, 8: 5})
combined["conventions_score_norm"] = (combined["conventions_score"] - scale_min) / scale_range

out_path = OUT / "asap_pp_conventions.csv"
combined.to_csv(out_path, index=False)
print(f"Saved {len(combined)} rows → {out_path}\n")

# Summary stats
print("=" * 60)
print("ESSAY COUNTS PER PROMPT")
print("=" * 60)
print(combined.groupby("prompt_id").size().rename("n_essays").to_string())

print()
print("=" * 60)
print("CONVENTIONS SCORE DISTRIBUTIONS (normalised 0–1)")
print("=" * 60)
for pid in [1, 2, 7, 8]:
    scores = combined.loc[combined["prompt_id"] == pid, "conventions_score_norm"]
    print(f"\nPrompt {pid}  (n={len(scores)})")
    print(f"  range : {scores.min():.3f} – {scores.max():.3f}")
    print(f"  mean  : {scores.mean():.3f}   std: {scores.std():.3f}")
    vc = scores.value_counts().sort_index()
    print(f"  value counts:\n{vc.to_string()}")

print()

# Visualise conventions score distributions
g = sns.displot(
    combined,
    x="conventions_score_norm",
    col="prompt_id",
    col_wrap=2,
    binwidth=0.1,
    binrange=(0, 1.0),
    height=4,
    aspect=1.2,
)
g.set(xlim=(0, 1))
g.set_titles("Prompt {col_name}")
g.set_axis_labels("Conventions Score (normalised)", "Count")
g.figure.suptitle("Conventions Score Distributions by Prompt", y=1.02)
g.savefig(OUT / "conventions_distributions.png", dpi=150)
print(f"Plot saved → {OUT / 'conventions_distributions.png'}")
