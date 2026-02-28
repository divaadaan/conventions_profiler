"""End-to-end analysis orchestrator."""

import logging
from pathlib import Path

import pandas as pd

from conventions_profiler.analysis.validation import run_full_analysis, cross_prompt_ranking
from conventions_profiler.analysis.correlation import prune_correlated_features
from conventions_profiler.analysis.cut_points import compute_cut_points

logger = logging.getLogger(__name__)

METADATA_COLS = {"essay_id", "prompt_id", "score", "word_count"}


def run_validation_pipeline(
    feature_matrix_path: Path,
    output_dir: Path,
    target: str = "score",
    prompts: list | None = None,
    corr_threshold: float = 0.85,
    consistency_threshold: float = 0.75,
    discrete_scores: bool = False,
) -> dict:
    """Run full analysis: per-prompt validation, cross-prompt ranking,
    correlation pruning, cut point derivation. Exports CSVs.
    Returns dict with keys: prompt_results, cross_ranking, pruned_features,
    dropped_pairs, feature_cuts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(feature_matrix_path)
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]

    if prompts is None:
        prompts = sorted(df["prompt_id"].unique())

    # Per-prompt analysis
    prompt_results = {}
    for pid in prompts:
        subset = df[df["prompt_id"] == pid]
        result = run_full_analysis(subset, feature_cols, target)
        result["prompt_id"] = pid
        prompt_results[pid] = result

        n_sp = result["spearman_sig"].sum()
        n_mw = result["mwu_sig"].sum()
        logger.info(
            f"Prompt {pid}: N={len(subset)}, "
            f"Spearman sig={n_sp}/{len(feature_cols)}, "
            f"MWU sig={n_mw}/{len(feature_cols)}"
        )

    # Cross-prompt ranking
    ranking = cross_prompt_ranking(prompt_results, feature_cols)
    n_prompts = len(prompts)

    # Filter to consistent features
    import math
    min_prompts = math.ceil(n_prompts * consistency_threshold)
    consistent_mask = (
        (ranking["n_significant"] >= min_prompts) & ranking["sign_consistent"]
    )
    consistent_feats = ranking[consistent_mask]["feature"].tolist()
    logger.info(f"Cross-prompt consistent features: {len(consistent_feats)}/{len(feature_cols)}")

    # Correlation pruning
    strength_lookup = ranking.set_index("feature")["min_abs_rho"].to_dict()
    pruned_features, dropped_pairs = prune_correlated_features(
        df, consistent_feats, strength_lookup, threshold=corr_threshold,
    )
    logger.info(
        f"Correlation pruning (|r| > {corr_threshold}): "
        f"{len(consistent_feats)} -> {len(pruned_features)}"
    )

    # Cut points
    feature_cuts = compute_cut_points(
        df, pruned_features, target, ranking,
        discrete_scores=discrete_scores, prompts=prompts,
    )
    logger.info(f"Cut points derived for {len(feature_cuts)} features")

    # Export
    for pid in prompts:
        prompt_results[pid].to_csv(output_dir / f"spearman_mwu_prompt_{pid}.csv", index=False)
    ranking.to_csv(output_dir / "cross_prompt_ranking.csv", index=False)
    if dropped_pairs:
        pd.DataFrame(dropped_pairs).to_csv(output_dir / "correlation_pruning.csv", index=False)
    cut_cols = [c for c in feature_cuts.columns if c.startswith("cut_")]
    deliverable_cols = ["feature", "description"] + cut_cols + ["direction", "module"]
    feature_cuts[deliverable_cols].to_csv(output_dir / "feature_cuts.csv", index=False)

    logger.info(f"Results exported to {output_dir}")

    return {
        "prompt_results": prompt_results,
        "cross_ranking": ranking,
        "pruned_features": pruned_features,
        "dropped_pairs": dropped_pairs,
        "feature_cuts": feature_cuts,
    }
