"""Inter-feature correlation analysis and redundancy pruning."""

import pandas as pd


def compute_feature_correlations(
    data: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Spearman correlation matrix between features."""
    return data[features].corr(method="spearman")


def prune_correlated_features(
    data: pd.DataFrame,
    features: list[str],
    strength_lookup: dict[str, float],
    threshold: float = 0.85,
) -> tuple[list[str], list[dict]]:
    """Iterative greedy pruning of highly correlated features.
    For each pair above threshold, drops the feature with the weaker
    cross-prompt signal (lower value in strength_lookup)."""
    pruned = set(features)
    dropped_pairs = []

    while True:
        remaining = sorted(pruned)
        corr_mat = data[remaining].corr(method="spearman")

        pairs = []
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                r = abs(corr_mat.iloc[i, j])
                if r > threshold:
                    pairs.append((r, remaining[i], remaining[j]))

        if not pairs:
            break

        pairs.sort(reverse=True)
        r_val, feat_a, feat_b = pairs[0]

        rho_a = strength_lookup.get(feat_a, 0)
        rho_b = strength_lookup.get(feat_b, 0)
        if rho_a >= rho_b:
            keep, drop = feat_a, feat_b
        else:
            keep, drop = feat_b, feat_a

        pruned.discard(drop)
        dropped_pairs.append({
            "kept": keep,
            "dropped": drop,
            "correlation": round(r_val, 3),
            "kept_strength": round(strength_lookup.get(keep, 0), 3),
            "dropped_strength": round(strength_lookup.get(drop, 0), 3),
        })

    return sorted(pruned), dropped_pairs
