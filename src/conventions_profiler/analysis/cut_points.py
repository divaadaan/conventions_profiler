"""Cut point derivation for feature scoring bands."""

import numpy as np
import pandas as pd

from conventions_profiler.features.registry import FEATURE_REGISTRY


def _quartile_banding(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    n_bands: int,
    prompts: list,
) -> dict[str, dict[int, list[float]]]:
    """Band essays by quartile boundaries on the target contiunuous score."""
    band_medians_all: dict[str, dict[int, list[float]]] = {}

    for pid in prompts:
        subset = data[data["prompt_id"] == pid].copy()
        scores = subset[target]

        raw_bounds = scores.quantile([0.25, 0.5, 0.75]).values
        boundaries = np.unique(raw_bounds)
        subset["_band"] = np.digitize(scores, boundaries, right=False)
        actual_bands = sorted(subset["_band"].unique())

        for feat in features:
            if feat not in band_medians_all:
                band_medians_all[feat] = {b: [] for b in range(n_bands)}
            for target_band in range(n_bands):
                mapped = actual_bands[
                    min(target_band * len(actual_bands) // n_bands, len(actual_bands) - 1)
                ]
                band_data = subset[subset["_band"] == mapped][feat].fillna(0)
                band_medians_all[feat][target_band].append(band_data.median())

    return band_medians_all


def _discrete_banding(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    prompts: list,
) -> dict[str, dict[int, list[float]]]:
    """Band essays directly by discrete score levels."""
    levels = sorted(data[target].unique())
    n_bands = len(levels)
    band_medians_all: dict[str, dict[int, list[float]]] = {}

    for pid in prompts:
        subset = data[data["prompt_id"] == pid]
        for feat in features:
            if feat not in band_medians_all:
                band_medians_all[feat] = {b: [] for b in range(n_bands)}
            for band_idx, level in enumerate(levels):
                band_data = subset[subset[target] == level][feat].fillna(0)
                if len(band_data) > 0:
                    band_medians_all[feat][band_idx].append(band_data.median())

    return band_medians_all


def compute_cut_points(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    cross_ranking: pd.DataFrame,
    discrete_scores: bool = False,
    n_bands: int = 4,
    prompts: list | None = None,
) -> pd.DataFrame:
    """Derive feature cut points from band medians - either from a discrete or conitnuous score label set in the scoringconfig"""
    if prompts is None:
        prompts = sorted(data["prompt_id"].unique())

    if discrete_scores:
        band_medians_all = _discrete_banding(data, features, target, prompts)
        n_bands = len(sorted(data[target].unique()))
    else:
        band_medians_all = _quartile_banding(data, features, target, n_bands, prompts)

    cr_lookup = cross_ranking.set_index("feature")
    cut_rows = []
    for feat in features:
        avg_medians = [
            np.mean(band_medians_all[feat][b])
            for b in range(n_bands)
            if band_medians_all[feat][b]
        ]
        if len(avg_medians) < 2:
            continue

        cuts = [(avg_medians[i] + avg_medians[i + 1]) / 2 for i in range(len(avg_medians) - 1)]

        cr_row = cr_lookup.loc[feat]
        spec = FEATURE_REGISTRY.get(feat)
        description = spec.description if spec else ""

        row = {
            "feature": feat,
            "description": description,
            **{f"cut_{i+1}": cuts[i] for i in range(len(cuts))},
            **{f"band_{i}_median": avg_medians[i] for i in range(len(avg_medians))},
            "direction": cr_row["direction"],
            "module": cr_row["module"],
            "min_abs_rho": cr_row["min_abs_rho"],
            "mean_abs_rho": cr_row["mean_abs_rho"],
        }
        cut_rows.append(row)

    result = pd.DataFrame(cut_rows).sort_values("min_abs_rho", ascending=False).reset_index(drop=True)

    # Exclude features with degenerate cut points
    cut_cols = [c for c in result.columns if c.startswith("cut_")]
    if len(cut_cols) >= 2:
        degenerate_mask = pd.Series(False, index=result.index)
        for i in range(len(cut_cols)):
            for j in range(i + 1, len(cut_cols)):
                degenerate_mask |= (result[cut_cols[i]] == result[cut_cols[j]])
        if degenerate_mask.any():
            result = result[~degenerate_mask].reset_index(drop=True)

    return result
