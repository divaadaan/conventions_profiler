"""Core statistical tests: Spearman correlation, Mann-Whitney U, FDR correction."""

import numpy as np
import pandas as pd
from scipy import stats


MODULE_MAP = {
    "dep_": "Dependency",
    "syn_": "Syntactic",
    "punct_": "Punctuation",
    "err_": "Error Severity",
    "spell_": "Spelling",
    "agg_": "Aggregation",
}


def get_module(feature_name: str) -> str:
    for prefix, name in MODULE_MAP.items():
        if feature_name.startswith(prefix):
            return name
    return "Unknown"


def spearman_analysis(
    data: pd.DataFrame,
    features: list[str],
    target: str,
) -> pd.DataFrame:
    """Spearman rho + p-value for each feature vs target."""
    rows = []
    y = data[target].values
    for feat in features:
        x = data[feat].fillna(0).values
        if np.std(x) == 0:
            rows.append({"feature": feat, "spearman_rho": 0.0, "spearman_p": 1.0})
            continue
        rho, p = stats.spearmanr(x, y)
        rows.append({"feature": feat, "spearman_rho": rho, "spearman_p": p})
    return pd.DataFrame(rows)


def mann_whitney_analysis(
    data: pd.DataFrame,
    features: list[str],
    target: str,
) -> pd.DataFrame:
    """Mann-Whitney U with median split. Rank-biserial r = 1 - 2U/(n1*n2)."""
    median_val = data[target].median()
    low = data[data[target] < median_val]
    high = data[data[target] > median_val]
    n_excluded = len(data) - len(low) - len(high)

    rows = []
    for feat in features:
        x_low = low[feat].fillna(0).values
        x_high = high[feat].fillna(0).values
        if len(x_low) < 2 or len(x_high) < 2:
            rows.append({
                "feature": feat, "mwu_U": np.nan, "mwu_p": np.nan,
                "rank_biserial_r": np.nan, "n_low": len(x_low),
                "n_high": len(x_high), "n_excluded": n_excluded,
            })
            continue
        if np.std(x_low) == 0 and np.std(x_high) == 0 and x_low[0] == x_high[0]:
            rows.append({
                "feature": feat, "mwu_U": len(x_low) * len(x_high) / 2,
                "mwu_p": 1.0, "rank_biserial_r": 0.0,
                "n_low": len(x_low), "n_high": len(x_high),
                "n_excluded": n_excluded,
            })
            continue
        U, p = stats.mannwhitneyu(x_low, x_high, alternative="two-sided")
        n1, n2 = len(x_low), len(x_high)
        r = 1 - (2 * U) / (n1 * n2)
        rows.append({
            "feature": feat, "mwu_U": U, "mwu_p": p,
            "rank_biserial_r": r, "n_low": n1, "n_high": n2,
            "n_excluded": n_excluded,
        })
    return pd.DataFrame(rows)


def apply_fdr(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction. Returns (q_values, significant_mask)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)

    nan_mask = np.isnan(p)
    p_clean = np.where(nan_mask, 1.0, p)

    order = np.argsort(p_clean)
    ranked = np.empty(n, dtype=float)
    ranked[order] = np.arange(1, n + 1)

    q = p_clean * n / ranked
    rev_order = np.argsort(-ranked)
    q_sorted = q[rev_order]
    q_cummin = np.minimum.accumulate(q_sorted)
    q[rev_order] = q_cummin
    q = np.clip(q, 0, 1)

    q[nan_mask] = np.nan
    sig = np.where(nan_mask, False, q <= alpha)

    return q, sig


def run_full_analysis(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run Spearman + MWU + FDR, return merged DataFrame sorted by |rho|."""
    sp = spearman_analysis(data, features, target)
    mw = mann_whitney_analysis(data, features, target)
    merged = sp.merge(mw, on="feature")

    sp_q, sp_sig = apply_fdr(merged["spearman_p"].values, alpha)
    merged["spearman_q"] = sp_q
    merged["spearman_sig"] = sp_sig

    mw_q, mw_sig = apply_fdr(merged["mwu_p"].fillna(1).values, alpha)
    merged["mwu_q"] = mw_q
    merged["mwu_sig"] = mw_sig

    merged["abs_rho"] = merged["spearman_rho"].abs()
    merged["abs_r"] = merged["rank_biserial_r"].abs()
    merged["module"] = merged["feature"].map(get_module)

    return merged.sort_values("abs_rho", ascending=False).reset_index(drop=True)


def cross_prompt_ranking(
    prompt_results: dict[int | str, pd.DataFrame],
    features: list[str],
) -> pd.DataFrame:
    """Rank features by cross-prompt consistency of Spearman rho."""
    prompts = sorted(prompt_results.keys())
    rows = []
    for feat in features:
        rhos = []
        n_sig = 0
        signs = []
        for pid in prompts:
            row = prompt_results[pid][prompt_results[pid]["feature"] == feat].iloc[0]
            rho_val = row["spearman_rho"]
            rhos.append(rho_val)
            if row["spearman_sig"] or row["mwu_sig"]:
                n_sig += 1
            if rho_val != 0:
                signs.append(np.sign(rho_val))

        abs_rhos = [abs(r) for r in rhos]
        sign_consistent = len(set(signs)) <= 1
        mean_rho = np.mean(rhos)

        rows.append({
            "feature": feat,
            "module": get_module(feat),
            "mean_rho": mean_rho,
            "mean_abs_rho": np.mean(abs_rhos),
            "min_abs_rho": np.min(abs_rhos),
            "max_abs_rho": np.max(abs_rhos),
            "n_significant": n_sig,
            "n_prompts": len(prompts),
            "sign_consistent": sign_consistent,
            "direction": "higher_is_better" if mean_rho > 0 else "lower_is_better",
        })

    return (
        pd.DataFrame(rows)
        .sort_values("min_abs_rho", ascending=False)
        .reset_index(drop=True)
    )
