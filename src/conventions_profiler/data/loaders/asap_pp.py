"""
ASAP++ data loader.

Loads essay text from the original ASAP dataset and joins with ASAP++
conventions annotations. Handles the different scoring scales across prompts
and computes z-scored conventions labels for cross-prompt analysis.

ASAP++ conventions labels:
    Prompts 1, 2: "Conventions" trait, scale 1-6
    Prompts 7, 8: "Conventions" trait, scale 0-3
    Prompts 3-6:  "Language" trait (conventions + fluency blend) — excluded by default

Data sources:
    - Essay text: Original ASAP Kaggle competition CSV
    - Annotations: ASAP++ supplementary data (Mathias & Bhattacharyya, 2018)
"""

import logging
from pathlib import Path

import pandas as pd

from conventions_profiler.models import Essay

logger = logging.getLogger(__name__)

# Conventions score column names vary by ASAP++ data format.
# These are the known variants — the loader tries each in order.
CONVENTIONS_COLUMN_CANDIDATES = [
    "conventions",
    "Conventions",
    "trait_conventions",
    "conventions_score",
]

# The "Language" trait in prompts 3-6 blends conventions with fluency.
# We exclude these by default but allow loading them as noisy proxies.
LANGUAGE_COLUMN_CANDIDATES = [
    "language",
    "Language",
    "trait_language",
]

# Score ranges per prompt for validation
SCORE_RANGES = {
    1: (1, 6),
    2: (1, 6),
    7: (0, 3),
    8: (0, 3),
}


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name from a list of candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _z_score_within_prompt(df: pd.DataFrame) -> pd.Series:
    """Z-score normalize conventions scores within each prompt."""
    return df.groupby("prompt_id")["conventions_score"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
    )


def load_asap_pp(
    essays_path: str | Path,
    annotations_path: str | Path,
    prompts: list[int] | None = None,
) -> list[Essay]:
    """Load ASAP++ essays with conventions annotations.

    Parameters
    ----------
    essays_path
        Path to the ASAP essay text CSV. Expected columns: essay_id, essay_set
        (prompt_id), essay (text). This is the standard Kaggle ASAP format.
    annotations_path
        Path to the ASAP++ annotations CSV. Must contain essay_id and a
        conventions score column.
    prompts
        Which prompts to load. Defaults to [1, 2, 7, 8] — the prompts with
        dedicated conventions annotations.

    Returns
    -------
    list[Essay]
        Essay objects with text, prompt_id, conventions_score, and conventions_z.
    """
    if prompts is None:
        prompts = [1, 2, 7, 8]

    essays_path = Path(essays_path)
    annotations_path = Path(annotations_path)

    if not essays_path.exists():
        raise FileNotFoundError(f"Essays file not found: {essays_path}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    # Load essay text
    essays_df = pd.read_csv(essays_path, encoding="utf-8")

    # Normalize column names — ASAP format varies across mirrors
    col_map = {}
    for col in essays_df.columns:
        col_lower = col.lower().strip()
        if col_lower in ("essay_id", "essayid", "id"):
            col_map[col] = "essay_id"
        elif col_lower in ("essay_set", "essayset", "prompt_id", "prompt"):
            col_map[col] = "prompt_id"
        elif col_lower in ("essay", "essay_text", "text"):
            col_map[col] = "text"
    essays_df = essays_df.rename(columns=col_map)

    required = {"essay_id", "prompt_id", "text"}
    missing = required - set(essays_df.columns)
    if missing:
        raise ValueError(
            f"Essays CSV missing columns: {missing}. "
            f"Found: {list(essays_df.columns)[:10]}"
        )

    # Load annotations
    annotations_df = pd.read_csv(annotations_path, encoding="utf-8")

    # Find the conventions score column
    conv_col = _find_column(annotations_df, CONVENTIONS_COLUMN_CANDIDATES)
    if conv_col is None:
        # Fall back to Language trait as noisy proxy
        lang_col = _find_column(annotations_df, LANGUAGE_COLUMN_CANDIDATES)
        if lang_col is not None:
            logger.warning(
                f"No conventions column found. Using '{lang_col}' as noisy proxy."
            )
            conv_col = lang_col
        else:
            raise ValueError(
                f"No conventions or language column found in annotations. "
                f"Columns: {list(annotations_df.columns)}"
            )

    # Normalize annotation column names
    ann_id_col = None
    for col in annotations_df.columns:
        if col.lower().strip() in ("essay_id", "essayid", "id"):
            ann_id_col = col
            break
    if ann_id_col is None:
        raise ValueError("Annotations CSV missing essay_id column.")

    annotations_df = annotations_df.rename(columns={
        ann_id_col: "essay_id",
        conv_col: "conventions_score",
    })

    # Join essays with annotations
    merged = essays_df.merge(
        annotations_df[["essay_id", "conventions_score"]],
        on="essay_id",
        how="inner",
    )

    # Filter to requested prompts
    merged = merged[merged["prompt_id"].isin(prompts)].copy()

    if len(merged) == 0:
        raise ValueError(
            f"No essays found for prompts {prompts}. "
            f"Available prompts: {sorted(essays_df['prompt_id'].unique())}"
        )

    # Validate score ranges
    for pid in prompts:
        prompt_data = merged[merged["prompt_id"] == pid]
        if len(prompt_data) == 0:
            logger.warning(f"No essays found for prompt {pid}")
            continue
        expected_min, expected_max = SCORE_RANGES.get(pid, (0, 10))
        actual_min = prompt_data["conventions_score"].min()
        actual_max = prompt_data["conventions_score"].max()
        if actual_min < expected_min or actual_max > expected_max:
            logger.warning(
                f"Prompt {pid}: scores range [{actual_min}, {actual_max}], "
                f"expected [{expected_min}, {expected_max}]"
            )

    # Z-score within prompt
    merged["conventions_z"] = _z_score_within_prompt(merged)

    # Build Essay objects
    essays = []
    for _, row in merged.iterrows():
        essays.append(Essay(
            essay_id=str(row["essay_id"]),
            text=str(row["text"]),
            prompt_id=int(row["prompt_id"]),
            conventions_score=float(row["conventions_score"]),
            conventions_z=float(row["conventions_z"]),
        ))

    logger.info(
        f"Loaded {len(essays)} essays from ASAP++ "
        f"(prompts {prompts}, "
        f"score distributions: {merged.groupby('prompt_id')['conventions_score'].describe()[['mean', 'std']].to_dict()})"
    )

    return essays
