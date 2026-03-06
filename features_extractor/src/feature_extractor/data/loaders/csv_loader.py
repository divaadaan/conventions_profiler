"""Generic CSV dataset loader with configurable column mapping."""

import logging
from pathlib import Path

import pandas as pd

from feature_extractor.models import Essay

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"essay_id", "text", "score"}


def load_dataset(
    path: str | Path,
    columns: dict[str, str] | None = None,
) -> list[Essay]:
    """Load essays from a CSV file with configurable column mapping.
    Parameters
    path:Path to the CSV file.
    columns:Mapping from internal field names to CSV column names.
    Defaults: essay_id="essay_id", text="text", prompt_id="prompt_id", score="score".

    Returns: list[Essay] objects with text, prompt_id, and score populated."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {path}")

    col_map = {
        "essay_id": "essay_id",
        "text": "text",
        "prompt_id": "prompt_id",
        "score": "score",
    }
    if columns:
        col_map.update(columns)

    df = pd.read_csv(path, encoding="utf-8")

    # Validate required columns exist in the CSV
    missing = []
    for field in REQUIRED_FIELDS:
        csv_col = col_map[field]
        if csv_col not in df.columns:
            missing.append(f"{field} (mapped to '{csv_col}')")
    if missing:
        raise ValueError(
            f"Dataset CSV missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    # prompt_id is optional
    prompt_col = col_map["prompt_id"]
    has_prompt = prompt_col in df.columns

    essays = []
    for _, row in df.iterrows():
        essays.append(Essay(
            essay_id=str(row[col_map["essay_id"]]),
            text=str(row[col_map["text"]]),
            prompt_id=row[prompt_col] if has_prompt else 1,
            score=float(row[col_map["score"]]),
        ))

    prompts = sorted(df[prompt_col].unique()) if has_prompt else [1]
    logger.info(
        f"Loaded {len(essays)} essays from {path} "
        f"(prompts: {prompts})"
    )

    return essays
