import logging
from pathlib import Path

import pandas as pd

from conventions_profiler.data.deanonymize import deanonymize
from conventions_profiler.models import Essay

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = {"essay_id", "prompt_id", "essay", "conventions_score", "conventions_score_norm"}


def load_asap_pp(path: str | Path) -> list[Essay]:
    """Load ASAP++ essays from the pre-processed conventions CSV.

    Parameters
    ----------
    path
        Path to the processed CSV with columns: essay_id, prompt_id, essay,
        conventions_score, conventions_score_norm.

    Returns
    -------
    list[Essay]
        Essay objects with text, prompt_id, conventions_score, and
        conventions_score_norm.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed CSV not found: {path}")

    df = pd.read_csv(path, encoding="utf-8")

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Processed CSV missing columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    essays = []
    for _, row in df.iterrows():
        essays.append(Essay(
            essay_id=str(row["essay_id"]),
            text=deanonymize(str(row["essay"])),
            prompt_id=int(row["prompt_id"]),
            conventions_score=float(row["conventions_score"]),
            conventions_score_norm=float(row["conventions_score_norm"]),
        ))

    logger.info(
        f"Loaded {len(essays)} essays from {path} "
        f"(prompts {sorted(df['prompt_id'].unique())})"
    )

    return essays
