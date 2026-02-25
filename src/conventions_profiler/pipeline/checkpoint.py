"""
Checkpoint layer for pass results.

Saves per-pass results as parquet files after each pass completes, enabling
resumption without re-running expensive passes. Each checkpoint stores
essay_id and JSON-serialized pass results, and validates coverage before
loading.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from conventions_profiler.models import Essay

logger = logging.getLogger(__name__)


def save_checkpoint(
    essays: list[Essay], pass_name: str, checkpoint_dir: Path
) -> None:
    """Serialize pass results for all essays to a parquet checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for essay in essays:
        data = essay.pass_results.get(pass_name, {})
        rows.append({
            "essay_id": essay.essay_id,
            "pass_results": json.dumps(data, default=str),
        })
    df = pd.DataFrame(rows)
    path = checkpoint_dir / f"{pass_name}.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(
    essays: list[Essay], pass_name: str, checkpoint_dir: Path
) -> bool:
    """Load a checkpoint if it exists and covers all essay IDs.

    Returns True if the checkpoint was loaded successfully, False otherwise.
    """
    path = checkpoint_dir / f"{pass_name}.parquet"
    if not path.exists():
        return False

    df = pd.read_parquet(path)
    checkpoint_ids = set(df["essay_id"].astype(str))
    essay_ids = {e.essay_id for e in essays}

    if not essay_ids.issubset(checkpoint_ids):
        missing = essay_ids - checkpoint_ids
        logger.info(
            f"Checkpoint {path.name} missing {len(missing)} essay(s) — rerunning"
        )
        return False

    # Build lookup and populate
    lookup = dict(zip(df["essay_id"].astype(str), df["pass_results"]))
    for essay in essays:
        essay.pass_results[pass_name] = json.loads(lookup[essay.essay_id])

    logger.info(f"Checkpoint loaded: {path} ({len(essays)} essays)")
    return True
