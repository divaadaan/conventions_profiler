"""Checkpoint pass results."""

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
    """Load a checkpoint if it exists and covers all essay IDs."""
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

    # Build lookup table and populate
    lookup = dict(zip(df["essay_id"].astype(str), df["pass_results"]))
    for essay in essays:
        essay.pass_results[pass_name] = json.loads(lookup[essay.essay_id])

    logger.info(f"Checkpoint loaded: {path} ({len(essays)} essays)")
    return True


def save_checkpoint_incremental(
    essays: list[Essay], pass_name: str, checkpoint_dir: Path
) -> None:
    """Append pass results for a chunk of essays to the checkpoint file."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{pass_name}.parquet"

    new_rows = []
    for essay in essays:
        data = essay.pass_results.get(pass_name, {})
        new_rows.append({
            "essay_id": essay.essay_id,
            "pass_results": json.dumps(data, default=str),
        })
    new_df = pd.DataFrame(new_rows)

    if path.exists():
        existing_df = pd.read_parquet(path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="essay_id", keep="last")
    else:
        combined = new_df

    combined.to_parquet(path, index=False)
    logger.debug(
        f"Incremental checkpoint saved: {path} "
        f"({len(new_rows)} new, {len(combined)} total)"
    )


def load_partial_checkpoint(
    essays: list[Essay], pass_name: str, checkpoint_dir: Path
) -> list[Essay]:
    """Load checkpoint and return only the essays NOT yet checkpointed."""
    path = checkpoint_dir / f"{pass_name}.parquet"
    if not path.exists():
        return essays

    df = pd.read_parquet(path)
    lookup = dict(zip(df["essay_id"].astype(str), df["pass_results"]))

    remaining = []
    restored = 0
    for essay in essays:
        if essay.essay_id in lookup:
            essay.pass_results[pass_name] = json.loads(lookup[essay.essay_id])
            restored += 1
        else:
            remaining.append(essay)

    if restored:
        logger.info(
            f"Partial checkpoint for '{pass_name}': "
            f"restored {restored}, remaining {len(remaining)}"
        )

    return remaining
