"""
Caching layer for intermediate pass results and final feature matrices.

Stores per-pass results as parquet files so extraction only runs once.
The feature matrix is stored separately and regenerated when feature
computation logic changes without re-running passes.
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class PipelineCache:
    """Manages caching of pass results and feature matrices."""

    def __init__(self, cache_dir: str | Path, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _pass_cache_path(self, pass_name: str) -> Path:
        return self.cache_dir / f"pass_{pass_name}.json"

    def _feature_matrix_path(self) -> Path:
        return self.cache_dir / "feature_matrix.parquet"

    def has_pass_results(self, pass_name: str) -> bool:
        if not self.enabled:
            return False
        return self._pass_cache_path(pass_name).exists()

    def save_pass_results(self, pass_name: str, results: dict[str, dict]) -> None:
        """Save pass results keyed by essay_id.

        Parameters
        ----------
        pass_name
            Name of the pass (e.g., "spacy", "languagetool").
        results
            Dict mapping essay_id -> pass result dict.
            Results must be JSON-serializable (no spaCy Doc objects).
        """
        if not self.enabled:
            return

        path = self._pass_cache_path(pass_name)
        with open(path, "w") as f:
            json.dump(results, f)
        logger.info(f"Cached {len(results)} results for pass '{pass_name}' -> {path}")

    def load_pass_results(self, pass_name: str) -> dict[str, dict]:
        """Load cached pass results."""
        path = self._pass_cache_path(pass_name)
        if not path.exists():
            raise FileNotFoundError(f"No cache found for pass '{pass_name}' at {path}")
        with open(path) as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} cached results for pass '{pass_name}'")
        return results

    def save_feature_matrix(self, df: pd.DataFrame) -> None:
        """Save the complete feature matrix as parquet."""
        if not self.enabled:
            return
        path = self._feature_matrix_path()
        df.to_parquet(path, index=False)
        logger.info(f"Saved feature matrix ({df.shape}) -> {path}")

    def load_feature_matrix(self) -> pd.DataFrame:
        """Load the cached feature matrix."""
        path = self._feature_matrix_path()
        if not path.exists():
            raise FileNotFoundError(f"No feature matrix cache at {path}")
        df = pd.read_parquet(path)
        logger.info(f"Loaded feature matrix ({df.shape}) from cache")
        return df

    def has_feature_matrix(self) -> bool:
        if not self.enabled:
            return False
        return self._feature_matrix_path().exists()

    def clear(self) -> None:
        """Remove all cached files."""
        if self.cache_dir.exists():
            for f in self.cache_dir.iterdir():
                f.unlink()
            logger.info(f"Cleared cache at {self.cache_dir}")
