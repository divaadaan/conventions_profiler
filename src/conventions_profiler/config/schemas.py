"""
Configuration schema and loader for the conventions profiler pipeline.

Loads from YAML, validates with pydantic, and resolves hardware settings.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class HardwareConfig(BaseModel):
    device: str = "auto"
    batch_size: int = 16

    def resolve_device(self) -> str:
        if self.device == "auto":
            try:
                import torch
                return "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device


class SpacyModelConfig(BaseModel):
    model: str = "en_core_web_trf"
    add_pipes: list[str] = Field(default_factory=lambda: ["textdescriptives/dependency_distance"])


class LanguageToolConfig(BaseModel):
    server_port: int = 8081
    language: str = "en-US"


class ModelsConfig(BaseModel):
    spacy: SpacyModelConfig = Field(default_factory=SpacyModelConfig)
    languagetool: LanguageToolConfig = Field(default_factory=LanguageToolConfig)


class CacheConfig(BaseModel):
    enabled: bool = True
    directory: str = "./cache"
    format: str = "parquet"

    @property
    def cache_path(self) -> Path:
        path = Path(self.directory)
        path.mkdir(parents=True, exist_ok=True)
        return path


class AsapPPConfig(BaseModel):
    path: str = "./src/conventions_profiler/data/processed/asap_pp_conventions.csv"


class DatasetsConfig(BaseModel):
    asap_pp: AsapPPConfig = Field(default_factory=AsapPPConfig)


class PipelineConfig(BaseModel):
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    datasets: DatasetsConfig = Field(default_factory=DatasetsConfig)


def load_config(config_path: str | Path = "config/default.yaml") -> PipelineConfig:
    """Load and validate pipeline configuration from a YAML file."""
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        return PipelineConfig(**raw)
    return PipelineConfig()
