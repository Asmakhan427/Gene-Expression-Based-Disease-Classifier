"""
config_loader.py - Central configuration loader.
"""

from pathlib import Path
from typing import Any
import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration file."""
    config_path = Path(path) if path else _CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(config: dict[str, Any]) -> None:
    """Create all directories defined in config."""
    for key in config.get("paths", {}):
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)