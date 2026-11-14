"""
Configuration loader.

Loads YAML config files and provides a unified configuration object
for the entire experiment pipeline.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# Project root is two levels up from this file (src/utils/config.py -> ds_project/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file and return as dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(
    config_path: Optional[str] = None,
    datasets_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and merge the default config and datasets config.

    Args:
        config_path: Path to default.yaml. Uses configs/default.yaml if None.
        datasets_path: Path to datasets.yaml. Uses configs/datasets.yaml if None.

    Returns:
        Merged configuration dictionary.
    """
    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "default.yaml")
    if datasets_path is None:
        datasets_path = str(PROJECT_ROOT / "configs" / "datasets.yaml")

    config = load_yaml(config_path)
    datasets_config = load_yaml(datasets_path)
    config.update(datasets_config)

    # Resolve relative paths to absolute
    for key in ["raw_dir", "processed_dir"]:
        if key in config.get("data", {}):
            config["data"][key] = str(PROJECT_ROOT / config["data"][key])

    config["results"]["base_dir"] = str(
        PROJECT_ROOT / config["results"]["base_dir"]
    )
    config["_project_root"] = str(PROJECT_ROOT)

    return config
