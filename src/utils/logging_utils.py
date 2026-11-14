"""
Logging utilities for experiment tracking.

Provides structured logging for training progress, results, and
experiment metadata.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(
    name: str = "mambatab",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create and configure a logger.

    Args:
        name: Logger name.
        log_dir: Directory for log files. If None, logs to stdout only.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{name}_{timestamp}.log")
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_experiment_meta(
    save_path: str,
    config: Dict[str, Any],
    setting: str,
    models: list,
    datasets: list,
) -> None:
    """Save experiment metadata to a JSON file.

    Args:
        save_path: Path to save the metadata file.
        config: Configuration dictionary.
        setting: Experimental setting (vanilla, ssl, fil).
        models: List of model names.
        datasets: List of dataset names.
    """
    meta = {
        "timestamp": datetime.now().isoformat(),
        "setting": setting,
        "models": models,
        "datasets": datasets,
        "num_seeds": config["training"]["num_seeds"],
        "seeds": config["training"]["seeds"],
        "config": {
            k: v
            for k, v in config.items()
            if k not in ("datasets", "_project_root")
        },
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
