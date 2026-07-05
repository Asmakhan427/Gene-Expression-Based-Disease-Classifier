"""
logger.py - Centralised logging configuration.
"""

import logging
from pathlib import Path


def setup_logger(
    name: str = "gene_classifier",
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure and return a project-wide logger."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(name)
    log.setLevel(numeric_level)
    if not log.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(numeric_level)
        ch.setFormatter(formatter)
        log.addHandler(ch)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(numeric_level)
            fh.setFormatter(formatter)
            log.addHandler(fh)
    return log