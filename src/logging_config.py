"""Centralized logging setup used by every script, module, and notebook.

Usage
-----
>>> from src.logging_config import get_logger
>>> log = get_logger(__name__)
>>> log.info("Loaded %d samples", n)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIGURED = False


def setup_logging(level: str = "INFO", log_dir: str = "logs", log_file: str = "pipeline.log") -> None:
    """Configure the root logger once (idempotent across repeated calls)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_path = REPO_ROOT / log_dir
    log_path.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path / log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
