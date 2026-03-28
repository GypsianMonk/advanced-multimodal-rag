"""
utils/logger.py
───────────────
Centralised Loguru logger configuration.
Writes to stdout (dev) and rotating file (production).
"""

from __future__ import annotations

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: str | None = "logs/rag.log",
    rotation: str = "10 MB",
    retention: str = "14 days",
) -> None:
    logger.remove()  # Remove default handler

    # Console handler — coloured, human-readable
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
        colorize=True,
    )

    # File handler — structured JSON for log aggregators
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
            rotation=rotation,
            retention=retention,
            compression="gz",
            serialize=False,
        )

    logger.info(f"Logger initialised — level={log_level}")
