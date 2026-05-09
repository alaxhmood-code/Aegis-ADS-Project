from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def configure_logging(
    *,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    max_bytes: int = 5 * 1024 * 1024,
    backups: int = 3,
) -> logging.Logger:
    """
    Configure a clean logger for the capture layer.

    - Console handler always enabled.
    - Optional rotating file handler.
    """

    logger = logging.getLogger("aegis.capture")
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers if called multiple times.
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)sZ %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=str(path),
            maxBytes=max_bytes,
            backupCount=backups,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger

