"""Central logging configuration."""
from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def setup_logging(level: int = logging.INFO) -> None:
    logger = logging.getLogger()
    logger.setLevel(level)
    if any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        return  # already configured
    fh = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=2_000_000, backupCount=3)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
