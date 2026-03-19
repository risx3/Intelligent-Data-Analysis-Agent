"""
Centralized logging configuration for the Intelligent Data Analysis Agent.

All modules should import the logger via:
    from utils.logger import get_logger
    logger = get_logger(__name__)
"""
import logging
import os
import sys
from config import BASE_DIR

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE  = os.getenv("LOG_FILE", os.path.join(BASE_DIR, "outputs", "agent.log"))

_fmt = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(_fmt)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_fmt)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    handlers=[_file_handler, _console_handler],
)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger sharing the root configuration."""
    return logging.getLogger(name)
