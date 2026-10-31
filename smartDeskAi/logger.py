"""
logger.py — Centralized logging configuration for SmartDeskAiAssistant.
"""

import logging
import logging.config
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "log"

# 1. Define your centralized logging dictionary configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False, # Keeps third-party package loggers alive
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
    },
    "handlers": {
        "console": {
            # Handled errors are reported to the employee politely; details go to log/app.log
            "class": "logging.StreamHandler",
            "level": "CRITICAL",
            "formatter": "standard",
            "stream": "ext://sys.stderr",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler", # Rotates files when they get large
            "level": "DEBUG",
            "formatter": "standard",
            "filename": str(LOG_DIR / "app.log"),
            "maxBytes": 10485760, # 10MB limit
            "backupCount": 3,     # Keep up to 3 old log files
            "encoding": "utf8"
        },
    },
    "loggers": {
        "root": { # Root logger configuration
            "handlers": ["console", "file"],
            "level": "DEBUG",
        },
        # Keep chatty third-party libraries out of the debug log
        **{name: {"level": "WARNING"} for name in ("httpx", "httpcore", "openai", "urllib3", "jira", "fastembed")},
    },
}

_configured = False


def setup_logger(name: str) -> logging.Logger:
    """Return a named logger, applying the logging configuration once per process."""
    global _configured
    if not _configured:
        LOG_DIR.mkdir(exist_ok=True)
        logging.config.dictConfig(LOGGING_CONFIG)
        _configured = True
    return logging.getLogger(name)


def set_console_level(level: int) -> None:
    """Change how much is echoed to the console (e.g. DEBUG with --verbose)."""
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(level)
