"""
telemetry/logger.py
Structured logging setup for the Edge AI Navigation System.

Features:
  - JSON-formatted log records (structlog-style without the dependency)
  - Rotating file handler with configurable size/backup count
  - Console handler with colour coding (when a TTY is attached)
  - Module-level loggers inherit the root configuration automatically
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.config_loader import LoggingConfig


# ANSI colour codes for TTY console output
_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}
_RESET = "\033[0m"


class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":      time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
            "module":  record.module,
            "line":    record.lineno,
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class _ColourFormatter(logging.Formatter):
    """Human-readable coloured formatter for TTY output."""

    FMT = "{colour}[{level}]{reset} {ts}  {name}:{line}  {msg}"

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, "")
        return self.FMT.format(
            colour=colour,
            reset=_RESET,
            level=record.levelname[0],          # single letter
            ts=time.strftime("%H:%M:%S", time.localtime(record.created)),
            name=record.name,
            line=record.lineno,
            msg=record.getMessage(),
        )


def setup_logging(cfg: LoggingConfig) -> None:
    """
    Configure the root logger once at startup.

    Args:
        cfg: LoggingConfig instance from config_loader.
    """
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(cfg.level)

    # Remove any handlers added by earlier imports
    root.handlers.clear()

    # ── File handler (rotating, JSON) ──────────────────────────────────────
    log_file = log_dir / "edge_ai_nav.log"
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=cfg.max_bytes,
        backupCount=cfg.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(cfg.level)
    file_handler.setFormatter(
        _JSONFormatter() if cfg.json_format else logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s:%(lineno)d  %(message)s"
        )
    )
    root.addHandler(file_handler)

    # ── Console handler ────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(cfg.level)
    if sys.stdout.isatty():
        console_handler.setFormatter(_ColourFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
        )
    root.addHandler(console_handler)

    # Silence noisy third-party loggers
    for noisy in ("uvicorn.access", "picamera2", "libcamera"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging initialised: level=%s  file=%s  json=%s",
        cfg.level, log_file, cfg.json_format,
    )
