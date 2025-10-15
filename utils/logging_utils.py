"""Structured logging helpers for the federated Mixture of Experts project."""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class LoggingArtifacts:
    """Container with paths created during logging configuration."""

    experiment_dir: Path
    log_file: Path
    checkpoints_dir: Path
    artifacts_dir: Path


class JsonFormatter(logging.Formatter):
    """Format log records as JSON with consistent structure."""

    default_time_format = "%Y-%m-%dT%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        log_record: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.default_time_format),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key.startswith("_") or key in {
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
            }:
                continue
            log_record[key] = value

        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log_record, default=str)


def create_experiment_directories(root: Path, experiment_name: str) -> LoggingArtifacts:
    """Create timestamped experiment directories under ``root``."""

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    experiment_dir = root / experiment_name / timestamp
    checkpoints_dir = experiment_dir / "checkpoints"
    artifacts_dir = experiment_dir / "artifacts"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log_file = experiment_dir / "experiment.log"
    return LoggingArtifacts(
        experiment_dir=experiment_dir,
        log_file=log_file,
        checkpoints_dir=checkpoints_dir,
        artifacts_dir=artifacts_dir,
    )


def setup_logging(
    config: Optional[Dict[str, Any]] = None,
    logger_name: Optional[str] = None,
) -> Tuple[logging.Logger, LoggingArtifacts]:
    """Configure structured logging and create experiment directories.

    Args:
        config: Configuration dictionary (typically ``config["logging"]``)
            containing ``output_dir``, ``experiment_name``, and ``log_level``.
        logger_name: Optional name for the logger. Defaults to the value from
            configuration or ``"fed_moe"``.

    Returns:
        A tuple of the configured logger and the directories created for the
        experiment run.
    """

    config = config or {}
    log_root = Path(config.get("output_dir", "./results")).expanduser().resolve()
    experiment_name = config.get("experiment_name", "experiment")
    log_level = str(config.get("log_level", "INFO")).upper()
    logger_name = logger_name or config.get("logger_name", "fed_moe")

    artifacts = create_experiment_directories(log_root, experiment_name)

    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.propagate = False

    # Remove existing handlers to avoid duplicate logs when reconfiguring.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = JsonFormatter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(artifacts.log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.debug("Logging configured", extra={"experiment_dir": str(artifacts.experiment_dir)})

    return logger, artifacts


__all__ = [
    "JsonFormatter",
    "LoggingArtifacts",
    "create_experiment_directories",
    "setup_logging",
]
