"""Logging configuration: concise console output + a detailed rotating file log.

The console shows the pipeline trace at the configured level; the file log
(<data_dir>/logs/personica.log) always captures DEBUG with timestamps and the
session id stamped on every record, so any past session can be reconstructed.
"""

from __future__ import annotations

import logging
import os
import warnings
from logging.handlers import RotatingFileHandler

NOISY_LOGGERS = (
    "LiteLLM", "litellm", "httpx", "httpcore", "chromadb",
    "urllib3", "sentence_transformers", "posthog",
)


class _SessionFilter(logging.Filter):
    """Stamps the session id onto every log record."""

    def __init__(self, session_id: str) -> None:
        super().__init__()
        self.session_id = session_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = self.session_id
        return True


def setup_logging(level_name: str, data_dir: str, session_id: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    session_filter = _SessionFilter(session_id)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter("  [%(name)s] %(message)s"))
    console.addFilter(session_filter)
    root.addHandler(console)

    try:
        log_dir = os.path.join(data_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, "personica.log"),
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s [session=%(session_id)s] %(message)s"
        ))
        file_handler.addFilter(session_filter)
        root.addHandler(file_handler)
    except OSError:
        root.warning("Could not create file log under %s", data_dir)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # litellm 1.81 triggers spurious pydantic serializer warnings on every
    # completion; they are harmless and drown out the real output.
    warnings.filterwarnings(
        "ignore", message="Pydantic serializer warnings", category=UserWarning)
