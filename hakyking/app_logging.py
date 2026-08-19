from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


PROJECT_ROOT = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parents[1]
)
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "hakyking.log"


def configure_logging() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if not any(
        isinstance(handler, RotatingFileHandler)
        and getattr(handler, "baseFilename", "") == str(LOG_FILE)
        for handler in root_logger.handlers
    ):
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)

    logging.getLogger(__name__).info("Logging initialized: %s", LOG_FILE)
    return LOG_FILE


def install_excepthook() -> None:
    original_hook = sys.excepthook

    def _hook(exc_type, exc_value, traceback):  # noqa: ANN001
        logging.getLogger("hakyking.uncaught").exception(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, traceback),
        )
        original_hook(exc_type, exc_value, traceback)

    sys.excepthook = _hook
