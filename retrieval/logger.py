"""
Logging utilities for the Paper2Fig-2026 Retrieval Framework.

Author: Nguyen Khanh
Project: Paper2Fig-2026
"""

import logging
from pathlib import Path


def setup_logger(
    log_file: Path,
    logger_name: str = "Paper2Fig",
) -> logging.Logger:
    """
    Create a logger that writes to both console and file.

    Parameters
    ----------
    log_file : Path
        Path to log file.

    logger_name : str
        Logger name.

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(logger_name)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---------------------------------------------------------
    # Console
    # ---------------------------------------------------------

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # ---------------------------------------------------------
    # File
    # ---------------------------------------------------------

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)

    # ---------------------------------------------------------

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger