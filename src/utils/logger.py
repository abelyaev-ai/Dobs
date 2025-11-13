"""Structured logging setup for the shipping price search tool."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Final

# Log format constants
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


class SensitiveDataFilter(logging.Filter):
    """Filter to prevent logging of sensitive data patterns.

    This filter redacts common patterns that might contain API keys,
    tokens, or other sensitive information.
    """

    REDACT_PATTERNS: Final[tuple[str, ...]] = (
        "api_key",
        "apikey",
        "token",
        "password",
        "secret",
        "authorization",
        "bearer",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record to redact sensitive information.

        Args:
            record: Log record to filter.

        Returns:
            True to allow the record, False to drop it.
        """
        message = record.getMessage().lower()

        # Check if any sensitive pattern appears in the message
        for pattern in self.REDACT_PATTERNS:
            if pattern in message:
                # Redact the entire message if it contains sensitive data
                record.msg = "[REDACTED - Contains sensitive data]"
                record.args = ()
                break

        return True


def setup_logger(
    name: str = "shipping_price_search",
    level: int = logging.INFO,
    log_file: Path | str | None = None,
    enable_console: bool = True,
    structured: bool = True,
) -> logging.Logger:
    """Configure and return a logger instance.

    Args:
        name: Logger name (typically module name or application name).
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file. If provided, logs will be written to file.
        enable_console: If True, logs will be written to console (stdout).
        structured: If True, use structured logging format with timestamps.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logger(__name__, level=logging.DEBUG)
        >>> logger.info("Processing started", extra={"document_id": "doc123"})
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    if structured:
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    else:
        formatter = logging.Formatter("%(levelname)s - %(message)s")

    # Add sensitive data filter
    sensitive_filter = SensitiveDataFilter()
    logger.addFilter(sensitive_filter)

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(sensitive_filter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(sensitive_filter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.

    This is a convenience function for getting module-level loggers
    that inherit configuration from the root application logger.

    Args:
        name: Logger name (typically __name__ from calling module).

    Returns:
        Logger instance.

    Example:
        >>> # In a module file
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)


def set_log_level(logger: logging.Logger, level: int | str) -> None:
    """Change the logging level for a logger and its handlers.

    Args:
        logger: Logger instance to modify.
        level: New logging level (can be int or string like "DEBUG", "INFO").

    Example:
        >>> logger = get_logger(__name__)
        >>> set_log_level(logger, "DEBUG")
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def log_exception(
    logger: logging.Logger,
    message: str,
    exc: Exception,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log an exception with context information.

    Args:
        logger: Logger instance to use.
        message: Human-readable error message.
        exc: Exception instance to log.
        extra: Optional dictionary with additional context.

    Example:
        >>> try:
        ...     process_document(doc_id)
        ... except ValueError as e:
        ...     log_exception(logger, "Document processing failed", e,
        ...                   extra={"doc_id": doc_id})
    """
    context = extra or {}
    logger.error(
        f"{message}: {exc.__class__.__name__}: {exc}",
        extra=context,
        exc_info=True,
    )
