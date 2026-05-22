"""
logger.py — Structured logging using structlog.
All modules import `get_logger()` from here.
"""
import logging
import structlog
from backend.config import get_settings


def setup_logging() -> None:
    """Configure structlog with JSON rendering and stdlib integration."""
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
    )

    # structlog configuration
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(),  # human-friendly output; swap to JSONRenderer in prod
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__):
    """Return a bound structlog logger for the given module name."""
    return structlog.get_logger().bind(module=name)
