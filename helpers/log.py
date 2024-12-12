import logging
import os
from enum import Enum


class LogLevel(Enum):
    CRITICAL = 50
    FATAL = 50
    ERROR = 40
    WARNING = 30
    WARN = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0

    @classmethod
    def from_string(cls, level: str) -> "LogLevel":
        return cls[level.upper()]

    @classmethod
    def from_env(cls) -> "LogLevel":
        return cls.from_string(os.getenv("LOG_LEVEL", "INFO"))


def build_logger(
    logger_name: str,
    log_level: int = LogLevel.from_env().value,
    show_time: bool = False,
    rich_tracebacks: bool = False,
    tracebacks_show_locals: bool = False,
) -> logging.Logger:
    from rich.logging import RichHandler

    rich_handler = RichHandler(
        show_time=show_time,
        rich_tracebacks=rich_tracebacks,
        tracebacks_show_locals=tracebacks_show_locals,
    )
    rich_handler.setFormatter(
        logging.Formatter(
            fmt="%(message)s",
            datefmt="[%X]",
        )
    )

    _logger = logging.getLogger(logger_name)
    _logger.addHandler(rich_handler)
    _logger.setLevel(log_level)
    _logger.propagate = False
    return _logger


# Default logger instance
logger: logging.Logger = build_logger("llm-os")

__all__ = ["logger"]
