"""
Logging configuration and utilities
"""
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger
from src.config import settings


class InterceptHandler(logging.Handler):
    """
    Intercept standard logging messages and redirect to loguru
    """
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def json_serializer(record: Dict[str, Any]) -> str:
    """
    Custom JSON serializer for loguru that handles datetime objects
    """
    subset = {
        "timestamp": record["time"].strftime("%Y-%m-%d %H:%M:%S"),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"]
    }
    
    # Add extra fields if present
    if record.get("extra"):
        subset.update(record["extra"])
    
    # Add exception info if present
    if record.get("exception"):
        subset["exception"] = record["exception"]
    
    return json.dumps(subset, default=str)


def setup_logging(
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Configure application logging
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format (json or plain)
        log_file: Optional log file path
    """
    # Use settings if not provided
    log_level = log_level or settings.log_level
    log_format = log_format or settings.log_format
    
    # Remove default handler
    logger.remove()
    
    # Configure format based on settings
    if log_format == "json":
        # For JSON format, use serialization
        logger.add(
            sys.stdout,
            format="{message}",
            level=log_level,
            serialize=True,
            enqueue=True
        )
    else:
        # For plain format
        log_fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stdout,
            format=log_fmt,
            level=log_level,
            serialize=False
        )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if log_format == "json":
            logger.add(
                log_file,
                format="{message}",
                level=log_level,
                rotation="100 MB",
                retention="30 days",
                compression="zip",
                serialize=True,
                enqueue=True
            )
        else:
            logger.add(
                log_file,
                format=log_fmt,
                level=log_level,
                rotation="100 MB",
                retention="30 days",
                compression="zip",
                serialize=False,
                enqueue=True
            )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    
    # Intercept uvicorn logs
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        logging.getLogger(logger_name).handlers = [InterceptHandler()]


def get_logger(name: str) -> logger:
    """
    Get a logger instance with the given name
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Setup logging on import
setup_logging()