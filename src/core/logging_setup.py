"""
Logging configuration for the crypto trading system.
"""

import os
import sys
from pathlib import Path
from loguru import logger
from src.core.config import config


def setup_logging():
    """Set up logging configuration."""
    
    # Remove default handler
    logger.remove()
    
    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=config.monitoring.log_level,
        colorize=True
    )
    
    # File handler for general logs
    logger.add(
        log_dir / "trading_system.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="1 day",
        retention="30 days",
        compression="zip"
    )
    
    # File handler for trading-specific logs
    logger.add(
        log_dir / "trading.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="1 day",
        retention="90 days",
        compression="zip",
        filter=lambda record: "trading" in record["name"].lower() or "order" in record["name"].lower()
    )
    
    # File handler for errors
    logger.add(
        log_dir / "errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
        level="ERROR",
        rotation="1 week",
        retention="90 days",
        compression="zip"
    )
    
    logger.info("Logging system initialized")
    return logger
