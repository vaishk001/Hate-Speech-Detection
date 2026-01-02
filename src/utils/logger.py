# src/utils/logger.py
"""Logging configuration for KRIXION project."""
import logging
import sys
from pathlib import Path

LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str = __name__, level: int = logging.INFO, log_file: bool = False) -> logging.Logger:
    """Get or create a logger with console and optional file output.
    
    Args:
        name: Logger name (typically __name__).
        level: Logging level (default INFO).
        log_file: If True, also log to file in logs/ directory.
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(LOG_DIR / f'{name.replace(".", "_")}.log')
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    
    return logger


if __name__ == '__main__':
    # Usage example
    logger = get_logger('example', log_file=True)
    logger.info('This is an info message')
    logger.warning('This is a warning')
    logger.error('This is an error')
