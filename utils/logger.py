import logging
import sys
from config.settings import settings

def get_log_level(level_str: str) -> int:
    """Convert log level string to logging level constant"""
    return getattr(logging, level_str.upper(), logging.INFO)

def setup_logger(name: str = __name__) -> logging.Logger:
    """Setup logger with consistent formatting"""
    logger = logging.getLogger(name)
    
    # Set level from settings
    level = get_log_level(settings.log_level)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Ensure handler level is also updated
    for handler in logger.handlers:
        handler.setLevel(level)
        
    return logger

def update_log_level():
    """Update all loggers to the level specified in settings"""
    level = get_log_level(settings.log_level)
    
    # Update the root logger
    logging.getLogger().setLevel(level)

    # Update all existing loggers
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)

# Default logger for the application
logger = setup_logger("dumb-llm-router")
