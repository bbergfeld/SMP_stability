# -*- coding: utf-8 -*-
"""
#%% Routines to process a pnt file in order to do mechanical calculation on the measured snow profile
@author: Bergfeld Bastian
"""
import logging
import functools

class LoggerConfig:
    """Configures logging based on user settings."""
    @staticmethod
    def setup_logging(log_to_file=False, log_filename="pipeline.log"):
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        
        # Clear existing handlers to prevent duplicate logs
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        logging.basicConfig(level=logging.INFO, format=log_format)

        if log_to_file:
            file_handler = logging.FileHandler(log_filename)
            file_handler.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(file_handler)
            logging.info("File logging enabled: %s", log_filename)

        # Suppress logs from external libraries
        logging.getLogger("snowmicropyn").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)


def error_handling_decorator(func):
    """Decorator to handle errors and log them automatically."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}", exc_info=True)
            return None
    return wrapper
