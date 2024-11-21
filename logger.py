import logging
from logging.handlers import RotatingFileHandler

def get_logger(name):
    """
    Returns a logger instance with the specified name.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # File Handler
    file_handler = RotatingFileHandler("app.log", maxBytes=10000, backupCount=3)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Stream Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_formatter = logging.Formatter("%(levelname)s - %(message)s")
    stream_handler.setFormatter(stream_formatter)

    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
