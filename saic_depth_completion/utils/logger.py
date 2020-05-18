import logging
from colorlog import ColoredFormatter


def setup_logger():

    formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(yellow)s%(name)s: %(white)s%(message)s",
        "%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG': 'green',
            'INFO': 'green',
            'WARNING': 'red',
            'ERROR': 'red',
            'CRITICAL': 'red',
        }
    )

    logger = logging.getLogger('saic-dc')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger

