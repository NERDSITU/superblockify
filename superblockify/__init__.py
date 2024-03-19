"""superblockify init."""
from os import environ

environ["USE_PYGEOS"] = "0"

from ._api import *
from ._version import __version__
from .config import logger

logger.info("superblockify version %s", __version__)
