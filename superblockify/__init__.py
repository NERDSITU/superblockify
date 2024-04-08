"""superblockify init."""
from os import environ

environ["USE_PYGEOS"] = "0"  # pylint: disable=wrong-import-position

from ._api import *
from ._version import __version__
from .config import logger

logger.info("superblockify version %s", __version__)
