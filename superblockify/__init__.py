"""superblockify init."""

from ._api import *
from ._version import __version__
from .config import logger

logger.info("superblockify version %s", __version__)
