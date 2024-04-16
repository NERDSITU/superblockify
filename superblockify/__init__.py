"""superblockify init."""

from os import environ

environ["USE_PYGEOS"] = "0"  # pylint: disable=wrong-import-position

from ._api import *
from ._version import __version__
from .config import logger, Config

logger.info("superblockify version %s", __version__)
logger.debug("Using graph directory %s", Config.GRAPH_DIR)
logger.debug("Using results directory %s", Config.RESULTS_DIR)
logger.debug("Using GHSL directory %s", Config.GHSL_DIR)
