"""superblockify init."""

import logging.config

from ._api import *
from ._version import __version__

# Logging configuration using the setup.cfg file
logging.config.fileConfig("setup.cfg")
# Get the logger for this module
logger = logging.getLogger("superblockify")
logger.info("superblockify version %s", __version__)
