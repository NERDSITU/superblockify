"""Load this module to fetch test data needed for certain tests."""

# pylint: disable=wrong-import-position
import sys
from os import path

import osmnx as ox

# Add the package folder to python path, so this module can be run from anywhere
sys.path.append(path.join(path.dirname(__file__), "..", ".."))

from superblockify.config import logger, Config
from superblockify.utils import load_graph_from_place

# turn on logging
ox.settings.log_console = True
# turn response caching off as this only loads graphs to files
ox.settings.use_cache = False

if __name__ == "__main__":
    for place in Config.PLACES_SMALL + Config.PLACES_GENERAL:
        logger.info(
            "Downloading graph for %s, with search string %s", place[0], place[1]
        )
        load_graph_from_place(
            f"./tests/test_data/cities/{place[0]}.graphml",
            place[1],
            add_population=True,
            custom_filter=Config.NETWORK_FILTER,
        )
