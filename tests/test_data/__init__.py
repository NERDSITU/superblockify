"""Load this module to fetch test data needed for certain tests."""
import logging
from ast import literal_eval
from configparser import ConfigParser
from os.path import join, dirname

import osmnx as ox

from superblockify.utils import load_graph_from_place

logger = logging.getLogger("superblockify")

# turn on logging
ox.settings.log_console = True
# turn response caching off as this only loads graphs to files
ox.settings.use_cache = False

config = ConfigParser()
config.read(join(dirname(__file__), "..", "..", "config.ini"))

if __name__ == "__main__":
    for place in literal_eval(config["tests"]["places_small"]) + literal_eval(
        config["tests"]["places_general"]
    ):
        logger.info(
            "Downloading graph for %s, with search string %s", place[0], place[1]
        )
        load_graph_from_place(
            f"./tests/test_data/cities/{place[0]}.graphml",
            place[1],
            network_type="drive",
        )
