"""Load this module to fetch test data needed for certain tests."""
from ast import literal_eval
from configparser import ConfigParser

import osmnx as ox

from superblockify.utils import load_graph_from_place

# turn response caching off as this only loads graphs to files
ox.config(use_cache=False, log_console=True)

config = ConfigParser()
config.read("config.ini")
PLACES_GENERAL = literal_eval(config["tests"]["places_general"])
PLACES_SMALL = literal_eval(config["tests"]["places_small"])

if __name__ == "__main__":
    for place in PLACES_SMALL + PLACES_GENERAL:
        load_graph_from_place(
            f"./tests/test_data/cities/{place[0]}.graphml",
            place[1],
            network_type="drive",
        )
