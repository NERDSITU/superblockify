"""Load this module to fetch test data needed for certain tests."""

import osmnx as ox

from superblockify.config import logger, PLACES_GENERAL, PLACES_SMALL, NETWORK_FILTER
from superblockify.utils import load_graph_from_place

# turn on logging
ox.settings.log_console = True
# turn response caching off as this only loads graphs to files
ox.settings.use_cache = False

if __name__ == "__main__":
    for place in PLACES_SMALL + PLACES_GENERAL:
        logger.info(
            "Downloading graph for %s, with search string %s", place[0], place[1]
        )
        load_graph_from_place(
            f"./tests/test_data/cities/{place[0]}.graphml",
            place[1],
            add_population=True,
            custom_filter=NETWORK_FILTER,
        )
