"""Cache graphs from cities.yml before processing.

This script is used to download the graphs for the cities in cities.yml.
"""
from os.path import exists, join, dirname
from sys import path

from tqdm import tqdm

path.append(join(dirname(__file__), "..", ".."))

import osmnx as ox

from superblockify.config import logger, NETWORK_FILTER, GRAPH_DIR, PLACES_100_CITIES
from superblockify.utils import load_graph_from_place

# turn on logging
ox.settings.log_console = False
# turn response caching off as this only loads graphs to files
ox.settings.use_cache = True
# turn on to force reloading the graph
RELOAD_GRAPHS = False

if __name__ == "__main__":
    # PLACES_100_CITIES is a dictionary of the form {name: place<dict>}
    for place_name, place in tqdm(
        list(PLACES_100_CITIES.items()),
        desc="Downloading graphs",
        unit="city",
    ):
        logger.info(
            "Caching graph for %s (%s)",
            place_name,
            place["query"],
        )
        # Check if the graph already exists
        graph_path = join(GRAPH_DIR, place_name + ".graphml")
        if exists(graph_path) and not RELOAD_GRAPHS:
            logger.debug("Graph already exists, skipping")
        else:
            try:
                logger.debug("Get graph from OSM relation IDs %s", place["osm_id"])
                load_graph_from_place(
                    save_as=graph_path,
                    search_string=["R" + str(osm_id) for osm_id in place["osm_id"]],
                    add_population=True,
                    custom_filter=NETWORK_FILTER,
                    simplify=True,
                    only_cache=True,
                )
            except Exception as exc:
                logger.error("Could not download graph for %s: %s", place_name, exc)
