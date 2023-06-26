"""Import cities script for the analysis.

This script is used to download the graphs for the cities in cities.yml.

Cities are downloaded with the :func:`load_graph_from_place` function.
The query is used to find the city in Nominatim.
The city name determines the save folder and file names.
If a graph already exists, it is skipped.

Parameters
----------
SLURM_ARRAY_TASK_ID : int
    The task ID of the SLURM job scheduler.
SLURM_ARRAY_TASK_COUNT : int
    The number of SLURM job scheduler tasks.

Notes
-----
By the parameters, this script knows which cities to download.
Also, the population is added to the graph, which is an expensive operation.
"""
from os import environ
from os.path import exists, join, dirname

# from sys import path

# path.append(join(dirname(__file__), "..", ".."))

import osmnx as ox

from scripts.analysis.utils import get_hpc_subset
from superblockify.config import logger, NETWORK_FILTER, GRAPH_DIR, PLACES_100_CITIES
from superblockify.utils import load_graph_from_place

# turn on logging
ox.settings.log_console = False
# turn response caching off as this only loads graphs to files
ox.settings.use_cache = True
# turn on to force reloading the graph
RELOAD_GRAPHS = False

if __name__ == "__main__":
    # Throw out places that are already in GRAPH_DIR, so the rest is distributed
    if not RELOAD_GRAPHS:
        PLACES_100_CITIES = {
            place_name: place
            for place_name, place in PLACES_100_CITIES.items()
            if not exists(join(GRAPH_DIR, place_name + ".graphml"))
        }
    logger.info("There are %s graphs left to download", len(PLACES_100_CITIES))
    subset = get_hpc_subset(list(PLACES_100_CITIES.items()))
    logger.info(
        "Task %s/%s: Downloading graphs %s",
        environ["SLURM_ARRAY_TASK_ID"],
        environ["SLURM_ARRAY_TASK_COUNT"],
        [place_name for place_name, _ in subset],
    )

    # PLACES_100_CITIES is a list of city dictionaries
    for place_name, place in subset:
        logger.info(
            "Downloading graph for %s (%s)",
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
                )
            except Exception as exc:
                logger.error("Could not download graph for %s: %s", place_name, exc)
                # raise exc
