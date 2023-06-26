"""Cache graphs from cities.yml before processing.

This script is used to download the graphs for the cities in cities.yml.

Parameters
----------
SLURM_ARRAY_TASK_ID : int
    The task ID of the SLURM job scheduler.
SLURM_ARRAY_TASK_COUNT : int
    The number of SLURM job scheduler tasks.

"""
from datetime import timedelta
from os import environ
from os.path import exists, join, dirname
from sys import path
from time import time

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
    # check $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT are set
    if "SLURM_ARRAY_TASK_ID" not in environ:
        raise ValueError("SLURM_ARRAY_TASK_ID not set")
    if "SLURM_ARRAY_TASK_COUNT" not in environ:
        raise ValueError("SLURM_ARRAY_TASK_COUNT not set")

    # Throw out places that are already in GRAPH_DIR, so the rest is distributed
    if not RELOAD_GRAPHS:
        PLACES_100_CITIES = {
            place_name: place
            for place_name, place in PLACES_100_CITIES.items()
            if not exists(join(GRAPH_DIR, place_name + ".graphml"))
        }

    # Determine with slice of the cities to download
    # from (task_num * num_cities // num_tasks)
    # to ((task_num + 1) * num_cities // num_tasks)
    subset = slice(
        int(environ["SLURM_ARRAY_TASK_ID"])
        * len(PLACES_100_CITIES)
        // int(environ["SLURM_ARRAY_TASK_COUNT"]),
        (int(environ["SLURM_ARRAY_TASK_ID"]) + 1)
        * len(PLACES_100_CITIES)
        // int(environ["SLURM_ARRAY_TASK_COUNT"]),
    )

    logger.info("There are %s graphs left to download", len(PLACES_100_CITIES))
    logger.info(
        "Task %s/%s: Downloading graphs %s %s",
        environ["SLURM_ARRAY_TASK_ID"],
        environ["SLURM_ARRAY_TASK_COUNT"],
        subset,
        list(PLACES_100_CITIES.keys())[subset],
    )

    # PLACES_100_CITIES is a dictionary of the form {name: place<dict>}
    for place_name, place in list(PLACES_100_CITIES.items())[subset]:
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
                t_start = time()
                load_graph_from_place(
                    save_as=graph_path,
                    search_string=["R" + str(osm_id) for osm_id in place["osm_id"]],
                    add_population=True,
                    custom_filter=NETWORK_FILTER,
                    simplify=True,
                    only_cache=True,
                )
                logger.info(
                    "Cached graph for %s in %s",
                    place_name,
                    timedelta(seconds=time() - t_start),
                )

            except Exception as exc:
                logger.error("Could not download graph for %s: %s", place_name, exc)
