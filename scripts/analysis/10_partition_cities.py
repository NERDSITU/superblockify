"""Partition cities and save the results to disk.

For each city we use different parametrizations.
- Unit - to minimize in shortest path search
    - time:
        - with imputated speed limits
        - imputated limits + slowed down speed limits in LTNs)
    (- distance: plain distance)

- Approach
    - ResidentialPartitioner
        - one run
    - BetweennessPartitioner
        - percentile: 90, 85, 80, 70, 50
        - betweenness scaling: normal, (length, linear)
        - betweenness range: global, (3km radius)

Parameters
----------
SLURM_ARRAY_TASK_ID : int
    The task ID of the SLURM job scheduler.
SLURM_ARRAY_TASK_COUNT : int
    The number of SLURM job scheduler tasks.
"""
from itertools import product
from os.path import join, dirname, exists, getsize
from random import shuffle
from sys import path

import osmnx as ox

path.append(join(dirname(__file__), "..", ".."))

from scripts.analysis.utils import get_hpc_subset

import superblockify as sb
from superblockify.config import (
    logger,
    GRAPH_DIR,
    RESULTS_DIR,
    PLACES_100_CITIES,
    PLACES_GERMANY,
)

# turn on logging
ox.settings.log_console = False
# turn response caching off as this only loads graphs to files
ox.settings.use_cache = True
# turn on to force reloading the graph
RELOAD_GRAPHS = False
# make plots
MAKE_PLOTS = False

parameters = {
    "partitioner": {
        "residential": {
            "partitioner": sb.ResidentialPartitioner,
        },
        "betweenness": {
            "partitioner": sb.BetweennessPartitioner,
            "kwargs": {
                "percentile": [90, 85, 80, 70, 50],
                "scaling": ["normal"],  # , ("length", "linear")],
                "max_range": [None, 3000],
            },
        },
    },
    "distance": [
        {"unit": "time", "replace_max_speeds": False},
        # {"unit": "time", "replace_max_speeds": True},
        # {"unit": "distance", "replace_max_speeds": False},
    ],
}

combinations = (  # generator of all combinations
    # partitioner_name, partitioner, unit, replace_max_speeds, part_kwargs
    {
        "part_name": partitioner_name,
        "part_class": part_data["partitioner"],
        "unit": unit_data["unit"],
        "replace_max_speeds": unit_data["replace_max_speeds"],
        "part_kwargs": kwarg_combination,
    }
    for partitioner_name, part_data in parameters["partitioner"].items()
    for unit_data in parameters["distance"]
    for kwarg_combination in [
        dict(zip(part_data.get("kwargs", {}).keys(), comb))
        for comb in product(*part_data.get("kwargs", {}).values())
    ]
)


def short_name_combination(combination):
    """Create a unique name for a combination of parameters."""
    short_name = combination["part_name"]
    short_name += "_unit-" + combination["unit"]
    short_name += "_rms-" + str(combination["replace_max_speeds"])[0]  # T/F
    if combination["part_name"] == "betweenness":
        short_name += "_per-" + str(combination["part_kwargs"]["percentile"])
        short_name += "_scl-" + str(combination["part_kwargs"]["scaling"])
        short_name += "_rng-" + str(combination["part_kwargs"]["max_range"])
    return short_name


if __name__ == "__main__":
    # Shuffle the dict of cities to redistribute the load
    shuffled = list(PLACES_100_CITIES.items())
    # shuffle(shuffled)
    subset = get_hpc_subset(shuffled)
    # Sort by graph size on disk (GRAPH_DIR/PLACE_NAME.graphml)
    subset = sorted(subset, key=lambda x: getsize(join(GRAPH_DIR, x[0] + ".graphml")))

    logger.info("Processing %s graphs", len(subset))
    combinations = list(combinations)
    logger.debug(
        "There are %s combinations for each graph: %s",
        len(combinations),
        [short_name_combination(comb) for comb in combinations],
    )

    for place_name, place in subset:
        logger.info(
            "Processing graph for %s (%s) (OSM ID(s) %s)",
            place_name,
            place["query"],
            place["osm_id"],
        )
        # If graph not downloaded, skip
        graph_path = join(GRAPH_DIR, place_name + ".graphml")
        if not exists(graph_path):
            logger.info("Graph not found in %s, skipping!", graph_path)
            continue
        logger.info("Loading graph from %s", graph_path)

        for comb in combinations:
            logger.info("City %s, combination %s", place_name, comb)
            name = place_name + "_" + short_name_combination(comb)
            # If partitioner already done, skip
            if exists(join(RESULTS_DIR, name, "done")):
                logger.info("Partitioner %s has already been run, skipping!", name)
            part = comb["part_class"](  # instantiate partitioner
                name=name,
                city_name=place_name,
                search_str=["R" + str(osmid) for osmid in place["osm_id"]],
                unit=comb["unit"],
            )
            logger.debug("Initialized partitioner %s, now running", part)
            part.run(
                calculate_metrics=True,
                make_plots=MAKE_PLOTS,
                replace_max_speeds=comb["replace_max_speeds"],
                **comb["part_kwargs"],
            )
            logger.info("Finished partitioning %s, saving to disk", part)
            part.save(
                save_graph_copy=False, dismiss_distance_matrix=True, key_figures=True
            )
            logger.info("Saved partitioner %s to disk", part)
            # check that the partitioner can be loaded from the disk
            try:
                part = comb["part_class"].load(part.name)
            except Exception as err:
                logger.error(
                    "Loading partitioner %s from disk failed, marking as "
                    "load_err: %s",
                    part,
                    err,
                )
                # mark the partitioner as failed - write file `load_err` in the
                # partitioner dir
                with open(
                    join(RESULTS_DIR, part.name, "load_err"), "w", encoding="utf-8"
                ) as file:
                    file.write(str(err))
            else:
                logger.debug(
                    "Loading partitioner %s from disk worked, marking as done", part
                )
                # mark the partitioner as done - write file `done` in the partitioner
                # dir
                with open(
                    join(RESULTS_DIR, part.name, "done"), "w", encoding="utf-8"
                ) as file:
                    file.write("done")
