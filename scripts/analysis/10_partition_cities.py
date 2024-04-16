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

from itertools import zip_longest
from os import rmdir
from os.path import join, dirname, exists, getsize
from sys import path

path.append(join(dirname(__file__), "..", ".."))

# Set run configurations in ./config.py
from scripts.analysis.config import (
    get_hpc_subset,
    combinations,
    MAKE_PLOTS,
    ONLY_PLOT,
    short_name_combination,
)

from superblockify.config import logger, Config

if __name__ == "__main__":
    # Add list(PLACES_100_CITIES.items()) + list(PLACES_GERMANY.items()) together, but
    # one from the first list, then one from the second list, etc.
    # note: the lists are not the same length
    city_list = [
        city
        for pair in zip_longest(
            list(Config.PLACES_100_CITIES.items()), list(Config.PLACES_GERMANY.items())
        )
        for city in pair
        if city is not None
    ]
    logger.info("Found %s cities", len(city_list))

    combinations = list(combinations)
    logger.info(
        "There are %s combinations for each graph: %s",
        len(combinations),
        [short_name_combination(comb) for comb in combinations],
    )
    # combine each city and combination - this is the cartesian product
    city_combination = [
        (place_name, place, comb)
        for place_name, place in city_list
        for comb in combinations
    ]
    # reduce list - if partitioner already done, skip
    city_combination = [
        (place_name, place, comb)
        for place_name, place, comb in city_combination
        if not exists(
            join(
                Config.RESULTS_DIR,
                place_name + "_" + short_name_combination(comb),
                "done",
            )
        )
    ]
    logger.info(
        "There are %s city-combination pairs to process (of possible %s)",
        len(city_combination),
        len(city_list) * len(combinations),
    )

    subset = get_hpc_subset(city_combination)
    # Sort by graph size on disk (GRAPH_DIR/PLACE_NAME.graphml)
    subset = sorted(
        subset, key=lambda x: getsize(join(Config.GRAPH_DIR, x[0] + ".graphml"))
    )

    for place_name, place, comb in subset:
        logger.info(
            "Processing graph for %s (%s) (OSM ID(s) %s)\n %s",
            place_name,
            place["query"],
            place["osm_id"],
            short_name_combination(comb),
        )
        # If graph not downloaded, skip
        graph_path = join(Config.GRAPH_DIR, place_name + ".graphml")
        if not exists(graph_path):
            logger.info("Graph not found in %s, skipping!", graph_path)
            continue
        logger.info("Loading graph from %s", graph_path)

        name = place_name + "_" + short_name_combination(comb)
        # If partitioner already done, skip
        if ONLY_PLOT:
            logger.info("Only plotting, not saving partitioner to disk.")
        elif exists(join(Config.RESULTS_DIR, name, "done")):
            logger.info("Partitioner %s has already been run, skipping!", name)
            continue
        elif exists(join(Config.RESULTS_DIR, name, "load_err")):
            logger.info(
                "Partitioner %s has already been run, but loading failed. "
                "Deleting and rerunning!",
            )
            # remove folder
            rmdir(join(Config.RESULTS_DIR, name))
        try:
            part = comb["part_class"](  # instantiate partitioner
                name=name,
                city_name=place_name,
                search_str=["R" + str(osmid) for osmid in place["osm_id"]],
                unit=comb["unit"],
            )
        except FileExistsError as err:
            logger.error(
                "FileExistsError when initializing partitioner %s: %s", name, err
            )
            continue
        logger.debug("Initialized partitioner %s, now running", part)
        part.run(
            calculate_metrics=not ONLY_PLOT,
            make_plots=MAKE_PLOTS,
            replace_max_speeds=comb["replace_max_speeds"],
            **comb["part_kwargs"],
        )
        if ONLY_PLOT:
            logger.info("Only plotting, skipping saving to disk")
            continue
        logger.info("Finished partitioning %s, saving to disk", part)
        part.save(save_graph_copy=False, dismiss_distance_matrix=True, key_figures=True)
        logger.info("Saved partitioner %s to disk", part)
        # check that the partitioner can be loaded from the disk
        try:
            part = comb["part_class"].load(part.name)
        except Exception as err:
            logger.error(
                "Loading partitioner %s from disk failed, marking as " "load_err: %s",
                part,
                err,
            )
            # mark the partitioner as failed - write file `load_err` in the
            # partitioner dir
            with open(
                join(Config.RESULTS_DIR, part.name, "load_err"), "w", encoding="utf-8"
            ) as file:
                file.write(str(err))
        else:
            logger.debug(
                "Loading partitioner %s from disk worked, marking as done", part
            )
            # mark the partitioner as done - write file `done` in the partitioner
            # dir
            with open(
                join(Config.RESULTS_DIR, part.name, "done"), "w", encoding="utf-8"
            ) as file:
                file.write("done")
