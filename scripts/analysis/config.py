"""Utility functions for analysis scripts."""
from os import environ

from itertools import product

import osmnx as ox

import superblockify as sb
from superblockify.config import logger

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


def get_hpc_subset(joblist):
    """Return subset of jobs for a HPC array job.

    Parameters
    ----------
    joblist : list
        List of jobs to be distributed.

    Returns
    -------
    list
        Subset of jobs for this job.
    """

    # check $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT are set
    if "SLURM_ARRAY_TASK_ID" not in environ:
        logger.warning("SLURM_ARRAY_TASK_ID not set, returning all jobs!")
        return joblist
    if "SLURM_ARRAY_TASK_COUNT" not in environ:
        logger.warning("SLURM_ARRAY_TASK_COUNT not set, returning all jobs!")
        return joblist

    # Determine with slice of the cities to download
    # from (task_num * num_cities // num_tasks)
    # to ((task_num + 1) * num_cities // num_tasks)
    subset = slice(
        int(environ["SLURM_ARRAY_TASK_ID"])
        * len(joblist)
        // int(environ["SLURM_ARRAY_TASK_COUNT"]),
        (int(environ["SLURM_ARRAY_TASK_ID"]) + 1)
        * len(joblist)
        // int(environ["SLURM_ARRAY_TASK_COUNT"]),
    )
    subset = joblist[subset]
    logger.debug(
        "Job %s/%s, with %s of %s jobs.",
        environ["SLURM_ARRAY_TASK_ID"],
        environ["SLURM_ARRAY_TASK_COUNT"],
        len(subset),
        len(joblist),
    )
    return subset
