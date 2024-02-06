"""Utility functions for analysis scripts."""

from glob import glob
from itertools import product
from os import environ
from os.path import join, dirname

import numpy as np
import osmnx as ox
import pandas as pd
from tqdm import tqdm

import superblockify as sb
from superblockify.config import logger

KEY_FIGURES_DIR = join(dirname(__file__), "..", "..", "data", "results", "key_figures")

# turn on logging
ox.settings.log_console = False
# turn response caching off as this only loads graphs to files
ox.settings.use_cache = True
# turn on to force reloading the graph
RELOAD_GRAPHS = False
# make plots
MAKE_PLOTS = True
ONLY_PLOT = False  # skips metric calculation and partitioner saving

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


def conf_to_params(conf_str):
    """Extract parameters from configuration string."""
    # get the filename without folder
    conf_str = conf_str.split("/")[-1].split(".")[0]
    # remove extension (_key_figures...)
    conf_str = conf_str.split("_key_figures")[0]
    partitioner_str = conf_str.split("_")[0]
    # every parameter afterwards is like _key-val_key-val...
    params = {}
    for param in conf_str.split("_")[1:]:
        key, val = param.split("-")
        if val == "F":
            val = False
        elif val == "T":
            val = True
        elif val == "None":
            val = None
        params[key] = val
    return {
        "partitioner": partitioner_str,
        **params,
    }


def load_subset(columns=None, params=None):
    """Load a subset of the key figures into a dataframe."""
    # Find configuration files
    config_files = glob(join(KEY_FIGURES_DIR, "*.feather"))
    # Filter by parameters
    if params is not None:
        config_files = [  # for given values matching the values in the config file
            file_path
            for file_path in config_files
            if all(conf_to_params(file_path)[key] == val for key, val in params.items())
        ]
    # Load all files
    key_figures_df = []
    conf_keys = set()
    for file_path in tqdm(config_files, desc="Loading key figures"):
        conf_str = file_path.split("/")[-1].split(".")[0].split("_key_figures")[0]
        conf_df = pd.read_feather(file_path, columns=columns)
        # Components is a list of dicts, calculate mean, median, std, min, max of them
        if "components" in conf_df.columns:
            # component_len from len(components)
            conf_df["component_len"] = conf_df["components"].apply(len)

            for col in [
                "n",
                "m",
                "k_avg",
                "mean_edge_betweenness_normal",
                "mean_edge_betweenness_normal_restricted",
                "mean_edge_betweenness_length",
                "mean_edge_betweenness_length_restricted",
                "mean_edge_betweenness_linear",
                "mean_edge_betweenness_linear_restricted",
                "change_mean_edge_betweenness_normal",
                "change_mean_edge_betweenness_length",
                "change_mean_edge_betweenness_linear",
                "population",
                "population_density",
                "area",
                "streets_per_node_avg",
                "intersection_count",
                "street_length_total",
                "circuity_avg",
                "intersection_density_km",
                "street_orientation_order",
            ]:
                conf_df["comp_" + col + "_mean"] = conf_df["components"].apply(
                    lambda x: np.mean([y[col] for y in x])
                )
                conf_df["comp_" + col + "_std"] = conf_df["components"].apply(
                    lambda x: np.std([y[col] for y in x])
                )
                conf_df["comp_" + col + "_median"] = conf_df["components"].apply(
                    lambda x: np.median([y[col] for y in x])
                )
                conf_df["comp_" + col + "_min"] = conf_df["components"].apply(
                    lambda x: np.min([y[col] for y in x])
                )
                conf_df["comp_" + col + "_max"] = conf_df["components"].apply(
                    lambda x: np.max([y[col] for y in x])
                )
                # percentiles 10, 90
                conf_df["comp_" + col + "_p10"] = conf_df["components"].apply(
                    lambda x: np.percentile([y[col] for y in x], 10)
                )
                conf_df["comp_" + col + "_p90"] = conf_df["components"].apply(
                    lambda x: np.percentile([y[col] for y in x], 90)
                )

            # remove components
            conf_df = conf_df.drop(columns=["components"])
        key_figures_df.append(conf_df.assign(**conf_to_params(conf_str)))
        conf_keys.update(conf_to_params(conf_str).keys())
    key_figures_df = pd.concat(key_figures_df, ignore_index=True)
    # Categorize some columns
    for column in [
        "created_with",
        "crs",
        "unit",
        "attribute_label",
        "attribute_dtype",
        "partitioner",
    ] + list(conf_keys):
        if column in key_figures_df.columns:
            key_figures_df[column] = key_figures_df[column].astype("category")
    if "simplified" in key_figures_df.columns:
        key_figures_df["simplified"] = key_figures_df["simplified"].astype("bool")

    if "graph_stats" in key_figures_df.columns:
        # flatten `graph_stats` dict into columns categorical
        key_figures_df = pd.concat(
            [key_figures_df, key_figures_df["graph_stats"].apply(pd.Series)], axis=1
        )
        key_figures_df.drop(columns=["graph_stats"], inplace=True)
    if "metric" in key_figures_df.columns:
        # flatten `metric` dict into columns
        key_figures_df = pd.concat(
            [key_figures_df, key_figures_df["metric"].apply(pd.Series)], axis=1
        )
        key_figures_df.drop(columns=["metric"], inplace=True)
    return key_figures_df
