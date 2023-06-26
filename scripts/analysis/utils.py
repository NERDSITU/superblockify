"""Utility functions for analysis scripts."""
from os import environ

from superblockify.config import logger


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
        "Job %s/%s, with $s of %s jobs.",
        environ["SLURM_ARRAY_TASK_ID"],
        environ["SLURM_ARRAY_TASK_COUNT"],
        len(subset),
        len(joblist),
    )
    return subset
