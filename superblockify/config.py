"""Configuration file for superblockify.

This module contains the configuration class for superblockify.

Notes
-----
Logger configuration is done using the :mod:`logging.cfg` file.
The logger for this module is named ``superblockify``.
"""

import logging.config
from os import getcwd
from os.path import join, dirname, exists

from ruamel.yaml import YAML

# Logging configuration using the logging.cfg file
logging.config.fileConfig(join(dirname(__file__), "logging.cfg"))
# Get the logger for this module
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)
logger = logging.getLogger("superblockify")


def set_log_level(level: int | str) -> None:
    """Set the logging level for the superblockify logger.

    Also sets the logging level for the first handler of the root logger.

    Parameters
    ----------
    level : int | str
        The logging level. See the :mod:`logging` module for more information.

    Raises
    ------
    ValueError
        If the level is not a valid logging level.
    """
    logger.setLevel(level)
    logging.getLogger().handlers[0].setLevel(level)


class Config:  # pylint: disable=too-few-public-methods
    """Configuration class for superblockify.

    Attributes
    ----------

    WORK_DIR
        The working directory of the package.
        This is used to store the graphs and results in subdirectories of this
        directory. By default, this is the current working directory when the package
        is imported. This is only used to define the following directories.
    GRAPH_DIR
        The directory where the graphs are stored.
    RESULTS_DIR
        The directory where the results are stored.
    GHSL_DIR
        The directory where the GHSL population data is stored when downloaded.

    V_MAX_LTN
        The maximum speed in km/h for the restricted calculation of travel times.
    V_MAX_SPARSE
        The maximum speed in km/h for the restricted calculation of travel times for the
        sparsified graph.

    NETWORK_FILTER
        The filter used to filter the OSM data for the graph. This is a string that is
        passed to the :func:`osmnx.graph_from_place` function.

    CLUSTERING_PERCENTILE
        The percentile used to determine the betweenness centrality threshold for the
        spatial clustering and anisotropy nodes.
    NUM_BINS
        The number of bins used for the histograms in the entropy calculation.

    FULL_RASTER
        The path and filename of the full GHSL raster.
        If None, tiles of the needed area are downloaded from the JRC FTP server and
        stored in the GHSL_DIR directory.
        <https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2025_GLOBE_R2023A_54009_100/V1-0/GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0.zip>
    DOWNLOAD_TIMEOUT
        The timeout in seconds for downloading the GHSL raster tiles.

    logger
        The logger for this module. This is used to log information, warnings and errors
        throughout the package.

    TEST_DATA_PATH
        The path to the test data directory.
    HIDE_PLOTS
        Whether to hide the plots in the tests.

    PLACES_GENERAL
        A list of tuples of the form ``(name, place)`` where ``name`` is the name of the
        place and ``place`` is the place string that is passed to the
        :func:`superblockify.utils.load_graph_from_place` function.
    PLACES_SMALL
        Same as ``PLACES_GENERAL`` but for places of which the graph is small enough to
        be used in the tests.
    PLACES_100_CITIES
        100 cities from Boeing et al. (2019) <https://doi.org/10.1007/s41109-019-0189-1>
        A dictionary of the form ``{name: place}`` where ``name`` is the name of the
        place, and ``place`` is a dictionary of various attributes. One of them is the
        ``query`` attribute which is the place string or a list of place strings.
        Find the extensive list in the ``./cities.yml`` file.
    PLACES_GERMANY
        List of cities in Germany by population. All cities with more than 100,000
        inhabitants are included. Data from the German Federal Statistical Office.

    PLOT_SUFFIX
        The format of the plots. Can be ``"png"``, ``"jpg"``, ``"pdf"``, ``"svg"``, etc.
        Matplotlib uses the Pillow library to save the plots,
        so all formats supported by Pillow are supported by Matplotlib.
        <https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html>

    MAX_NODES
        The maximum number of nodes in the graph. If the graph has more nodes, it is
        reduced. See :func:`superblockify.partitioning.utils.reduce_graph`.
    """

    # General
    WORK_DIR = getcwd()  # current working directory when the package is imported
    GRAPH_DIR = join(WORK_DIR, "data", "graphs")
    RESULTS_DIR = join(WORK_DIR, "data", "results")
    GHSL_DIR = join(WORK_DIR, "data", "ghsl")

    # Superblock
    # Max speeds in km/h for the restricted calculation of travel times
    V_MAX_LTN = 15.0
    V_MAX_SPARSE = 50.0

    # Graph
    NETWORK_FILTER = (
        '["highway"]["area"!~"yes"]["access"!~"private"]'
        '["highway"!~"abandoned|bridleway|bus_guideway|busway|construction|corridor|'
        "cycleway|elevator|escalator|footway|path|pedestrian|planned|platform|proposed|"
        'raceway|service|steps|track"]'
        '["motor_vehicle"!~"no"]["motorcar"!~"no"]'
        '["service"!~"alley|driveway|emergency_access|parking|parking_aisle|private"]'
    )

    # Metrics
    CLUSTERING_PERCENTILE = 90
    NUM_BINS = 36

    # Population data (GHSL)
    FULL_RASTER = (
        None  # join(GHSL_DIR, "GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0.tif")
    )
    DOWNLOAD_TIMEOUT = 60

    # Tests
    TEST_DATA_PATH = join(dirname(__file__), "..", "tests", "test_data")
    HIDE_PLOTS = True

    # Places
    PLACES_FILE = join(dirname(__file__), "cities.yml")
    # see if the file is available
    if exists(PLACES_FILE):
        with open(PLACES_FILE, "r", encoding="utf-8") as file:
            yaml = YAML(typ="safe")
            places = yaml.load(file)
            PLACES_GENERAL = [
                (name, data["query"])
                for name, data in places["place_lists"]["test_general"][
                    "cities"
                ].items()
            ]
            PLACES_SMALL = [
                (name, data["query"])
                for name, data in places["place_lists"]["test_small"]["cities"].items()
            ]
            PLACES_100_CITIES = places["place_lists"]["100_cities_boeing"]["cities"]
            PLACES_GERMANY = places["place_lists"]["germany_by_pop"]["cities"]

    # Plot format
    PLOT_SUFFIX = "pdf"

    # Reduce graph
    MAX_NODES = 20000
