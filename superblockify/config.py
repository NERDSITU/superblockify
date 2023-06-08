"""Configuration file for superblockify.

This module does not contain any functions or classes, but only variables that are
used throughout the package.

Attributes
----------

WORK_DIR
    The working directory of the package. This is used to store the graphs and results
    in subdirectories of this directory.
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
PLACES_OTHER
    Same as ``PLACES_GENERAL`` but for a variety of places that are used in the tests.

Notes
-----
Logger configuration is done using the :mod:`setup.cfg` file. The logger for this
module is named ``superblockify``.
"""

import logging.config
from os.path import join, dirname

# General
WORK_DIR = "./"
GRAPH_DIR = "./data/graphs/"
RESULTS_DIR = "./data/results/"
GHSL_DIR = "./data/ghsl/"

# LTN
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
FULL_RASTER = GHSL_DIR + "GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0.tif"
DOWNLOAD_TIMEOUT = 60

# Logging configuration using the setup.cfg file
logging.config.fileConfig(join(dirname(__file__), "..", "setup.cfg"))
# Get the logger for this module
logger = logging.getLogger("superblockify")

# Tests
TEST_DATA_PATH = "./tests/test_data/"
HIDE_PLOTS = True

PLACES_GENERAL = [
    ("Barcelona", "Barcelona, Catalonia, Spain"),
    ("Brooklyn", "Brooklyn, New York, United States"),
    ("Copenhagen", ["Københavns Kommune, Denmark", "Frederiksberg Kommune, Denmark"]),
    ("Resistencia", "Resistencia, Chaco, Argentina"),
    ("Liechtenstein", "Liechtenstein, Europe"),
]

PLACES_SMALL = [
    ("Adliswil", "Adliswil, Bezirk Horgen, Zürich, Switzerland"),
    ("MissionTown", "团城山街道, Xialu, Hubei, China"),
    ("Scheveningen", "Scheveningen, The Hague, Netherlands"),
]

PLACES_OTHER = [
    ("Strasbourg", "R4630050"),
    ("Milan", "Milan, Lombardy, Italy"),
    ("Palma", "Palma, Balearic Islands, Spain"),
    ("New York", "New York, New York, United States"),
    ("Paris", "Paris, Île-de-France, France"),
    ("Rome", "Rome, Lazio, Italy"),
    ("San Francisco", "San Francisco, California, United States"),
    ("Tokyo", "Tokyo, Japan"),
]
