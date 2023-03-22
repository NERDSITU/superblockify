"""Module for test fixtures available for all test files"""
import inspect
from ast import literal_eval
from configparser import ConfigParser
from os import listdir, path, remove
from os.path import getsize
from shutil import rmtree

import osmnx as ox
import pytest

from superblockify import partitioning
from superblockify.partitioning import BasePartitioner

config = ConfigParser()
config.read("config.ini")
TEST_DATA = config["tests"]["test_data_path"]
RESULTS_DIR = config["general"]["results_dir"]


@pytest.fixture(
    params=sorted(
        [city for city in listdir(f"{TEST_DATA}cities/") if city.endswith(".graphml")],
        key=lambda city: getsize(f"{TEST_DATA}cities/" + city),
    )
)
def test_city_all(request):
    """Fixture for loading and parametrizing all cities with bearing test_data."""
    # return request.param without .graphml
    return request.param[:-8], ox.load_graphml(
        filepath=f"{TEST_DATA}cities/" + request.param
    )


@pytest.fixture(
    params=[
        city
        for city in listdir(f"{TEST_DATA}cities/")
        if city
        in [
            city[0] + ".graphml"
            for city in literal_eval(config["tests"]["places_small"])
        ]
    ]
)
def test_city_small(request):
    """Fixture for loading and parametrizing all cities with bearing and length
    test_data."""
    return request.param[:-8], ox.load_graphml(
        filepath=f"{TEST_DATA}cities/" + request.param
    )


@pytest.fixture(
    params=inspect.getmembers(
        partitioning,
        predicate=lambda o: inspect.isclass(o)
        and issubclass(o, BasePartitioner)
        and o is not BasePartitioner,
    )
)
def partitioner_class(request):
    """Fixture for parametrizing all partitioners inheriting from BasePartitioner."""
    return request.param[1]


@pytest.fixture(scope="class")
def _teardown_test_graph_io():
    """Delete Adliswil_tmp.graphml file and directory."""
    yield None
    work_cities = ["Adliswil_tmp", "Adliswil_tmp_save_load"]
    for city in work_cities:
        test_graph = path.join(config["general"]["graph_dir"], city + ".graphml")
        if path.exists(test_graph):
            remove(test_graph)
        results_dir = path.join(RESULTS_DIR, city + "_name")
        if path.exists(results_dir):
            rmtree(results_dir)


@pytest.fixture(scope="session", autouse=True)
def _teardown_test_folders():
    """Delete all test data folders."""
    yield None
    # Delete all folders in RESULTS_DIR that end with _test
    for folder in listdir(RESULTS_DIR):
        if folder.endswith("_test"):
            rmtree(path.join(RESULTS_DIR, folder))
