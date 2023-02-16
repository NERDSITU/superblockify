"""Module for test fixtures available for all test files"""
import inspect
from configparser import ConfigParser
from os import listdir

import osmnx as ox
import pytest

from superblockify import partitioning
from superblockify.partitioning import BasePartitioner

config = ConfigParser()
config.read("config.ini")
TEST_DATA = config["tests"]["test_data_path"]


@pytest.fixture(
    params=[
        city
        for city in listdir(f"{TEST_DATA}cities/")
        if city.endswith(".graphml")
    ]
)
def test_city_all(request):
    """Fixture for loading and parametrizing all cities with bearing test_data."""
    return request.param, ox.load_graphml(
        filepath=f"{TEST_DATA}cities/" + request.param
    )


@pytest.fixture(
    params=[
        city
        for city in listdir(f"{TEST_DATA}cities/")
        if city.endswith("_small.graphml")
    ]
)
def test_city_small(request):
    """Fixture for loading and parametrizing all cities with bearing and length
    test_data."""
    return request.param, ox.load_graphml(
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
