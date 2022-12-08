"""Module for test fixtures available for all test files"""
from configparser import ConfigParser
import inspect
from os import listdir

import osmnx as ox
import pytest

from superblockify.partitioning import BasePartitioner
from superblockify import partitioning

config = ConfigParser()
config.read("config.ini")
TEST_DATA = config["tests"]["test_data_path"]


@pytest.fixture(
    params=[
        city
        for city in listdir(f"{TEST_DATA}cities/")
        if city.endswith("_bearing.graphml")
    ]
)
def test_city_bearing(request):
    """Fixture for loading and parametrizing all cities with bearing test_data."""
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
