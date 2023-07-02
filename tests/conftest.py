"""Module for test fixtures available for all test files"""
from copy import deepcopy
from functools import wraps
from glob import glob
from os import listdir, remove
from os.path import getsize, join, exists
from shutil import rmtree

import pytest
from networkx import set_node_attributes
from requests.exceptions import ConnectTimeout
from urllib3.exceptions import MaxRetryError

from superblockify.config import (
    TEST_DATA_PATH,
    RESULTS_DIR,
    PLACES_SMALL,
    GRAPH_DIR,
    HIDE_PLOTS,
)
from superblockify.partitioning import __all_partitioners__
from superblockify.utils import load_graphml_dtypes

TEST_CITY_PATH = join(TEST_DATA_PATH, "cities")
ALL_CITIES_SORTED = sorted(
    [city for city in listdir(TEST_CITY_PATH) if city.endswith(".graphml")],
    key=lambda city: getsize(join(TEST_CITY_PATH, city)),
)
SMALL_CITIES = [
    city
    for city in listdir(TEST_CITY_PATH)
    if city in [city[0] + ".graphml" for city in PLACES_SMALL]
]


# Redefining names for extending fixtures
# pylint: disable=redefined-outer-name

# *****************************************************
# Partitioner Fixtures
# *****************************************************


@pytest.fixture(
    scope="session",
    params=[
        part
        if not getattr(part, "__deprecated__", False)
        else pytest.param(part, marks=pytest.mark.xfail(reason=part.__deprecated__))
        for part in __all_partitioners__
        if not getattr(part, "__exclude_test_fixture__", False)
    ],
    # use partitioner name, but cut off "Partitioner"-> "Part" if it is there
    ids=[
        part.__name__.replace("Partitioner", "Part")
        for part in __all_partitioners__
        if not getattr(part, "__exclude_test_fixture__", False)
    ],
)
def partitioner_class(request):
    """Fixture for parametrizing all partitioners inheriting from BasePartitioner."""
    return request.param


# *****************************************************
# City Fixtures
# *****************************************************
# All cities
# **********


@pytest.fixture(
    scope="session",
    params=ALL_CITIES_SORTED,
    ids=[city[:-8] for city in ALL_CITIES_SORTED],
)
def test_city_all(request):
    """Fixture for loading and parametrizing all city graphs from test_data."""
    # return request.param without .graphml
    return request.param[:-8], load_graphml_dtypes(join(TEST_CITY_PATH, request.param))


@pytest.fixture(scope="function")
def test_city_all_copy(test_city_all):
    """Fixture for getting a copy of all city graphs from test_data."""
    city_name, graph = test_city_all
    return city_name, graph.copy()


@pytest.fixture(scope="session")
def test_city_all_preloaded_save(test_city_all, partitioner_class):
    """Fixture for saving preloaded partitioners for all cities with bearing and
    length test_data. Without metrics. Shared across all tests."""
    city_name, graph = test_city_all
    part = partitioner_class(
        name=f"{city_name}_{partitioner_class.__name__}_preloaded_test",
        city_name=city_name,
        graph=graph.copy(),
    )
    part.save(save_graph_copy=True)
    assert exists(join(part.results_dir, part.name + ".partitioner"))
    assert exists(join(part.results_dir, part.name + ".metrics"))
    return part.name, part.__class__


@pytest.fixture(scope="function")
def test_city_all_preloaded(test_city_all_preloaded_save):
    """Fixture for preloaded partitioners for all cities with bearing and length.
    Without metrics. Loaded for each test."""
    name, cls = test_city_all_preloaded_save
    return cls.load(name=name)


@pytest.fixture(scope="session")
def test_city_all_precalculated_save(test_city_all, partitioner_class):
    """Fixture for saving precalculated partitioners for all cities with bearing and
    length test_data. Without metrics. Shared across all tests."""
    city_name, graph = test_city_all
    part = partitioner_class(
        name=f"{city_name}_{partitioner_class.__name__}_precalculated_test",
        city_name=city_name,
        graph=graph.copy(),
        max_nodes=None,
    )
    part.run(calculate_metrics=False)
    part.save(save_graph_copy=True)
    return part.name, part.__class__


@pytest.fixture(scope="function")
def test_city_all_precalculated(test_city_all_precalculated_save):
    """Fixture for precalculated partitioners for all cities with bearing and length.
    Without metrics. Loaded for each test."""
    name, cls = test_city_all_precalculated_save
    return cls.load(name=name)


@pytest.fixture(scope="session")
def test_city_all_reduced_precalculated_save(test_city_all, partitioner_class):
    """Fixture for saving a reduced precalculated partitioners for all cities with
    bearing and length test_data. Without metrics. Shared across all tests.
    The graphs are reduced to the half of the nodes but max 1000 nodes.
    """
    city_name, graph = test_city_all
    part = partitioner_class(
        name=f"{city_name}_{partitioner_class.__name__}_precalculated_test",
        city_name=city_name,
        graph=graph.copy(),
        max_nodes=min(1000, graph.number_of_nodes() // 2),
    )
    part.run(calculate_metrics=False)
    part.save(save_graph_copy=True)
    return part.name, part.__class__


@pytest.fixture(scope="function")
def test_city_all_reduced_precalculated(test_city_all_reduced_precalculated_save):
    """Fixture for reduced precalculated partitioners for all cities with bearing and
    length. Without metrics. Loaded for each test. Reduced to half or max 1000 nodes."""
    name, cls = test_city_all_reduced_precalculated_save
    return cls.load(name=name)


# ************
# Small cities
# ************


@pytest.fixture(
    scope="session", params=SMALL_CITIES, ids=[city[:-8] for city in SMALL_CITIES]
)
def test_city_small(request):
    """Fixture for loading and parametrizing small city graphs from test_data."""
    return request.param[:-8], load_graphml_dtypes(join(TEST_CITY_PATH, request.param))


@pytest.fixture(scope="function")
def test_city_small_copy(test_city_small):
    """Fixture for getting a copy of small city graphs from test_data."""
    city_name, graph = test_city_small
    return city_name, graph.copy()


@pytest.fixture(scope="session")
def test_city_small_precalculated(test_city_small, partitioner_class):
    """Fixture for loading and parametrizing small cities with bearing and length
    test_data. With metrics."""
    city_name, graph = test_city_small
    part = partitioner_class(
        name=f"{city_name}_{partitioner_class.__name__}_precalculated_test",
        city_name=city_name,
        graph=graph.copy(),
    )
    part.run(calculate_metrics=True)
    part.save(save_graph_copy=True)
    return part.name, part.__class__


@pytest.fixture(scope="function")
def test_city_small_precalculated_copy(test_city_small_precalculated):
    """Return a copy of small cities with bearing and length test_data. Without
    metrics. Loaded for each test."""
    name, cls = test_city_small_precalculated
    return cls.load(name=name)


@pytest.fixture(scope="session")
def test_city_small_preloaded(test_city_small, partitioner_class):
    """Fixture for loading and parametrizing small cities not run yet."""
    city_name, graph = test_city_small
    part = partitioner_class(
        name=f"{city_name}_{partitioner_class.__name__}_preloaded_test",
        city_name=city_name,
        graph=graph.copy(),
    )
    return part


@pytest.fixture(scope="function")
def test_city_small_preloaded_copy(test_city_small_preloaded):
    """Return a copy of small cities not run yet."""
    return deepcopy(test_city_small_preloaded)


@pytest.fixture(scope="module")
def test_city_small_osmid(test_city_small):
    """Return a graph with the osmid baked down to a single value."""
    _, graph = test_city_small
    # Some osmid attributes return lists, not ints, just take first element
    set_node_attributes(
        graph,
        {node: node for node in graph.nodes()},
        "osmid",
    )
    return graph


@pytest.fixture(scope="function")
def test_city_small_osmid_copy(test_city_small_osmid):
    """Return a copy of the graph with the osmid baked down to a single value."""
    return test_city_small_osmid.copy()


# ********
# One city
# ********


@pytest.fixture(scope="session")
def test_one_city():
    """Fixture for loading and parametrizing one small city."""
    return SMALL_CITIES[0][:-8], load_graphml_dtypes(
        join(TEST_CITY_PATH, SMALL_CITIES[0])
    )


@pytest.fixture(scope="function")
def test_one_city_copy(test_one_city):
    """Fixture for getting a copy of one small city."""
    city_name, graph = test_one_city
    return city_name, graph.copy()


@pytest.fixture(scope="session")
def test_one_city_precalculated(partitioner_class, test_one_city):
    """Fixture for loading and parametrizing one small city with bearing and length
    test_data. Without metrics."""
    city_name, graph = test_one_city
    part = partitioner_class(
        name=f"{city_name}_{partitioner_class.__name__}_precalculated_test",
        city_name=city_name,
        graph=graph.copy(),
    )
    part.run(calculate_metrics=False)
    part.save(save_graph_copy=True)
    return part.name, part.__class__


@pytest.fixture(scope="function")
def test_one_city_precalculated_copy(test_one_city_precalculated):
    """Return a copy of one city with bearing and length test_data. Without metrics.
    Loaded for each test."""
    name, cls = test_one_city_precalculated
    return cls.load(name=name)


@pytest.fixture(scope="session")
def test_one_city_preloaded(partitioner_class, test_one_city):
    """Fixture for loading and parametrizing one small city not run yet."""
    city_name, graph = test_one_city
    part = partitioner_class(
        name=f"{city_name}_{partitioner_class.__name__}_preloaded_test",
        city_name=city_name,
        graph=graph.copy(),
    )
    return part


@pytest.fixture(scope="function")
def test_one_city_preloaded_copy(test_one_city_preloaded):
    """Return a copy of one city not run yet."""
    return deepcopy(test_one_city_preloaded)


# *****************************************************
# Other Fixtures
# *****************************************************
# Clean up test folders after tests are done
# ******************************************


@pytest.fixture(scope="class")
def _teardown_test_graph_io():
    """Delete Adliswil_tmp.graphml file and directory."""
    yield None
    work_cities = ["Adliswil_tmp", "Adliswil_tmp_save_load"]
    for city in work_cities:
        test_graph = join(GRAPH_DIR, city + ".graphml")
        if exists(test_graph):
            remove(test_graph)
        results_dir = join(RESULTS_DIR, city + "_name")
        if exists(results_dir):
            rmtree(results_dir)


@pytest.fixture(scope="session", autouse=True)
def _teardown_test_folders():
    """Delete all test data folders."""
    yield None
    # Delete all folders in RESULTS_DIR that end with _test
    if exists(RESULTS_DIR):
        for folder in listdir(RESULTS_DIR):
            if folder.endswith("_test"):
                rmtree(join(RESULTS_DIR, folder))


@pytest.fixture(scope="module")
def _delete_ghsl_tifs():
    """Delete GHSL tifs."""
    yield
    for filepath in glob(join(TEST_DATA_PATH, "GHS_POP*.tif")):
        remove(filepath)


# ***********************
# Hide plots during tests
# ***********************


@pytest.fixture(scope="function", autouse=HIDE_PLOTS)
def _patch_plt_show(monkeypatch):
    """Patch plt.show() and plt.Figure.show() to prevent plots from showing during
    tests."""
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    monkeypatch.setattr("matplotlib.pyplot.Figure.show", lambda _: None)


# ****************************
# Mark download tests as xfail
# ****************************


def mark_xfail_flaky_download(test_func):
    """Decorator to mark flaky tests that download data from OSM as xfail."""

    # https://stackoverflow.com/questions/43937748/wrap-each-pytest-test-function-into-try-except
    @wraps(test_func)
    def test_func_wrapper(*args, **kwargs):
        try:
            test_func(*args, **kwargs)
        except (MaxRetryError, ConnectTimeout) as err:
            pytest.xfail(f"Download failed for {test_func.__name__}: {err}")
            raise err
        # Also for Exception if the test includes "Bad Gateway"
        except Exception as err:  # pylint: disable=broad-except
            if "Bad Gateway" in str(err):  # specific error
                pytest.xfail(f"Download failed for {test_func.__name__}: {err}")
                raise err
            raise err  # broader error - not marked as xfail

    return test_func_wrapper
