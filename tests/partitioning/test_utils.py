"""Tests for the partitioning utils."""
from copy import deepcopy
from os.path import join, exists

import pytest

from superblockify import save_to_gpkg


@pytest.mark.parametrize("save_path", [None, "test.gpkg"])
def test_save_to_gpkg(test_city_small_precalculated, save_path, _teardown_test_folders):
    """Test saving to geopackage."""
    save_path = (
        None
        if save_path is None
        else join(
            test_city_small_precalculated.results_dir,
            test_city_small_precalculated.name + "-filepath.gpkg",
        )
    )
    save_to_gpkg(deepcopy(test_city_small_precalculated), save_path=save_path)
    # Check that the file exists
    assert exists(
        join(
            test_city_small_precalculated.results_dir,
            test_city_small_precalculated.name
            + ("-filepath.gpkg" if save_path else ".gpkg"),
        )
    )


@pytest.mark.parametrize(
    "replace_attibute",
    [
        [("sparsified", None)],  # no sparsified graph
        [("sparsified", 1)],  # wrong type
        [("components", None), ("partitions", None)],  # no components or partitions
        [("components", None), ("partitions", 1)],  # wrong type
        [("components", 1)],  # wrong type
        [("components", [1])],  # wrong type
        [("components", [{"subgraph": None}])],  # no 'name' attribute
        [("components", [{"name": None}])],  # no 'subgraph' attribute
    ],
)
def test_save_to_gpkg_faulty_subgraphs(test_one_city_precalculated, replace_attibute):
    """Test saving to geopackage with faulty subgraphs."""
    graph = deepcopy(test_one_city_precalculated)
    for attribute, value in replace_attibute:
        setattr(graph, attribute, value)
    with pytest.raises(ValueError):
        save_to_gpkg(graph)
