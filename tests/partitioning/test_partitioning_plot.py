"""Tests for the partitioning plot module."""
import pytest
from matplotlib import pyplot as plt
from matplotlib.pyplot import Figure, Axes

from superblockify.partitioning.plot import (
    plot_partition_graph,
    plot_subgraph_component_size,
    plot_component_rank_size,
    plot_component_graph,
)


def test_plot_partition_graph(test_city_small_precalculated_copy):
    """Test `plot_partition_graph` by design."""
    part = test_city_small_precalculated_copy
    fig, axe = plot_partition_graph(part)
    assert isinstance(fig, Figure)
    assert isinstance(axe, Axes)
    plt.close("all")


def test_plot_component_graph(test_city_all_precalculated):
    """Test `plot_component_graph` by design."""
    part = test_city_all_precalculated
    if part.components is not None:
        fig, axe = plot_component_graph(part)
        assert isinstance(fig, Figure)
        assert isinstance(axe, Axes)
        plt.close("all")


def test_plot_partition_graph_unpartitioned(
    test_city_all_preloaded, test_city_all_precalculated
):
    """Test `plot_partition_graph` exception handling."""
    part = test_city_all_preloaded
    with pytest.raises(AssertionError):
        plot_partition_graph(part)
    part = test_city_all_precalculated
    part.attribute_label = None
    with pytest.raises(AssertionError):
        plot_partition_graph(part)


def test_plot_partitions_unpartitioned(
    test_city_all_preloaded, test_city_all_precalculated
):
    """Test `plot_partition_graph` exception handling."""
    part = test_city_all_preloaded
    with pytest.raises(AssertionError):
        plot_partition_graph(part)
    part = test_city_all_precalculated
    part.attribute_label = None
    with pytest.raises(AssertionError):
        plot_partition_graph(part)


def test_plot_subgraph_component_size(
    test_city_all_preloaded, test_city_all_precalculated
):
    """Test `plot_subgraph_component_size` by design."""
    part = test_city_all_preloaded
    with pytest.raises(AssertionError):
        plot_subgraph_component_size(part, "nodes")
    part = test_city_all_precalculated
    plot_subgraph_component_size(part, "nodes")
    plot_subgraph_component_size(part, "edges")
    plt_len, _ = plot_subgraph_component_size(part, "length")
    assert isinstance(plt_len, Figure)
    plt_len.show()
    part.components = None
    plot_subgraph_component_size(part, "nodes")
    plt.close("all")


@pytest.mark.parametrize(
    "invalid_measure",
    ["", "invalid", "node", None, 10, 1.0, True, False],
)
def test_plot_subgraph_component_size_invalid_measure(
    test_city_small_precalculated_copy, invalid_measure
):
    """Test `plot_subgraph_component_size` with unavailable measure."""
    part = test_city_small_precalculated_copy
    with pytest.raises(ValueError):
        plot_subgraph_component_size(part, invalid_measure)


def test_plot_component_rank_size(test_city_all_preloaded, test_city_all_precalculated):
    """Test `plot_component_rank_size` by design."""
    part = test_city_all_preloaded
    with pytest.raises(AssertionError):
        plot_component_rank_size(part, "nodes")
    part = test_city_all_precalculated
    plot_component_rank_size(part, "nodes")
    plot_component_rank_size(part, "edges")
    plt_len, _ = plot_component_rank_size(part, "length")
    assert isinstance(plt_len, Figure)
    plt_len.show()
    part.components = None
    plot_component_rank_size(part, "nodes")
    plt.close("all")


@pytest.mark.parametrize(
    "invalid_measure",
    ["", "invalid", "node", None, 10, 1.0, True, False],
)
def test_plot_component_rank_size_invalid_measure(
    test_city_small_precalculated_copy, invalid_measure
):
    """Test `plot_component_rank_size` with unavailable measure."""
    part = test_city_small_precalculated_copy
    with pytest.raises(ValueError):
        plot_component_rank_size(part, invalid_measure)
