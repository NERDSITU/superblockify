"""Tests for the metrics plotting module."""

import pytest

from superblockify.metrics.metric import Metric
from superblockify.metrics.plot import (
    plot_distance_matrices,
    plot_distance_matrices_pairwise_relative_difference,
    plot_component_wise_travel_increase,
    plot_relative_difference,
    plot_relative_increase_on_graph,
)


def test_plot_distance_matrices(test_city_small_precalculated_copy):
    """Test plotting distance matrices."""
    part = test_city_small_precalculated_copy
    plt, _ = plot_distance_matrices(part.metric)
    plt.show()


def test_plot_distance_matrices_missing(test_city_small_precalculated_copy):
    """Test plotting distance matrices with missing matrix."""
    part = test_city_small_precalculated_copy
    part.metric.distance_matrix = None
    with pytest.raises(ValueError):
        plot_distance_matrices(part.metric)


def test_plot_distance_matrices_pairwise_relative_difference(
    test_city_small_precalculated_copy,
):
    """Test plotting pairwise relative difference."""
    part = test_city_small_precalculated_copy
    plt, _ = plot_distance_matrices_pairwise_relative_difference(part.metric)
    plt.show()


def test_plot_distance_matrices_pairwise_relative_difference_missing(
    test_city_small_precalculated_copy,
):
    """Test plotting pairwise relative difference with missing matrix."""
    part = test_city_small_precalculated_copy
    part.metric.distance_matrix = None
    with pytest.raises(ValueError):
        plot_distance_matrices_pairwise_relative_difference(part.metric)


def test_plot_relative_difference(test_city_small_precalculated_copy):
    """Test plotting pairwise relative difference."""
    part = test_city_small_precalculated_copy
    plt, _ = plot_relative_difference(part.metric, "N", "S")
    plt.show()


def test_plot_component_wise_travel_increase(test_city_small_precalculated_copy):
    """Test plotting pairwise relative difference."""
    part = test_city_small_precalculated_copy
    plt, _ = plot_component_wise_travel_increase(
        part,
        part.metric.distance_matrix,
        part.metric.node_list,
        measure1="N",
        measure2="S",
        unit=part.metric.unit_symbol(),
    )
    plt.show()


def test_plot_relative_increase_on_graph(test_city_small_precalculated_copy):
    """Test plotting pairwise relative difference."""
    part = test_city_small_precalculated_copy
    plt, _ = plot_relative_increase_on_graph(part.graph, part.metric.unit_symbol())
    plt.show()


def test_plot_distance_matrices_uncalculated():
    """Test plotting distance matrices when they have not been calculated."""
    metric = Metric()
    with pytest.raises(ValueError):
        plot_distance_matrices(metric)
