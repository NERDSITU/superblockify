"""Tests for the metrics plotting module."""
import pytest

from superblockify.metrics import plot_distance_matrices
from superblockify.metrics.metric import Metric


def test_plot_distance_matrices_uncalculated():
    """Test plotting distance matrices when they have not been calculated."""
    metric = Metric()
    with pytest.raises(ValueError):
        plot_distance_matrices(metric)
