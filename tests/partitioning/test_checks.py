"""Test the partitioner checks that are not covered by other tests."""

from networkx import DiGraph

from superblockify.partitioning.checks import is_valid_partitioning


def test_is_valid_partitioning_not_connected(test_city_small_precalculated_copy):
    """Test the `is_valid_partitioning` function."""
    part = test_city_small_precalculated_copy
    part.sparsified = DiGraph([(0, 1), (2, 3)])
    # only add two separate edges to the sparsified graph
    part.get_ltns()[0]["subgraph"] = part.graph.edge_subgraph(
        (list(part.graph.edges)[0], list(part.graph.edges)[30])
    )
    # Move `rep_node` outside component for second component
    part.get_ltns()[1]["rep_node"] = next(
        node for node in part.graph.nodes if node not in part.get_ltns()[1]["subgraph"]
    )  # yields first node of part.graph that is not in the subgraph
    assert is_valid_partitioning(part) is False
