"""Load this module to fetch test data needed for certain tests."""
from itertools import chain

import osmnx as ox

# turn response caching off as this only loads graphs to files
ox.config(use_cache=False, log_console=True)

# General cities/neighborhoods
places_bearing = [
    ("Barcelona", "Barcelona, Catalonia, Spain"),
    ("Brooklyn", "Brooklyn, New York, United States"),
    ("Copenhagen", ["Københavns Kommune, Denmark", "Frederiksberg Kommune, Denmark"]),
    ("MissionTown", "团城街道, Xialu, Hubei, China"),
    ("Resistencia", "Resistencia, Chaco, Argentina"),
    ("Scheveningen", "Scheveningen, The Hague, Netherlands"),
]

places_bearing_length = [
    ("Adliswil", "Adliswil, Bezirk Horgen, Zürich, Switzerland"),
    ("Liechtenstein", "Liechtenstein, Europe"),
    ("MissionTown", "团城街道, Xialu, Hubei, China"),
    ("Scheveningen", "Scheveningen, The Hague, Netherlands"),
]


def extract_attributes(graph, edge_attributes, node_attributes):
    """Extract only the specified attributes from a graph.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph to extract attributes from.
    edge_attributes : set
        The edge attributes to keep.
    node_attributes : set
        The node attributes to keep.

    Returns
    -------
    networkx.MultiDiGraph
        The graph with only the specified attributes.
    """

    # Get all unique edge attributes
    all_attributes = set(
        chain.from_iterable(d.keys() for *_, d in graph.edges(data=True))
    )

    excess_attributes = all_attributes - edge_attributes
    # Delete all excess attributes
    for _, _, attr_dict in graph.edges(data=True):
        for att in excess_attributes:
            attr_dict.pop(att, None)

    # Get all unique node attributes
    all_attributes = set(
        chain.from_iterable(d.keys() for *_, d in graph.nodes(data=True))
    )

    excess_attributes = all_attributes - node_attributes
    # Delete all excess attributes
    for _, attr_dict in graph.nodes(data=True):
        for att in excess_attributes:
            attr_dict.pop(att, None)

    return graph


if __name__ == "__main__":
    for place in places_bearing:
        test_graph = ox.graph_from_place(place[1], network_type="drive")

        test_graph = extract_attributes(
            test_graph,
            edge_attributes={"geometry", "osmid"},
            node_attributes={"y", "x", "osmid"},
        )

        # Add edge bearings - the precision >1 is important for binning
        test_graph = ox.add_edge_bearings(test_graph, precision=2)

        ox.io.save_graphml(
            test_graph,
            filepath=f"./tests/test_data/cities" f"/{place[0]}_bearing.graphml",
        )

    for place in places_bearing_length:
        test_graph = ox.graph_from_place(place[1], network_type="drive")

        test_graph = ox.distance.add_edge_lengths(test_graph)

        test_graph = extract_attributes(
            test_graph,
            edge_attributes={"geometry", "osmid", "length"},
            node_attributes={"y", "x", "osmid"},
        )

        # Add edge bearings - the precision >1 is important for binning
        test_graph = ox.add_edge_bearings(test_graph, precision=2)

        ox.io.save_graphml(
            test_graph,
            filepath=f"./tests/test_data/cities" f"/{place[0]}_bearing_length.graphml",
        )
