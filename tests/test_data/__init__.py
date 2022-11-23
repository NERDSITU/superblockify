"""Load this module to fetch test data needed for certain tests."""
from itertools import chain

import osmnx as ox

# turn response caching off as this only loads graphs to files
ox.config(use_cache=False)

# General cities/neighborhoods
places = [
    ('Brooklyn', 'Brooklyn, New York, United States'),
    ('MissionTown', '团城街道, Xialu, Hubei, China'),
    ('Resistencia', 'Resistencia, Chaco, Argentina'),
    ('Scheveningen', 'Scheveningen, The Hague, Netherlands')
]

if __name__ == "__main__":
    for place in places:
        graph = ox.graph_from_place(place[1], network_type='drive')

        # Get all unique edge attributes
        all_attributes = set(
            chain.from_iterable(d.keys() for *_, d in graph.edges(data=True)))
        keep_attributes = {'geometry', 'osmid'}  # only keep these
        excess_attributes = all_attributes - keep_attributes
        # Delete all excess attributes
        for n1, n2, d in graph.edges(data=True):
            for att in excess_attributes:
                d.pop(att, None)

        # Get all unique node attributes
        all_attributes = set(
            chain.from_iterable(d.keys() for *_, d in graph.nodes(data=True)))
        keep_attributes = {'y', 'x', 'osmid'}  # only keep these
        excess_attributes = all_attributes - keep_attributes
        # Delete all excess attributes
        for n, d in graph.nodes(data=True):
            for att in excess_attributes:
                d.pop(att, None)

        # Add edge bearings
        graph = ox.add_edge_bearings(graph)

        ox.io.save_graphml(graph, filepath=f"./cities/{place[0]}_bearing.graphml")
