"""Load this module to fetch test data needed for certain tests."""
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
        graph = ox.graph_from_place(place[1])
        ox.io.save_graphml(graph, filepath=f"./cities/{place[0]}.graphml")
