"""Load this module to fetch test data needed for certain tests."""

import osmnx as ox

from superblockify.utils import load_graph_from_place

# turn response caching off as this only loads graphs to files
ox.config(use_cache=False, log_console=True)

# General cities/neighborhoods
places_general = [
    ("Barcelona", "Barcelona, Catalonia, Spain"),
    ("Brooklyn", "Brooklyn, New York, United States"),
    ("Copenhagen", ["Københavns Kommune, Denmark", "Frederiksberg Kommune, Denmark"]),
    ("Resistencia", "Resistencia, Chaco, Argentina"),
]

places_small = [
    ("Adliswil", "Adliswil, Bezirk Horgen, Zürich, Switzerland"),
    ("Liechtenstein", "Liechtenstein, Europe"),
    ("MissionTown", "团城街道, Xialu, Hubei, China"),
    ("Scheveningen", "Scheveningen, The Hague, Netherlands"),
]

if __name__ == "__main__":
    for place in places_small + places_general:
        load_graph_from_place(
            f"./tests/test_data/cities/{place[0]}_"
            + ("general" if place in places_general else "small")
            + ".graphml",
            place[1],
            network_type="drive",
        )
