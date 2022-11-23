"""Script on how to use paint streets function"""
# Belongs to 20221018-Painting_Grids notebook.

import osmnx as ox

from superblockify.plot import paint_streets

if __name__ == "__main__":
    CITY_NAME = "Brooklyn_bearing.graphml"
    graph = ox.load_graphml(filepath="../../tests/test_data/cities/" + CITY_NAME)

    paint_streets(
        graph,
        save=True,
        edge_linewidth=0.5,
        filepath=f"../../tests/test_data/output/{CITY_NAME[:-8]}.pdf",
    )
