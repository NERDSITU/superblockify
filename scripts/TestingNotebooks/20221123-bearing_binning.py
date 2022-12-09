"""First try on neighborhood finding using binning of street bearings."""
# Bearing binning from 20221025-Find_Grids notebook.

import osmnx as ox

import superblockify as sb

if __name__ == "__main__":
    # graph should have bearings with precision of 2.
    # CITY_NAME = "Brooklyn_bearing.graphml"
    CITY_NAME = "Brooklyn_bearing.graphml"
    graph = ox.load_graphml(filepath="./tests/test_data/cities/" + CITY_NAME)

    part = sb.BearingPartitioner(graph)
    part.run(show_analysis_plots=True)
    part.plot_partition_graph()
