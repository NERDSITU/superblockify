"""If you downloaded cities with scripts/analysis/03_load_cities.py, and you had a 
larger `MAX_NODES` than you want to work with, you can use this script to reduce the
already saved graphs.
"""

from os import listdir
from os.path import join
from shutil import move

from osmnx import save_graphml

from superblockify.partitioning.utils import reduce_graph
from superblockify.config import logger, GRAPH_DIR
from superblockify.utils import load_graphml_dtypes

# from sys import path
# path.append(join(dirname(__file__), "..", ".."))

MAX_NODES = 20_000

if __name__ == "__main__":
    # Loop over all graphs in GRAPH_DIR (.graphml files)
    for graph_file in listdir(GRAPH_DIR):
        if not graph_file.endswith(".graphml"):
            continue
        graph_path = join(GRAPH_DIR, graph_file)
        logger.info("Loading graph from %s", graph_path)
        graph = load_graphml_dtypes(graph_path)
        if graph.number_of_nodes() <= MAX_NODES:
            logger.info("Graph has %s nodes, skipping", graph.number_of_nodes())
            continue
        logger.info("Loaded graph with %s nodes", len(graph.nodes))
        # Reduce graph
        graph = reduce_graph(graph, max_nodes=MAX_NODES)
        # Backup graph - move to subfolder "unreduced"
        logger.info("Moving old graph to subfolder 'unreduced'")
        move(graph_path, join(GRAPH_DIR, "unreduced", graph_file))
        # Save graph
        logger.info("Saving graph to %s", graph_path)
        save_graphml(graph, graph_path)
        logger.info("Saved graph to %s with ", graph_path)
