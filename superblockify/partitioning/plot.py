"""Plotting functions for the partitioners."""
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from networkx import set_edge_attributes
from numpy import linspace, array

from .. import plot
from ..config import logger


def plot_partition_graph(partitioner, **pba_kwargs):
    """Plotting the partitions with color on graph.

    Plots the partitioned graph, just like `plot.paint_streets` but that the
    *partitions* have a uniform color.

    Parameters
    ----------
    partitioner : BasePartitioner
        The partitioner to plot.
    pba_kwargs
        Keyword arguments to pass to `superblockify.plot_by_attribute`.

    Returns
    -------
    fig, axe : tuple
        matplotlib figure, axis

    Raises
    ------
    AssertionError
        If BasePartitioner has not been run yet (the partitions are not defined).

    """

    partitioner.check_has_been_run()

    # Log plotting
    logger.info(
        "Plotting partitions graph for %s with attribute %s",
        partitioner.name,
        partitioner.attribute_label,
    )
    return plot.plot_by_attribute(
        partitioner.graph,
        edge_attr=partitioner.attribute_label,
        edge_minmax_val=partitioner.attr_value_minmax,
        **pba_kwargs,
    )


def plot_component_graph(partitioner, **pba_kwargs):
    """Plotting the components with color on graph.

    Plots the graph with the components, just like `plot.paint_streets` but that
    the *components* have a uniform color.

    Parameters
    ----------
    partitioner : BasePartitioner
        The partitioner to plot.
    pba_kwargs
        Keyword arguments to pass to `superblockify.plot_by_attribute`.

    Returns
    -------
    fig, axe : tuple
        matplotlib figure, axis

    Raises
    ------
    AssertionError
        If BasePartitioner has not been run yet (the partitions are not defined).
    AssertionError
        If `partitioner.components` is not defined (the subgraphs have not been split
        into components).

    """

    partitioner.check_has_been_run()

    if partitioner.components is None:
        raise AssertionError(
            f"Components have not been defined for {partitioner.name}. "
            f"Run `make_subgraphs_from_attribute` with `split_disconnected` "
            f"set to True."
        )

    # Log plotting
    logger.info(
        "Plotting component graph for %s with attribute %s",
        partitioner.name,
        partitioner.attribute_label,
    )
    # Bake component labels into graph
    for component in partitioner.components:
        if not component["ignore"]:
            set_edge_attributes(
                component["subgraph"],
                component["name"],
                "component_name",
            )

    cmap = plt.get_cmap("prism")
    # So markers for representative nodes are not the same color as the edges,
    # where they are placed on, construct a new color map from the prism color
    # map, but which is darker. Same colors as cmap, but all values are
    # multiplied by 0.75, except the alpha value, which is set to 1.
    dark_cmap = ListedColormap(
        array([cmap(i) for i in range(cmap.N)]) * array([0.75, 0.75, 0.75, 1])
    )

    return plot.plot_by_attribute(
        partitioner.graph,
        edge_attr="component_name",
        edge_attr_types="categorical",
        edge_cmap=cmap,
        edge_minmax_val=None,
        node_attr="representative_node_name",
        node_attr_types="categorical",
        node_cmap=dark_cmap,
        node_minmax_val=None,
        node_size=40,
        node_zorder=2,
        **pba_kwargs,
    )


def plot_component_rank_size(partitioner, measure):
    """Plot a rank distribution of the component sizes.

    Scatter plot of the component sizes, sorted after the rank of the component.

    Parameters
    ----------
    partitioner : BasePartitioner
        The partitioner to plot.
    measure : str, optional
        Way to measure component size. Can be 'edges', 'length' or 'nodes'.
    xticks : list of numbers or strings, optional
        List of xticks to use. If None, the xticks are seven evely spaced numbers
        between the partitioner.attr_value_minmax.

    Returns
    -------
    fig, axe : tuple
        matplotlib figure, axis

    Raises
    ------
    AssertionError
        If BasePartitioner has not been run yet (the partitions are not defined).
    ValueError
        If measure is not 'edges', 'length' or 'nodes'.

    """

    partitioner.check_has_been_run()

    if measure not in ["edges", "length", "nodes"]:
        raise ValueError(
            f"Measure '{measure}' is not supported, "
            f"use 'edges', 'length' or 'nodes'."
        )

    # Find number of edges in each component for each partition
    key_name = "length_total" if measure == "length" else f"num_{measure}"
    component_size = []
    ignore = []

    # If subgraphs were split, use components
    if partitioner.components:
        logger.debug("Using components for plotting.")
        for comp in partitioner.components:
            component_size.append(comp[key_name])
            ignore.append(comp["ignore"])
    # Else use partitions
    else:
        logger.debug("Using partitions for plotting.")
        for part in partitioner.partitions:
            component_size.append(part[key_name])
            ignore = None

    # Sort component sizes, ignore ignored components
    component_size = array(component_size)
    if ignore:
        component_size = component_size[~array(ignore)]
    component_size = sorted(component_size, reverse=True)

    # Log plotting
    logger.info(
        "Plotting component size rank for %s with attribute %s",
        partitioner.name,
        partitioner.attribute_label,
    )

    # Plot
    fig, axe = plt.subplots(figsize=(8, 6))

    # Plot component size rank
    axe.scatter(
        range(len(component_size)),
        component_size,
        s=10,
        marker="o",
        color="k",
        zorder=2,
    )
    axe.set_xlabel("Component rank", fontsize=12)
    axe.set_ylabel(f"Component size ({measure} [m])", fontsize=12)
    axe.set_title(
        f"Component size rank for {partitioner.name} with attribute "
        f"{partitioner.attribute_label}",
        fontsize=13,
    )
    axe.set_yscale("log")
    axe.set_ylim(bottom=1)
    axe.set_xlim(left=-1)
    axe.grid(True)
    fig.tight_layout()
    return fig, axe


def plot_subgraph_component_size(partitioner, measure, xticks=None, **pcs_kwargs):
    """Plot the size of the subgraph components of the partitions.

    Scatter plot of the size of the subgraph components of each partition type.

    Parameters
    ----------
    partitioner : BasePartitioner
        The partitioner to plot.
    measure : str, optional
        Way to measure component size. Can be 'edges', 'length' or 'nodes'.
    xticks : list of numbers or strings, optional
        List of xticks to use. If None, the xticks are seven evely spaced numbers
        between the partitioner.attr_value_minmax.
    pcs_kwargs
        Keyword arguments to pass to `superblockify.plot.plot_component_size`.

    Returns
    -------
    fig, axe : tuple
        matplotlib figure, axis

    Raises
    ------
    AssertionError
        If BasePartitioner has not been run yet (the partitions are not defined).
    ValueError
        If measure is not 'edges', 'length' or 'nodes'.

    """

    partitioner.check_has_been_run()

    if measure not in ["edges", "length", "nodes"]:
        raise ValueError(
            f"Measure '{measure}' is not supported, "
            f"use 'edges', 'length' or 'nodes'."
        )

    # Find number of edges in each component for each partition
    key_name = "length_total" if measure == "length" else f"num_{measure}"
    component_size = []
    component_values = []
    ignore = []

    # If subgraphs were split, use components
    if partitioner.components:
        logger.debug("Using components for plotting.")
        for comp in partitioner.components:
            component_size.append(comp[key_name])
            component_values.append(comp["value"])
            ignore.append(comp["ignore"])
    # Else use partitions
    else:
        logger.debug("Using partitions for plotting.")
        for part in partitioner.partitions:
            component_size.append(part[key_name])
            component_values.append(part["value"])
            ignore = None

    if xticks is None:
        xticks = list(linspace(*partitioner.attr_value_minmax, 7))

    # Plot
    return plot.plot_component_size(
        graph=partitioner.graph,
        attr=partitioner.attribute_label,
        component_size=component_size,
        component_values=component_values,
        size_measure_label=f"Component size ({measure})",
        ignore=ignore,
        title=partitioner.name,
        minmax_val=partitioner.attr_value_minmax,
        xticks=xticks,
        **pcs_kwargs,
    )
