"""Plotting functions for the partitioners."""

from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from networkx import set_edge_attributes
from numpy import linspace, array

from .. import plot
from ..attribute import determine_minmax_val
from ..config import logger, Config


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
        "Plotting partitions graph for %s with attribute `%s`",
        partitioner.name,
        partitioner.attribute_label,
    )
    fig, axe = plot.plot_by_attribute(
        partitioner.graph,
        edge_attr=partitioner.attribute_label,
        edge_minmax_val=partitioner.attr_value_minmax,
        **pba_kwargs,
    )
    axe.set_title(f"Street network for {partitioner.name}")
    return fig, axe


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
        "Plotting component graph for %s with attribute `%s`",
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

    fig, axe = plot.plot_by_attribute(
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
    axe.set_title(f"Street network with colored Superblocks for {partitioner.name}")
    # Add a legend with dummy entries
    # - black line: Sparse Network
    # - colored line: Superblocks
    # - circle of same color: Representative node
    legend_handles = [
        plt.Line2D([0], [0], color="black", lw=2, label="Sparse Network"),
    ]
    # Extract colors from colormap
    num_colors = len(partitioner.components)
    colors = [cmap(i / num_colors) for i in range(num_colors)]
    for i, color in enumerate(colors[:2]):  # Only show first two Superblocks
        legend_handles.append(
            plt.Line2D([0], [0], color=color, lw=2, label=f"Superblock {i + 1}")
        )
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=color,
                lw=0,
                marker="o",
                label=f"Representative node {i + 1}",
            )
        )
    # Add a "..." entry to indicate more Superblocks
    legend_handles.append(plt.Line2D([0], [0], color="gray", lw=2, label="..."))
    # place legend outside of plot
    axe.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1, 1))
    # tight layout
    fig.tight_layout()
    return fig, axe


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
    key_name = (
        "length_total" if measure == "length" else "n" if measure == "nodes" else "m"
    )
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
        "Plotting component size rank for %s with attribute `%s`",
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
    axe.set_xlabel("Superblock rank", fontsize=12)
    unit = "m" if measure == "length" else "nodes" if measure == "nodes" else "edges"
    measure = "street length" if measure == "length" else measure
    axe.set_ylabel(f"Superblock size ({measure} ({unit}))", fontsize=12)
    axe.set_title(
        f"Superblock size rank for {partitioner.name} with attribute "
        f"`{partitioner.attribute_label}`",
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
    key_name = (
        "length_total" if measure == "length" else "n" if measure == "nodes" else "m"
    )
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


def plot_speed_un_restricted(
    graph,
    sparsified,
    v_s=Config.V_MAX_SPARSE,
    v_ltn=Config.V_MAX_LTN,
    cmap="viridis",
):
    """Plot the speed limit of the edges of a graph before and after restrictions.

    Side-by-side plot of the speed limits of the edges of a graph, before and after
    restrictions. Both maps use a shared colorbar.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph to plot the speed limits of.
        Needs to have `speed_kph` as the unrestricted speed limit.
    sparsified : networkx.MultiDiGraph
        The sparsified graph, optimally a view of the original graph.
    v_s : float
        The max speed for the edges of the sparsified graph.
    v_ltn : float
        The max speed for the remaining edges.
    cmap : str
        The colormap to use.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    axe : matplotlib.axes.Axes
        The axes containing the plot.
    """

    # Create a wide figure with two subplots and the specified size
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Find min and max speed limits
    minmax_val = determine_minmax_val(graph, None, "speed_kph", attr_type="edge")
    # include v_s and v_ltn in the minmax_val tuple
    minmax_val = (min(minmax_val[0], v_s, v_ltn), max(minmax_val[1], v_s, v_ltn))

    # Write a new attribute to the graph with the restricted speed limits - all edges in
    # sparsified graph have speed limit v_s, all other edges have speed limit v_ltn
    for edge in graph.edges:
        if edge in sparsified.edges:
            graph.edges[edge]["speed_kph_restricted"] = v_s
        else:
            graph.edges[edge]["speed_kph_restricted"] = v_ltn

    # Plot original max speed limits on left side
    plot.plot_by_attribute(
        graph,
        "speed_kph",
        edge_cmap=cmap,
        edge_minmax_val=minmax_val,
        ax=axes[0],
    )

    # Plot restricted max speed limits on right side
    plot.plot_by_attribute(
        graph,
        "speed_kph_restricted",
        edge_cmap=cmap,
        edge_minmax_val=minmax_val,
        ax=axes[1],
    )

    # Set titles
    axes[0].set_title("Original speed limits", fontsize=20)
    axes[1].set_title("Restricted speed limits", fontsize=20)

    # Set shared colorbar
    cbar = fig.colorbar(
        ScalarMappable(
            norm=Normalize(vmin=minmax_val[0], vmax=minmax_val[1]),
            cmap=cmap,
        ),
        ax=axes,
    )
    cbar.set_label("Speed limit (km/h)", fontsize=20, labelpad=20)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.tick_params(labelsize=15)

    return fig, axes
