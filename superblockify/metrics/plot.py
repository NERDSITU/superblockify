"""Plotting functions for the network measures."""

from itertools import product

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from .measures import rel_increase
from ..plot import plot_by_attribute


def plot_distance_distributions(
    dist_matrix, dist_title, coords, coord_title, labels, distance_unit="m"
):
    """Plot the distributions of the Euclidean distances and coordinates.

    Parameters
    ----------
    dist_matrix : ndarray
        The distance matrix for the partitioning. dist_matrix[i, j] is the Euclidean
        distance between node i and node j.
    dist_title : str
        The title of the histogram of the Euclidean distances.
    coords : tuple
        The coordinates of the nodes. coords[0] is the x-coordinates, coords[1] is
        the y-coordinates. Can be either angular or Euclidean coordinates.
    coord_title : str
        The title of the scatter plot of the coordinates.
    labels : tuple
        The labels of the coordinates. labels[0] is the label of the x-coordinate,
        labels[1] is the label of the y-coordinate.
    distance_unit : str, optional
        The unit of the distance. Defaults to "m".

    """
    _, axe = plt.subplots(1, 2, figsize=(10, 5))
    # Plot distribution of distances
    axe[0].hist(dist_matrix.flatten(), bins=100)
    axe[0].set_title(dist_title)
    axe[0].set_xlabel(
        "Travel time (s)" if distance_unit == "s" else f"Distance ({distance_unit})"
    )
    axe[0].set_ylabel("Count")
    # Plot scatter plot of lat/lon, aspect ratio should be 1:1
    axe[1].set_aspect("equal")
    axe[1].scatter(coords[0], coords[1], alpha=0.5, s=1)
    axe[1].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    axe[1].set_title(coord_title)
    axe[1].set_xlabel(labels[0])
    axe[1].set_ylabel(labels[1])
    plt.tight_layout()


def plot_distance_matrices(metric, name=None):
    """Show the distance matrices for the network measures.

    Plots all available distance matrices in a single figure.

    Parameters
    ----------
    metric : metrics.Metric
        The metric to plot the distance matrices for.
    name : str
        The name to put into the title of the plot.

    Returns
    -------
    fig, axes : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes of the plot.

    Raises
    ------
    ValueError
        If no distance matrices are available.
    """

    if metric.distance_matrix is None:
        raise ValueError("No distance matrices available.")

    # Make figure with the fitting amount of subplots
    fig, axes = plt.subplots(
        1, len(metric.distance_matrix), figsize=(len(metric.distance_matrix) * 5, 5)
    )

    # Find maximal, non-inf value for the colorbar
    max_val = max(
        np.max(value[value != np.inf]) for value in metric.distance_matrix.values()
    )
    dist_im = None
    # Subplots with shared colorbar, title, and y-axis label
    for axe, (key, value) in zip(axes, metric.distance_matrix.items()):
        dist_im = axe.imshow(value, vmin=0, vmax=max_val)
        axe.set_title(f"$d_{key}(i, j)$", fontsize=14)
        axe.set_xlabel("Node $j$")
        axe.set_aspect("equal")
    # Share y-axis
    axes[0].set_ylabel("Node $i$")
    for axe in axes[1:]:
        axe.get_shared_y_axes().joined(axes[0], axe)

    # Plot colorbar on the right side of the figure
    fig.colorbar(dist_im, ax=axes, fraction=0.046, pad=0.04)
    # Label colorbar
    # dist_im.set_label(f"Distance [{unit}]") not working
    # Label at the right of the plot, next to colorbar
    fig.text(
        0.925,
        0.5,
        f"Distance [{metric.unit_symbol()}]",
        va="center",
        rotation="vertical",
        fontsize=14,
    )

    # Title above all subplots
    fig.suptitle(
        f"Distance matrices for the network measures ({metric.unit_symbol()})\n"
        f"{'(' + name + ')' if name else ''}",
        fontsize=16,
    )

    return fig, axes


def plot_distance_matrices_pairwise_relative_difference(metric, name=None):
    """Show the pairwise relative difference between the distance matrices.

    Plots the pairwise relative difference between the distance matrices in a
    single figure. Only plots the lower triangle of the distance matrices.
    On the diagonal the distance matrices are plotted as in
    `plot_distance_matrices`.

    Parameters
    ----------
    metric : metrics.Metric
        The metric to plot the distance matrices for.
    name : str
        The name to put into the title of the plot.

    Returns
    -------
    fig, axes : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes of the plot.

    Raises
    ------
    ValueError
        If no distance matrices are available.
    """
    # pylint: disable=too-many-locals

    if metric.distance_matrix is None:
        raise ValueError("No distance matrices available.")

    # Make figure with the fitting amount of subplots
    # We need len(metric.distance_matrix)^2 subplots, but we only plot the lower
    # triangle, the rest will be empty. On the diagonal we plot the distance
    # matrices.
    fig, axes = plt.subplots(
        len(metric.distance_matrix),
        len(metric.distance_matrix),
        figsize=(len(metric.distance_matrix) * 5, len(metric.distance_matrix) * 5),
    )
    # Find maximal, non-inf value for the colorbar for the diagonal
    max_val = max(
        np.max(value[value != np.inf]) for value in metric.distance_matrix.values()
    )
    # Calculate the pairwise relative difference between the distance matrices
    # save the relative difference and the minimal value for the colorbar regarding
    # the absolute value
    rel_diff = {}
    max_val_rel = 0
    # For the lower triangle
    for i, (key_i, value_i) in enumerate(metric.distance_matrix.items()):
        for _, (key_j, value_j) in enumerate(metric.distance_matrix.items()):
            if f"{key_i}{key_j}" not in metric.directness.keys():
                continue
            # Calculate the pairwise relative difference
            # Use np.inf if either value is np.inf or if the denominator is 0
            # pylint: disable=arguments-out-of-order
            rel_diff[key_j, key_i] = rel_increase(value_j, value_i)
            # Find the maximal value for the colorbar regarding the absolute value
            # (ignoring np.inf)
            max_val_rel = max(
                max_val_rel,
                np.max(rel_diff[key_j, key_i][rel_diff[key_j, key_i] != np.inf]),
            )

    # Plot distance matrices on diagonal axes and relative difference on the
    # lower triangle axes
    # Iterate over all combinations of keys, for the upper triangle make the axes
    # invisible
    # Only write labels on the left and bottom axes
    dist_im = None
    diff_im = None
    for i, (key_i, key_j) in enumerate(product(metric.distance_matrix, repeat=2)):
        axe = axes[i // len(metric.distance_matrix), i % len(metric.distance_matrix)]

        # Make the upper triangle axes invisible
        if i // len(metric.distance_matrix) < i % len(metric.distance_matrix):
            axe.set_visible(False)
        # On the diagonal plot the distance matrices
        elif i // len(metric.distance_matrix) == i % len(metric.distance_matrix):
            # Use colormap viridis for the distance matrices
            dist_im = axe.imshow(
                metric.distance_matrix[key_i], vmin=0, vmax=max_val, cmap="viridis"
            )
            axe.set_title(f"$d_{key_i}(i, j)$")
            axe.set_aspect("equal")
        # On the lower triangle plot the pairwise relative difference
        else:
            # The relative differences are all negative, the colormap will go from
            # min_val to 0, a fitting colormap is RdYlGn
            diff_im = axe.imshow(
                rel_diff[key_i, key_j],
                # vmin=1,
                # vmax=max_val_rel,
                # logaritmic scale
                norm=LogNorm(vmin=1, vmax=max_val_rel),
                cmap="RdYlGn_r",
            )
            axe.set_title(f"$\\frac{{d_{{{key_i}}}(i, j)}}{{d_{{{key_j}}}(i, j)}}$")
            axe.set_xlabel("Node $j$")
            axe.set_ylabel("Node $i$")
            axe.set_aspect("equal")
        # Only write labels on the left and bottom axes
        if i // len(metric.distance_matrix) != len(metric.distance_matrix) - 1:
            axe.set_xticklabels([])
        if i % len(metric.distance_matrix) != 0:
            axe.set_yticklabels([])
    # Set the labels for all x and y axes
    for axe in axes[-1, :]:
        axe.set_xlabel("Node $j$")
    for axe in axes[:, 0]:
        axe.set_ylabel("Node $i$")

    # Plot the two colorbars on the right side of the figure
    # Colorbar for the diagonal
    fig.colorbar(dist_im, ax=axes, fraction=0.046, pad=0.04)
    # Colorbar for the lower triangle
    fig.colorbar(diff_im, ax=axes, fraction=0.046, pad=0.04)
    # Label colorbar
    dist_im.set_label(f"Distance [{metric.unit_symbol()}]")
    # Title above all subplots
    fig.suptitle(
        f"Pairwise relative difference between the distance matrices "
        f"({metric.unit_symbol()}) {'(' + name + ')' if name else ''}"
    )

    return fig, axes


def plot_component_wise_travel_increase(
    partitioner, distance_matrix, node_list, measure1, measure2, unit, **pg_kwargs
):
    """Calculate and plot the component-wise travel increase.

    For every component of the partitioner, calculate the relative travel increase.
    This is the mean of the relative travel increase for all pairs of nodes in the
    component. Relative travel increase is the percentual increase of the travel between
    measure 1 and measure 2.

    Mean of d_1(i, j) / d_1(i, j) if i in the concerning component.

    Parameters
    ----------
    partitioner : Partitioner
        The partitioner to calculate the component-wise travel increase for.
    distance_matrix : dict
        The distance matrix to calculate the component-wise travel increase for.
    node_list : list
        The list of nodes the distance matrices are sorted by.
    measure1 : str
        The name of the first measure. Key for the distance matrix.
    measure2 : str
        The name of the second measure. Key for the distance matrix.
    unit : str
        The unit of the distance matrices.
    pg_kwargs : dict
        Keyword arguments for the plot_graph function.

    Returns
    -------
    fig, axes : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes of the plot.
    """  # pylint: disable=too-many-locals

    partition_nodes = partitioner.get_partition_nodes()

    rel_incr = rel_increase(distance_matrix[measure1], distance_matrix[measure2])

    for component in partition_nodes:
        # Calculate mean of the relative travel increase: Take all increases from inside
        # the component to all other nodes and reverse. All rows and columns with
        # Indices in the component
        nodes_idx = [
            node_list.index(node)
            for node in component["subgraph"].nodes
            if node not in partitioner.sparsified.nodes
        ]
        # column from nodes to all other nodes
        col = rel_incr[:, nodes_idx]
        # row from all other nodes to nodes
        row = rel_incr[nodes_idx, :]
        # Take the mean of the union where the values are not np.inf
        rel_increase_component = np.mean(
            np.union1d(
                col[~np.isinf(col)],
                row[~np.isinf(row)],
            )
        )
        component["rel_increase"] = rel_increase_component
        # Write this to the edges of the subgraph
        for edge in component["subgraph"].edges:
            partitioner.graph.edges[edge]["rel_increase_comp"] = rel_increase_component

    _, axes = plt.subplots(1, 1, figsize=(10, 10))
    # Add colorbar
    sc_mapbl = plt.cm.ScalarMappable(
        cmap="RdYlGn_r",
        norm=plt.Normalize(
            vmin=1,
            vmax=np.max([component["rel_increase"] for component in partition_nodes]),
        ),
    )
    sc_mapbl._A = []  # pylint: disable=protected-access
    # add colorbar to the figure
    cbar = plt.colorbar(
        sc_mapbl, ax=axes, fraction=0.046, pad=0.04, orientation="vertical"
    )
    # Label the colorbar, vertical alignment, latex math mode
    cbar.ax.set_ylabel(
        rf"Travel distance increase in superblock ({unit}) "
        rf"$d_{{{measure1}}} (i, j)$ / $d_{{{measure2}}} (i, j)$",
        rotation=270,
        labelpad=20,
        fontsize=17,
    )

    # Plot by attribute
    return plot_by_attribute(
        partitioner.graph,
        edge_attr="rel_increase_comp",
        edge_attr_types="numerical",
        edge_cmap="RdYlGn_r",
        edge_minmax_val=(
            1,
            np.max([component["rel_increase"] for component in partition_nodes]),
        ),
        ax=axes,
        **pg_kwargs,
    )


def plot_relative_difference(metric, key_i, key_j, title=None):
    """Plot the relative difference between two distance matrices.

    Parameters
    ----------
    metric : Metric
        The metric to plot the relative difference for.
    key_i : str
        The key of the first distance matrix. This is the baseline.
    key_j : str
        The key of the second distance matrix. This is the comparison.
    title : str, optional
        The title of the plot.

    Returns
    -------
    fig, axe : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axe of the plot.
    """
    # Calculate the relative difference
    rel_diff = rel_increase(
        metric.distance_matrix[key_i], metric.distance_matrix[key_j]
    )

    # Plot the relative difference
    fig, axe = plt.subplots(1, 1, figsize=(8, 8))
    # The relative differences are all negative, the colormap will go from min_val to 0,
    # a fitting colormap is RdYlGn
    diff_im = plt.imshow(
        rel_diff,
        # vmin=1,
        # vmax=np.max(rel_diff),
        cmap="RdYlGn_r",
        # log-scale
        norm=LogNorm(
            vmin=1,
            # max_val is the maximum value of the relative difference, but not np.inf
            vmax=np.max(rel_diff[~np.isinf(rel_diff)]),
        ),
    )
    axe.set_title(
        f"{title} ({metric.unit_symbol()}) | "
        f"$\\frac{{d_{{{key_i}}}(i, j)}}{{d_{{{key_j}}}(i, j)}}$"
    )
    axe.set_xlabel("Node $j$")
    axe.set_ylabel("Node $i$")
    axe.set_aspect("equal")

    # Plot the two colorbar
    fig.colorbar(diff_im, ax=axe, fraction=0.046, pad=0.04)

    return fig, axe


def plot_relative_increase_on_graph(graph, unit_symbol, **pg_kwargs):
    """Plot the relative increase edge wise from attribute `rel_increase`.

    Use `:func:metric.measures.write_relative_increase_to_edges` to write the relative
    increase to the edges.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to plot the relative increase on, must have the attribute
        `rel_increase` on the edges.
    unit_symbol : str
        The unit symbol of the metric.
    pg_kwargs : dict
        Keyword arguments for the plot_graph function.

    Returns
    -------
    fig, axes : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes of the plot.
    """

    _, axes = plt.subplots(1, 1, figsize=(10, 10))
    # Add colorbar
    sc_mapbl = plt.cm.ScalarMappable(
        cmap="RdYlGn_r",
        norm=plt.Normalize(
            vmin=1, vmax=np.max([edge["rel_increase"] for edge in graph.edges.values()])
        ),
    )
    sc_mapbl._A = []  # pylint: disable=protected-access
    # add colorbar to the figure
    cbar = plt.colorbar(
        sc_mapbl, ax=axes, fraction=0.046, pad=0.04, orientation="vertical"
    )
    # Label the colorbar, vertical alignment, latex math mode
    cbar.ax.set_ylabel(
        f"Travel distance increase ({unit_symbol}) "
        "(all to all demand) $d_{N}(i, j)$ / $d_{S}(i, j)$",
        rotation=270,
        labelpad=20,
        fontsize=15,
    )

    return plot_by_attribute(
        graph,
        edge_attr="rel_increase",
        edge_attr_types="numerical",
        edge_cmap="RdYlGn_r",
        edge_minmax_val=(
            1,
            np.max([edge["rel_increase"] for edge in graph.edges.values()]),
        ),
        ax=axes,
        **pg_kwargs,
    )
