"""Plotting functions."""

from os import path

import networkx as nx
import osmnx as ox
from matplotlib import patches
from matplotlib import pyplot as plt

from .attribute import determine_minmax_val, new_edge_attribute_by_function
from .config import logger

plt.set_loglevel("info")


def paint_streets(graph, cmap="hsv", **pg_kwargs):
    """Plot a graph with (cyclic) colormap related to edge direction.

    Color will be chosen based on edge bearing, cyclic in 90 degree.
    Function is a wrapper around `osmnx.plot_graph`.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph
    cmap : string, optional
        name of a matplotlib colormap
    pg_kwargs
        keyword arguments to pass to `osmnx.plot_graph`.

    Returns
    -------
    fig, ax : tuple
        matplotlib figure, axis

    Raises
    ------
    ValueError
        If no bearing attribute is found in the graph.

    Examples
    --------
    _See example in `scripts/TestingNotebooks/20221122-painting_grids.py`._

    """

    # Check for bearing attribute.
    if "bearing" not in nx.get_edge_attributes(graph, "bearing"):
        logger.warning(
            "No edge attribute `bearing` found. "
            "Use `osmnx.add_edge_bearings` to add them on the unprojected graph."
        )

    # Write attribute where bearings are baked down modulo 90 degrees.
    new_edge_attribute_by_function(
        graph, lambda bear: bear % 90, "bearing", "bearing_90"
    )

    return plot_by_attribute(
        graph,
        edge_attr="bearing_90",
        edge_attr_types="numerical",
        edge_cmap=cmap,
        **pg_kwargs,
    )


def plot_by_attribute(
    graph,
    edge_attr=None,
    edge_attr_types="numerical",
    edge_cmap="hsv",
    edge_linewidth=1,
    edge_minmax_val=None,
    node_attr=None,
    node_attr_types="numerical",
    node_cmap="hsv",
    node_size=15,
    node_minmax_val=None,
    edge_color=None,
    node_color=None,
    **pg_kwargs,
):
    """Plot a graph based on an edge attribute and colormap.

    Color will be chosen based on the specified edge attribute passed to a colormap.
    Function is a direct wrapper around `osmnx.plot_graph`.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        Input graph
    edge_attr : string, optional
        Graph's edge attribute to select colors by
    edge_attr_types : string, optional
        Type of the edge attribute to be plotted, can be 'numerical' or 'categorical'
    edge_cmap : string or matplotlib.colors.ListedColormap, optional
        Name of a matplotlib colormap to use for the edge colors or a colormap object
    edge_linewidth : float, optional
        Width of the edges' lines
    edge_minmax_val : tuple, optional
        Tuple of (min, max) values of the edge attribute to be plotted
        (default: min and max of edge attr)
    node_attr : string, optional
        Graph's node attribute to select colors by
    node_attr_types : string, optional
        Type of the node attribute to be plotted, can be 'numerical' or 'categorical'
    node_cmap : string or matplotlib.colors.ListedColormap, optional
        Name of a matplotlib colormap to use for the node colors or a colormap object
    node_size : int, optional
        Size of the nodes
    node_minmax_val : tuple, optional
        Tuple of (min, max) values of the node attribute to be plotted
        (default: min and max of node attr)
    pg_kwargs
        Keyword arguments to pass to `osmnx.plot_graph`.
    edge_color : string, optional
        Do not pass this attribute if `edge_attr` is set, as it is set by the
        edge attribute and colormap.
    node_color : string, optional
        Do not pass this attribute if `node_attr` is set, as it is set by the
        node attribute and colormap.

    Raises
    ------
    ValueError
        If `edge_color`/`node_color` is set while `edge_attr`/`node_attr` is set.
    ValueError
        If `edge_linewidth` and `node_size` both <= 0, otherwise the plot will be empty.
    ValueError
        If `edge_attr` and `node_attr` are both None.

    Returns
    -------
    fig, axe : tuple
        matplotlib figure, axis

    Notes
    -----
    At least one of `edge_attr` or `node_attr` must be set.

    """
    # pylint: disable=too-many-arguments, too-many-locals

    # Check if edge_attr and node_attr are both None
    if edge_attr is None and node_attr is None:
        raise ValueError("At least one of `edge_attr` or `node_attr` must be set.")
    # If edge_attr/node_attr is set it cannot be in pg_kwargs, check respectively
    if edge_attr and edge_color is not None:
        raise ValueError(
            f"The `edge_color` attribute was set to {edge_color}, it will be "
            f"overwritten by the colors determined with the bearings and colormap."
        )
    if node_attr and node_color is not None:
        raise ValueError(
            f"The `node_color` attribute was set to {node_color}, it will be "
            f"overwritten by the colors determined with the bearings and colormap."
        )

    e_c, n_c = None, None

    if edge_attr:
        # Choose the color for each edge based on the edge's attribute value,
        # if `None`, set to black.
        # Make list of edge colors, order is the same as in graph.edges()
        e_c = list(
            make_edge_color_list(
                graph,
                attr=edge_attr,
                cmap=(
                    plt.get_cmap(edge_cmap) if isinstance(edge_cmap, str) else edge_cmap
                ),
                attr_types=edge_attr_types,
                minmax_val=edge_minmax_val,
                none_color=(0, 0, 0, 1),  # black
            )
        )
        # Print list of unique colors in the colormap, with a set comprehension
        logger.debug(
            "Unique colors in the edge colormap %s (len %s): %s",
            edge_cmap,
            len(e_c),
            {tuple(c) for c in e_c},
        )
    if node_attr:
        # Choose the color for each node based on the node's attribute value,
        # if `None`, set to black.
        # Make list of node colors, order is the same as in graph.nodes()
        n_c = list(
            make_node_color_list(
                graph,
                attr=node_attr,
                cmap=(
                    plt.get_cmap(node_cmap) if isinstance(node_cmap, str) else node_cmap
                ),
                attr_types=node_attr_types,
                minmax_val=node_minmax_val,
                none_color=(0, 0, 0, 0),  # transparent
            )
        )
        # Print list of unique colors in the colormap, with a set comprehension
        # logger.debug(
        #     "Unique colors in the node colormap %s (len %s): %s",
        #     node_cmap,
        #     len(n_c),
        #     {tuple(c) for c in n_c},
        # )

    # If only edge_attr is set
    if e_c and not n_c:
        # Plot graph with osmnx's function, pass further attributes
        return ox.plot_graph(
            graph,
            node_size=node_size,
            edge_color=e_c,
            node_color=node_color if node_color else (0, 0, 0, 0),
            edge_linewidth=edge_linewidth,
            bgcolor=(0, 0, 0, 0),
            show=False,
            **pg_kwargs,
        )
    # If only node_attr is set
    if n_c and not e_c:
        # Plot graph with osmnx's function, pass further attributes
        return ox.plot_graph(
            graph,
            node_size=node_size,
            edge_color=edge_color if edge_color else (0, 0, 0, 0),
            node_color=n_c,
            edge_linewidth=edge_linewidth,
            bgcolor=(0, 0, 0, 0),
            show=False,
            **pg_kwargs,
        )
    # If both edge_attr and node_attr are set
    return ox.plot_graph(
        graph,
        node_size=node_size,
        edge_color=e_c,
        node_color=n_c,
        edge_linewidth=edge_linewidth,
        bgcolor=(0, 0, 0, 0),
        show=False,
        **pg_kwargs,
    )


def make_edge_color_list(
    graph,
    attr,
    cmap,
    attr_types="numerical",
    minmax_val=None,
    none_color=(0.5, 0.5, 0.5, 1),
):
    """Make a list of edge colors based on an edge attribute and colormap.

    Color will be chosen based on the specified edge attribute passed to a colormap.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        Input graph
    attr : string
        Graph's edge attribute to select colors by
    attr_types : string, optional
        Type of the edge attribute to be plotted, can be 'numerical' or 'categorical'
    cmap : matplotlib.colors.Colormap
        Colormap to use for the edge colors
    minmax_val : tuple, optional
        If `attr_types` is 'numerical', tuple of (min, max) values of the attribute
        to be plotted (default: min and max of attr)
    none_color : tuple, optional
        Color to use for edges with `None` attribute value

    Returns
    -------
    list
        List of edge colors, order is the same as in graph.edges()

    """
    return make_color_list(
        graph,
        attr,
        cmap,
        obj_type="edge",
        attr_types=attr_types,
        minmax_val=minmax_val,
        none_color=none_color,
    )


def make_node_color_list(
    graph,
    attr,
    cmap,
    attr_types="numerical",
    minmax_val=None,
    none_color=(0.5, 0.5, 0.5, 1),
):
    """Make a list of node colors based on a node attribute and colormap.

    Color will be chosen based on the specified node attribute passed to a colormap.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        Input graph
    attr : string
        Graph's node attribute to select colors by
    attr_types : string, optional
        Type of the node attribute to be plotted, can be 'numerical' or 'categorical'
    cmap : matplotlib.colors.Colormap
        Colormap to use for the node colors
    minmax_val : tuple, optional
        If `attr_types` is 'numerical', tuple of (min, max) values of the attribute
        to be plotted (default: min and max of attr)
    none_color : tuple, optional
        Color to use for nodes with `None` attribute value

    Raises
    ------
    ValueError
        If `attr_types` is not 'numerical' or 'categorical'
    ValueError
        If `attr_types` is 'categorical' and `minmax_val` is not None

    Returns
    -------
    list
        List of node colors, order is the same as in graph.nodes()

    """
    return make_color_list(
        graph,
        attr,
        cmap,
        obj_type="node",
        attr_types=attr_types,
        minmax_val=minmax_val,
        none_color=none_color,
    )


def make_color_list(
    graph,
    attr,
    cmap,
    obj_type="edge",
    attr_types="numerical",
    minmax_val=None,
    none_color=(0.5, 0.5, 0.5, 1),
):
    """Make a list of colors based on an attribute and colormap.

    Color will be chosen based on the specified attribute passed to a colormap.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        Input graph
    attr : string
        Graph's attribute to select colors by
    cmap : matplotlib.colors.Colormap
        Colormap to use for the colors
    obj_type : string, optional
        Type of the object to take the attribute from, can be 'edge' or 'node'
    attr_types : string, optional
        Type of the attribute to be plotted, can be 'numerical' or 'categorical'
    minmax_val : tuple, optional
        If `attr_types` is 'numerical', tuple of (min, max) values of the attribute
        to be plotted (default: min and max of attr)
    none_color : tuple, optional
        Color to use for objects with `None` attribute value

    Raises
    ------
    ValueError
        If `attr_types` is not 'numerical' or 'categorical'
    ValueError
        If `attr_types` is 'categorical' and `minmax_val` is not None
    ValueError
        If `obj_type` is not "edge" or "node"

    Returns
    -------
    list
        List of colors, order is the same as in graph.nodes() or graph.edges()
    """

    if attr_types == "categorical" and minmax_val is not None:
        raise ValueError(
            f"The `minmax_val` attribute was set to {minmax_val}, "
            f"it should be None."
        )

    if obj_type not in ["edge", "node"]:
        raise ValueError(
            f"The `obj_type` attribute was set to {obj_type}, "
            f"it should be either 'edge' or 'node'."
        )

    if attr_types == "numerical":
        minmax_val = determine_minmax_val(graph, minmax_val, attr, attr_type=obj_type)
        if obj_type == "edge":
            return [
                (
                    cmap((attr_val - minmax_val[0]) / (minmax_val[1] - minmax_val[0]))
                    if attr_val is not None
                    else none_color
                )
                for u, v, k, attr_val in graph.edges(keys=True, data=attr)
            ]
        # obj_type == "node"
        return [
            (
                cmap((attr_val - minmax_val[0]) / (minmax_val[1] - minmax_val[0]))
                if attr_val is not None
                else none_color
            )
            for u, attr_val in graph.nodes(data=attr)
        ]
    if attr_types == "categorical":
        # Enumerate through the unique values of the attribute
        # and assign a color to each value, `None` will be assigned to `none_color`.
        unique_vals = (
            set(attr_val for u, v, k, attr_val in graph.edges(keys=True, data=attr))
            if obj_type == "edge"
            else set(attr_val for u, attr_val in graph.nodes(data=attr))
        )

        # To sort the values, remove None from the set, sort the values,
        # and add None back to the set.
        unique_vals.discard(None)
        unique_vals = list(unique_vals)
        try:
            unique_vals = sorted(unique_vals)
        except TypeError:
            # If the values are not sortable, just leave them as they are.
            logger.debug(
                "The values of the attribute %s are not sortable, "
                "the order of the colors in the colormap will be random.",
                attr,
            )
        finally:
            unique_vals.append(None)

        if obj_type == "edge":
            return [
                (
                    cmap(unique_vals.index(attr_val) / (len(unique_vals) - 1))
                    if attr_val is not None
                    else none_color
                )
                for u, v, k, attr_val in graph.edges(keys=True, data=attr)
            ]
        # obj_type == "node"
        return [
            (
                cmap(unique_vals.index(attr_val) / (len(unique_vals) - 1))
                if attr_val is not None
                else none_color
            )
            for u, attr_val in graph.nodes(data=attr)
        ]
    # If attr_types is not 'numerical' or 'categorical', raise an error
    raise ValueError(
        f"The `attr_types` attribute was set to {attr_types}, "
        f"it should be 'numerical' or 'categorical'."
    )


def plot_component_size(
    graph,
    attr,
    component_size,
    component_values,
    size_measure_label,
    ignore=None,
    title=None,
    cmap="hsv",
    minmax_val=None,
    num_component_log_scale=True,
    show_legend=None,
    xticks=None,
    **kwargs,
):  # pylint: disable=too-many-locals
    """Plot the distribution of component sizes for each partition value.

    x-axis: values of the partition
    y-axis: size of the component (e.g. number of edges, nodes or length)
    color: value of the partition

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        Input graph
    attr : string
        Graph's attribute to select colormap min and max values by
        if `minmax_val` is incomplete
    component_size : list
        Number of edges in each component
    component_values : list
        Value of the partition for each component
    size_measure_label : str
        Label of the size measure (e.g. "Number of edges", "Number of nodes",
        "Length (m)")
    ignore : list, optional
        List of values to ignore, plot in gray. If None, no values are ignored.
    title : str, optional
        Title of the plot
    cmap : string, optional
        Name of a matplotlib colormap
    minmax_val : tuple, optional
        Tuple of (min, max) values of the attribute to be plotted
        (default: min and max of attr)
    num_component_log_scale : bool, optional
        If True, the y-axis is plotted on a log scale
    show_legend : bool, optional
        If True, the legend is shown. If None, the legend is shown if the unique
        values of the partition are less than 23.
    xticks : list, optional
        List of xticks
    kwargs
        Keyword arguments to pass to `matplotlib.pyplot.plot`.

    Returns
    -------
    fig, axe : tuple
        matplotlib figure, axis
    """

    fig, axe = plt.subplots()

    # Choose color of each value
    minmax_val = determine_minmax_val(graph, minmax_val, attr)
    colormap = plt.get_cmap(cmap)

    # Plot
    logger.debug("Plotting component/partition sizes for %s.", title)
    # Labelling
    axe.set_xlabel(attr)
    axe.set_ylabel(size_measure_label)
    if title is not None:
        axe.set_title(f"Component size of {title}")

    # Scaling and grid
    if num_component_log_scale:
        axe.set_yscale("log")
    axe.grid(True)
    plt.xticks(xticks)

    # Make legend with unique colors
    sorted_unique_values = sorted(set(component_values))

    # Show legend if `show_legend` is True, not when it is False,
    # and if it is None, only if the number of unique values is less than 23
    if show_legend or (show_legend is None and len(sorted_unique_values) < 23):
        sorted_unique_colors = [
            colormap((v - minmax_val[0]) / (minmax_val[1] - minmax_val[0]))
            for v in sorted_unique_values
        ]
        # Place legend on the outside right without cutting off the plot
        axe.legend(
            handles=[
                patches.Patch(
                    color=sorted_unique_colors[i], label=sorted_unique_values[i]
                )
                for i in range(len(sorted_unique_values))
            ],
            fontsize="small",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
        plt.tight_layout()

    # Scatter plot
    axe.scatter(
        component_values,
        component_size,
        c=(
            [
                (
                    colormap((v - minmax_val[0]) / (minmax_val[1] - minmax_val[0]))
                    if i is False
                    else "gray"
                )
                for v, i in zip(component_values, ignore)
            ]
            if ignore is not None
            else [
                colormap((v - minmax_val[0]) / (minmax_val[1] - minmax_val[0]))
                for v in component_values
            ]
        ),
        alpha=0.5,
        zorder=2,
        **kwargs,
    )

    return fig, axe


def plot_road_type_for(graph, included_types, name, **plt_kwargs):
    """Plot the distribution of road types for the given graph.

    Find possible road types from the graph's edges, or at
    https://wiki.openstreetmap.org/wiki/Key:highway.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        Input graph
    included_types : list
        List of osm highway keys to be highlighted, all other edges have another color
    name : str
        Title of the plot
    plt_kwargs
        Keyword arguments to pass to `matplotlib.pyplot.plot`.

    Returns
    -------
    fig, axe : tuple
        matplotlib figure, axis
    """

    # Write to 'searched_road_type' attribute 1 if edge is in included_types,
    # 0 otherwise
    # graph.edges[edge]["highway"] could be list or string
    for edge in graph.edges:
        if (
            isinstance(graph.edges[edge]["highway"], list)
            and any(
                road_type in included_types
                for road_type in graph.edges[edge]["highway"]
            )
        ) or (
            isinstance(graph.edges[edge]["highway"], str)
            and graph.edges[edge]["highway"] in included_types
        ):
            graph.edges[edge]["residential"] = 1
        else:
            graph.edges[edge]["residential"] = 0

    # Plot
    logger.debug("Plotting residential/other edges for %s.", name)

    # Use plot_by_attribute to plot the distribution of residential edges
    return plot_by_attribute(
        graph,
        edge_attr="residential",
        edge_attr_types="numerical",
        edge_cmap="cool",
        edge_minmax_val=(0, 1),
        **plt_kwargs,
    )


def save_plot(results_dir, fig, filename, **sa_kwargs):
    """Save the plot `fig` to file.

    Saved in the results_dir/filename.

    Parameters
    ----------
    results_dir : str
        Directory to save to.
    fig : matplotlib.figure.Figure
        Figure to save.
    filename : str
        Filename to save to.
    sa_kwargs
        Keyword arguments to pass to `matplotlib.pyplot.savefig`.
    """

    filename = path.join(results_dir, filename)
    # Log saving
    logger.debug(
        "Saving plot (%s) to %s",
        fig.axes[0].get_title(),
        filename,
    )

    # Check if al axes are compatible with tight_layout
    # if there are more than one axes
    if len(fig.axes) > 1:
        for axe in fig.axes:
            # Also axe.get_subplotspec() might be None
            if axe.get_subplotspec() is None:
                logger.debug(
                    "Not using tight_layout because one of the axes "
                    "has no subplotspec."
                )
                break
        else:
            fig.tight_layout()
    else:
        fig.tight_layout()

    # Save
    fig.savefig(filename, **sa_kwargs)
