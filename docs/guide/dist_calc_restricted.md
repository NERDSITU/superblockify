---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.8.2
kernelspec:
  display_name: Python 3
  language: python
  name: python
---

# Restricted Distance Calculation

````{sidebar} [Partition Requirements](./partition_requirements)
Our street network graph $G = (V, E, v)$ has a partition
$v: G \rightarrow (G_s, G_1 \ldots G_k)$ and for every node.
````

For a valid partitioning we want to calculate the **distances and predecessors** between 
all nodes while respecting the restrictions of the partitioning. The restrictions 
are that on a path it is only allowed once to leave and enter a partition.

We will guide through the idea of the algorithm in the following sections.

## Small example

```{code-cell} ipython3
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
```

Imagine we have a directed graph with $G_s, G_1, G_2$ as partitions and the following 
edges:

```{code-cell} ipython3
:tags: [hide-input]
# Create graph
G = nx.DiGraph(
    [
        (1, 2, {"weight": 1}),
        (2, 1, {"weight": 1}),
        (2, 3, {"weight": 1}),
        (3, 2, {"weight": 1}),
        (1, 4, {"weight": 1}),
        (4, 1, {"weight": 1}),
        (2, 5, {"weight": 1}),
        (5, 2, {"weight": 1}),
        (5, 4, {"weight": 4}),
        (2, 4, {"weight": 1}),
        (3, 6, {"weight": 2}),
        (6, 3, {"weight": 2}),
        (6, 1, {"weight": 1}),
        (6, 2, {"weight": 6}),
        (6, 7, {"weight": 1}),
        (7, 6, {"weight": 1}),
    ]
)
# Draw a scaled down version of the graph
nx.draw(
    G,
    with_labels=True,
    font_color="white",
    pos=nx.kamada_kawai_layout(G),
    ax=plt.figure(figsize=(8, 3)).gca(),
)
```
The sparse graph $G_s$ is the subgraph of $G$ with nodes $1, 2, 3$.
```{code-cell} ipython3
n_sparse = [1, 2, 3]
```
The partitions $G_1$ and $G_2$ are the subgraphs of $G$ with nodes $4, 5$ and $6$
respectively.
```{code-cell} ipython3
partitions = {
    "G_s": {"nodes": n_sparse, "color": "black", "subgraph": G.subgraph(n_sparse)},
    "G_1": {"nodes": [4, 5], "color": "mediumseagreen"},
    "G_2": {"nodes": [6, 7], "color": "crimson"},
}
```
To each dictionary add a subgraph view, including all edges connecting to the nodes, 
and a node list with the connected nodes. Color the nodes according to the partition.
```{code-cell} ipython3
for name, part in partitions.items():
    if "subgraph" not in part:
        # subgraph for all edges from or to nodes in partition
        part["subgraph"] = G.edge_subgraph(
            [(u, v) for u, v in G.edges if u in part["nodes"] or v in part["nodes"]]
        )
    part["nodelist"] = part["subgraph"].nodes
    for node in part["nodes"]:
        G.nodes[node]["partition"] = part["color"]

nx.draw(G, with_labels=True, node_color=[G.nodes[n]["partition"] for n in G.nodes],
        font_color="white",
        pos=nx.kamada_kawai_layout(G),
        ax=plt.figure(figsize=(8,3)).gca(),
        # {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (0, 1), 5: (1, 1), 6: (2, -1)}
        )
```

To check the subgraphs are correct, draw these separately.

```{code-cell} ipython3
# Copy subgraphs, relabel them, merge them into one graph
composite_graph = nx.DiGraph()
for name, part in partitions.items():
    subgraph = part["subgraph"].copy()
    subgraph = nx.relabel_nodes(subgraph, {n: f"{name}_{n}" for n in subgraph.nodes})
    composite_graph = nx.compose(composite_graph, subgraph)
nx.draw(
    composite_graph,
    node_color=[composite_graph.nodes[n]["partition"] for n in composite_graph.nodes],
    with_labels=True,
    font_color="white",
    labels={n: n.split("_")[-1] for n in composite_graph.nodes},
    pos=nx.planar_layout(composite_graph),
    ax=plt.figure(figsize=(8, 3)).gca(),
)
```
## Distance calculation

Calculate all-pairs shortest path lengths for all subgraphs separately. This could be
easily done with `nx.floyd_warshall_predecessor_and_distance` for this graph. But as
we'll use this approach for larger graphs, we'll use `scipy.sparse.csgraph.dijkstra`.

```{code-cell} ipython3
pp = []
for name, part in partitions.items():
    part["distances"], part["predecessors"] = dijkstra(
        nx.to_scipy_sparse_array(part["subgraph"], nodelist=part["nodelist"],
                                 weight="weight"),
        directed=True,
        return_predecessors=True,
    )
    pp += [pd.DataFrame(part['distances'], index=part['nodelist'],
                        columns=part['nodelist'])]
    pp[-1].columns.name = name
display(*pp)
```
## Merge results

Merge distances and predecessors for all subgraphs.

```{code-cell} ipython3
:tags: [hide-input]
def colored_predecessors(preds):
    return pd.DataFrame(preds, index=node_order_indices,
                        columns=node_order_indices).style. \
        applymap(lambda x: f"color: {G.nodes[node_order[x]]['partition']}"
    if x != -9999 else "")
```

```{code-cell} ipython3
node_order = list(G.nodes)
node_order_indices = list(range(len(node_order)))
dist_matrix = np.full((len(node_order), len(node_order)), np.inf, dtype=np.half)
pred_matrix = np.full((len(node_order), len(node_order)), -9999, dtype=np.int32)
# do G_s last, as paths on the sparsified graph are not allowed
# to pass any other node
for name, part in [(n, p) for n, p in partitions.items() if n != "G_s"] + [
    ("G_s", partitions["G_s"])
]:
    part["node_order_ids"] = [node_order.index(n) for n in part["nodelist"]]
    dist_matrix[np.ix_(part["node_order_ids"], part["node_order_ids"])] = part[
        "distances"
    ]
    # as the predecessors have indices of the nodelist, we need to map them to the
    # indices of the full graph. node_order_indices.index(part["predecessors"])
    # Map every value, respect -9999 stays -9999
    pred_matrix[np.ix_(part["node_order_ids"], part["node_order_ids"])] = np.vectorize(
        lambda x: node_order_indices[part["node_order_ids"][x]]
        if x != -9999
        else x
    )(part["predecessors"])
node_order_indices = [node_order.index(n) for n in node_order]
display(pd.DataFrame(dist_matrix, index=node_order, columns=node_order),
        colored_predecessors(pred_matrix))
```

````{sidebar} Predecessor Matrix
The predecessor matrix is used to reconstruct the shortest paths.
Predecessor of $i$ to $j$ is the node with index $k$ that is on the shortest path 
from $i$ to $j$. So the predecessor of $i$ to $j$ is $k$ if $d_{ij} = d_{ik} + d_{kj}$. 
````
In column 5 (node id $6$) the value in row 0, 1 and 2 is 5. This means
that the predecessor of these is 5 itself. We can see the direct connection
in the graph. Column 4, row 3 is 4, but column 3, row 4 is -9999. We can 
see this, as between nodes with the id $4$ and $5$ we only have a directed edge from
$5$ to $4$.

## Fill up distances

Fill up remaining distances. The restriction of only entering/leave the sparsified
graph once is respected.
As the sparsified graph is required to be strongly connected, for its nodes, shortest
paths exist within the sparsified graph. So only node pairs from components have
to be filled up. So we can use the distances from the sparsified graph, to find the
shortest distances between the missing node pairs, respecting the restriction.

### 1. Partition to sparsified graph
For the nodes in the partitions nodelists, we calculate the distances to the nodes
in the sparsified graph as the minimum of its sum of distance to a exit node and the
distance to the node on the sparsified graph. For partition $n$:

$$ d_{ij} = \min_{k_n \in V_s\cup V_n} \left(d_{ik} + d_{kj}\right), \quad i \in V_n
\Leftrightarrow j \in V_s$$

### 2. Partition to partition
If the start and end nodes are both in partitions, we need another summation, as we
need to find the minimum of the sum of the distances to the exit nodes of the
start partition, the distance to the entry nodes of the end partition and the
node itself. For partition $n$ and $m$:

$$ d_{ij} = \min_{k_n \in V_s\cup V_n, l_m \in V_s\cup V_m} \left(d_{ik} + d_{kl} +
d_{lj}\right), \quad i \in V_n \Leftrightarrow j \in V_m$$

### 3. All-in-one simplified graph

For simplicity, we can construct a graph, where the sparse graph is fully connected
with the produced shortest paths. The remaining nodes ($V_1, \dots, V_n$) are
directly connected to the sparse graph, weighted with the shortest paths, but not 
connected under each other. The found distances respect the restriction of only 
entering/leaving the sparse graph once. Per construction, throughway are impossible
as the partition nodes are not connected to each other.

```{code-cell} ipython3
:tags: [hide-input]
# 3. All-in-one simplified graph
# Copy full distance graph and set all distances between partitions to inf
G_restricted = dist_matrix.copy()
n_sparse_indices = [node_order.index(n) for n in n_sparse]
n_partition_indices = [i for i in node_order_indices if i not in n_sparse_indices]
G_restricted[np.ix_(n_partition_indices, n_partition_indices)] = np.inf

# Construct Compressed Sparse Row matrix
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
data = G_restricted.flatten()
row, col = np.indices(G_restricted.shape)
row, col = row.flatten(), col.flatten()

# remove the diagonal
data = data[row != col]
row, col = row[row != col], col[row != col]
# remove inf values
data, row, col = data[data != np.inf], row[data != np.inf], col[data != np.inf]

G_restricted = csr_matrix((data, (row, col)), shape=(len(node_order), len(node_order)))

print(G_restricted)
pd.DataFrame(G_restricted.toarray(), index=node_order, columns=node_order).style. \
    apply(lambda x: ["background-color: lightgrey" if v == 0 else ""
                     for v in x])
```

On this graph, we can find the shortest paths between all nodes, respecting the
restriction of only entering/leaving the sparse graph once. This way we also find the
shortest paths in the case it is shorter to go through the sparse graph, returning
to the start partition.

Using a sparse graph representation might not seem that effective here, as in this 
example this only saves a small fraction of space. But for larger graphs, as we will
use it for, this can save a lot of space. Furthermore, the Dijkstra implementation
takes this representation as input and achieves its speedup through this.

```{code-cell} ipython3
# Calculate distances on this graph
dist_simple, pred_simple = dijkstra(G_restricted, directed=True,
                                    return_predecessors=True)

display(pd.DataFrame(dist_simple, index=node_order, columns=node_order))
G_restricted_nx = nx.from_scipy_sparse_array(G_restricted, create_using=nx.DiGraph)
G_restricted_nx = nx.relabel_nodes(G_restricted_nx,
                                   {i: n for i, n in enumerate(node_order)})
nx.draw(G_restricted_nx,
        with_labels=True,
        node_color=[G.nodes[n]["partition"] for n in G.nodes],
        font_color="white",
        pos=nx.kamada_kawai_layout(G_restricted_nx),
        ax=plt.figure(figsize=(8, 3)).gca(),
        )
```
Getting from $5$ to $4$ is now shorter, traversing the sparse graph $5
\overset{1}{\to} 2 \overset{1}{\to} 1 \overset{1}{\to} 4$, than going directly, with
weight $4$. Also, it is possible to get from the red partition to the green partition
in $6 \overset{1}{\to} 1 \overset{1}{\to} 2 \overset{1}{\to} 5$.
Controversely, the distance from $7$ to $6$ increased, as this is the distance over
the sparsified graph $7 \overset{3}{\to} 3 \overset{2}{\to} 6$.

To get the final distances we need to choose the minimum of the distances from the
original graph and the distances from the simplified graph.

```{code-cell} ipython3
min_mask = dist_simple < dist_matrix
dist_final = np.where(min_mask, dist_simple, dist_matrix)
# use colormap for distances from green to red (min to max)
pd.DataFrame(dist_final, index=node_order, columns=node_order).style. \
    background_gradient(cmap='Blues')
```

## Reconstruct Predecessor Matrix

As of now, we have the predecessor matrix from the original distance calculations
`pred_matrix` and the predecessor matrix from the restricted graph `pred_sparse`.
Now we want to augment the `pred_matrix` with the `pred_sparse` values. Here
we got to be careful, as `pred_sparse` relates to the constructed graph, that has all
the shortcuts. The predecessors in `pred_sparse` need to be used to look up the actual
predecessors in `pred_matrix`. Sounds confusing, but is actually quite simple.

When $p_\text{m}$ are the isolated predecessors with distances $d_\text{m}$ and
$p_\text{s}$ the simplified ones with distances $d_\text{s}$, this comes down to

$$p_{ij} = \begin{cases}
p_\text{m}[i, j] & \text{if } d_\text{m}[i, j] \leq d_\text{s}[i, j] \\
p_\text{m}\left[p_\text{s}[i, j], j\right] & \text{otherwise}
\end{cases}.$$

```{code-cell} ipython3
display(colored_predecessors(pred_matrix), colored_predecessors(pred_simple))
```

```{code-cell} ipython3
# Use min_mask for the i, j where dist_simple < dist_matrix
pred_final = pred_matrix.copy()
pred_final[min_mask] = pred_matrix[pred_simple[min_mask], np.where(min_mask)[1]]
display(colored_predecessors(pred_final))
```

```{code-cell} ipython3
:tags: [hide-input]
import timeit

test_matrix = pred_final.copy()

def using_loops():
    for i, j in zip(*np.where(min_mask)):
        test_matrix[i, j] = pred_matrix[pred_simple[i, j], j]
        
def using_numpy():  # uses broadcasting
    test_matrix[min_mask] = pred_matrix[pred_simple[min_mask], np.where(min_mask)[1]]

t_l = timeit.Timer(using_loops).timeit(number=10000)
t_np = timeit.Timer(using_numpy).timeit(number=10000)
print(f"loops: {t_l * 1000:.2f} ms\n"
      f"numpy: {t_np * 1000:.2f} ms\n"
      f"speedup: {t_l / t_np:.2f}")
```

## Implementation

Using the above described algorithm, the method 
`calculate_partitioning_distance_matrix()` is implemented. It takes a partitioner as 
the input and returns the distance matrix and the predecessor matrix.

In the [first example](#small-example) the distances are actually the same with and
without the restrictions. To see a difference we will show the difference with another
example, using the implementation of the algorithm.

TODO: add example
