Transport Network Graph Representation
--------------------------------------

There are several ways to represent transportation networks, mappings
from the real world to another representation, e.g., a visual
representation as a hiking map or a graph representation as a network.
From application to application, useful representations can differ. We
will use a directed multigraph :math:`G = (V, E, l)` with edges
:math:`e \in E` and vertices :math:`v \in V`. The edges :math:`e` are
weighted with length :math:`l`, but can have more attributes, like a
type or a name. Also, the vertices :math:`v` can have attributes, e.g.,
geographical latitude and longitude. Edges represent streets, and
vertices represent intersections, junctions, or dead ends. Streets are
specifically not the semantic entity of a road, but a part of a road
between exactly two intersections. Another way of dealing with a road
network is grouping edges to ways, inspired by the semantics of named
roads [elgouj2022]_, or a dual construction is defining
road sections drivable without turns as nodes and streets connected by a
turn as edges [lagesse2015]_. For the street graph,
:math:`G` we require a few more properties:

-  **Directed**: The edges have a direction, e.g., from intersection
   :math:`a` to intersection :math:`b`. In the case of two-way streets
   are represented by two edges, one from :math:`a` to :math:`b` and one
   from :math:`b` to :math:`a`.

-  **Strongly connected**: There is a path from every vertex to every other
   vertex. In a street graph, this means that every intersection is
   reachable from every other intersection.

-  **Loops**: An edge can start and end at the same vertex.

As the transportation network can have bridges and tunnels, the graph is
not necessarily planar. The Python package `osmnx <https://osmnx.readthedocs.io/en/stable/>`_
[boeing2017a]_ implements such functionality to
standardizedly retrieve OSM data and simplify the network into a graph
representation of the transportation network that the above requirements
after some filtering. It is based on the `NetworkX <https://networkx.org/>`_
[SciPyProceedings11]_ package, which implements graph
algorithms and data structures.

Partition Requirements
^^^^^^^^^^^^^^^^^^^^^^

The street graph :math:`G` will be split into partitions, one for each
LTN and one for the sparse network. This can be described by a
partitioning
:math:`\mathcal{P} : G \mapsto \left(G_\mathrm{sp} \cup G_1 \cup \dots \cup G_k\right)`
returning subgraphs :math:`G_i\subseteq G`, one sparse
:math:`G_\mathrm{sp}`, and :math:`k` :math:`G_i`. Such a partitioning
function :math:`\mathcal{P}` must satisfy a union property

.. math::

   \bigcup_{i=1}^k G_i \cup G_\mathrm{sp} = G

and an edge-wise disjoint property

.. math::

   \forall i, j \in \{\mathrm{sp}, 1, \dots, k\} : i \neq j \Rightarrow E_i \cap E_j = \emptyset,

where :math:`E_i` is the set of edges of :math:`G_i`. The union property states that the
partitioning function :math:`\mathcal{P}` must return a partitioning of
the whole graph :math:`G`, in other words, no street should be left out.
The disjoint property
states that the partitions must be edge-wise disjoint, i.e., no street
should be part of more than one partition. This also means that from the
set of edges :math:`E_i` we can exactly reconstruct the set of vertices
:math:`V_i`. Our goal is to compare performance of automatized
:math:`\mathcal{P}`, before any restrictions are applied, to restricting
paths to only use edges of the start and, end and sparse network. Such
for all paths :math:`p = (e_s, \dots, e_t)`, where
:math:`e_s \in E_\mathrm{s}` and :math:`e_t \in E_\mathrm{t}`, the path
is a subset
:math:`p \subseteq E_\mathrm{s} \cup E_\mathrm{sp} \cup E_\mathrm{t}`,
including paths starting or ending in the sparse network. To satisfy
connectivity for :math:`\mathcal{P}`, a sufficient condition is that the
sparse network is strongly connected and that the LTNs are connected to
the sparse network. From anywhere in a neighborhood, it must be possible
to reach anywhere else in a city, without passing a foreign LTN.
However, it is possible that with a start and end inside the same LTN
one must, by car, use the sparse network.


.. [elgouj2022] El Gouj, H., Rincón-Acosta, C. & Lagesse, C. Urban morphogenesis
   analysis based on geohistorical road data. Appl Netw Sci 7, 1–26 (2022).
   `DOI 10.1007/s41109-021-00440-0 <https://doi.org/10.1007/s41109-021-00440-0>`_
.. [lagesse2015] Lagesse, C.
   `Lire les Lignes de la Ville. <https://shs.hal.science/tel-01245898>`_
   (Universite Paris Diderot-Paris VII, 2015).
.. [boeing2017a] Boeing, G. OSMnx: New methods for acquiring, constructing,
   analyzing, and visualizing complex street networks. Computers, Environment and Urban
   Systems 65, 126–139 (2017).
   `DOI 10.1007/978-3-030-12381-9_12 <https://doi.org/10.1007/978-3-030-12381-9_12>`_
.. [SciPyProceedings11] Hagberg, A. A., Schult, D. A. & Swart, P. J. Exploring
   network structure, dynamics, and function using NetworkX. in Proceedings of the 7th
   python in science conference (eds. Varoquaux, G., Vaught, T. & Millman, J.) 11–15
   (2008).
