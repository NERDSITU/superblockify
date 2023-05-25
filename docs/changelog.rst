*********
Changelog
*********

Version 0.1.3 (2023-05-
**************************

* Added general graph statistics :mod:`superblockify.metric.graph_stats`.
  Including spatial clustering and anisotropy.

Version 0.1.2 (2023-05-18)
**************************

* Added Partitioner based on Betweenness Centrality.
* Fix segfault in betweenness centrality calculation caused by testcase with one node
  graph.

Version 0.1.1 (2023-05-15)
**************************

* Added Betweenness Centrality Calculation in measures, precompiled version works quick
  on metropolian sized city networks.
* Added speed limit: Routing and low traffic speed overwriting. Unit can be passed
  when initializing a partitioner.

Version 0.1.0 (2023-04-11)
**************************

* Initial release
* Full rework of the restricted distance calculation. Runs quicker and is more
  memory efficient. Also, path finding had a bug in the previous version.


Version 0.0.0
*************

* See changes before in the repository under the tag `0.0.0
  <https://github.com/cbueth/Superblockify/tags>`_.