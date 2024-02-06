*********
Changelog
*********

Version 1.0.0 (2024-02-
**************************

* ✨ First major release ✨
* 📊 IO operations enhanced with graph reduction.
* 🗒️ Logging improvements: silenced numba compilation, reprojected debug messages.
* ⚙️ Parallelization updates: removed `num_workers` and `chunk_size`.
* 🧪 Testing updates: increased util coverage, added response 502 as `xfail`.
* 🆕 New features: Betweenness Centrality Cutoff, Reduced path filling.
* 🐛 Fixes: notebook formatting, GEOSException in tesselation, missing attribute.
* 🔄 Merged pull request: `🌐 Added Betweenness Centrality Cutoff
  <https://github.com/cbueth/Superblockify/pull/82>`_.
* 📝 Misc changes: deactivated colormap logging,
  unified nodes and edges into one variable.
* 📊 Improved analysis scripts

Version 0.2.2 (2023-06-27)
**************************

* 📊 Unified Plot image format/suffix in config
* 🔢 Key Figures: lightweight results for analysis, see
  :func:`superblockify.partitioning.utils.get_key_figures`.
* 💾 Lightweigth metric saving
* 🆔 Added ISO 3166 country codes
* 🏙️ City Crawling: Get cities from Springer Website Table. Useful functions to add
  OSM relation IDs to the cities. Moved cities to `cities.yml` file.
* 🌆 City List format specification.
* 🗒️ Adjust logging for better usefullness. Add and remove some log messages.
* 📚️ Added `mamba` to the installation instructions and changed standard environment
  name.
* ⬆️ Demand Change: Added LTN aggregate statistics for the betweennesses.

Version 0.2.1 (2023-06-22)
**************************

* ✨ Second release ✨
* ⬆️ Integrated final graph statistics and LTN statistics.
* 🏡 Moved Coverage to Codecov |codecov-badge|.
* ⬆️ Display basic graph stats at Partitioner initialization.
  Abstract base class :class:`superblockify.partitioning.base.BasePartitioner`.
* ⬆️ Geopackage export: Resolve LTN cell option. If set to True, the LTN cells are
  resolved to polygons. Normally, only the edges are exported.
  Added general graph stats with OSM boundary polygon.

.. |codecov-badge| image:: https://codecov.io/gh/cbueth/Superblockify/branch/main/graph/badge.svg?token=AS72IFT2Q4
   :target: https://codecov.io/gh/cbueth/Superblockify
   :height: 2ex

Version 0.2.0 (2023-06-20)
**************************

* 🔧 Sped up population distribution in
  :func:`superblockify.population.approximation.get_edge_population`.
* ⬆️ Add population and density to LTNs
* 🐛 Fix: Graph import projection order. Un-skewed distance attribute.

Version 0.1.3 (2023-06-19)
**************************

* 📚️ Documented approaches in reference notebooks :ref:`Population Data`,
  :ref:`Street Tessellation`, and :ref:`Street Population Density`.
* ⬆️ Added population preprocessing for for every tesselated edge. This enables an
  efficient population density aggregation for any given superblock.
  See modules in :mod:`superblockify.population`.
* ⬆️ Automated population data download and preprocessing of the GHS-POP - R2023A dataset
  <https://ghsl.jrc.ec.europa.eu/ghs_pop2023.php>.
* ⬆️ Added graph attribute `boundary`, used for calculating the total area of the city.
* ⬆️ Added general graph statistics :mod:`superblockify.metric.graph_stats`.
  Including spatial clustering and anisotropy.

Version 0.1.2 (2023-05-18)
**************************

* ⬆️ Added Partitioner based on Betweenness Centrality.
* 🐛 Fix segfault in betweenness centrality calculation caused by testcase with one node
  graph.

Version 0.1.1 (2023-05-15)
**************************

* ⬆️ Added Betweenness Centrality Calculation in measures, precompiled version works
  quick on metropolitan sized city networks.
* ⬆️ Added speed limit: Routing and low traffic speed overwriting. Unit can be passed
  when initializing a partitioner.

Version 0.1.0 (2023-04-11)
**************************

* ✨ Initial release ✨
* 🔧 Full rework of the restricted distance calculation. Runs quicker and is more
  memory efficient. Also, path finding had a bug in the previous version.


Version 0.0.0
*************

* See changes before in the repository under the tag `0.0.0
  <https://github.com/cbueth/Superblockify/tags>`_.