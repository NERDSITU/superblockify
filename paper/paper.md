---
title: 'superblockify: A Python Package for Automated Generation, Visualization, and Analysis of Potential Superblocks in Cities'
tags:
  - Python
  - urban planning
  - low traffic neighborhood
  - geospatial analysis
  - network analysis
  - urban mobility
  - urban data
authors:
  - given-names: Carlson M.
    surname: Büth
    orcid: 0000-0003-2298-8438
    corresponding: true # (This is how to denote the corresponding author)
    #    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - given-names: Anastassia
    surname: Vybornova
    orcid: 0000-0001-6915-2561
    affiliation: 1
  - given-names: Michael
    surname: Szell
    orcid: 0000-0003-3022-2483
    affiliation: "1, 3, 4"
affiliations:
  - name: NEtwoRks, Data, and Society (NERDS), Computer Science Department, IT University of Copenhagen, 2300 Copenhagen, Denmark
    index: 1
  - name: Institute for Cross-Disciplinary Physics and Complex Systems (IFISC), University of the Balearic Islands (UIB) and Spanish National Research Council (CSIC), 07122 Palma de Mallorca, Spain
    index: 2
  - name: ISI Foundation, 10126 Turin, Italy
    index: 3
  - name: Complexity Science Hub Vienna, 1080 Vienna, Austria
    index: 4
date: 23 April 2024
bibliography: paper.bib
---

# Summary

`superblockify` is a Python package designed to assist in planning future Superblock
implementations by partitioning an urban street network into Superblock-like neighborhoods
and providing tools for visualizing and analyzing these partition results.
A Superblock is a set of adjacent urban blocks where vehicular through traffic is
prevented or pacified, giving priority to people walking and
cycling [@nieuwenhuijsen_superblock_2024]. The potential Superblock blueprints and descriptive
statistics generated by `superblockify` can be used by urban planners as a first step in
a data-driven planning pipeline for future urban transformations, or by urban data scientists as an efficient
computational method to evaluate potential Superblock partitions. The software is licensed under
AGPLv3 and is available at \url{https://superblockify.city}.

# Statement of need

The Superblock model is an urban planning intervention with
massive public health benefits that creates more liveable and sustainable
cities [@world2022walking;@mueller2020;@laverty2021low].
Superblocks form human-centric neighborhoods with reduced vehicular traffic. They are
safer, quieter, and more environmentally
friendly [@agenciadecologiaurbanadebarcelona2021; @martin2021; @mueller2020] than
car-centric urban landscapes which fully expose citizens to car harm [@miner2024car].
The scientific study of Superblocks has expanded quickly in recent years, summarized in
a review by @nieuwenhuijsen_superblock_2024.
The planning and implementation of Superblocks is an intricate process, requiring
extensive stakeholder involvement and careful consideration of
trade-offs [@nieuwenhuijsen2019; @transportforlondon2020; @stadtwien2021].
New computational tools and data sets, such as the `osmnx` Python library [@boeing2017]
and OpenStreetMap [@openstreetmapcontributors2023], provide the opportunity to simplify
this process by allowing to easily analyze and visualize urban street networks
computationally.
Recent quantitative studies on Superblocks have seized this opportunity with
different focuses, such as potential Superblock detection via network flow
on the abstract level [@eggimann_potential_2022] or in the local context of
Vienna [@frey2020potenziale]; development of interactive micro-level planning
tools [@tuneourblock;@abstreet]; green space [@eggimann_expanding_2022], social
factors [@yan_redefining_2023], health benefit modeling [@li_modeling_2023], or an
algorithmic taxonomy of designs [@feng_algorithmic_2022].
However, to our knowledge, none of these emerging research efforts have led to an open,
general-use, extendable software package for Superblock delineation, visualization, and
analysis.
`superblockify` fills this gap.

The software offers benefits for at least two use cases.
First, for urban planning, it provides a quick way to generate Superblock blueprints for
a city, together with descriptive statistics informing the planning process.
These blueprints can serve as a vision or first draft for potential future city
development.
In a planning pipeline, `superblockify` stands at the beginning, broadly delineating the
potential areas of study first.
Then, exported Superblocks can feed into an open geographic information system like
QGIS [@qgis] or into tools like A/B Street [@abstreet] or TuneOurBlock [@tuneourblock]
that allow finetuned modifications or traffic simulations.
This quick feedback can reduce the time and resources required to manually plan
Superblocks, which in turn can accelerate sustainable urban development.
Second, `superblockify` enables researchers to conduct large-scale studies across
multiple cities or regions, providing valuable insights into the potential impacts of
Superblocks at a broader scale, e.g. travel time changes.
In both cases, `superblockify` can help to identify best practices, algorithmic
approaches, and strategies for Superblock implementation.

The software has served in a preliminary analysis of potential Superblocks in 180 cities
worldwide [@buth2023master] and will be used in subsequent studies within the EU Horizon
Project JUST STREETS (\url{https://just-streets.eu}). With increased urbanization,
impacts of climate
change, and focus on reducing
car-dependence [@ritchie2018; @satterthwaite2009; @mattioli_political_2020], the need
for
sustainable urban planning tools like `superblockify` will only
increase [@nieuwenhuijsen_superblock_2024].

# Features

`superblockify` has three main features: Data access and partitioning,
Visualization, and Analysis.

## Data access and partitioning

`superblockify` leverages OpenStreetMap data [@openstreetmapcontributors2023] and
population
data GHS-POP R2023A [@pesaresi2023].
From a user-given search query, e.g., a city name,
`superblockify` retrieves the street network data of a city,
the necessary GHS-POP tile(s),
and distributes the population data onto a tesselation of the street network.

After the street network and optional metadata are loaded in,
the package partitions the street network into Superblocks.
In its current version 1.0.0, `superblockify` comes with two partitioners:

1. The residential approach uses the given residential street tag to decompose the
   street network into Superblocks.
2. The betweenness approach uses the streets with high betweenness centrality for the
   decomposition.

The choice between these two approaches depends on the data quality and the desired outcome.
The residential approach is appropriate for using residential data, if available and accurate.
The betweenness approach is an alternative based on traffic flow approximation.
The resulting Superblocks can be exported in GeoPackage (`.gpkg`) format for further
use.

## Visualization

After the partitioning, factors relevant for analysis and planning of Superblocks can be
calculated and visualized, e.g., area, population, population density, or demand change
by betweenness centrality.
Example Superblock configurations for two cities are shown in Fig.
\ref{fig:combined_graphs}.

![Automated generation of Superblocks. Athens, GR (top row) and Baltimore, MD, USA (bottom row) Superblocks generated using the residential partitioner (left column) and the betweenness partitioner (right column). Each Superblock is plotted in a different color, the rest of the streets are black. For easier visual recognition, each Superblock is also highlighted by a representative node of the same color. Map data from OpenStreetMap. \label{fig:combined_graphs}](combined_graphs.png)

## Analysis

For analysis, the package calculates various graph metrics of the street network, including:

- Global efficiency [@latora2001]: In the context of Superblocks, this measures how the overall ease of vehicular movement across the city might change after implementation.
- Directness [@szell2022]: This indicates how Superblock implementation might affect the directness of routes, potentially increasing or decreasing detours.
- Betweenness centrality [@brandes2008]: Identifies which streets might bear increased traffic load after Superblock implementation.
- Spatial clustering and anisotropy of high betweenness centrality nodes [@kirkley2018]: Describes how clustered and non-uniformly distributed the expected traffic bottlenecks are.
- Street orientation-order [@boeing2019]: Quantifies how grid-like each Superblock is.
- Average circuity [@boeing2019a]: Measures the length increase of routes on the street network compared to straight-line distances.

These metrics are calculated for the entire street network and for each Superblock
individually, providing insights into how the Superblock implementation might affect the
overall city structure and local neighborhood characteristics.
To facilitate further analysis, all of these metrics are included in the exportable
GeoPackage file.

# Design

`superblockify`'s design is object-oriented with a focus on modularity and
extensibility.
An abstract partitioner base class is provided to facilitate implementing new custom
approaches for Superblock generation.
At the core of the package, `superblockify` extends Dijkstra’s efficient distance
calculation approach with Fibonacci heaps on reduced graphs, ensuring optimal
performance when iterating various Superblock configurations while respecting the
Superblock restriction of no through traffic.
This restriction is checked via just-in-time (JIT) compilation
through `numba` [@siu_kwan_lam_2023_8087361] to speed up the calculation of betweenness
centrality on directed, large-scale street networks.
Central code dependencies are the `osmnx` [@boeing2017] and `networkx` [@hagberg2008]
packages for data acquisition, preprocessing, and network analysis, and
the `geopandas` [@joris_van_den_bossche_2023_8009629] package for spatial analysis.

# Acknowledgements

Michael Szell acknowledges funding from the EU Horizon Project JUST STREETS (Grant
agreement
ID: 101104240). All authors gratefully acknowledge all open source libraries on
which `superblockify` builds, and the open source data that this software makes use of:
Global Human Settlement Layer, and map data copyrighted by OpenStreetMap contributors
available from \url{https://www.openstreetmap.org}.

# Authors contributions with [CRediT](https://credit.niso.org/)

- Carlson M. Büth: Conceptualization, Software, Investigation, Methodology, Writing –
  original draft, Validation
- Anastassia Vybornova: Conceptualization, Supervision, Writing – review & editing,
  Validation
- Michael Szell: Conceptualization, Project administration, Writing – review & editing,
  Validation, Funding acquisition

# References
