:notoc:

.. image:: ../assets/superblockify_logo.png
  :width: 1121
  :alt: superblockify logo

***************************
superblockify documentation
***************************

On these pages you can find documentation for superblockify.

What is `superblockify`?
========================

`superblockify` is a Python package for partitioning an urban street network into
Superblock-like neighborhoods and for visualizing and analyzing the partition results. A
Superblock is a set of adjacent urban blocks where vehicular through traffic is
prevented or pacified, giving priority to people walking and cycling.

.. image:: ../assets/superblockify_concept.png
  :width: 1500
  :alt: superblockify partitions an urban street network into Superblock-like neighborhoods

Setup and use
=============

To set up superblockify, see the `Installation <installation>`__ page.
To use superblockify, the `Usage <usage>`__ page
is a good place to start.
More on the details of the inner workings can be found on
the `Reference pages <guide>`__.
Furthermore, you can also find the `API documentation <api/index.html>`__.

Statement of Need
=================

`superblockify` is designed to address the need for an open, general-use, and extendable software package for Superblock delineation, visualization, and analysis. The Superblock model is an urban planning intervention that creates more liveable and sustainable cities by forming human-centric neighborhoods with reduced vehicular traffic. However, the planning and implementation of Superblocks is a complex process that requires extensive stakeholder involvement and careful consideration of trade-offs.

With the advent of new computational tools and datasets, there is an opportunity to simplify this process by allowing for easy computational analysis and visualization of urban street networks. `superblockify` seizes this opportunity, filling a gap in the current landscape of research efforts.

The target audience for `superblockify` includes urban planners, researchers in urban studies, data scientists interested in urban data, and policymakers involved in urban development. By providing a tool for Superblock analysis, `superblockify` aims to support these professionals in their work towards creating safer, quieter, and more environmentally friendly urban environments.

Contributing
============
If you want to contribute to the development of superblockify, please read the
`CONTRIBUTING.md <https://github.com/NERDSITU/superblockify/blob/main/CONTRIBUTING.md>`__
file.

.. toctree::
   :caption: Overview
   :maxdepth: 1
   :glob:

   installation
   usage
   guide/index
   api/index
   changelog

