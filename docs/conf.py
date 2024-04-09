# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# autodoc_mock_imports = [
#     "modules that are not installed but referenced in the docs",
# ]

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "superblockify"
copyright = "2023-2024, Carlson Büth"
author = "Carlson Büth"
release = "1.0.0rc4"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}
master_doc = "index"

language = "en"

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
myst_heading_anchors = 3
nb_execution_timeout = 180

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_sidebars = {
    "index": [
        "navbar-logo.html",
        # "sbt-sidebar-nav.html",
        "search-field.html",
        # "globaltoc.html",
    ],
    # "api/*": [
    #     "navbar-logo.html",
    #     # "sbt-sidebar-nav.html",
    #     "sidebar-nav-bs",
    # ],
}
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/NERDSITU/superblockify/",
    "repository_branch": "main",
    "use_source_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "path_to_docs": "docs",
    # "home_page_in_toc": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com/",
        "deepnote_url": "https://deepnote.com/",
        "notebook_interface": "jupyterlab",
    },
}
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
    "osmnx": ("https://osmnx.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable", None),
    "numba": ("https://numba.readthedocs.io/en/stable", None),
    "rasterstats": ("https://pythonhosted.org/rasterstats/", None),
    "momepy": ("https://docs.momepy.org/en/stable", None),
}
