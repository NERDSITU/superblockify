"""Tests for superblockify module."""
from os import environ

# OSMNX: explicitly set the backend to use
environ["USE_PYGEOS"] = "1"
