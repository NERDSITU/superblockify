"""Population init, subpackage for the GHSL Population data"""

from .approximation import (
    add_edge_population,
    get_edge_population,
    get_population_area,
)
from .ghsl import resample_load_window, get_ghsl
from .tessellation import add_edge_cells, get_edge_cells
