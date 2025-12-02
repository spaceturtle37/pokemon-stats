# src/pokemon_stats/__init__.py
"""
pokemon_stats package public surface.

We expose submodules (not copy their names into the package namespace)
so callers use: pokemon_stats.viz.hist_stats(...)
This makes reloads simple and predictable during development.
"""

# Expose the submodule(s) — add other submodules here as needed
from . import viz, pca_utils, poke_colors, mat_utils

# Public API (visible when doing `from pokemon_stats import *`)
__all__ = ["viz", "pca_utils", "poke_colors", "mat_utils"]

# Optional: package version (useful if you manage versions)
# __version__ = "0.0.0"