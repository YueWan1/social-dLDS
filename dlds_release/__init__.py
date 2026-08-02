"""Shared helpers for the social-dLDS release.

Install in editable mode from the repository root so every stage script can
import it regardless of where it is run from::

    pip install -e .

Submodules:
    paths       shared data, raw-data, results and output path resolution
    plotting    portable plotting helpers used by figure scripts
"""

from . import paths  # noqa: F401

__version__ = "0.1.0"
