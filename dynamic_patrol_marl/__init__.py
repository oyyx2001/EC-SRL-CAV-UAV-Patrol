"""Compatibility package for the repository's historical import name.

The source directories live at the repository root, while modules import them
through ``dynamic_patrol_marl``. Extending the package path keeps those imports
working even when Git clones the repository as ``dynamic-patrol``.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1])]
