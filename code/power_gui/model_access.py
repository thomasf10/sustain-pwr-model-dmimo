"""Make the two model packages importable from the GUI.

The GUI is a thin presentation layer; it imports the models as libraries rather
than duplicating any physics. ``fr3_power`` (in ``../FR3_power_model``) is a
proper package, while the D-MIMO rate code (in ``../D_MIMO_rate``) uses flat
imports, so both directories are placed on ``sys.path``. Import this module once
at the top of every page before importing the models.
"""

from __future__ import annotations

import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parent.parent  # the code/ directory

# Order matters only for shadowing; these packages share no module names.
for _sub in ("FR3_power_model", "D_MIMO_rate"):
    _path = str(_CODE / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)
