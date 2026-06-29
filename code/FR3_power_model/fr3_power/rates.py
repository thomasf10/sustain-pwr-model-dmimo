"""Loader for the cached ergodic-rate table (``data/rates.json``).

The power figures consume rates produced (slowly, via Sionna) by the rate
model. To keep the power scripts fast and dependency-free, those rates are
cached on disk and loaded here. See :mod:`fr3_power.rate_model` to regenerate.

The cached rates are the *delivered* rates of eq (2.40): the rate model already
applies the full pre-log ``tau_i * (1 - tau_sig)`` (and load and effective
bandwidth). This loader applies no further factor, so the pre-log lives in
exactly one place and cannot be double-counted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "rates.json"


@dataclass
class RateTable:
    """Ergodic sum rates as a function of the number of RF chains."""

    M_RF: np.ndarray      # number of RF chains
    R_DL: np.ndarray      # DL ergodic sum rate [bit/s]
    R_UL: np.ndarray      # UL ergodic sum rate [bit/s]

    def index_of(self, m_rf: int) -> int:
        """Position of ``m_rf`` in the table (raises if absent)."""
        matches = np.where(self.M_RF == m_rf)[0]
        if matches.size == 0:
            raise KeyError(f"No cached rate for M_RF={m_rf}; available: {list(self.M_RF)}")
        return int(matches[0])


def load(path: Path | str = DEFAULT_PATH) -> RateTable:
    """Load the delivered ergodic-rate table (eq 2.40); no extra factor applied."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return RateTable(
        M_RF=np.asarray(data["M_RF"]),
        R_DL=np.asarray(data["R_DL_Gbps"], dtype=float) * 1e9,
        R_UL=np.asarray(data["R_UL_Gbps"], dtype=float) * 1e9,
    )
