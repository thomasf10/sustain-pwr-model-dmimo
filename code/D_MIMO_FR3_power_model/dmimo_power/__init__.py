"""Power model of a distributed massive MIMO (cell-free) FR3 deployment.

Implements the distributed extension of
``power_model_description/sections/dmimo_pwr_model.tex``, built on top of the
two packages that already exist in this repository rather than re-deriving
anything:

    ../FR3_power_model/   co-located component models, frame averaging, and the
                          single-site power breakdown, reused per AP
    ../D_MIMO_rate/       the rate model, which supplies the delivered sum rates
                          and the per-AP transmit powers

Layout:
    config    -- DMIMOPowerParams (extends fr3_power.PowerParams), Split, PASizing
    network   -- per-AP / fronthaul / CPU blocks and the network total (eq. pnet)
    scenarios -- builds matched rate configurations, runs them, caches the rates

Neither sibling package is installable, so importing this one puts them on
``sys.path``. That is the same idiom the ``FR3_power_model`` scripts use.
"""

from __future__ import annotations

import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parent.parent.parent
for _sibling in ("FR3_power_model", "D_MIMO_rate"):
    _path = str(_CODE_ROOT / _sibling)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from .config import (  # noqa: E402
    UNSOURCED,
    DMIMOPowerParams,
    PASizing,
    Split,
    unsourced_parameters,
)
from .network import (  # noqa: E402
    NetworkBreakdown,
    ap_analog,
    ap_digital,
    ap_operating_point,
    ap_pa,
    compute_colocated,
    compute_network,
    cpu,
    energy_efficiency,
    fronthaul,
    fronthaul_rate,
    xi_ap,
    xi_cpu,
    xi_precoder_centralized,
    xi_precoder_local,
)

__all__ = [
    "DMIMOPowerParams",
    "PASizing",
    "Split",
    "UNSOURCED",
    "unsourced_parameters",
    "NetworkBreakdown",
    "ap_analog",
    "ap_digital",
    "ap_operating_point",
    "ap_pa",
    "compute_colocated",
    "compute_network",
    "cpu",
    "energy_efficiency",
    "fronthaul",
    "fronthaul_rate",
    "xi_ap",
    "xi_cpu",
    "xi_precoder_centralized",
    "xi_precoder_local",
]
