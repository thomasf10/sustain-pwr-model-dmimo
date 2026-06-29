"""Parametric power consumption model of upper mid-band (FR3) base stations.

Clean re-implementation of the model from

    E. Peschiera, S. Yun, Y. Lee, L. Van der Perre and F. Rottenberg,
    "A parametric power model of upper mid-band (FR3) base stations for 6G,"
    in 2026 IEEE ICASSP, Barcelona, Spain, 2026, pp. 21476-21480.

The package is organised as:
    config         -- all model/HW parameters and the per-point operating state
    components     -- subcomponent power models (eqs 2.29-2.39)
    frame_average  -- operating-mode averaging + load-ind/load-dep split
    power_model    -- assembles digital / analog / PA consumption and totals
    beamforming    -- DFT codebooks and subarray beam selection (rate model)
    rate_model     -- Sionna channel generation + ZF hybrid beamforming -> rates
    plotting       -- the stacked-bar figures
"""

from .config import PowerParams, OperatingPoint
from .frame_average import LoadSplit, frame_average
from .power_model import (
    PowerBreakdown,
    digital,
    analog,
    pa,
    compute,
    energy_efficiency,
)

__all__ = [
    "PowerParams",
    "OperatingPoint",
    "LoadSplit",
    "frame_average",
    "PowerBreakdown",
    "digital",
    "analog",
    "pa",
    "compute",
    "energy_efficiency",
]
