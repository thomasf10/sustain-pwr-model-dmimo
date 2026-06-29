"""Assembled component consumption: digital, analog, power amplifier (eq 2.19).

Each ``digital`` / ``analog`` / ``pa`` function returns a :class:`LoadSplit`
(load-independent and load-dependent parts, in watts), already divided by the
relevant supply-and-cooling efficiency. :func:`compute` returns all three in a
:class:`PowerBreakdown`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import components as comp
from .config import OperatingPoint, PowerParams
from .frame_average import LoadSplit, frame_average


def _static_per_fpga(p: PowerParams, M_RF: int) -> float:
    """Static power of precoder/combiner/BB-filter: one FPGA per group of chains."""
    return np.ceil(M_RF / p.rf_chains_per_fpga)


def digital(p: PowerParams, op: OperatingPoint) -> LoadSplit:
    """Frame-averaged digital-processing consumption (eqs 2.32-2.35)."""
    M_RF = op.M_RF
    P_static = _static_per_fpga(p, M_RF)

    p_enc = comp.p_encoder(p, op.R_DL)
    p_dec = comp.p_decoder(p, op.R_UL)
    p_pre = comp.p_precoder(p, M_RF, P_static)
    p_com = comp.p_combiner(p, M_RF, P_static)
    p_ifft = comp.p_ifft(p, M_RF)
    p_dpd = comp.p_dpd(p, M_RF)
    p_fil = comp.p_filter_bb(p, M_RF, P_static)

    # Active DSP-chain consumption per direction (eqs 2.34-2.35).
    P_dig_DL = p_enc + p_pre + p_ifft + p_dpd + p_fil
    P_dig_UL = p_dec + p_com + p_ifft + p_fil

    avg_DL = frame_average(p.tau_DL, p.tau_DLsig, op.xbar_DL,
                           P_dig_DL, P_dig_DL, P_dig_DL,
                           p.delta_dig_micro, p.delta_dig_idle)
    avg_UL = frame_average(p.tau_UL, p.tau_ULsig, op.xbar_UL,
                           P_dig_UL, P_dig_UL, P_dig_UL,
                           p.delta_dig_micro, p.delta_dig_idle)
    return (avg_DL + avg_UL).scaled(1 / p.eta_dig_sc)


def analog(p: PowerParams, op: OperatingPoint) -> LoadSplit:
    """Frame-averaged analog-processing consumption (eqs 2.25-2.28)."""
    M_RF = op.M_RF
    M_PS = op.M_PS

    p_dac = comp.p_dac(p)
    p_adc = comp.p_adc(p)
    p_mix = comp.p_mixer(p)
    p_ps = comp.p_phase_shifter(p)
    p_lna = comp.p_lna(p)

    # Per active RF chain, then summed over chains plus the shared LO (eqs 2.27-2.28).
    P_ana_DL_chain = 2 * p_dac + 2 * p.P_filterRF + 2 * p_mix + M_PS * p_ps
    P_ana_DL = p.P_LO + M_RF * P_ana_DL_chain

    P_ana_UL_chain = (2 * p_adc + 2 * p.P_filterRF + 2 * p_mix
                      + M_PS * p_ps + (op.M_ant / M_RF) * p_lna)
    P_ana_UL = p.P_LO + M_RF * P_ana_UL_chain

    avg_DL = frame_average(p.tau_DL, p.tau_DLsig, op.xbar_DL,
                           P_ana_DL, P_ana_DL, P_ana_DL,
                           p.delta_ana_micro, p.delta_ana_idle)
    avg_UL = frame_average(p.tau_UL, p.tau_ULsig, op.xbar_UL,
                           P_ana_UL, P_ana_UL, P_ana_UL,
                           p.delta_ana_micro, p.delta_ana_idle)
    misc = LoadSplit(p.P_ana_misc, 0.0)
    return (avg_DL + avg_UL + misc).scaled(1 / p.eta_ana_sc)


def pa(p: PowerParams, op: OperatingPoint) -> LoadSplit:
    """Frame-averaged power-amplifier consumption (eqs 2.23-2.24).

    The PAs transmit only in DL; the frame average therefore uses the DL phase.
    The micro-sleep / idle base level is the zero-output consumption P_PA(0).
    """
    P_active = comp.p_pa(p, op.Pa)
    P_signaling = comp.p_pa(p, p.zeta_DLsig * p.P_max)
    P_zero = comp.p_pa(p, 0.0)

    avg = frame_average(p.tau_DL, p.tau_DLsig, op.xbar_DL,
                        P_active, P_signaling, P_zero,
                        p.delta_PA_micro, p.delta_PA_idle)
    # Sum over the M_ant active PAs, then apply supply-and-cooling efficiency.
    return avg.scaled(op.M_ant / p.eta_PA_sc)


@dataclass
class PowerBreakdown:
    """Per-component consumption of one operating point [W]."""

    digital: LoadSplit
    analog: LoadSplit
    pa: LoadSplit

    @property
    def total(self) -> float:
        """Total BS power consumption P_cons (eq 2.19) [W]."""
        return self.digital.total + self.analog.total + self.pa.total


def compute(p: PowerParams, op: OperatingPoint) -> PowerBreakdown:
    """Full per-component power breakdown for one operating point."""
    return PowerBreakdown(digital(p, op), analog(p, op), pa(p, op))


def energy_efficiency(op: OperatingPoint, breakdown: PowerBreakdown) -> float:
    """Energy efficiency [bit/J] = delivered load-weighted rate / total power."""
    delivered = op.R_UL * op.xbar_UL + op.R_DL * op.xbar_DL
    return delivered / breakdown.total
