"""Working-mode power of individual subcomponents (eqs 2.20-2.39).

Each function returns the consumption of a single subcomponent (or, for the
digital/analog ones, the contribution that is later summed per RF chain) when
in active/working mode. Frame averaging over the operating modes is handled
separately in :mod:`fr3_power.frame_average`.

All functions take the shared :class:`~fr3_power.config.PowerParams` as ``p``.
"""

from __future__ import annotations

import numpy as np

from .config import PowerParams


# --- Power amplifier ------------------------------------------------------
def p_pa(p: PowerParams, p_out: float) -> float:
    """Active-mode PA consumption for output power ``p_out`` (eq 2.21)."""
    static = p.xi * p.P_max / p.eta_PAmax
    dynamic = (1 - p.xi) * p.P_max ** (1 - p.alpha) * p_out ** p.alpha / p.eta_PAmax
    return static + dynamic


# --- Analog subcomponents (eqs 2.29-2.31) ---------------------------------
def p_dac(p: PowerParams) -> float:
    """Digital-to-analog converter (eq 2.29)."""
    return p.Xi_DAC_1 * 2 ** p.b_DAC + p.Xi_DAC_2 * p.b_DAC * p.f_DAC


def p_adc(p: PowerParams) -> float:
    """Analog-to-digital converter (eq 2.30)."""
    return p.W_ADC * p.f_ADC * 2 ** p.b_ADC


def p_mixer(p: PowerParams) -> float:
    """Mixer (eq 2.31)."""
    return p.Xi_mix * p.f_c


def p_phase_shifter(p: PowerParams) -> float:
    """Phase shifter (eq 2.31)."""
    return p.Xi_PS * p.B


def p_lna(p: PowerParams) -> float:
    """Low-noise amplifier (eq 2.31)."""
    return p.Xi_LNA * p.B


# --- Digital subcomponents (eqs 2.36-2.38) --------------------------------
# Static term + dynamic term = (complex GOPS) / (computational efficiency).
def p_encoder(p: PowerParams, R_DL: float) -> float:
    """Channel encoder (eq 2.36); scales with the DL rate."""
    dynamic = p.f_sI * 14 / (3 * 8) * R_DL / p.B_tilde
    return p.P_encoder_s + dynamic / p.eta_encoder


def p_decoder(p: PowerParams, R_UL: float) -> float:
    """Channel decoder (eq 2.36); scales with the UL rate."""
    dynamic = p.f_sI * (5 * 35) / (2 * 3) * R_UL / p.B_tilde
    return p.P_decoder_s + dynamic / p.eta_decoder


def p_precoder(p: PowerParams, M_RF: int, P_static: float) -> float:
    """MIMO precoder, ZF via Cholesky (eqs 2.37, 2.39)."""
    gops = M_RF * (p.f_sI * 2 * p.K
                   + p.f_sI / p.upsilon_coh
                   * (p.K ** 3 / (3 * M_RF) + 3 * p.K ** 2 + p.K))
    return P_static + gops / p.eta_precoder


def p_combiner(p: PowerParams, M_RF: int, P_static: float) -> float:
    """MIMO combiner, ZF (eqs 2.37, 2.39)."""
    gops = M_RF * p.f_sI * 2 * p.K
    return P_static + gops / p.eta_combiner


def p_ifft(p: PowerParams, M_RF: int) -> float:
    """IFFT (DL) / FFT (UL) for OFDM (de)modulation (eq 2.38)."""
    gops = M_RF * p.f_sII * 3 / 2 * np.log2(p.Q_IFFT)
    return p.P_IFFT_s + gops / p.eta_IFFT


def p_dpd(p: PowerParams, M_RF: int) -> float:
    """Digital predistortion (eq 2.38); DL only."""
    gops = M_RF * p.f_sII * p.Xi_DPD
    return p.P_DPD_s + gops / p.eta_DPD


def p_filter_bb(p: PowerParams, M_RF: int, P_static: float) -> float:
    """Baseband filter (eq 2.38)."""
    gops = M_RF * p.f_sII * p.Xi_filterBB
    return P_static + gops / p.eta_filterBB
