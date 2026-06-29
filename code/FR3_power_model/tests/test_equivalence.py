"""Behaviour-preserving check: the refactored model must reproduce the legacy
per-iteration formulas of Fig_2a.py and Fig_2b.py to floating-point precision.

The legacy formulas are inlined here verbatim (translated to the shared
parameter names) so the test is self-contained and does not import the old
scripts. Run with ``python tests/test_equivalence.py`` or under pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fr3_power import OperatingPoint, PowerParams, compute  # noqa: E402
from fr3_power import rates as rates_mod  # noqa: E402

# Reference rates exactly as hard-coded in the legacy scripts.
R_DL_REF = np.array([1.6, 3.64, 6.06, 8.10, 10.93, 12.10, 13.44]) * 1e9 * 0.93
R_UL_REF = np.array([1.02, 1.25, 1.5, 1.68, 1.98, 2.18, 2.35]) * 1e9 * 0.93
M_VEC = np.array([16, 32, 64, 128, 256, 512, 1024])


def legacy_point(M_ant, M_RF, P_T, R_DL, R_UL, xbar_DL=1.0, xbar_UL=1.0):
    """Verbatim legacy computation -> (dig_li, dig_ld, ana_li, ana_ld, pa_li, pa_ld)."""
    # --- parameters (copied from Fig_2a.py / Fig_2b.py) ---
    K = 8
    f_c = 10e9
    lda_c = 3e8 / f_c
    Delta_f = 120e3
    B = 400e6
    B_tilde = 0.9 * B
    Q_IFFT = 4096
    mu = 0.9
    f_sI = mu * B
    f_sII = Q_IFFT * Delta_f
    Pa = P_T / M_ant
    eta_dig_sc = eta_PA_sc = eta_ana_sc = 0.81
    tau_DL = 0.75
    tau_DLsig = 1 / 14
    zeta_DLsig = 1 / 12
    tau_UL = 1 - tau_DL
    tau_ULsig = 1 / 14
    P_max = 0.1
    xi = 0.1
    eta_PAmax = 0.15
    alpha = 0.75
    b_DAC = 8
    f_DAC = 5e9
    b_ADC = 8
    f_ADC = 5e9
    W_ADC = 70e-15
    Xi_LNA = 2.7e-11
    Xi_mix = 2.5e-13
    Xi_PS = 3.5e-11
    P_LO = 40e-3
    P_filterRF = 5e-3
    dist_max = 300
    vel = 30
    upsilon_coh = (3e8 / dist_max) * (lda_c / (2 * vel))
    Xi_DPD = 50
    n_filterBB = 20
    o_filterBB = 4
    Xi_filterBB = n_filterBB * o_filterBB
    P_encoder_s = P_decoder_s = P_IFFT_s = P_DPD_s = 0.1
    eta_encoder = 2e12
    eta_decoder = 2e12
    eta_precoder = 0.2e12
    eta_combiner = 0.2e12
    eta_IFFT = 2e12
    eta_DPD = 2e12
    eta_filterBB = 0.2e12
    delta_dig_micro = 0.5
    delta_dig_idle = 0.25
    delta_ana_micro = 0.75
    delta_ana_idle = 0.5
    delta_PA_micro = 0.5
    delta_PA_idle = 0.25

    def P_PA(p):
        return xi * P_max / eta_PAmax + (1 - xi) * P_max ** (1 - alpha) * p ** alpha / eta_PAmax

    M_PS = 0 if M_RF == M_ant else M_ant / M_RF
    P_precoder_s = 1 * np.ceil(M_RF / 32)
    P_combiner_s = 1 * np.ceil(M_RF / 32)
    P_filterBB_s = 1 * np.ceil(M_RF / 32)

    # Digital
    P_enc = P_encoder_s + 1 / eta_encoder * (f_sI * 14 / (3 * 8) * R_DL / B_tilde)
    P_dec = P_decoder_s + 1 / eta_decoder * (f_sI * (5 * 35) / (2 * 3) * R_UL / B_tilde)
    P_pre = P_precoder_s + 1 / eta_precoder * M_RF * (f_sI * 2 * K + f_sI / upsilon_coh * (K ** 3 / (3 * M_RF) + 3 * K ** 2 + K))
    P_com = P_combiner_s + 1 / eta_combiner * M_RF * f_sI * 2 * K
    P_IFFT = P_IFFT_s + 1 / eta_IFFT * M_RF * f_sII * 3 / 2 * np.log2(Q_IFFT)
    P_DPD = P_DPD_s + 1 / eta_DPD * M_RF * f_sII * Xi_DPD
    P_filBB = P_filterBB_s + 1 / eta_filterBB * M_RF * f_sII * Xi_filterBB
    P_digDL = P_enc + P_pre + P_IFFT + P_DPD + P_filBB
    Pbar_digDL = (xbar_DL * tau_DL * (1 - tau_DLsig) * P_digDL +
                  tau_DL * tau_DLsig * P_digDL +
                  tau_DL * (1 - xbar_DL) * (1 - tau_DLsig) * P_digDL * delta_dig_micro +
                  (1 - tau_DL) * P_digDL * delta_dig_idle)
    P_digUL = P_dec + P_com + P_IFFT + P_filBB
    Pbar_digUL = (xbar_UL * tau_UL * (1 - tau_ULsig) * P_digUL +
                  tau_UL * tau_ULsig * P_digUL +
                  tau_UL * (1 - xbar_UL) * (1 - tau_ULsig) * P_digUL * delta_dig_micro +
                  (1 - tau_UL) * P_digUL * delta_dig_idle)
    Pbar_dig = Pbar_digDL + Pbar_digUL
    dig_li = (Pbar_dig - xbar_DL * tau_DL * (1 - tau_DLsig) * P_digDL + xbar_DL * tau_DL * (1 - tau_DLsig) * P_digDL * delta_dig_micro -
              xbar_UL * tau_UL * (1 - tau_ULsig) * P_digUL + xbar_UL * tau_UL * (1 - tau_ULsig) * P_digUL * delta_dig_micro)
    dig_ld = Pbar_dig - dig_li
    dig_li /= eta_dig_sc
    dig_ld /= eta_dig_sc

    # Analog
    P_DAC = 1.5e-5 * 2 ** b_DAC + 1.5e-12 * b_DAC * f_DAC
    P_ADC = W_ADC * f_ADC * 2 ** b_ADC
    P_mix = Xi_mix * f_c
    P_PS = Xi_PS * B
    P_LNA = Xi_LNA * B
    P_anaDL1 = 2 * P_DAC + 2 * P_filterRF + 2 * P_mix + M_PS * P_PS
    P_anaDL = P_LO + M_RF * P_anaDL1
    Pbar_anaDL = (xbar_DL * tau_DL * (1 - tau_DLsig) * P_anaDL +
                  tau_DL * tau_DLsig * P_anaDL +
                  tau_DL * (1 - xbar_DL) * (1 - tau_DLsig) * P_anaDL * delta_ana_micro +
                  (1 - tau_DL) * P_anaDL * delta_ana_idle)
    P_anaUL1 = 2 * P_ADC + 2 * P_filterRF + 2 * P_mix + M_PS * P_PS + M_ant / M_RF * P_LNA
    P_anaUL = P_LO + M_RF * P_anaUL1
    Pbar_anaUL = (xbar_UL * tau_UL * (1 - tau_ULsig) * P_anaUL +
                  tau_UL * tau_ULsig * P_anaUL +
                  tau_UL * (1 - xbar_UL) * (1 - tau_ULsig) * P_anaUL * delta_ana_micro +
                  (1 - tau_UL) * P_anaUL * delta_ana_idle)
    Pbar_ana = Pbar_anaDL + Pbar_anaUL
    ana_li = (Pbar_ana - xbar_DL * tau_DL * (1 - tau_DLsig) * P_anaDL + xbar_DL * tau_DL * (1 - tau_DLsig) * P_anaDL * delta_ana_micro -
              xbar_UL * tau_UL * (1 - tau_ULsig) * P_anaUL + xbar_UL * tau_UL * (1 - tau_ULsig) * P_anaUL * delta_ana_micro)
    ana_ld = Pbar_ana - ana_li
    ana_li /= eta_ana_sc
    ana_ld /= eta_ana_sc

    # Power amplifier
    Pbar_PA1 = (xbar_DL * tau_DL * (1 - tau_DLsig) * P_PA(Pa) +
                tau_DL * tau_DLsig * P_PA(zeta_DLsig * P_max) +
                tau_DL * (1 - xbar_DL) * (1 - tau_DLsig) * P_PA(0) * delta_PA_micro +
                (1 - tau_DL) * P_PA(0) * delta_PA_idle)
    Pbar_PA = M_ant * Pbar_PA1
    pa_li = (Pbar_PA - M_ant * (xbar_DL * tau_DL * (1 - tau_DLsig) * P_PA(Pa) -
                                xbar_DL * tau_DL * (1 - tau_DLsig) * P_PA(0) * delta_PA_micro))
    pa_ld = Pbar_PA - pa_li
    pa_li /= eta_PA_sc
    pa_ld /= eta_PA_sc

    return dig_li, dig_ld, ana_li, ana_ld, pa_li, pa_ld


def _check(M_ant, M_RF, P_T, R_DL, R_UL, xbar_DL=1.0, xbar_UL=1.0):
    p = PowerParams()
    op = OperatingPoint(M_ant=M_ant, M_RF=M_RF, P_T=P_T, R_DL=R_DL, R_UL=R_UL,
                        xbar_DL=xbar_DL, xbar_UL=xbar_UL)
    b = compute(p, op)
    got = np.array([b.digital.load_ind, b.digital.load_dep,
                    b.analog.load_ind, b.analog.load_dep,
                    b.pa.load_ind, b.pa.load_dep])
    ref = np.array(legacy_point(M_ant, M_RF, P_T, R_DL, R_UL, xbar_DL, xbar_UL))
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-9), (
        f"mismatch at M_ant={M_ant}, M_RF={M_RF}:\n got={got}\n ref={ref}")


def test_fig2a_sweep():
    """Fig 2a: M_ant = 1024, sweep M_RF, P_T = 100 W."""
    for i, M_RF in enumerate(M_VEC):
        _check(1024, int(M_RF), 100.0, R_DL_REF[i], R_UL_REF[i])


def test_fig2b_sweep():
    """Fig 2b: fully digital (M_RF = M_ant), sweep M_ant, P_T scales with M_ant."""
    for i, M_ant in enumerate(M_VEC):
        _check(int(M_ant), int(M_ant), 100.0 * M_ant / 1024, R_DL_REF[i], R_UL_REF[i])


def test_partial_load():
    """Load-dependent split must hold for xbar < 1 too."""
    _check(1024, 64, 100.0, R_DL_REF[2], R_UL_REF[2], xbar_DL=0.3, xbar_UL=0.3)


def test_rate_loader_matches_reference():
    """Cached rates.json must reproduce the legacy hard-coded arrays."""
    table = rates_mod.load()
    assert np.allclose(table.M_RF, M_VEC)
    assert np.allclose(table.R_DL, R_DL_REF)
    assert np.allclose(table.R_UL, R_UL_REF)


if __name__ == "__main__":
    test_fig2a_sweep()
    test_fig2b_sweep()
    test_partial_load()
    test_rate_loader_matches_reference()
    print("All equivalence checks passed.")
