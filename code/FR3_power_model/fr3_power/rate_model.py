"""Ergodic DL/UL sum-rate computation (Fig 2c).

Generates 3GPP 38.901 Urban-Micro channels with Sionna and evaluates the
zero-forcing sum rate for fully-digital (``M_RF == M_ant``) and hybrid
partially-connected beamforming, then caches the result to ``data/rates.json``.

Sionna + TensorFlow are required only here; the power figures read the cached
table via :mod:`fr3_power.rates`. This is the computationally heavy part and
benefits greatly from a GPU.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.linalg as la

from .beamforming import hermitian, subarray_beam_sel
from .rates import DEFAULT_PATH


@dataclass
class RateConfig:
    """Parameters governing the rate simulation (defaults match Section 2.3)."""

    M_ant: int = 1024
    M_RF_vec: tuple = (16, 32, 64, 128, 256, 512, 1024)
    K: int = 8
    f_c: float = 10e9
    Delta_f: float = 120e3
    B: float = 400e6
    Q_IFFT: int = 4096
    Q: int = 3000                 # data subcarriers (Q_max)
    P_T_DL: float = 100.0         # total DL transmit power [W]
    P_T_UL: float = 100e-3        # per-user UL transmit power [W]
    tau_DL: float = 0.75
    tau_DLsig: float = 1 / 14
    tau_ULsig: float = 1 / 14
    F_n_dB: float = 9.0           # noise figure [dB]
    T_n: float = 290.0            # noise temperature [K]
    N_iter: int = 1000            # channel realisations
    seed: int = 42
    gpu_num: int = 0

    @property
    def B_tilde(self) -> float:
        return 0.9 * self.B

    @property
    def tau_UL(self) -> float:
        return 1 - self.tau_DL

    @property
    def sigma2_n(self) -> float:
        k_B = 1.3806491e-23
        return k_B * self.T_n * 10 ** (self.F_n_dB / 10) * self.B

    def planar_array(self):
        """Horizontal / vertical element counts of the uniform planar array."""
        root = np.sqrt(self.M_ant)
        if root - np.round(root) == 0:
            return int(root), int(root)
        m_hor = int(2 ** np.ceil(np.log2(root)))
        return m_hor, int(self.M_ant / m_hor)


def _import_sionna(cfg: RateConfig):
    """Import Sionna/TensorFlow lazily with a helpful error if unavailable."""
    if os.getenv("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_num)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    try:
        import sionna  # noqa: F401
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "rate_model requires sionna and tensorflow. Install with "
            "`pip install sionna tensorflow` (no ray tracing needed)."
        ) from exc
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as exc:  # pragma: no cover
            print(exc)
    tf.get_logger().setLevel("ERROR")
    sionna.phy.config.seed = cfg.seed
    return sionna, tf


def _gen_channel(sionna, tf, cfg: RateConfig, m_hor, m_ver, direction):
    """One 38.901 UMi OFDM channel realisation, shape (K, M_ant, fft_size)."""
    from sionna.phy.ofdm import ResourceGrid
    from sionna.phy.channel.tr38901 import Antenna, AntennaArray, UMi
    from sionna.phy.channel import gen_single_sector_topology as gen_topology
    from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel

    ut_array = Antenna(polarization="single", polarization_type="V",
                       antenna_pattern="omni", carrier_frequency=cfg.f_c)
    bs_array = AntennaArray(num_rows=m_ver, num_cols=m_hor, polarization="single",
                            polarization_type="V", antenna_pattern="38.901",
                            carrier_frequency=cfg.f_c)
    channel_model = UMi(carrier_frequency=cfg.f_c, o2i_model="low",
                        ut_array=ut_array, bs_array=bs_array, direction=direction,
                        enable_pathloss=True, enable_shadow_fading=False,
                        always_generate_lsp=False)
    topology = gen_topology(1, cfg.K, "umi")
    channel_model.set_topology(*topology)
    rg = ResourceGrid(num_ofdm_symbols=14, fft_size=cfg.Q_IFFT,
                      subcarrier_spacing=cfg.Delta_f, num_tx=int(cfg.K),
                      num_streams_per_tx=1, cyclic_prefix_length=20,
                      pilot_pattern="kronecker", pilot_ofdm_symbol_indices=[2, 11])
    frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
    cir = channel_model(1, 1)
    h = cir_to_ofdm_channel(frequencies, *cir, normalize=False)
    return tf.squeeze(h)


def _sum_rate_dl(Hmat, M_ant, M_RF, m_hor, m_ver, P_T_DL):
    """Zero-forcing DL sum rate over the users for one subcarrier [bit/s/Hz]."""
    if M_RF == M_ant:
        W = hermitian(Hmat) @ la.inv(Hmat @ hermitian(Hmat))
        W /= la.norm(W, "fro") / np.sqrt(P_T_DL)
        eff = Hmat @ W
    else:
        W_ana = subarray_beam_sel(m_hor, m_ver, M_RF, Hmat, "downlink")
        H_eff = Hmat @ W_ana
        W_dig = hermitian(H_eff) @ la.inv(H_eff @ hermitian(H_eff))
        W_dig /= la.norm(W_ana @ W_dig, "fro") / np.sqrt(P_T_DL)
        eff = H_eff @ W_dig
    rate = 0.0
    K = eff.shape[0]
    for k in range(K):
        ds = np.abs(eff[k, k]) ** 2
        iui = np.sum(np.abs(eff[k, :]) ** 2) - ds
        rate += np.log2(1 + ds / (iui + 1))
    return rate


def _sum_rate_ul(Hmat, M_ant, M_RF, m_hor, m_ver, P_T_UL):
    """Zero-forcing UL sum rate over the users for one subcarrier [bit/s/Hz]."""
    if M_RF == M_ant:
        V = la.inv(hermitian(Hmat) @ Hmat) @ hermitian(Hmat)
        eff = V @ Hmat
        noise = [la.norm(V[k, :]) ** 2 for k in range(eff.shape[0])]
    else:
        V_ana = subarray_beam_sel(m_hor, m_ver, M_RF, Hmat, "uplink")
        H_eff = V_ana @ Hmat
        V_dig = la.inv(hermitian(H_eff) @ H_eff) @ hermitian(H_eff)
        eff = V_dig @ H_eff
        noise = [la.norm(V_dig[k, :] @ V_ana) ** 2 for k in range(eff.shape[0])]
    rate = 0.0
    K = eff.shape[0]
    for k in range(K):
        ds = P_T_UL * np.abs(eff[k, k]) ** 2
        iui = P_T_UL * (np.sum(np.abs(eff[k, :]) ** 2) - np.abs(eff[k, k]) ** 2)
        rate += np.log2(1 + ds / (iui + noise[k]))
    return rate


def compute_rates(cfg: RateConfig | None = None, verbose: bool = True):
    """Run the Monte-Carlo simulation, returning (M_RF_vec, R_DL, R_UL) [bit/s]."""
    cfg = cfg or RateConfig()
    sionna, tf = _import_sionna(cfg)
    m_hor, m_ver = cfg.planar_array()
    M_RF_vec = np.array(cfg.M_RF_vec)
    R_DL = np.zeros(len(M_RF_vec))
    R_UL = np.zeros(len(M_RF_vec))

    for idx, M_RF in enumerate(M_RF_vec):
        for it in range(cfg.N_iter):
            if verbose:
                print(f"M_RF = {M_RF}, iter = {it} of {cfg.N_iter}")
            sionna.phy.config.seed = int(np.round(np.random.uniform(0, 5000)))
            h_dl = _gen_channel(sionna, tf, cfg, m_hor, m_ver, "downlink")
            h_ul = _gen_channel(sionna, tf, cfg, m_hor, m_ver, "uplink")
            H_DL = np.asarray(h_dl)[:, :, :cfg.Q]
            H_UL = np.asarray(h_ul)[:, :, :cfg.Q]
            for q in range(cfg.Q):
                Hmat_dl = H_DL[:, :, q] / np.sqrt(cfg.sigma2_n)
                R_DL[idx] += _sum_rate_dl(Hmat_dl, cfg.M_ant, int(M_RF),
                                          m_hor, m_ver, cfg.P_T_DL)
                Hmat_ul = H_UL[:, :, q] / np.sqrt(cfg.sigma2_n)
                R_UL[idx] += _sum_rate_ul(Hmat_ul, cfg.M_ant, int(M_RF),
                                          m_hor, m_ver, cfg.P_T_UL)

    # Single application of the eq (2.40) pre-log (full load): tau_i * (1 - tau_sig),
    # times effective bandwidth per data subcarrier. Owned here only; never reapplied
    # downstream, so the pre-log cannot be double-counted.
    pre_dl = cfg.tau_DL * (1 - cfg.tau_DLsig) * (cfg.B_tilde / cfg.Q)
    pre_ul = cfg.tau_UL * (1 - cfg.tau_ULsig) * (cfg.B_tilde / cfg.Q)
    R_DL = pre_dl * R_DL / cfg.N_iter
    R_UL = pre_ul * R_UL / cfg.N_iter
    return M_RF_vec, R_DL, R_UL


def compute_and_save(cfg: RateConfig | None = None,
                     path: Path | str = DEFAULT_PATH) -> None:
    """Compute the rates and write them to the cache used by the power figures."""
    cfg = cfg or RateConfig()
    M_RF_vec, R_DL, R_UL = compute_rates(cfg)
    data = {
        "description": "Generated by rate_model.compute_and_save. Delivered ergodic "
                       "sum rates [Gbit/s] including the full eq (2.40) pre-log; "
                       "consumed as-is by the power figures (no further factor).",
        "M_RF": [int(m) for m in M_RF_vec],
        "R_DL_Gbps": list(R_DL / 1e9),
        "R_UL_Gbps": list(R_UL / 1e9),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"Wrote rates to {path}")
