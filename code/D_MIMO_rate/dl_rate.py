"""Downlink spectral-efficiency / rate evaluation for distributed massive MIMO.

Orchestrates one Monte Carlo experiment for the downlink system model of
``sections/dmimo_sysmodel.tex``. The pipeline is fixed here; the physics lives in
:mod:`mimo_helpers`, where the channel model, precoding, and power control are
implemented. Each realization runs:

    positions -> channel realization (H, beta) -> CSI
              -> precoding directions -> power control -> precoders
              -> effective channel -> SINR -> spectral efficiency

The channel realization comes from the backend selected by ``cfg.channel_model``
(Sionna 38.901 UMi by default, or the analytical Rayleigh model).

Per-user SEs are averaged over realizations to give the ergodic result. The DL
sum rate in bit/s is ``sum_k SE_k * B`` and is what feeds the encoder/MIMO terms
of the power model (``\\eqref{eq:rate}`` in the manuscript).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config_dmimo import DMIMOConfig
import mimo_helpers as mh


@dataclass
class DownlinkResult:
    """Outcome of a downlink Monte Carlo run."""

    se_per_user: np.ndarray   # ergodic per-user SE [bit/s/Hz], shape (K,)
    ap_power: np.ndarray      # mean per-AP transmit power [W], shape (L,)
    cfg: DMIMOConfig

    @property
    def sum_se(self) -> float:
        """Downlink sum spectral efficiency [bit/s/Hz]."""
        return float(self.se_per_user.sum())

    @property
    def sum_rate(self) -> float:
        """Downlink ergodic sum rate [bit/s] = sum_se * B."""
        return self.sum_se * self.cfg.B


def simulate_downlink(cfg: DMIMOConfig,
                      rng: Optional[np.random.Generator] = None) -> DownlinkResult:
    """Run the downlink SE evaluation for a configuration.

    Args:
        cfg: System configuration.
        rng: Optional random generator; defaults to ``default_rng(cfg.seed)``.

    Returns:
        A :class:`DownlinkResult` with the ergodic per-user SE and mean per-AP
        transmit power.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    se_acc = np.zeros(cfg.K)
    power_acc = np.zeros(cfg.L)

    # Build the channel backend once (Sionna UMi model or analytical Rayleigh).
    channel = mh.build_channel(cfg)

    for _ in range(cfg.n_realizations):
        # --- Channel model (mimo_helpers) --------------------------------
        ap_pos, ue_pos = mh.draw_positions(cfg, rng)
        H, beta = mh.channel_realization(cfg, channel, ap_pos, ue_pos, rng)
        H_hat = mh.estimate_channels(cfg, rng, H)
        print(f"Realization:H_hat.shape={H_hat.shape}")

        # --- Precoding + power control (mimo_helpers) --------------------
        Wbar = mh.precoding_directions(cfg, H_hat)
        rho = mh.power_control(cfg, beta, Wbar)
        W = mh.normalize_precoder(cfg, Wbar, rho)

        # --- Signal processing (implemented) -----------------------------
        G = mh.effective_channel(H, W)
        sinr = mh.downlink_sinr(G, cfg.noise_power)
        se_acc += mh.spectral_efficiency(sinr, cfg.dl_prelog)
        power_acc += mh.ap_powers(cfg, W)

    n = cfg.n_realizations
    return DownlinkResult(se_per_user=se_acc / n,
                          ap_power=power_acc / n,
                          cfg=cfg)


def main() -> None:
    cfg = DMIMOConfig()
    print(cfg.summary())
    print()
    result = simulate_downlink(cfg)
    print(f"DL sum SE   : {result.sum_se:.2f} bit/s/Hz")
    print(f"DL sum rate : {result.sum_rate/1e9:.3f} Gbit/s")
    print(f"max AP power: {result.ap_power.max():.3f} W (budget {cfg.rho_max:.3f} W)")


if __name__ == "__main__":
    main()
