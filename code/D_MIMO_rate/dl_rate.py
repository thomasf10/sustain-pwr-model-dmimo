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
sum rate in bit/s is ``B_tilde * sum_k SE_k`` on the effective bandwidth
``B_tilde = 0.9 B``, and is what feeds the encoder/MIMO terms of the power model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from tqdm import tqdm

from config_dmimo import DMIMOConfig
import mimo_helpers as mh


@dataclass
class DownlinkResult:
    """Outcome of a downlink Monte Carlo run."""

    se_per_user: np.ndarray   # ergodic per-user SE [bit/s/Hz], shape (K,)
    ap_power: np.ndarray      # mean per-AP transmit power [W], shape (L,)
    se_samples: np.ndarray    # per-user SE per realization [bit/s/Hz], shape (n, K)
    cfg: DMIMOConfig

    @property
    def sum_se(self) -> float:
        """Downlink sum spectral efficiency [bit/s/Hz]."""
        return float(self.se_per_user.sum())

    @property
    def sum_rate(self) -> float:
        """Downlink ergodic sum rate [bit/s], ``R_DL = B_tilde * sum_k SE_k``.

        The system model carries the delivered rate on the *effective* bandwidth
        ``B_tilde = 0.9 B``, not on the full ``B``. The encoder and decoder
        models of the power package are driven by ``R / B_tilde``, so using
        ``B`` here would inflate both the rate and the coding power by 1/0.9.
        """
        return self.sum_se * self.cfg.B_tilde

    @property
    def se_5pct(self) -> float:
        """95%-likely per-user SE [bit/s/Hz] (5th percentile over users, drops)."""
        return float(np.percentile(self.se_samples, 5))

    @property
    def se_median(self) -> float:
        """Median per-user SE [bit/s/Hz] (over users and realizations)."""
        return float(np.median(self.se_samples))


def simulate_downlink(cfg: DMIMOConfig,
                      rng: Optional[np.random.Generator] = None,
                      visualize: bool = False,
                      plot_cdf: bool = False,
                      report_fronthaul: bool = False,
                      progress: bool = True) -> DownlinkResult:
    """Run the downlink SE evaluation for a configuration.

    Args:
        cfg: System configuration.
        rng: Optional random generator; defaults to ``default_rng(cfg.seed)``.
        visualize: If true, plot the network (APs, UEs, CPU) for the first drop
            via :func:`mimo_helpers.plot_network` before continuing.
        plot_cdf: If true, plot the per-user SE CDF over all realizations via
            :func:`mimo_helpers.plot_se_cdf` once the run finishes.
        report_fronthaul: If true, print the CPU-to-AP fronthaul length overview
            (:func:`mimo_helpers.fronthaul_summary`) for the first drop.
        progress: Show the per-drop progress bar. Set false when this run is one
            step of an outer sweep that draws its own bar, so the two do not
            write to the same terminal line.

    Returns:
        A :class:`DownlinkResult` with the ergodic per-user SE, mean per-AP
        transmit power, and the per-user SE samples of every realization.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    se_acc = np.zeros(cfg.K)
    power_acc = np.zeros(cfg.L)
    se_samples = np.empty((cfg.n_realizations, cfg.K))

    # Build the channel backend once (Sionna UMi model or analytical Rayleigh).
    channel = mh.build_channel(cfg)

    for i in tqdm(range(cfg.n_realizations), desc="Channel realizations",
                  unit="drop", disable=not progress):
        # --- Channel model (mimo_helpers) --------------------------------
        ap_pos, ue_pos = mh.draw_positions(cfg, rng)
        if i == 0:
            if report_fronthaul:
                tqdm.write(mh.fronthaul_summary(cfg, ap_pos))
            if visualize:
                mh.plot_network(cfg, ap_pos, ue_pos)
        H, beta = mh.channel_realization(cfg, channel, ap_pos, ue_pos, rng)
        H_hat = mh.estimate_channels(cfg, rng, H)  # perfect CSI for now

        # --- Precoding + power control (mimo_helpers) --------------------
        Wbar = mh.precoding_directions(cfg, H_hat)
        rho = mh.power_control(cfg, beta, Wbar)
        W = mh.normalize_precoder(cfg, Wbar, rho)

        # --- Signal processing (implemented) -----------------------------
        G = mh.effective_channel(H, W)
        # Per-subcarrier noise: normalize_precoder spreads rho_k over the Q
        # subcarriers, so the SINR must meet sigma^2 / Q, not the full-band one.
        sinr = mh.downlink_sinr(G, cfg.noise_power_sc)
        se_k = mh.spectral_efficiency(sinr, cfg.dl_prelog)
        se_samples[i] = se_k
        se_acc += se_k
        power_acc += mh.ap_powers(cfg, W)

    n = cfg.n_realizations
    result = DownlinkResult(se_per_user=se_acc / n,
                            ap_power=power_acc / n,
                            se_samples=se_samples,
                            cfg=cfg)
    if plot_cdf:
        mh.plot_se_cdf(result.se_samples)
    return result


def main() -> None:
    cfg = DMIMOConfig()
    print(cfg.summary())
    print()
    result = simulate_downlink(cfg, visualize=True, plot_cdf=True, report_fronthaul=True)
    print(f"DL sum SE      : {result.sum_se:.2f} bit/s/Hz")
    print(f"DL sum rate    : {result.sum_rate/1e9:.3f} Gbit/s")
    print(f"per-user SE     : median {result.se_median:.2f}, "
          f"95%-likely {result.se_5pct:.2f} bit/s/Hz")
    print(f"max AP power   : {result.ap_power.max():.3f} W (budget {cfg.rho_max:.3f} W)")


if __name__ == "__main__":
    main()
