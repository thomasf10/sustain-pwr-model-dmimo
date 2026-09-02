"""Uplink spectral-efficiency / rate evaluation for distributed massive MIMO.

Orchestrates one Monte Carlo experiment for the uplink system model of
``sections/dmimo_ul_sysmodel.tex``. The pipeline is fixed here; the physics lives
in :mod:`mimo_helpers`, where the channel model, receive combining, fusion, and
power control are implemented. Each realization runs:

    positions -> channel realization (H, beta) -> CSI
              -> uplink power control -> combining directions
              -> CPU fusion weights -> effective channel -> SINR
              -> spectral efficiency

The channel realization comes from the backend selected by ``cfg.channel_model``
(Sionna 38.901 UMi by default, or the analytical Rayleigh model), and it is the
same channel the downlink uses: under TDD reciprocity ``h_k[q]`` carries user
``k``'s signal to the ``LM`` distributed antennas.

Per-user SEs are averaged over realizations to give the ergodic result. The UL
sum rate in bit/s is ``sum_k SE_k * B`` and is what feeds the channel-decoder
term of the power model and, through ``\\eqref{eq:fh_ul}``, the traffic-dependent
part of the fronthaul (``\\eqref{eq:ul-rate-user}`` in the manuscript).

Three things differ structurally from :mod:`dl_rate`:

* **Power control comes first.** The MMSE and L-MMSE combiners are loaded with
  the transmit powers (``sigma^2 P^{-1}``), so ``p`` has to be known before the
  combiners are built. In the downlink the order is the other way round.
* **There is no normalization stage.** Each user owns its budget ``p_max``
  instead of sharing a network one, so the powers are final as they leave
  :func:`mimo_helpers.uplink_power_control` and no counterpart of
  ``normalize_precoder`` exists. The SINR is invariant to the scale of ``v_k``.
* **The uplink phase has to carry data.** The uplink prelog is the data fraction
  ``tau_UL (1 - tau_ULsig) xbar_UL`` of the frame, so ``tau_ULsig = 1`` (or a
  zero load) leaves the uplink carrying nothing but pilots and every uplink SE
  is zero by construction.

The uplink transmit power is spent at the users and so never enters the network
consumption ``P_net``, which counts the APs, the fronthaul, and the CPU only.
Uplink power control acts on the delivered rate and on nothing else in the energy
efficiency, which makes it a pure spectral-efficiency lever rather than an energy
one.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from tqdm import tqdm

from config_dmimo import DMIMOConfig
import mimo_helpers as mh


@dataclass
class UplinkResult:
    """Outcome of an uplink Monte Carlo run."""

    se_per_user: np.ndarray   # ergodic per-user SE [bit/s/Hz], shape (K,)
    ue_power: np.ndarray      # mean per-UE transmit power [W], shape (K,)
    se_samples: np.ndarray    # per-user SE per realization [bit/s/Hz], shape (n, K)
    cfg: DMIMOConfig

    @property
    def sum_se(self) -> float:
        """Uplink sum spectral efficiency [bit/s/Hz]."""
        return float(self.se_per_user.sum())

    @property
    def sum_rate(self) -> float:
        """Uplink ergodic sum rate [bit/s], ``R_UL = B_tilde * sum_k SE_k``.

        On the effective bandwidth ``B_tilde = 0.9 B`` of the system model, the
        same convention the decoder model of the power package assumes.
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


def simulate_uplink(cfg: DMIMOConfig,
                    rng: Optional[np.random.Generator] = None,
                    visualize: bool = False,
                    plot_cdf: bool = False,
                    report_fronthaul: bool = False,
                    progress: bool = True) -> UplinkResult:
    """Run the uplink SE evaluation for a configuration.

    The random generator is used for the AP/UE drop, exactly as in
    :func:`dl_rate.simulate_downlink`, so running both with the same ``cfg.seed``
    puts the two directions on the same sequence of network layouts and makes
    them paired on geometry.

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
            step of an outer sweep that draws its own bar.

    Returns:
        An :class:`UplinkResult` with the ergodic per-user SE, mean per-UE
        transmit power, and the per-user SE samples of every realization.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    if cfg.ul_prelog <= 0:
        warnings.warn(
            f"tau_UL={cfg.tau_UL:.2f}, tau_ULsig={cfg.tau_ULsig:.2f}, "
            f"xbar_UL={cfg.xbar_UL:.2f}: the uplink phase carries no data, so "
            "ul_prelog = 0 and every uplink SE is zero by construction.",
            stacklevel=2,
        )

    se_acc = np.zeros(cfg.K)
    power_acc = np.zeros(cfg.K)
    se_samples = np.empty((cfg.n_realizations, cfg.K))

    # Build the channel backend once (Sionna UMi model or analytical Rayleigh).
    # TDD reciprocity: the downlink channel is also the uplink one, so the
    # backend is built for the downlink direction and H is used as it comes.
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

        # --- Power control + combining (mimo_helpers) --------------------
        # Powers first: the MMSE/L-MMSE loading sigma^2 P^{-1} needs them.
        p = mh.uplink_power_control(cfg, beta)
        V = mh.combining_directions(cfg, H_hat, p)
        a = mh.fusion_weights(cfg, V, H_hat, p)
        V = mh.apply_fusion_weights(cfg, V, a)

        # --- Signal processing -------------------------------------------
        G = mh.uplink_effective_channel(V, H)
        v_norm2 = mh.combiner_norms(V)
        se_k = mh.uplink_spectral_efficiency(cfg, G, v_norm2, p)
        se_samples[i] = se_k
        se_acc += se_k
        power_acc += p

    n = cfg.n_realizations
    result = UplinkResult(se_per_user=se_acc / n,
                          ue_power=power_acc / n,
                          se_samples=se_samples,
                          cfg=cfg)
    if plot_cdf:
        mh.plot_se_cdf(result.se_samples, title="Uplink per-user SE distribution")
    return result


def main() -> None:
    cfg = DMIMOConfig()
    print(cfg.summary())
    print()
    result = simulate_uplink(cfg, visualize=True, plot_cdf=True, report_fronthaul=True)
    print(f"UL sum SE      : {result.sum_se:.2f} bit/s/Hz")
    print(f"UL sum rate    : {result.sum_rate/1e9:.3f} Gbit/s")
    print(f"per-user SE     : median {result.se_median:.2f}, "
          f"95%-likely {result.se_5pct:.2f} bit/s/Hz")
    print(f"UE power       : mean {result.ue_power.mean()*1e3:.1f} mW, "
          f"max {result.ue_power.max()*1e3:.1f} mW (budget {cfg.p_max*1e3:.0f} mW)")


if __name__ == "__main__":
    main()
