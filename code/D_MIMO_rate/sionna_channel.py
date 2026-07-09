"""Sionna-backed 3GPP TR 38.901 Urban Microcell channel for the D-MIMO rate model.

Wraps Sionna's system-level UMi geometry-based stochastic model
(:class:`sionna.phy.channel.tr38901.UMi`) so the rate pipeline can draw realistic
cell-free channel realizations instead of the analytical correlated-Rayleigh
model. One :class:`SionnaUMiChannel` is built per configuration (the antenna
arrays and scenario are created once); each call to :meth:`generate` sets the
AP/UE topology for a fresh drop and returns the collective downlink channel.

The APs are the 38.901 "base stations" (each an ``M``-element half-wavelength
ULA of omnidirectional elements) and the single-antenna UEs are the "user
terminals". All links are forced to NLOS (``los=False``) so the channel stays
zero-mean, consistent with the correlated-Rayleigh assumptions the analytical
system model and its precoding/SINR derivations rest on; the path loss, shadow
fading, and spatial correlation still follow 38.901 UMi.

Unlike the analytical model, Sionna does not expose an explicit correlation
matrix ``R_kl``. The large-scale fading ``beta[l, k]`` returned here is therefore
estimated from the realization as the average ``|h_kl|^2`` over the ``M``
antennas and ``Q`` subcarriers, which matches the definition
``beta_kl = (1/N) Tr(R_kl)`` (the average per-antenna channel gain). Distributed
precoders that need ``R_kl`` explicitly (L-MMSE, LP-MMSE) would have to estimate
it by Monte Carlo averaging of ``h h^H``.

The backend is heavy (PyTorch plus the Dr.Jit/Mitsuba ``sionna-rt`` stack), so
this module is imported lazily by :func:`mimo_helpers.build_channel` only when
``cfg.channel_model`` selects Sionna.
"""

from __future__ import annotations

import numpy as np

from config_dmimo import DMIMOConfig


class SionnaUMiChannel:
    """3GPP TR 38.901 UMi channel generator for one :class:`DMIMOConfig`.

    Args:
        cfg: System configuration (carrier, bandwidth, array size, heights, ...).
        direction: ``"downlink"`` (APs transmit) or ``"uplink"`` (UEs transmit).
    """

    def __init__(self, cfg: DMIMOConfig, direction: str = "downlink") -> None:
        import torch
        from sionna.phy import config as sn_config
        from sionna.phy.channel import subcarrier_frequencies
        from sionna.phy.channel.tr38901 import UMi, PanelArray

        # Seed Sionna's RNG so the shadowing/ray realizations are reproducible.
        sn_config.seed = cfg.seed

        self._torch = torch
        self.cfg = cfg
        self.direction = direction

        # UEs: single omnidirectional antenna. APs: M-element half-wavelength ULA
        # (the horizontal axis is the panel columns; spacing defaults to 0.5 lambda).
        ut_array = PanelArray(
            num_rows_per_panel=1, num_cols_per_panel=1,
            polarization="single", polarization_type="V",
            antenna_pattern="omni", carrier_frequency=cfg.f_c, precision="double",
        )
        bs_array = PanelArray(
            num_rows_per_panel=1, num_cols_per_panel=cfg.M,
            polarization="single", polarization_type="V",
            antenna_pattern=cfg.antenna_pattern, carrier_frequency=cfg.f_c,
            precision="double",
        )
        self._model = UMi(
            carrier_frequency=cfg.f_c, o2i_model=cfg.o2i_model,
            ut_array=ut_array, bs_array=bs_array, direction=direction,
            enable_pathloss=True, enable_shadow_fading=True, precision="double",
        )
        # Q flat subcarriers spaced by Delta_f, centered at the carrier.
        self._freqs = subcarrier_frequencies(cfg.Q, cfg.Delta_f, precision="double")

    def generate(self, ap_pos, ue_pos, rng: np.random.Generator):
        """Draw one channel realization for the given AP/UE drop.

        Args:
            ap_pos: AP coordinates ``(L, 2)`` or ``(L, 3)`` [m]; the height column
                is ignored and replaced by ``cfg.ap_height``.
            ue_pos: UE coordinates ``(K, 2)`` or ``(K, 3)`` [m]; likewise clamped
                to ``cfg.ue_height``.
            rng: Unused here (Sionna owns its own RNG, seeded in ``__init__``);
                kept for signature parity with the analytical channel model.

        Returns:
            Tuple ``(H, beta)`` with the collective downlink channel
            ``H`` shaped ``(Q, K, M_tot)`` (``H[q, k, :] = h_k[q]``, complex128)
            and the large-scale fading ``beta`` shaped ``(L, K)`` [linear].
        """
        del rng  # Sionna randomness is seeded once in __init__.
        torch = self._torch
        from sionna.phy.channel import cir_to_ofdm_channel

        cfg = self.cfg
        L, K, M, Q = cfg.L, cfg.K, cfg.M, cfg.Q

        ap = np.asarray(ap_pos, dtype=float)
        ue = np.asarray(ue_pos, dtype=float)
        bs_loc = np.column_stack([ap[:, 0], ap[:, 1], np.full(L, cfg.ap_height)])
        ut_loc = np.column_stack([ue[:, 0], ue[:, 1], np.full(K, cfg.ue_height)])

        # Leading dimension is Sionna's batch size; one drop -> batch of 1.
        bs_t = torch.tensor(bs_loc[None], dtype=torch.float64)
        ut_t = torch.tensor(ut_loc[None], dtype=torch.float64)
        zeros_ut = torch.zeros((1, K, 3), dtype=torch.float64)
        zeros_bs = torch.zeros((1, L, 3), dtype=torch.float64)
        in_state = torch.zeros((1, K), dtype=torch.bool)  # all UEs outdoor
        los = False if cfg.force_nlos else None

        self._model.set_topology(
            ut_loc=ut_t, bs_loc=bs_t, ut_orientations=zeros_ut,
            bs_orientations=zeros_bs, ut_velocities=zeros_ut,
            in_state=in_state, los=los,
        )

        # Static snapshot (no Doppler): one time sample, then to the frequency domain.
        a, tau = self._model(num_time_samples=1, sampling_frequency=cfg.Delta_f)
        h = cir_to_ofdm_channel(self._freqs, a, tau, normalize=False)
        # Sionna layout: (batch, num_rx=K, rx_ant=1, num_tx=L, tx_ant=M, sym=1, Q).
        h_np = h.detach().cpu().numpy()[0, :, 0, :, :, 0, :]  # (K, L, M, Q)

        # Collective channel: stack the per-AP blocks so column index is l*M + n,
        # matching the AP-major M_tot ordering used elsewhere (see ap_powers).
        H = np.moveaxis(h_np, -1, 0).reshape(Q, K, L * M).astype(np.complex128)

        # beta[l, k] = average |h_kl|^2 over the M antennas and Q subcarriers.
        beta = (np.abs(h_np) ** 2).mean(axis=(2, 3)).T  # (L, K)

        return H, beta
