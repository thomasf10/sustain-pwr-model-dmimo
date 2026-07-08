"""MIMO-specific building blocks for the distributed massive MIMO rate model.

The functions here are split into two kinds:

* **Stubs you implement** -- the channel model, the transmit precoding, and the
  power control. These raise :class:`NotImplementedError` and document the
  expected inputs, outputs, and array shapes so the surrounding pipeline stays
  fixed while you fill in the physics.
* **Implemented signal-processing** -- the effective-channel, SINR, spectral
  efficiency, and per-AP power computations of the downlink system model. These
  are generic given the channels and precoders and are ready to use.

Array-shape conventions (complex128 unless noted), with ``M_tot = L * M`` the
total number of distributed antennas:

    H     channels                (Q, K, M_tot)   H[q, k, :] = h_k[q]
    Wbar  precoding directions    (Q, M_tot, K)   Wbar[q, :, k] = wbar_k[q]
    W     normalized precoders    (Q, M_tot, K)   W[q, :, k] = w_k[q]
    G     effective channels      (Q, K, K)       G[q, k, i] = h_k[q]^H w_i[q]
    beta  large-scale fading      (L, K)          beta[l, k]
    rho   power coefficients      (K,) or (L, K)

Columns of ``H`` store the collective channel vector ``h_k[q]`` (not conjugated);
the Hermitian in ``h_k^H w_i`` is applied inside :func:`effective_channel`.
"""

from __future__ import annotations

import numpy as np

from config_dmimo import DMIMOConfig, PrecodingScheme

# ======================================================================
# Channel model  --  IMPLEMENT YOURSELF
# ======================================================================


def draw_positions(cfg: DMIMOConfig, rng: np.random.Generator):
    """Draw AP and UE locations in the coverage area.

    APs and UEs are dropped independently and uniformly over the square
    ``[0, area_size) x [0, area_size)`` [m] (the wrap-around torus of
    :meth:`DMIMOConfig`, so no location is disadvantaged by an edge). Only the
    horizontal ``(x, y)`` coordinates are random; the fixed ``ap_height`` and
    ``ue_height`` set the vertical separation and are folded into the 3-D
    distance by :func:`large_scale_fading`.

    Args:
        cfg: System configuration.
        rng: Random generator (use it for reproducibility).

    Returns:
        Tuple ``(ap_pos, ue_pos)`` of arrays shaped ``(L, 2)`` and ``(K, 2)``
        holding the horizontal AP and UE coordinates [m].
    """
    ap_pos = rng.uniform(0.0, cfg.area_size, size=(cfg.L, 2))
    ue_pos = rng.uniform(0.0, cfg.area_size, size=(cfg.K, 2))
    return ap_pos, ue_pos


def large_scale_fading(cfg: DMIMOConfig, ap_pos, ue_pos,
                       rng: np.random.Generator) -> np.ndarray:
    """Large-scale fading (average channel gain) beta[l, k] between AP l and UE k.

    Implements the 3GPP Urban Microcell model of the cell-free monograph
    (Bjornson & Sanguinetti, Sec. 2.5.2): the channel gain in dB is
    ``beta_kl [dB] = -PL(d_kl) + F_kl``, with ``PL`` the log-distance path loss
    of :meth:`DMIMOConfig.path_loss_dB`, ``d_kl`` the 3-D AP-UE distance (the
    horizontal separation combined with the ``ap_height - ue_height`` vertical
    offset), and ``F_kl ~ N(0, shadow_std_dB^2)`` the log-normal shadowing. The
    returned coefficients are ``beta = 10 ** (beta_dB / 10)`` in linear scale.

    The shadowing terms are drawn independently across ``(l, k)``. The book also
    specifies inter-UE shadow correlation for a common AP (eq. for
    ``E{F_kl F_ij}``); that spatial correlation is not modeled here.
    [MODEL: correlated shadowing omitted]

    Args:
        cfg: System configuration.
        ap_pos: AP coordinates ``(L, 2)`` [m] from :func:`draw_positions`.
        ue_pos: UE coordinates ``(K, 2)`` [m] from :func:`draw_positions`.
        rng: Random generator (used for the shadowing realizations).

    Returns:
        Array ``(L, K)`` of linear large-scale fading coefficients.
    """
    ap = np.asarray(ap_pos, dtype=float)   # (L, 2)
    ue = np.asarray(ue_pos, dtype=float)   # (K, 2)
    horizontal = np.linalg.norm(ap[:, None, :] - ue[None, :, :], axis=2)  # (L, K)
    height_diff = cfg.ap_height - cfg.ue_height
    dist_3d = np.sqrt(horizontal ** 2 + height_diff ** 2)                 # (L, K)

    path_loss_dB = cfg.path_loss_dB(dist_3d)                              # (L, K)
    shadowing_dB = rng.normal(0.0, cfg.shadow_std_dB, size=path_loss_dB.shape)
    beta_dB = -path_loss_dB + shadowing_dB
    return 10.0 ** (beta_dB / 10.0)


def spatial_correlation(cfg: DMIMOConfig, ap_pos, ue_pos, beta: np.ndarray) -> np.ndarray:
    """Per-AP spatial correlation matrices R_kl (Hermitian PSD, Tr(R_kl) = M*beta).

    Returns:
        Array ``(L, K, M, M)`` with ``R[l, k]`` the correlation matrix of the
        channel from AP l to UE k. For uncorrelated fading this is
        ``beta[l, k] * I_M``.
    """
    raise NotImplementedError("channel model: spatial correlation matrices R_kl")


def generate_channels(cfg: DMIMOConfig, rng: np.random.Generator,
                      R: np.ndarray) -> np.ndarray:
    """Draw correlated Rayleigh-fading channel realizations.

    Each collective channel is ``h_k[q] ~ CN(0, R_k[q])`` with the block-diagonal
    ``R_k = diag(R_k1, ..., R_kL)``; realizations are independent across the Q
    subcarriers.

    Returns:
        Array ``(Q, K, M_tot)`` of channel realizations ``H[q, k, :] = h_k[q]``.
    """
    raise NotImplementedError("channel model: correlated Rayleigh channel realizations")


def estimate_channels(cfg: DMIMOConfig, rng: np.random.Generator,
                      H: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Return the CSI used to build the precoders.

    For perfect CSI simply return ``H``. For a pilot-based study, return the MMSE
    estimates (including pilot contamination when ``tau_p < K``).

    Returns:
        Array ``(Q, K, M_tot)`` of channel estimates, same layout as ``H``.
    """
    raise NotImplementedError("channel model: channel estimation (or return H for perfect CSI)")


# ======================================================================
# Transmit precoding  --  IMPLEMENT YOURSELF
# ======================================================================


def precoding_directions(cfg: DMIMOConfig, H_hat: np.ndarray) -> np.ndarray:
    """Unnormalized precoding directions wbar_k[q] for the configured scheme.

    Dispatch on ``cfg.precoding`` (see :class:`PrecodingScheme`). Centralized
    schemes (ZF/RZF/MMSE/P-*) use the full collective estimates; distributed
    schemes (MR/L-MMSE/LP-MMSE) act per AP block of ``M`` rows.

    Args:
        cfg: System configuration.
        H_hat: Channel estimates ``(Q, K, M_tot)``.

    Returns:
        Directions ``Wbar`` shaped ``(Q, M_tot, K)`` (power/normalization applied
        later in :func:`normalize_precoder`).
    """
    raise NotImplementedError(f"precoding: directions for {cfg.precoding.value}")


# ======================================================================
# Power control  --  IMPLEMENT YOURSELF
# ======================================================================


def power_control(cfg: DMIMOConfig, beta: np.ndarray, Wbar: np.ndarray) -> np.ndarray:
    """Downlink power-control coefficients.

    Return per-user powers ``rho`` shaped ``(K,)`` for centralized operation, or
    per-AP-per-user powers shaped ``(L, K)`` for distributed operation (e.g. the
    fractional rule rho_kl = rho_max * beta_kl^kappa / sum_i beta_il^kappa).

    Returns:
        Array ``(K,)`` or ``(L, K)`` of non-negative powers [W].
    """
    raise NotImplementedError("power control: power coefficients rho")


def normalize_precoder(cfg: DMIMOConfig, Wbar: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Scale directions into precoders that meet the per-AP power budget.

    Applies the direction/power split ``w_k = sqrt(rho_k) * wbar_k / ||wbar_k||``
    (per-user), or the per-AP variant, and enforces
    ``sum_q sum_k ||w_kl[q]||^2 <= rho_max`` for every AP l (see
    :func:`ap_powers` to check it).

    Args:
        cfg: System configuration.
        Wbar: Directions ``(Q, M_tot, K)``.
        rho: Power coefficients ``(K,)`` or ``(L, K)`` from :func:`power_control`.

    Returns:
        Normalized precoders ``W`` shaped ``(Q, M_tot, K)``.
    """
    raise NotImplementedError("power control: normalize directions into precoders")


# ======================================================================
# Signal processing  --  IMPLEMENTED (downlink system model)
# ======================================================================


def effective_channel(H: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Effective-channel matrices G[q, k, i] = h_k[q]^H w_i[q].

    Args:
        H: Channels ``(Q, K, M_tot)``.
        W: Precoders ``(Q, M_tot, K)``.

    Returns:
        Array ``(Q, K, K)`` with the desired gains on the diagonal and the
        inter-user leakage off-diagonal.
    """
    return np.conj(H) @ W


def downlink_sinr(G: np.ndarray, noise_power: float) -> np.ndarray:
    """Effective downlink SINR per user and subcarrier (eq. downlink-sinr).

    ``SINR_k[q] = |G_kk|^2 / (sum_{i != k} |G_ki|^2 + sigma^2)``.

    Args:
        G: Effective-channel matrices ``(Q, K, K)``.
        noise_power: Receiver noise power sigma^2 [W].

    Returns:
        Array ``(Q, K)`` of linear SINR values.
    """
    gain = np.abs(G) ** 2                        # (Q, K, K)
    desired = np.diagonal(gain, axis1=1, axis2=2)  # (Q, K)
    total_received = gain.sum(axis=2)            # (Q, K)
    interference = total_received - desired
    return desired / (interference + noise_power)


def spectral_efficiency(sinr: np.ndarray, prelog: float) -> np.ndarray:
    """Per-user spectral efficiency [bit/s/Hz], averaged over the subcarriers.

    ``SE_k = prelog * mean_q log2(1 + SINR_k[q])``. Average the returned values
    over Monte Carlo realizations to obtain the ergodic SE.

    Args:
        sinr: Linear SINRs ``(Q, K)``.
        prelog: Coherence-block prelog factor (e.g. ``cfg.dl_prelog``).

    Returns:
        Array ``(K,)`` of per-user spectral efficiencies.
    """
    return prelog * np.log2(1.0 + sinr).mean(axis=0)


def ap_powers(cfg: DMIMOConfig, W: np.ndarray) -> np.ndarray:
    """Transmit power radiated by each AP, sum_q sum_k ||w_kl[q]||^2 [W].

    Use this to verify the per-AP constraint ``ap_powers(cfg, W) <= rho_max``.

    Args:
        cfg: System configuration (for L, M).
        W: Precoders ``(Q, M_tot, K)``.

    Returns:
        Array ``(L,)`` of per-AP transmit powers.
    """
    per_antenna = (np.abs(W) ** 2).sum(axis=(0, 2))       # (M_tot,)
    return per_antenna.reshape(cfg.L, cfg.M).sum(axis=1)  # (L,)
