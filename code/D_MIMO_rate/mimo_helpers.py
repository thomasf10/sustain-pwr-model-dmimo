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

from config_dmimo import ChannelModel, DMIMOConfig, PrecodingScheme

# ======================================================================
# Channel model
# ======================================================================
#
# The channel model has two backends, selected by ``cfg.channel_model``:
#
# * ``SIONNA_UMI`` -- realistic 3GPP TR 38.901 UMi channels from Sionna. This is
#   the default and is fully implemented via :mod:`sionna_channel`.
# * ``RAYLEIGH`` -- the analytical correlated-Rayleigh model, whose per-step
#   pieces (``spatial_correlation``, ``generate_channels``, ``estimate_channels``)
#   are left as stubs to implement.
#
# Both share :func:`draw_positions`. Use :func:`build_channel` once, then
# :func:`channel_realization` per drop, to get ``(H, beta)`` regardless of backend.


def build_channel(cfg: DMIMOConfig):
    """Build the channel backend selected by ``cfg.channel_model`` (once).

    Returns a stateful generator for :func:`channel_realization` when Sionna is
    selected (Sionna is imported lazily so the analytical path needs no heavy
    dependencies), or ``None`` for the analytical Rayleigh backend.
    """
    if cfg.channel_model is ChannelModel.SIONNA_UMI:
        from sionna_channel import SionnaUMiChannel
        return SionnaUMiChannel(cfg)
    return None


def channel_realization(cfg: DMIMOConfig, channel, ap_pos, ue_pos,
                        rng: np.random.Generator):
    """One channel realization ``(H, beta)`` for a drop, for either backend.

    Args:
        cfg: System configuration.
        channel: Object from :func:`build_channel` (Sionna generator or ``None``).
        ap_pos, ue_pos: AP/UE coordinates from :func:`draw_positions`.
        rng: Random generator for the analytical backend.

    Returns:
        Tuple ``(H, beta)`` with channels ``(Q, K, M_tot)`` and large-scale
        fading ``(L, K)``.
    """
    if channel is not None:
        return channel.generate(ap_pos, ue_pos, rng)

    raise NotImplementedError("channel model: analytical correlated-Rayleigh use Sionna channel model instead or implement the stubs in mimo_helpers.py")
    # # Analytical correlated-Rayleigh backend.
    # beta = large_scale_fading(cfg, ap_pos, ue_pos, rng)
    # R = spatial_correlation(cfg, ap_pos, ue_pos, beta)
    # H = generate_channels(cfg, rng, R)
    # return H, beta


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
                      H: np.ndarray) -> np.ndarray:
    """Return the CSI used to build the precoders.

    Currently perfect CSI: the true channels ``H`` are returned unchanged. For a
    pilot-based study, replace this with the MMSE estimates (including pilot
    contamination when ``tau_p < K``), which for the analytical backend needs the
    correlation matrices ``R`` and for the Sionna backend needs them estimated
    from Monte Carlo averaging.

    Returns:
        Array ``(Q, K, M_tot)`` of channel estimates, same layout as ``H``.
    """
    print("mimo_helpers.estimate_channels: perfect CSI (H_hat = H)")

    return H


# ======================================================================
# Transmit precoding
# ======================================================================
#
# The centralized precoders (MR, ZF, RZF, MMSE) act on the M_tot collective
# antennas jointly; the local ones (L-RZF, L-MMSE) build each AP's block from its
# own CSI. Both families are implemented below. The scalable partial variants
# (P-MMSE, P-RZF, LP-MMSE), which need user-centric clustering, remain stubs.


def precoding_directions(cfg: DMIMOConfig, H_hat: np.ndarray) -> np.ndarray:
    """Unnormalized precoding directions wbar_k[q] for the configured scheme.

    Implements the centralized and local precoders of the cell-free downlink from
    the collective channel estimates. Writing ``E = H_hat[q].T`` for the
    ``(M_tot, K)`` matrix whose columns are the user estimates ``h_k[q]`` and
    ``G = E^H E`` for their ``(K, K)`` Gram matrix, the centralized directions are

    * ``MR``   : ``wbar_k = h_k``                          (maximum ratio)
    * ``ZF``   : ``Wbar   = E (E^H E)^{-1}``               (zero forcing)
    * ``RZF``  : ``Wbar   = E (E^H E + lambda I_K)^{-1}``  (regularized ZF)
    * ``MMSE`` : ``Wbar   = E (E^H E + sigma^2 I_K)^{-1}``

    and the local (per-AP, distributed) directions ``L-RZF`` / ``L-MMSE`` build
    each AP's block from only its own local CSI (see :func:`_local_regularized_zf`).

    ZF/RZF/MMSE use the ``K x K`` signal-domain form, which is the push-through
    equivalent of the ``M_tot x M_tot`` form ``(E E^H + lambda I)^{-1} E`` but
    only inverts a ``K x K`` matrix. The RZF loading is ``cfg.rzf_regularization``
    (default ``sigma^2``); MMSE uses the noise power ``sigma^2``. Under the
    current perfect-CSI setup the error-aware MMSE precoder reduces to this
    regularized-ZF form, so MMSE and RZF coincide when the RZF loading is left at
    its ``sigma^2`` default; they diverge once that loading is retuned or a
    channel-estimation-error covariance is modeled (the same holds for the local
    pair L-MMSE / L-RZF). The loading is the noise power for a unit-power dual
    uplink; for physically-scaled channels you may want to set ``cfg.rzf_reg`` to
    ``sigma^2 / p`` for the intended per-user power ``p``.

    Args:
        cfg: System configuration.
        H_hat: Channel estimates ``(Q, K, M_tot)``.

    Returns:
        Directions ``Wbar`` shaped ``(Q, M_tot, K)`` (power/normalization applied
        later in :func:`normalize_precoder`).
    """
    scheme = cfg.precoding
    # E[q] has columns h_k[q]; this is already the (Q, M_tot, K) direction layout.
    E = H_hat.transpose(0, 2, 1)                     # (Q, M_tot, K)

    if scheme is PrecodingScheme.MR:
        return E.copy()

    # Local (per-AP) regularized-ZF family: each AP designs from its own block.
    if scheme is PrecodingScheme.L_RZF:
        return _local_regularized_zf(cfg, H_hat, E, cfg.rzf_regularization)
    if scheme is PrecodingScheme.L_MMSE:
        return _local_regularized_zf(cfg, H_hat, E, cfg.noise_power)

    # Centralized regularized-ZF family: Wbar = E (E^H E + load * I_K)^{-1}, the
    # members differing only in the diagonal loading of the Gram matrix.
    if scheme is PrecodingScheme.ZF:
        load = 0.0                       # no loading; gram must be full rank (K <= M_tot)
    elif scheme is PrecodingScheme.RZF:
        load = cfg.rzf_regularization    # heuristic loading (default sigma^2)
    elif scheme is PrecodingScheme.MMSE:
        load = cfg.noise_power           # noise-power loading sigma^2
    else:
        raise NotImplementedError(f"precoding: directions for {scheme.value}")

    gram = np.conj(H_hat) @ E                        # (Q, K, K), [k,i] = h_k^H h_i
    if load:
        gram = gram + load * np.eye(cfg.K)

    # Wbar = E @ inv(gram). Solved stably as (inv(gram) @ E^H)^H, using that gram
    # is Hermitian and E^H (per subcarrier) is conj(H_hat).
    X = np.linalg.solve(gram, np.conj(H_hat))        # (Q, K, M_tot) = inv(gram) @ E^H
    return np.conj(X).transpose(0, 2, 1)             # (Q, M_tot, K)


def _local_regularized_zf(cfg: DMIMOConfig, H_hat: np.ndarray, E: np.ndarray,
                          load: float) -> np.ndarray:
    """Local (per-AP) regularized-ZF directions for L-RZF / L-MMSE.

    Each AP ``l`` designs its ``M x K`` precoder from only its own channel block
    ``E_l`` (the ``M`` rows of ``E`` for that AP, whose columns are the local
    estimates ``h_kl``), as ``W_l = (E_l E_l^H + load * I_M)^{-1} E_l``, and the
    blocks are stacked into the collective ``(Q, M_tot, K)`` precoder. This uses
    the ``M x M`` antenna-domain form (the per-AP array is small) and the loading
    keeps it invertible even when ``M < K``, unlike a plain local ZF.

    Under perfect CSI L-MMSE reduces to this form (its estimation-error
    covariance term ``C_il`` vanishes), so L-MMSE and L-RZF differ only in the
    loading and coincide at the ``sigma^2`` default. As with the centralized
    schemes, the loading assumes a unit-power dual uplink; retune
    ``cfg.rzf_reg`` to ``sigma^2 / p`` for physically-scaled channels.

    Args:
        cfg: System configuration.
        H_hat: Channel estimates ``(Q, K, M_tot)``.
        E: Direction-layout estimates ``H_hat.transpose(0, 2, 1)`` ``(Q, M_tot, K)``.
        load: Diagonal loading (``sigma^2`` for L-MMSE, ``cfg.rzf_reg`` for L-RZF).

    Returns:
        Directions ``Wbar`` shaped ``(Q, M_tot, K)``.
    """
    Q, K, M, L = cfg.Q, cfg.K, cfg.M, cfg.L
    Wbar = np.empty((Q, cfg.M_tot, K), dtype=np.complex128)
    loadI = load * np.eye(M)
    for l in range(L):
        blk = slice(l * M, (l + 1) * M)
        E_l = E[:, blk, :]                           # (Q, M, K), columns h_kl
        H_l = H_hat[:, :, blk]                        # (Q, K, M), rows h_kl
        A_l = E_l @ np.conj(H_l) + loadI              # (Q, M, M) = sum_k h_kl h_kl^H + load*I
        Wbar[:, blk, :] = np.linalg.solve(A_l, E_l)   # (Q, M, K) = inv(A_l) @ E_l
    return Wbar



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
