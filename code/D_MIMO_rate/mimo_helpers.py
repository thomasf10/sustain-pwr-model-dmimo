"""MIMO-specific building blocks for the distributed massive MIMO rate model.

The functions here are split into two kinds:

* **Stubs you implement** -- the analytical Rayleigh channel model. These raise
  :class:`NotImplementedError` and document the expected inputs, outputs, and
  array shapes so the surrounding pipeline stays fixed while you fill in the
  physics.
* **Implemented signal-processing** -- the precoding, combining, power control,
  effective-channel, SINR, spectral efficiency, and per-AP power computations of
  the downlink and uplink system models. These are generic given the channels
  and are ready to use.

Array-shape conventions (complex128 unless noted), with ``M_tot = L * M`` the
total number of distributed antennas:

    H     channels                (Q, K, M_tot)   H[q, k, :] = h_k[q]
    Wbar  precoding directions    (Q, M_tot, K)   Wbar[q, :, k] = wbar_k[q]
    W     normalized precoders    (Q, M_tot, K)   W[q, :, k] = w_k[q]
    V     receive combiners       (Q, M_tot, K)   V[q, :, k] = v_k[q]
    G     DL effective channels   (Q, K, K)       G[q, k, i] = h_k[q]^H w_i[q]
    G_ul  UL effective channels   (Q, K, K)       G[q, k, i] = v_k[q]^H h_i[q]
    beta  large-scale fading      (L, K)          beta[l, k]
    rho   DL power coefficients   (K,) or (L, K)
    p     UL transmit powers      (K,)
    a     LSFD fusion weights     (L, K)          a[l, k] = a_kl

Columns of ``H`` store the collective channel vector ``h_k[q]`` (not conjugated);
the Hermitian in ``h_k^H w_i`` and ``v_k^H h_i`` is applied inside
:func:`effective_channel` / :func:`uplink_effective_channel`.

By TDD reciprocity the same ``H`` serves both directions: the downlink applies
the array response as ``h_k[q]^H``, so ``h_k[q]`` itself is the channel that
carries user ``k``'s uplink signal to the ``LM`` distributed antennas.

**Powers and the subcarrier dimension.** All transmit budgets are totals over
the OFDM block, and both system models are written per subcarrier, so a
per-subcarrier signal power has to meet the per-subcarrier noise
``cfg.noise_power_sc = sigma^2 / Q`` rather than the full-band
``cfg.noise_power``. The two directions reach that differently:

* Downlink: :func:`normalize_precoder` normalizes ``sum_q ||w_k[q]||^2 = rho_k``,
  leaving about ``rho_k / Q`` per subcarrier, so :func:`downlink_sinr` must be
  given ``cfg.noise_power_sc``. The downlink SINR is *not* invariant to a common
  rescaling of signal and noise, so getting this wrong scales it by ``Q``.
* Uplink: eq. ul-sinr and the loading of eq. ul-centralized-rzf are both
  invariant to a common factor on ``(p, sigma^2)``, so the split
  ``p_k[q] = p_k / Q`` cancels against ``sigma^2 / Q`` exactly. The uplink
  routines therefore work with the block totals ``p`` and ``cfg.noise_power``
  and never divide by ``Q``; passing the per-subcarrier pair instead gives
  bit-identical results.
"""

from __future__ import annotations

import numpy as np

from config_dmimo import (
    APPlacement,
    ChannelModel,
    CombiningScheme,
    DMIMOConfig,
    FusionRule,
    OperationMode,
    PowerControlScheme,
    PrecodingScheme,
    SEBound,
)

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

    UEs are always dropped uniformly over the square
    ``[0, area_size) x [0, area_size)`` [m] (the wrap-around torus of
    :meth:`DMIMOConfig`, so no location is disadvantaged by an edge). The APs
    follow ``cfg.ap_placement``: ``RANDOM`` drops them uniformly too, which is
    the cell-free convention, while ``CENTER`` puts them at the area centre and
    exists for the ``L = 1`` co-located baseline, where a uniform drop would
    sometimes park the only site in a corner. Only the horizontal ``(x, y)``
    coordinates vary; the fixed ``ap_height`` and ``ue_height`` set the vertical
    separation and are folded into the 3-D distance by
    :func:`large_scale_fading`.

    Args:
        cfg: System configuration.
        rng: Random generator (use it for reproducibility).

    Returns:
        Tuple ``(ap_pos, ue_pos)`` of arrays shaped ``(L, 2)`` and ``(K, 2)``
        holding the horizontal AP and UE coordinates [m].
    """
    if cfg.ap_placement is APPlacement.CENTER:
        ap_pos = np.tile(cpu_position(cfg), (cfg.L, 1))
    else:
        ap_pos = rng.uniform(0.0, cfg.area_size, size=(cfg.L, 2))
    ue_pos = rng.uniform(0.0, cfg.area_size, size=(cfg.K, 2))
    return ap_pos, ue_pos


def cpu_position(cfg: DMIMOConfig, cpu_pos=None) -> np.ndarray:
    """Coordinate ``(2,)`` [m] of the central unit (CPU/CU).

    The CPU is a logical entity with no location in the system model; absent an
    explicit ``cpu_pos`` it is placed at the coverage-area centre. This position
    is used only for display (:func:`plot_network`) and for the fronthaul-length
    bookkeeping (:func:`fronthaul_lengths`), never in the signal model.
    """
    if cpu_pos is None:
        return np.array([cfg.area_size / 2, cfg.area_size / 2])
    return np.asarray(cpu_pos, dtype=float)


def fronthaul_lengths(cfg: DMIMOConfig, ap_pos, cpu_pos=None) -> np.ndarray:
    """Planar CPU-to-AP fronthaul link lengths ``(L,)`` [m] for one drop.

    Euclidean distance in the horizontal plane from the CPU (see
    :func:`cpu_position`) to each AP. This is the straight-line separation; it
    ignores AP height and any real cable/fibre routing, so it is a lower bound on
    the physical fronthaul run. Because :func:`draw_positions` redraws the APs
    each realization, these lengths describe the given drop, not a fixed
    deployment.

    Args:
        cfg: System configuration (``area_size`` for the default CPU location).
        ap_pos: AP coordinates ``(L, 2)`` [m] from :func:`draw_positions`.
        cpu_pos: Optional CPU coordinate ``(2,)`` [m]; defaults to the centre.

    Returns:
        Array ``(L,)`` of link lengths [m].
    """
    ap = np.asarray(ap_pos, dtype=float)
    return np.linalg.norm(ap - cpu_position(cfg, cpu_pos), axis=1)


def fronthaul_summary(cfg: DMIMOConfig, ap_pos, cpu_pos=None) -> str:
    """Human-readable overview of the per-link and total fronthaul lengths.

    Lists the CPU-to-AP length of every fronthaul link and the aggregate, mean,
    and maximum, for the AP drop in ``ap_pos`` (see :func:`fronthaul_lengths`).
    """
    cpu = cpu_position(cfg, cpu_pos)
    d = fronthaul_lengths(cfg, ap_pos, cpu_pos)
    lines = [f"Fronthaul overview (CPU at ({cpu[0]:.1f}, {cpu[1]:.1f}) m, {cfg.L} links)"]
    for l, dl in enumerate(d):
        lines.append(f"  AP {l:2d} -> CPU : {dl:7.1f} m")
    total = d.sum()
    lines.append(f"  total fronthaul length : {total:.1f} m ({total/1e3:.3f} km)")
    lines.append(f"  mean / max link        : {d.mean():.1f} / {d.max():.1f} m")
    return "\n".join(lines)


def plot_network(cfg: DMIMOConfig, ap_pos, ue_pos, cpu_pos=None, ax=None,
                 annotate: bool = False, show: bool = True, save_path=None):
    """Scatter the cell-free network drop: APs, UEs, and the central unit (CPU).

    Draws the ``L`` access points, the ``K`` users, and the CPU over the square
    coverage area, with light fronthaul links from the CPU to every AP. The CPU
    is a logical entity with no coordinate in the system model; absent an
    explicit ``cpu_pos`` it is placed at the area centre for display only. The
    deployment lives on a wrap-around torus, so the straight fronthaul segments
    are drawn for readability and are not the wrap-around distances.

    Matplotlib is imported lazily so importing this module stays dependency-free.

    Args:
        cfg: System configuration (``L``, ``K``, ``area_size``).
        ap_pos: AP coordinates ``(L, 2)`` [m] from :func:`draw_positions`.
        ue_pos: UE coordinates ``(K, 2)`` [m] from :func:`draw_positions`.
        cpu_pos: Optional CPU coordinate ``(2,)`` [m]; defaults to the area centre.
        ax: Optional Matplotlib ``Axes`` to draw into; a new figure is made if
            omitted.
        annotate: If true, label each AP and UE with its index.
        show: If true, call ``plt.show()`` before returning.
        save_path: If given, save the figure to this path (dpi 150).

    Returns:
        The Matplotlib ``Axes`` the network was drawn on.
    """
    import matplotlib.pyplot as plt

    ap = np.asarray(ap_pos, dtype=float)
    ue = np.asarray(ue_pos, dtype=float)
    cpu = cpu_position(cfg, cpu_pos)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Fronthaul links CPU <-> each AP (drawn under the markers).
    for l in range(ap.shape[0]):
        ax.plot([cpu[0], ap[l, 0]], [cpu[1], ap[l, 1]],
                color="0.8", lw=0.8, zorder=1)

    ax.scatter(ue[:, 0], ue[:, 1], marker="o", s=40, c="tab:blue",
               edgecolors="k", linewidths=0.4, label=f"UEs (K={cfg.K})", zorder=3)
    ax.scatter(ap[:, 0], ap[:, 1], marker="^", s=90, c="tab:red",
               edgecolors="k", linewidths=0.5, label=f"APs (L={cfg.L})", zorder=4)
    ax.scatter(cpu[0], cpu[1], marker="s", s=150, c="tab:green",
               edgecolors="k", linewidths=0.6, label="CPU", zorder=5)

    if annotate:
        for l in range(ap.shape[0]):
            ax.annotate(str(l), ap[l], textcoords="offset points", xytext=(4, 4),
                        fontsize=8, color="tab:red")
        for k in range(ue.shape[0]):
            ax.annotate(str(k), ue[k], textcoords="offset points", xytext=(4, 4),
                        fontsize=8, color="tab:blue")

    ax.set_xlim(0, cfg.area_size)
    ax.set_ylim(0, cfg.area_size)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Cell-free network: {cfg.L} APs, {cfg.K} UEs "
                 f"({cfg.area_size:.0f} m x {cfg.area_size:.0f} m)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, ls=":", alpha=0.4)

    if save_path is not None:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return ax


def plot_se_cdf(se_samples, ax=None, label=None, show: bool = True, save_path=None,
                title: str = "Downlink per-user SE distribution"):
    """Empirical CDF of the per-user spectral efficiency.

    Each sample is one user's SE in one channel realization; pooling them over
    users and realizations gives the standard cell-free SE distribution over
    random user locations and fading, which the ergodic sum SE hides. The 5th
    percentile (the 95%-likely SE) and the median are marked, since the lower
    tail is the fairness metric that matters in cell-free deployments.

    Matplotlib is imported lazily so importing this module stays dependency-free.

    Args:
        se_samples: Per-user SE values [bit/s/Hz]; any shape, flattened here
            (e.g. :attr:`dl_rate.DownlinkResult.se_samples`, shape ``(n, K)``).
        ax: Optional Matplotlib ``Axes`` to draw into (for overlaying schemes);
            a new figure is made if omitted.
        label: Optional curve label (shown in a legend when given).
        show: If true, call ``plt.show()`` before returning.
        save_path: If given, save the figure to this path (dpi 150).
        title: Axes title; override it for the uplink or for a comparison plot.

    Returns:
        The Matplotlib ``Axes`` the CDF was drawn on.
    """
    import matplotlib.pyplot as plt

    se = np.sort(np.asarray(se_samples, dtype=float).ravel())
    if se.size == 0:
        raise ValueError("se_samples is empty")
    cdf = np.arange(1, se.size + 1) / se.size

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    line, = ax.plot(se, cdf, lw=2, label=label)
    color = line.get_color()
    p05, p50 = np.percentile(se, 5), np.percentile(se, 50)
    for x, y in ((p05, 0.05), (p50, 0.50)):
        ax.plot([x, x, 0], [0, y, y], ls=":", color=color, lw=1, zorder=1)
    ax.annotate(f"5% = {p05:.2f}", (p05, 0.05), textcoords="offset points",
                xytext=(6, -2), fontsize=8, color=color)
    ax.annotate(f"median = {p50:.2f}", (p50, 0.50), textcoords="offset points",
                xytext=(6, -10), fontsize=8, color=color)

    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("per-user spectral efficiency [bit/s/Hz]")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.4)
    if label is not None:
        ax.legend(loc="lower right")

    if save_path is not None:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return ax


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
# Downlink power control
# ======================================================================


def power_control(cfg: DMIMOConfig, beta: np.ndarray, Wbar: np.ndarray) -> np.ndarray:
    """Downlink power-control coefficients from the large-scale fading.

    The allocation is dispatched on ``cfg.operation`` so that the level of
    cooperation and the power-control rule stay consistent (the same coupling
    that :class:`DMIMOConfig` enforces between ``operation`` and ``precoding``):

    * ``CENTRALIZED`` returns a per-user vector ``rho`` shaped ``(K,)``. Two
      heuristics are available through ``cfg.power_alloc``: ``EQUAL`` splits the
      total network power equally, ``rho_k = P_tot / K``; ``FRACTIONAL`` weights
      users by their aggregate large-scale gain, ``rho_k = P_tot * beta_k^v /
      sum_i beta_i^v`` with ``beta_k = sum_l beta_kl``. Both sum to the total
      power budget ``P_tot = L * rho_max`` (every AP transmitting at full power).
    * ``DISTRIBUTED`` returns a per-AP-per-user matrix ``rho`` shaped ``(L, K)``
      using the (only) local rule, the per-AP fractional allocation
      ``rho_kl = rho_max * beta_kl^v / sum_i beta_il^v``. This satisfies
      ``sum_k rho_kl = rho_max`` for every AP by construction, so each AP meets
      its per-antenna-array power budget with equality.

    The exponent ``v`` interpolates between equal power (``v = 0``),
    gain-proportional allocation that favours strong users (``v > 0``), and
    fairness-oriented allocation that boosts weak users (``v < 0``). These
    coefficients are nominal targets defined from the large-scale statistics;
    :func:`normalize_precoder` performs the final direction/power split and is
    responsible for enforcing the exact per-AP budget on the realized precoders.

    Args:
        cfg: System configuration (``operation``, ``power_alloc``, ``v``,
            ``rho_max``, ``L``, ``K``).
        beta: Large-scale fading ``(L, K)`` [linear], e.g. from
            :func:`channel_realization`.
        Wbar: Precoding directions ``(Q, M_tot, K)``. Unused by these
            statistics-only heuristics; kept for signature parity with rules
            that shape power from the instantaneous directions.

    Returns:
        Array ``(K,)`` (centralized) or ``(L, K)`` (distributed) of non-negative
        powers [W].
    """
    del Wbar  # heuristics depend only on the large-scale fading beta
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (cfg.L, cfg.K):
        raise ValueError(
            f"beta must have shape (L, K)=({cfg.L}, {cfg.K}), got {beta.shape}"
        )

    if cfg.operation is OperationMode.CENTRALIZED:
        return _centralized_power_control(cfg, beta)
    if cfg.operation is OperationMode.DISTRIBUTED:
        return _local_power_control(cfg, beta)
    raise NotImplementedError(f"power control: operation {cfg.operation.value}")


def _centralized_power_control(cfg: DMIMOConfig, beta: np.ndarray) -> np.ndarray:
    """Per-user centralized allocation ``rho`` shaped ``(K,)`` summing to P_tot.

    ``EQUAL`` gives ``rho_k = P_tot / K``; ``FRACTIONAL`` gives ``rho_k = P_tot *
    beta_k^v / sum_i beta_i^v`` with the aggregate user gain
    ``beta_k = sum_l beta_kl`` and ``P_tot = L * rho_max``.
    """
    P_tot = cfg.L * cfg.rho_max
    if cfg.power_alloc is PowerControlScheme.EQUAL:
        return np.full(cfg.K, P_tot / cfg.K)
    if cfg.power_alloc is PowerControlScheme.FRACTIONAL:
        beta_k = beta.sum(axis=0)                 # (K,) aggregate gain per user
        weights = beta_k ** cfg.v                 # (K,)
        return P_tot * weights / weights.sum()
    raise NotImplementedError(
        f"power control: centralized rule {cfg.power_alloc.value}"
    )


def _local_power_control(cfg: DMIMOConfig, beta: np.ndarray) -> np.ndarray:
    """Per-AP fractional allocation ``rho`` shaped ``(L, K)``.

    ``rho_kl = rho_max * beta_kl^v / sum_i beta_il^v``; the denominator is
    the per-AP sum over the served users, so ``sum_k rho_kl = rho_max`` and each
    AP spends its full budget. This is the only local (distributed) rule.
    """
    weights = beta ** cfg.v                        # (L, K)
    denom = weights.sum(axis=1, keepdims=True)    # (L, 1) per-AP normalizer
    return cfg.rho_max * weights / denom


def normalize_precoder(cfg: DMIMOConfig, Wbar: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Scale directions into precoders that meet the per-AP power budget.

    The direction/power split depends on the shape of ``rho`` (which encodes the
    cooperation level, see :func:`power_control`):

    * Per-user ``rho`` shaped ``(K,)`` (centralized). Each user direction is first
      normalized over all antennas and subcarriers to radiate ``rho_k`` in total,
      ``w_k = sqrt(rho_k) * wbar_k / sqrt(sum_q ||wbar_k[q]||^2)``. Because the
      centralized precoder is designed jointly across APs, its columns cannot be
      rescaled per AP without breaking the interference nulling, so the per-AP
      budget is met by a single global scaling ``sqrt(rho_max / max_l P_l)`` that
      brings the busiest AP to ``rho_max``. With ``sum_k rho_k = L * rho_max`` the
      per-AP powers average to ``rho_max``, so this factor is at most one and the
      constraint is always feasible.
    * Per-AP-per-user ``rho`` shaped ``(L, K)`` (distributed). Each AP normalizes
      its own ``M``-antenna block independently to radiate ``rho_kl`` for user k,
      ``w_kl = sqrt(rho_kl) * wbar_kl / sqrt(sum_q ||wbar_kl[q]||^2)``. The per-AP
      power is then ``sum_k rho_kl``, which the local fractional rule sets to
      ``rho_max``, so every AP meets its budget with equality using only local
      quantities (no CPU-side global scaling).

    A direction with zero energy (e.g. a local block that a per-AP precoder left
    empty) is left at zero rather than divided by zero. Verify the result with
    ``ap_powers(cfg, W) <= rho_max``.

    Args:
        cfg: System configuration.
        Wbar: Directions ``(Q, M_tot, K)``.
        rho: Power coefficients ``(K,)`` or ``(L, K)`` from :func:`power_control`.

    Returns:
        Normalized precoders ``W`` shaped ``(Q, M_tot, K)``.
    """
    Wbar = np.asarray(Wbar)
    rho = np.asarray(rho, dtype=float)
    Q, M_tot, K = Wbar.shape
    L, M = cfg.L, cfg.M

    if rho.shape == (K,):
        # Centralized: per-user total power, then one global scaling so the
        # busiest AP sits exactly at rho_max (preserves the joint directions).
        energy = (np.abs(Wbar) ** 2).sum(axis=(0, 1))     # (K,) sum_q ||wbar_k||^2
        W = Wbar * _power_scale(rho, energy)[None, None, :]
        peak = ap_powers(cfg, W).max()                    # busiest AP power [W]
        if peak > 0:
            W = W * np.sqrt(cfg.rho_max / peak)
        return W

    if rho.shape == (L, K):
        # Distributed: each AP normalizes its own block to radiate rho_kl, so
        # sum_k rho_kl = rho_max is met per AP with only local information.
        W = np.empty_like(Wbar)
        for l in range(L):
            blk = slice(l * M, (l + 1) * M)
            Wbar_l = Wbar[:, blk, :]                       # (Q, M, K)
            energy = (np.abs(Wbar_l) ** 2).sum(axis=(0, 1))  # (K,) sum_q ||wbar_kl||^2
            W[:, blk, :] = Wbar_l * _power_scale(rho[l], energy)[None, None, :]
        return W

    raise ValueError(
        f"rho must have shape (K,)=({K},) or (L, K)=({L}, {K}), got {rho.shape}"
    )


def _power_scale(power: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Per-column scale ``sqrt(power / energy)``, guarding zero-energy directions.

    Columns with zero direction energy get a zero scale (they radiate nothing)
    instead of producing ``inf``/``nan``.
    """
    scale = np.zeros_like(energy, dtype=float)
    nz = energy > 0
    scale[nz] = np.sqrt(power[nz] / energy[nz])
    return scale


# ======================================================================
# Downlink signal processing
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

    This is a per-subcarrier expression and ``G`` is built from precoders that
    :func:`normalize_precoder` has spread over the ``Q`` subcarriers, so
    ``noise_power`` must be the per-subcarrier ``cfg.noise_power_sc`` and not the
    full-band ``cfg.noise_power``. Unlike the uplink, the downlink SINR is not
    invariant to a common rescaling of signal and noise, so the mismatch does not
    cancel: it scales the SINR by ``Q``.

    Args:
        G: Effective-channel matrices ``(Q, K, K)``.
        noise_power: Per-subcarrier receiver noise power sigma^2 [W], i.e.
            ``cfg.noise_power_sc``.

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


# ======================================================================
# Uplink power control
# ======================================================================
#
# The uplink is not the mirror image of the downlink here. Each user is its own
# transmitter with its own budget p_max (eq. ul-power-constraint), so there is no
# shared budget to divide and no counterpart of the common rescaling that
# normalize_precoder applies. The powers are therefore final as they leave
# uplink_power_control, and the combiners that follow are scale-invariant.


def uplink_power_control(cfg: DMIMOConfig, beta: np.ndarray) -> np.ndarray:
    """Uplink transmit powers p_k from the large-scale fading (eq. ul-fractional-power).

    Two rules, selected by ``cfg.ul_power_alloc``:

    * ``EQUAL`` is full power, ``p_k = p_max`` for every user. It is the natural
      reference and the worst case for fairness, since a user close to an AP then
      swamps a distant one on the same resources.
    * ``FRACTIONAL`` weights users by their aggregate large-scale gain
      ``beta_k = sum_l beta_kl``, ``p_k = p_max * beta_k^v / max_i beta_i^v``.

    The normalization by a *maximum* rather than a sum is what differs from the
    downlink rule :func:`_centralized_power_control`: there a shared budget had
    to be divided among the users, whereas here the denominator only keeps the
    strongest user inside its own budget. This gives ``p_k <= p_max`` for either
    sign of the exponent ``cfg.v_ul``, since the maximum is attained by the
    strongest user when ``v > 0`` and by the weakest when ``v < 0``. ``v = 0``
    recovers full power and ``v = -1`` inverts the channel statistically, so that
    every user arrives at the network with the same average total gain
    ``p_k beta_k``.

    Args:
        cfg: System configuration (``ul_power_alloc``, ``v_ul``, ``p_max``).
        beta: Large-scale fading ``(L, K)`` [linear] from
            :func:`channel_realization`.

    Returns:
        Array ``(K,)`` of per-user transmit powers [W], each at most ``p_max``.
    """
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (cfg.L, cfg.K):
        raise ValueError(
            f"beta must have shape (L, K)=({cfg.L}, {cfg.K}), got {beta.shape}"
        )

    if cfg.ul_power_alloc is PowerControlScheme.EQUAL:
        return np.full(cfg.K, cfg.p_max)
    if cfg.ul_power_alloc is PowerControlScheme.FRACTIONAL:
        beta_k = beta.sum(axis=0)                 # (K,) aggregate gain per user
        weights = beta_k ** cfg.v_ul              # (K,)
        return cfg.p_max * weights / weights.max()
    raise NotImplementedError(
        f"uplink power control: rule {cfg.ul_power_alloc.value}"
    )


# ======================================================================
# Uplink receive combining
# ======================================================================
#
# The centralized combiners (ZF, RZF, MMSE) act on the collective y[q] over all
# M_tot antennas; the local ones (MR, L-RZF, L-MMSE) build each AP's block from
# its own CSI, and the CPU then fuses the L resulting scalars (see
# fusion_weights). Because eq. ul-sinr is invariant to a per-user rescaling of
# v_k, the combiners returned here need no normalization stage.


def combining_directions(cfg: DMIMOConfig, H_hat: np.ndarray,
                         p: np.ndarray) -> np.ndarray:
    """Receive combining vectors v_k[q] for the configured scheme.

    Writing ``E = H_hat[q].T`` for the ``(M_tot, K)`` matrix whose columns are
    the user estimates ``h_k[q]`` and ``E^H E`` for their ``(K, K)`` Gram matrix,
    the centralized combiners are

    * ``ZF``   : ``V = E (E^H E)^{-1}``                    (eq. ul-zf-combiner)
    * ``RZF``  : ``V = E (E^H E + lambda I_K)^{-1}``       heuristic loading
    * ``MMSE`` : ``V = E (E^H E + sigma^2 P^{-1})^{-1}``   (eq. ul-centralized-rzf)

    and the local ones are ``MR`` (``v_kl = h_kl``, eq. ul-mr-combiner) and the
    per-AP regularized family of :func:`_local_regularized_combining`.

    The uplink MMSE loading is a *diagonal matrix* ``sigma^2 P^{-1}``, not a
    scalar: this is the push-through form of the ``M_tot x M_tot`` combiner
    ``(H P H^H + sigma^2 I)^{-1} h_k`` of eq. ul-mmse-combiner with the diagonal
    factor dropped, which the scale invariance of eq. ul-sinr makes irrelevant.
    That physically determined loading is the one difference from the downlink
    regularized family, where ``lambda`` is a free heuristic; for equal powers
    ``p_k = p`` the two coincide at ``lambda = sigma^2 / p``, which is the sense
    in which the downlink RZF precoder is the dual of MMSE combining. ``RZF``
    keeps the scalar heuristic ``cfg.ul_rzf_regularization`` (default
    ``sigma^2 / p_max``) and so coincides with ``MMSE`` when every user
    transmits at full power. ``ZF`` is the ``sigma^2 / p -> 0`` high-SNR limit
    and needs ``K <= M_tot`` for the Gram matrix to be invertible.

    Args:
        cfg: System configuration.
        H_hat: Channel estimates ``(Q, K, M_tot)``.
        p: Uplink transmit powers ``(K,)`` [W] from
            :func:`uplink_power_control`. Used by the MMSE and L-MMSE loadings;
            ignored by MR, ZF, RZF, and L-RZF, whose directions do not depend on
            the power allocation.

    Returns:
        Combiners ``V`` shaped ``(Q, M_tot, K)``, arbitrarily scaled per user.
    """
    scheme = cfg.combining
    p = np.asarray(p, dtype=float)
    if p.shape != (cfg.K,):
        raise ValueError(f"p must have shape (K,)=({cfg.K},), got {p.shape}")
    if np.any(p <= 0):
        raise ValueError("uplink powers must be strictly positive to load the combiner")

    # E[q] has columns h_k[q]; this is already the (Q, M_tot, K) combiner layout.
    E = H_hat.transpose(0, 2, 1)                     # (Q, M_tot, K)

    if scheme is CombiningScheme.MR:
        return E.copy()

    # Local (per-AP) family: each AP designs from its own M-antenna block.
    if scheme is CombiningScheme.L_MMSE:
        return _local_regularized_combining(cfg, H_hat, E, cfg.noise_power, p)
    if scheme is CombiningScheme.L_RZF:
        return _local_regularized_combining(cfg, H_hat, E,
                                            cfg.ul_rzf_regularization, None)

    # Centralized family: V = E (E^H E + diag(load))^{-1}, the members differing
    # only in the diagonal loading of the Gram matrix.
    if scheme is CombiningScheme.ZF:
        load = np.zeros(cfg.K)                            # high-SNR limit
    elif scheme is CombiningScheme.RZF:
        load = np.full(cfg.K, cfg.ul_rzf_regularization)  # heuristic sigma^2 / p_max
    elif scheme is CombiningScheme.MMSE:
        load = cfg.noise_power / p                        # sigma^2 P^{-1}
    else:
        raise NotImplementedError(f"combining: directions for {scheme.value}")

    gram = np.conj(H_hat) @ E                        # (Q, K, K), [k,i] = h_k^H h_i
    if np.any(load):
        gram = gram + np.diag(load)

    # V = E @ inv(gram). Solved stably as (inv(gram) @ E^H)^H, using that gram is
    # Hermitian (the loading is real diagonal) and E^H (per subcarrier) is
    # conj(H_hat).
    X = np.linalg.solve(gram, np.conj(H_hat))        # (Q, K, M_tot) = inv(gram) @ E^H
    return np.conj(X).transpose(0, 2, 1)             # (Q, M_tot, K)


def _local_regularized_combining(cfg: DMIMOConfig, H_hat: np.ndarray, E: np.ndarray,
                                 load: float, p) -> np.ndarray:
    """Local (per-AP) regularized combiners for L-MMSE / L-RZF.

    AP ``l`` combines its own observation using only its own channel block
    ``E_l = [h_1l ... h_Kl]`` (eq. ul-local-rzf):

    * ``L-MMSE`` : ``V_l = (E_l P E_l^H + sigma^2 I_M)^{-1} E_l``
    * ``L-RZF``  : ``V_l = (E_l E_l^H + lambda I_M)^{-1} E_l``

    the difference being that L-MMSE uses the physically determined weighting by
    the transmit powers and the noise-power loading, whereas L-RZF keeps the
    heuristic scalar loading of the downlink form :func:`_local_regularized_zf`.
    Both invert an ``M x M`` matrix, because the per-AP array is small, and the
    loading is what keeps that inverse well conditioned in the usual regime
    ``M < K`` where a plain local ZF would fail. An AP can suppress only the
    interference it observes itself, so the coherent cross-AP rejection of the
    centralized combiner is lost.

    Args:
        cfg: System configuration.
        H_hat: Channel estimates ``(Q, K, M_tot)``.
        E: Combiner-layout estimates ``H_hat.transpose(0, 2, 1)`` ``(Q, M_tot, K)``.
        load: Diagonal loading (``sigma^2`` for L-MMSE, ``cfg.ul_rzf_reg`` for L-RZF).
        p: Uplink powers ``(K,)`` to weight the local Gram matrix (L-MMSE), or
            ``None`` to leave it unweighted (L-RZF).

    Returns:
        Combiners ``V`` shaped ``(Q, M_tot, K)``.
    """
    Q, K, M, L = cfg.Q, cfg.K, cfg.M, cfg.L
    V = np.empty((Q, cfg.M_tot, K), dtype=np.complex128)
    loadI = load * np.eye(M)
    for l in range(L):
        blk = slice(l * M, (l + 1) * M)
        E_l = E[:, blk, :]                            # (Q, M, K), columns h_kl
        H_l = H_hat[:, :, blk]                        # (Q, K, M), rows h_kl
        # sum_k p_k h_kl h_kl^H + load * I  (p_k = 1 when p is None).
        weighted = E_l if p is None else E_l * p
        A_l = weighted @ np.conj(H_l) + loadI         # (Q, M, M)
        V[:, blk, :] = np.linalg.solve(A_l, E_l)      # (Q, M, K) = inv(A_l) @ E_l
    return V


# ======================================================================
# Uplink fusion of the per-AP soft estimates
# ======================================================================


def fusion_weights(cfg: DMIMOConfig, V: np.ndarray, H_hat: np.ndarray,
                   p: np.ndarray) -> np.ndarray:
    """CPU weights a_kl that fuse the L local soft estimates (eq. ul-lsfd).

    In distributed operation AP ``l`` forwards the scalar
    ``s_kl = v_kl^H y_l`` and the CPU forms ``s_k = sum_l a_kl^* s_kl``. Because
    linear combining is distributive over the APs (eq. ul-distributive), this is
    exactly the collective combiner whose ``l``-th block is ``a_kl v_kl``, so the
    fusion rule is part of the combiner rather than a separate stage and
    eq. ul-sinr continues to apply unchanged.

    * ``EQUAL`` returns all-ones: the local estimates are simply added, and the
      CPU needs nothing beyond the streams themselves.
    * ``LSFD`` returns the large-scale fading decoding weights of
      eq. ul-lsfd-weights, which maximize the resulting use-and-then-forget
      SINR. These are statistical, so they are refreshed once per large-scale
      fading realization and cost the fronthaul nothing per coherence block.

    Centralized operation combines the raw samples at the CPU and has no fusion
    stage, so all-ones is returned there too and applying it is a no-op.

    The expectations in eq. ul-lsfd-weights are over the small-scale fading for
    fixed user positions. One drop holds the positions fixed and gives ``Q``
    frequency-domain realizations, so they are estimated by averaging over the
    ``Q`` subcarriers. That estimator is only as good as the subcarriers are
    numerous and decorrelated across the band; with a small ``Q`` the weights
    carry sampling noise. [MODEL: LSFD expectations estimated over subcarriers]

    [VERIFY: eq. ul-lsfd-weights carries an open ``\\todo`` in
    ``sections/dmimo_ul_sysmodel.tex`` asking that it be checked against the LSFD
    expression of Demir, Bjornson & Sanguinetti (2021) before it is used in the
    evaluation. This implementation transcribes the manuscript as written; the
    check has not been done here, which is why ``FusionRule.EQUAL`` is the
    default.]

    Args:
        cfg: System configuration (``operation``, ``fusion``).
        V: Local combiners ``(Q, M_tot, K)`` from :func:`combining_directions`.
        H_hat: Channel estimates ``(Q, K, M_tot)``; the CPU builds the statistics
            from the CSI it has, which under perfect CSI is the true channel.
        p: Uplink transmit powers ``(K,)`` [W].

    Returns:
        Array ``(L, K)`` of fusion weights (complex for LSFD, real ones otherwise).
    """
    if (cfg.operation is OperationMode.CENTRALIZED
            or cfg.fusion is FusionRule.EQUAL):
        return np.ones((cfg.L, cfg.K))
    if cfg.fusion is not FusionRule.LSFD:
        raise NotImplementedError(f"fusion: rule {cfg.fusion.value}")

    L, M, K, Q = cfg.L, cfg.M, cfg.K, cfg.Q
    sigma2 = cfg.noise_power
    p = np.asarray(p, dtype=float)

    # Per-AP effective gains g[q, k, i, l] = v_kl[q]^H h_il[q], and the per-AP
    # combiner norms d[q, k, l] = ||v_kl[q]||^2.
    g = np.empty((Q, K, K, L), dtype=np.complex128)
    d = np.empty((Q, K, L))
    for l in range(L):
        blk = slice(l * M, (l + 1) * M)
        V_l = V[:, blk, :]                                    # (Q, M, K)
        H_l = H_hat[:, :, blk]                                # (Q, K, M)
        g[..., l] = np.conj(V_l).transpose(0, 2, 1) @ H_l.transpose(0, 2, 1)
        d[..., l] = (np.abs(V_l) ** 2).sum(axis=1)            # (Q, K)

    # A_k = sum_i p_i E{g_ki g_ki^H} + sigma^2 E{D_k}, with E{.} estimated over q.
    # D_k is diagonal because the noise is independent across APs.
    A = np.einsum("qkil,qkim,i->klm", g, np.conj(g), p, optimize=True) / Q  # (K, L, L)
    A += sigma2 * (d.mean(axis=0)[:, :, None] * np.eye(L))                 # (K, L, L)

    b = np.sqrt(p)[:, None] * np.diagonal(g, axis1=1, axis2=2).mean(axis=0).T
    #   diagonal(...) is (Q, L, K) -> mean over q -> (L, K) -> .T is (K, L) = E{g_kk}
    return np.linalg.solve(A, b[..., None])[..., 0].T                      # (L, K)


def apply_fusion_weights(cfg: DMIMOConfig, V: np.ndarray,
                         a: np.ndarray) -> np.ndarray:
    """Fold the CPU fusion weights into the collective combiner.

    Replaces the ``l``-th block of ``v_k[q]`` by ``a_kl v_kl[q]``, which by
    eq. ul-distributive is the collective combiner that realizes the weighted sum
    ``sum_l a_kl^* v_kl^H y_l``. After this the uplink SINR routines apply to
    ``V`` without knowing which cooperation level produced it.

    Args:
        cfg: System configuration (for ``L``, ``M``).
        V: Local combiners ``(Q, M_tot, K)``.
        a: Fusion weights ``(L, K)`` from :func:`fusion_weights`.

    Returns:
        Fused combiners ``(Q, M_tot, K)``.
    """
    a = np.asarray(a)
    if a.shape != (cfg.L, cfg.K):
        raise ValueError(f"a must have shape (L, K)=({cfg.L}, {cfg.K}), got {a.shape}")
    return V * np.repeat(a, cfg.M, axis=0)[None, :, :]   # (1, M_tot, K) broadcast


# ======================================================================
# Uplink signal processing
# ======================================================================


def uplink_effective_channel(V: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Effective-channel matrices G[q, k, i] = v_k[q]^H h_i[q].

    The uplink transpose of :func:`effective_channel`: row ``k`` collects what
    the combiner for user ``k`` sees of every user's channel, so the diagonal
    carries the desired gains and the off-diagonal the multi-user interference
    of eq. ul-estimate.

    Args:
        V: Combiners ``(Q, M_tot, K)``.
        H: Channels ``(Q, K, M_tot)`` (the true ones, not the estimates).

    Returns:
        Array ``(Q, K, K)``.
    """
    return np.conj(V).transpose(0, 2, 1) @ H.transpose(0, 2, 1)


def combiner_norms(V: np.ndarray) -> np.ndarray:
    """Squared combiner norms ||v_k[q]||^2, shape ``(Q, K)``.

    This is the factor that scales the noise in eq. ul-sinr: unlike the downlink,
    where the noise power is fixed at the single-antenna receiver, the uplink
    noise is collected by the combiner itself, so a combiner cannot be made
    better simply by being made larger.
    """
    return (np.abs(V) ** 2).sum(axis=1)


def uplink_sinr(G: np.ndarray, v_norm2: np.ndarray, p: np.ndarray,
                noise_power: float) -> np.ndarray:
    """Effective uplink SINR per user and subcarrier (eq. ul-sinr).

    ``SINR_k[q] = p_k |G_kk|^2 / (sum_{i != k} p_i |G_ki|^2 + sigma^2 ||v_k[q]||^2)``.

    This is the instantaneous (genie-aided) expression, valid when the effective
    channel ``v_k^H h_k`` is known wherever the decoding happens. It is the
    uplink counterpart of :func:`downlink_sinr` and is therefore the consistent
    choice when the two directions are compared. Under local operation with
    statistical fusion weights the CPU does not know that effective channel, and
    this expression is then optimistic by the amount of residual channel
    hardening; :func:`uplink_sinr_uatf` is the achievable bound there.

    Both ``p`` and ``noise_power`` are block totals; see the module docstring for
    why the per-subcarrier factor ``1/Q`` cancels out of this ratio.

    Args:
        G: Effective-channel matrices ``(Q, K, K)`` from
            :func:`uplink_effective_channel`.
        v_norm2: Squared combiner norms ``(Q, K)`` from :func:`combiner_norms`.
        p: Uplink transmit powers ``(K,)`` [W].
        noise_power: Receiver noise power sigma^2 [W] at the APs.

    Returns:
        Array ``(Q, K)`` of linear SINR values.
    """
    received = (np.abs(G) ** 2) * np.asarray(p)[None, None, :]  # (Q, K, K), p_i weights i
    desired = np.diagonal(received, axis1=1, axis2=2)           # (Q, K)
    interference = received.sum(axis=2) - desired               # (Q, K)
    return desired / (interference + noise_power * v_norm2)


def uplink_sinr_uatf(G: np.ndarray, v_norm2: np.ndarray, p: np.ndarray,
                     noise_power: float) -> np.ndarray:
    """Use-and-then-forget uplink SINR, one value per user.

    ``SINR_k = p_k |E{g_kk}|^2 / (sum_i p_i E{|g_ki|^2} - p_k |E{g_kk}|^2
    + sigma^2 E{||v_k||^2})`` with ``g_ki = v_k^H h_i``.

    Only the *mean* effective channel is treated as useful gain; its fluctuation
    around that mean stays in the denominator as extra interference, which is
    what makes this a rigorous achievable rate when the decoder knows the channel
    statistics but not the realization. That is the situation whenever the fusion
    weights are statistical (LSFD or equal-weight local operation), where the
    instantaneous expression of :func:`uplink_sinr` would be optimistic. The gap
    between the two grows as ``M`` shrinks and the channel hardens less.

    The expectation is over the small-scale fading for fixed user positions and
    is estimated by averaging over the ``Q`` subcarriers of the drop, the same
    estimator :func:`fusion_weights` uses. [MODEL: UatF expectations estimated
    over subcarriers]

    Args:
        G: Effective-channel matrices ``(Q, K, K)``.
        v_norm2: Squared combiner norms ``(Q, K)``.
        p: Uplink transmit powers ``(K,)`` [W].
        noise_power: Receiver noise power sigma^2 [W] at the APs.

    Returns:
        Array ``(1, K)`` of linear SINR values. The leading axis is a singleton
        because the expectation has already consumed the subcarrier dimension;
        keeping it lets the result feed :func:`spectral_efficiency` unchanged.
    """
    p = np.asarray(p, dtype=float)
    mean_gain = np.diagonal(G, axis1=1, axis2=2).mean(axis=0)   # (K,) E{g_kk}
    desired = p * np.abs(mean_gain) ** 2                        # (K,)
    total = (np.abs(G) ** 2).mean(axis=0) @ p                   # (K,) sum_i p_i E{|g_ki|^2}
    noise = noise_power * v_norm2.mean(axis=0)                  # (K,)
    # total - desired keeps both the interference and the p_k Var(g_kk) fluctuation.
    return (desired / (total - desired + noise))[None, :]       # (1, K)


def uplink_spectral_efficiency(cfg: DMIMOConfig, G: np.ndarray,
                               v_norm2: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-user uplink SE [bit/s/Hz] under the configured rate expression.

    Dispatches on ``cfg.ul_se_bound`` between the instantaneous SINR of
    eq. ul-sinr and the use-and-then-forget bound, then applies the uplink prelog
    ``cfg.ul_prelog`` of eq. ul-rate-user. The result is a *delivered* SE: do not
    apply a prelog again downstream.

    Args:
        cfg: System configuration (``ul_se_bound``, ``noise_power``, ``ul_prelog``).
        G: Effective-channel matrices ``(Q, K, K)``.
        v_norm2: Squared combiner norms ``(Q, K)``.
        p: Uplink transmit powers ``(K,)`` [W].

    Returns:
        Array ``(K,)`` of per-user spectral efficiencies.
    """
    if cfg.ul_se_bound is SEBound.UATF:
        sinr = uplink_sinr_uatf(G, v_norm2, p, cfg.noise_power)
    elif cfg.ul_se_bound is SEBound.INSTANTANEOUS:
        sinr = uplink_sinr(G, v_norm2, p, cfg.noise_power)
    else:
        raise NotImplementedError(f"uplink SE: bound {cfg.ul_se_bound.value}")
    return spectral_efficiency(sinr, cfg.ul_prelog)
