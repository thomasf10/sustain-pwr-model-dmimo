"""Analytical channel model of the cell-free monograph's running example.

Implements the correlated-Rayleigh channel that Bjornson & Sanguinetti,
*Foundations of User-Centric Cell-Free Massive MIMO* (arXiv:2108.02541), use in
all of its simulations, so that the rate pipeline of this directory can be run on
the book's own propagation model rather than on the 3GPP TR 38.901 backend:

* toroidal (wrap-around) geometry, step 3 of the Monte Carlo methodology in
  Sec. 5.6;
* large-scale fading ``beta_kl [dB] = -30.5 - 36.7 log10(d_kl / 1 m) + F_kl``,
  eq. (2.20), evaluated through :meth:`DMIMOConfig.path_loss_dB`;
* shadow fading correlated across UEs seen by a common AP,
  ``E{F_kl F_il} = sigma_sf^2 2^{-delta_ki / 9 m}``, eq. (2.21);
* spatial correlation from the Gaussian local scattering model, eq. (2.23) with
  the angular PDF (2.24), on a half-wavelength ULA;
* channel realizations ``h_kl ~ CN(0, R_kl)``, eq. (2.18).

The public entry point is :class:`BookChannel`, whose :meth:`BookChannel.generate`
has the same ``(ap_pos, ue_pos, rng) -> (H, beta)`` signature as
:class:`sionna_channel.SionnaUMiChannel`, so both can be dropped into the same
Monte Carlo loop. Only NumPy is required.

Shape conventions follow :mod:`mimo_helpers`: ``H`` is ``(Q, K, M_tot)`` with
``H[q, k, :]`` the collective channel ``h_k[q]`` stacked AP-major
(``column l*N + n``), and ``beta`` is ``(L, K)`` in linear scale.
"""

from __future__ import annotations

import numpy as np

from config_cellfree_book import BookExtras
from config_dmimo import DMIMOConfig

# Nodes/weights of the Gauss-Hermite rule used for the angular integral of
# eq. (2.23). Twenty nodes per angular dimension resolve the integrand to well
# below the plotting accuracy for the ASDs considered here (5-40 deg).
_GH_NODES = 20


def wrapped_displacement(a: np.ndarray, b: np.ndarray, area_size: float) -> np.ndarray:
    """Shortest displacement ``b - a`` on the wrap-around torus [m].

    Each coordinate difference is reduced to the interval
    ``[-area_size/2, area_size/2)``, which selects the shortest of the paths that
    do or do not cross an edge of the square coverage area.

    Args:
        a: Coordinates ``(..., 2)`` [m].
        b: Coordinates ``(..., 2)`` [m], broadcastable against ``a``.
        area_size: Side of the square coverage area [m].

    Returns:
        Displacement array of the broadcast shape, ``(..., 2)``.
    """
    delta = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return delta - area_size * np.round(delta / area_size)


def pair_geometry(cfg: DMIMOConfig, ap_pos, ue_pos, wrap_around: bool = True):
    """Per-link geometry between every AP and every UE.

    Args:
        cfg: System configuration (``area_size``, ``ap_height``, ``ue_height``).
        ap_pos: AP coordinates ``(L, 2)`` [m].
        ue_pos: UE coordinates ``(K, 2)`` [m].
        wrap_around: Use the toroidal metric of :func:`wrapped_displacement`.

    Returns:
        Tuple ``(d_3d, azimuth, elevation)``, each ``(L, K)``: the 3-D AP-UE
        distance [m], the azimuth of the UE seen from the AP measured from the
        array broadside [rad], and the elevation of the UE relative to the AP
        [rad] (negative, since the APs are above the UE plane).
    """
    ap = np.asarray(ap_pos, dtype=float)[:, None, :]   # (L, 1, 2)
    ue = np.asarray(ue_pos, dtype=float)[None, :, :]   # (1, K, 2)
    if wrap_around:
        delta = wrapped_displacement(ap, ue, cfg.area_size)   # (L, K, 2)
    else:
        delta = ue - ap
    d_2d = np.linalg.norm(delta, axis=2)                      # (L, K)
    height_diff = cfg.ap_height - cfg.ue_height
    d_3d = np.sqrt(d_2d ** 2 + height_diff ** 2)

    # The ULAs lie along the y-axis, so the array broadside is the +x direction
    # and the azimuth of the UE is measured from it.
    azimuth = np.arctan2(delta[:, :, 1], delta[:, :, 0])
    elevation = -np.arctan2(height_diff, np.maximum(d_2d, 1e-9))
    return d_3d, azimuth, elevation


def shadow_covariance(cfg: DMIMOConfig, ue_pos, extras: BookExtras) -> np.ndarray:
    """UE-by-UE shadow-fading covariance ``(K, K)`` [dB^2] at a common AP.

    Eq. (2.21) of the monograph: ``E{F_kl F_il} = sigma_sf^2 2^{-delta_ki / 9 m}``
    with ``delta_ki`` the UE-UE distance, measured here with the same wrap-around
    metric as the AP-UE distances.
    """
    ue = np.asarray(ue_pos, dtype=float)
    if extras.wrap_around:
        delta = wrapped_displacement(ue[:, None, :], ue[None, :, :], cfg.area_size)
    else:
        delta = ue[None, :, :] - ue[:, None, :]
    d_ue = np.linalg.norm(delta, axis=2)                       # (K, K)
    return cfg.shadow_std_dB ** 2 * 2.0 ** (-d_ue / extras.shadow_decorr_m)


def _psd_factor(C: np.ndarray) -> np.ndarray:
    """Lower-triangular-like factor ``A`` with ``A A^T = C`` for a PSD matrix.

    Uses a Cholesky factorization, falling back to a clipped eigendecomposition
    when the matrix is only positive *semi*definite (which the wrap-around
    distance metric can produce, since the exponential kernel is guaranteed
    positive definite on the plane but not on the torus).
    """
    try:
        return np.linalg.cholesky(C + 1e-12 * np.eye(C.shape[0]) * np.trace(C) / C.shape[0])
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(C)
        return V * np.sqrt(np.maximum(w, 0.0))


def large_scale_fading(cfg: DMIMOConfig, ap_pos, ue_pos, rng: np.random.Generator,
                       extras: BookExtras) -> np.ndarray:
    """Large-scale fading ``beta`` ``(L, K)`` [linear] of eq. (2.20).

    ``beta_kl [dB] = -PL(d_kl) + F_kl`` with ``PL`` the log-distance path loss of
    :meth:`DMIMOConfig.path_loss_dB` (the book's ``30.5 + 36.7 log10(d/1m)``) and
    ``F_kl`` a zero-mean Gaussian shadowing term with the UE-UE covariance of
    :func:`shadow_covariance`, drawn independently for each AP.
    """
    d_3d, _, _ = pair_geometry(cfg, ap_pos, ue_pos, extras.wrap_around)
    return _shadowed_beta(cfg, d_3d, ue_pos, rng, extras)


def _shadowed_beta(cfg: DMIMOConfig, d_3d: np.ndarray, ue_pos,
                   rng: np.random.Generator, extras: BookExtras) -> np.ndarray:
    """Apply the correlated shadowing of eq. (2.21) to precomputed distances."""
    A = _psd_factor(shadow_covariance(cfg, ue_pos, extras))        # (K, K)
    F = (A @ rng.standard_normal((cfg.K, cfg.L))).T                # (L, K) [dB]
    return 10.0 ** ((-cfg.path_loss_dB(d_3d) + F) / 10.0)


def local_scattering_correlation(cfg: DMIMOConfig, azimuth: np.ndarray,
                                 elevation: np.ndarray, beta: np.ndarray,
                                 asd_deg: float | None) -> np.ndarray:
    """Spatial correlation matrices ``R`` ``(L, K, N, N)`` of eq. (2.23).

    For a half-wavelength ULA the ``(m, n)`` entry of the correlation matrix is

        ``[R]_{mn} = beta * E{ exp(j pi (m - n) sin(phi_bar) cos(theta_bar)) }``

    where the azimuth ``phi_bar ~ N(phi, sigma_phi^2)`` and elevation
    ``theta_bar ~ N(theta, sigma_theta^2)`` of the multipath components follow the
    Gaussian local scattering PDF (2.24) around the nominal AP-UE angles. The
    expectation is evaluated with a two-dimensional Gauss-Hermite rule, which
    makes the matrix Hermitian Toeplitz and positive semidefinite by construction
    (it is built from a characteristic function). The normalization
    ``Tr(R_kl)/N = beta_kl`` of eq. (2.19) holds exactly, since the ``m = n``
    entry is ``beta``.

    Args:
        cfg: System configuration (``M`` antennas per AP).
        azimuth: Nominal azimuth angles ``(L, K)`` [rad] from :func:`pair_geometry`.
        elevation: Nominal elevation angles ``(L, K)`` [rad].
        beta: Large-scale fading ``(L, K)`` [linear].
        asd_deg: ASD ``sigma_phi = sigma_theta`` [deg], or ``None`` for
            uncorrelated fading ``R_kl = beta_kl I_N``.

    Returns:
        Array ``(L, K, N, N)`` of Hermitian PSD correlation matrices.
    """
    L, K, N = cfg.L, cfg.K, cfg.M
    if asd_deg is None or N == 1:
        # N = 1 makes R_kl the scalar beta_kl, so the ASD is irrelevant there.
        return beta[:, :, None, None] * np.eye(N)[None, None]

    sigma = np.deg2rad(asd_deg)
    x, w = np.polynomial.hermite.hermgauss(_GH_NODES)
    # phi_bar = phi + sqrt(2) sigma x_i and theta_bar = theta + sqrt(2) sigma y_j,
    # so E{g} = (1/pi) sum_ij w_i w_j g(phi_bar_i, theta_bar_j).
    phi = azimuth[:, :, None, None] + np.sqrt(2) * sigma * x[None, None, :, None]
    theta = elevation[:, :, None, None] + np.sqrt(2) * sigma * x[None, None, None, :]
    u = np.sin(phi) * np.cos(theta)                    # (L, K, n, n) directional cosine
    weight = (w[:, None] * w[None, :]) / np.pi         # (n, n)

    # First column of the Toeplitz matrix: c[d] = E{exp(j pi d u)} for d = 0..N-1.
    c = np.empty((L, K, N), dtype=np.complex128)
    for d in range(N):
        c[:, :, d] = np.einsum("ij,lkij->lk", weight, np.exp(1j * np.pi * d * u))

    idx = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])   # |m - n|
    upper = np.arange(N)[:, None] <= np.arange(N)[None, :]        # m <= n
    R = c[:, :, idx]                                              # (L, K, N, N)
    # [R]_{mn} = c[m-n] for m >= n and its conjugate above the diagonal, which
    # makes R Hermitian Toeplitz.
    R = np.where(upper, np.conj(R), R)
    return beta[:, :, None, None] * R


def correlated_rayleigh(cfg: DMIMOConfig, R: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """Draw ``h_kl ~ CN(0, R_kl)`` and stack them into ``H`` ``(Q, K, M_tot)``.

    Realizations are independent across the ``Q`` blocks; the running example is
    frequency-flat, so the benchmark uses ``Q = 1``.

    Args:
        cfg: System configuration.
        R: Correlation matrices ``(L, K, N, N)``.
        rng: Random generator.

    Returns:
        Array ``(Q, K, M_tot)`` with ``H[q, k, l*N:(l+1)*N] = h_kl``.
    """
    L, K, N, Q = cfg.L, cfg.K, cfg.M, cfg.Q
    # Jitter relative to the mean diagonal keeps the Cholesky well-posed for the
    # near-rank-deficient matrices that small ASDs produce.
    jitter = 1e-10 * np.trace(R, axis1=2, axis2=3).real[:, :, None, None] / N
    A = np.linalg.cholesky(R + jitter * np.eye(N))                # (L, K, N, N)
    z = (rng.standard_normal((Q, L, K, N)) + 1j * rng.standard_normal((Q, L, K, N)))
    z /= np.sqrt(2.0)
    h = np.einsum("lkmn,qlkn->qlkm", A, z)                        # (Q, L, K, N)
    return np.moveaxis(h, 1, 2).reshape(Q, K, L * N).astype(np.complex128)


class BookChannel:
    """Correlated-Rayleigh channel generator for the monograph's running example.

    Mirrors the interface of :class:`sionna_channel.SionnaUMiChannel` so the two
    backends are interchangeable inside a Monte Carlo loop.

    Args:
        cfg: System configuration, normally from
            :func:`config_cellfree_book.book_config`.
        extras: Book parameters that :class:`DMIMOConfig` has no field for.
    """

    def __init__(self, cfg: DMIMOConfig, extras: BookExtras) -> None:
        self.cfg = cfg
        self.extras = extras

    def generate(self, ap_pos, ue_pos, rng: np.random.Generator):
        """One channel realization for a drop.

        Args:
            ap_pos: AP coordinates ``(L, 2)`` [m].
            ue_pos: UE coordinates ``(K, 2)`` [m].
            rng: Random generator, used for both shadowing and small-scale fading.

        Returns:
            Tuple ``(H, beta)`` with ``H`` shaped ``(Q, K, M_tot)`` (complex128)
            and ``beta`` shaped ``(L, K)`` [linear].
        """
        cfg, extras = self.cfg, self.extras
        d_3d, azimuth, elevation = pair_geometry(cfg, ap_pos, ue_pos, extras.wrap_around)
        beta = _shadowed_beta(cfg, d_3d, ue_pos, rng, extras)
        R = local_scattering_correlation(cfg, azimuth, elevation, beta, extras.asd_deg)
        return correlated_rayleigh(cfg, R, rng), beta
