"""Sanity checks for the distributed massive MIMO rate model.

Run this after changing the config, channel model, or precoding to confirm the
building blocks still behave. It checks:

* config: derived quantities, path-loss model, enum coercion;
* signal processing: effective channel, SINR, spectral efficiency, AP powers
  (against hand computations on controlled inputs);
* precoding: exact algebraic invariants of MR/ZF/RZF/MMSE and the local
  L-RZF/L-MMSE schemes, plus that unimplemented schemes raise;
* power control: centralized (K,) and distributed (L,K) allocations and that the
  per-AP power budget holds after normalization;
* channel model: Sionna UMi shapes, beta consistency, and reproducibility
  (skipped with a note if Sionna is not installed);
* end-to-end: one realization run through the power-control pipeline
  precoding -> normalize -> SINR -> SE, checking finite positive SE.

Usage (from this directory, with the project venv):

    python sanity_checks.py

Exits non-zero if any check fails.
"""

from __future__ import annotations

import sys

import numpy as np

from config_dmimo import (
    ChannelModel,
    DMIMOConfig,
    OperationMode,
    PrecodingScheme,
    dbm_to_watt,
)
import mimo_helpers as mh


class Checks:
    """Minimal pass/fail harness."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.failures: list[str] = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> None:
        ok = bool(ok)
        tag = "PASS" if ok else "FAIL"
        line = f"  [{tag}] {name}"
        if detail:
            line += f"  ({detail})"
        print(line)
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            self.failures.append(name)

    def section(self, title: str) -> None:
        print(f"\n=== {title} ===")

    def summary(self) -> int:
        print(f"\n{self.passed} passed, {self.failed} failed")
        if self.failures:
            print("FAILED: " + ", ".join(self.failures))
        return 1 if self.failed else 0


def synthetic_channel(cfg: DMIMOConfig, seed: int = 1) -> np.ndarray:
    """Well-conditioned complex Gaussian channel ``(Q, K, M_tot)`` for algebra checks."""
    rng = np.random.default_rng(seed)
    shape = (cfg.Q, cfg.K, cfg.M_tot)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def interference_to_desired(H: np.ndarray, W: np.ndarray) -> float:
    """Aggregate off-diagonal / diagonal power ratio of the effective channel."""
    G = mh.effective_channel(H, W)
    desired = np.abs(np.diagonal(G, axis1=1, axis2=2)) ** 2
    off = (np.abs(G) ** 2).sum(axis=2) - desired
    return float(off.mean() / desired.mean())


def check_config(chk: Checks) -> None:
    chk.section("Config")
    cfg = DMIMOConfig(L=6, M=4, K=4, Q=8)

    chk("M_tot = L * M", cfg.M_tot == cfg.L * cfg.M, f"{cfg.M_tot}")
    chk("noise_power matches dBm->W", np.isclose(cfg.noise_power, dbm_to_watt(cfg.noise_power_dBm)))
    chk("dl_prelog = (tau_c - tau_p)/tau_c",
        np.isclose(cfg.dl_prelog, (cfg.tau_c - cfg.tau_p) / cfg.tau_c))
    chk("rzf_regularization defaults to sigma^2",
        np.isclose(cfg.rzf_regularization, cfg.noise_power))

    # Path loss: equals the reference loss at d0, and increases with distance.
    chk("path_loss_dB(d0) = pathloss_ref_loss_dB",
        np.isclose(float(cfg.path_loss_dB(cfg.ref_distance)), cfg.pathloss_ref_loss_dB))
    chk("path_loss increases with distance",
        cfg.path_loss_dB(100.0) > cfg.path_loss_dB(10.0) > cfg.path_loss_dB(1.0))
    chk("path_loss floors at min_ap_ue_distance",
        np.isclose(float(cfg.path_loss_dB(0.01)), float(cfg.path_loss_dB(cfg.min_ap_ue_distance))))

    # String -> enum coercion.
    cfg2 = DMIMOConfig(L=6, M=4, K=4, Q=8, precoding="L-RZF", operation="distributed",
                       channel_model="rayleigh")
    chk("enum coercion (precoding/operation/channel_model)",
        cfg2.precoding is PrecodingScheme.L_RZF
        and cfg2.operation is OperationMode.DISTRIBUTED
        and cfg2.channel_model is ChannelModel.RAYLEIGH)


def check_signal_processing(chk: Checks) -> None:
    chk.section("Signal processing")
    rng = np.random.default_rng(0)

    # effective_channel: G[q,k,i] = h_k^H w_i, matched against an explicit einsum.
    H = (rng.standard_normal((3, 4, 5)) + 1j * rng.standard_normal((3, 4, 5)))
    W = (rng.standard_normal((3, 5, 4)) + 1j * rng.standard_normal((3, 5, 4)))
    G = mh.effective_channel(H, W)
    G_ref = np.einsum("qkm,qmi->qki", np.conj(H), W)
    chk("effective_channel = conj(H) @ W", np.allclose(G, G_ref), f"shape {G.shape}")

    # downlink_sinr on a controlled 2-user G.
    sigma2 = 0.5
    Gc = np.array([[[2.0 + 0j, 1.0 + 0j], [0.5 + 0j, 3.0 + 0j]]])  # (1, 2, 2)
    sinr = mh.downlink_sinr(Gc, sigma2)
    exp0 = 2.0 ** 2 / (1.0 ** 2 + sigma2)
    exp1 = 3.0 ** 2 / (0.5 ** 2 + sigma2)
    chk("downlink_sinr matches |g_kk|^2/(sum interf + sigma^2)",
        np.allclose(sinr[0], [exp0, exp1]))
    chk("SINR non-negative", np.all(sinr >= 0))

    # spectral_efficiency = prelog * mean_q log2(1 + sinr).
    sinr2 = np.array([[1.0, 3.0], [7.0, 0.0]])  # (Q=2, K=2)
    se = mh.spectral_efficiency(sinr2, prelog=0.9)
    se_ref = 0.9 * np.log2(1 + sinr2).mean(axis=0)
    chk("spectral_efficiency = prelog * mean_q log2(1+SINR)", np.allclose(se, se_ref))

    # ap_powers: for all-ones precoders each AP radiates M * Q * K.
    cfg = DMIMOConfig(L=6, M=4, K=4, Q=8)
    W1 = np.ones((cfg.Q, cfg.M_tot, cfg.K), dtype=complex)
    p = mh.ap_powers(cfg, W1)
    chk("ap_powers sums |w|^2 per AP", np.allclose(p, cfg.M * cfg.Q * cfg.K),
        f"per-AP {p[0]:.0f}")


def check_precoding(chk: Checks) -> None:
    chk.section("Precoding (exact invariants on a synthetic channel)")
    cfg = DMIMOConfig(L=6, M=4, K=4, Q=8)   # M_tot=24 >= K, ZF well-posed
    H = synthetic_channel(cfg)
    Hh = H  # perfect CSI

    local = {"MR", "L-RZF", "L-MMSE", "LP-MMSE"}  # distributed (per-AP) schemes

    def directions(scheme, **over):
        c = DMIMOConfig(L=6, M=4, K=4, Q=8, precoding=scheme,
                        operation="distributed" if scheme in local else "centralized",
                        **over)
        return mh.precoding_directions(c, Hh)

    # MR direction is exactly h_k (columns of E).
    Wmr = directions("MR")
    chk("MR: wbar_k = h_k", np.allclose(Wmr, H.transpose(0, 2, 1)))

    # ZF nulls inter-user interference and matches the explicit pseudoinverse.
    Wzf = directions("ZF")
    chk("ZF nulls interference", interference_to_desired(H, Wzf) < 1e-18,
        f"ratio {interference_to_desired(H, Wzf):.1e}")
    E0 = Hh[0].T
    Wzf0 = E0 @ np.linalg.inv(E0.conj().T @ E0)
    chk("ZF matches explicit E (E^H E)^-1", np.allclose(Wzf[0], Wzf0))

    # RZF == MMSE at the default (sigma^2) loading; they differ when RZF is retuned.
    chk("RZF == MMSE at default loading", np.allclose(directions("RZF"), directions("MMSE")))
    chk("RZF != ZF with a large loading",
        not np.allclose(directions("RZF", rzf_reg=1.0), Wzf))

    # Local schemes: L-MMSE == L-RZF at default loading.
    chk("L-MMSE == L-RZF at default loading",
        np.allclose(directions("L-MMSE"), directions("L-RZF")))

    # Locality: zeroing AP j's channel block only affects that AP's precoder rows.
    j, M = 2, cfg.M
    Wloc = directions("L-RZF")
    H2 = H.copy()
    H2[:, :, j * M:(j + 1) * M] = 0.0
    c2 = DMIMOConfig(L=6, M=4, K=4, Q=8, precoding="L-RZF", operation="distributed")
    Wloc2 = mh.precoding_directions(c2, H2)
    other = np.r_[0:j * M, (j + 1) * M:cfg.M_tot]
    chk("local: other AP blocks unchanged when AP j zeroed",
        np.allclose(Wloc[:, other, :], Wloc2[:, other, :]))
    chk("local: zeroed AP block gives zero precoder",
        np.allclose(Wloc2[:, j * M:(j + 1) * M, :], 0.0))

    # Unimplemented schemes must raise.
    for scheme in ("P-MMSE", "P-RZF", "LP-MMSE"):
        op = "distributed" if scheme.startswith("L") else "centralized"
        c = DMIMOConfig(L=6, M=4, K=4, Q=8, precoding=scheme, operation=op)
        try:
            mh.precoding_directions(c, Hh)
            chk(f"{scheme} raises NotImplementedError", False, "did not raise")
        except NotImplementedError:
            chk(f"{scheme} raises NotImplementedError", True)


def check_power_control(chk: Checks) -> None:
    chk.section("Power control + precoder normalization")
    rng = np.random.default_rng(3)
    tol = 1 + 1e-9

    def make_beta(cfg: DMIMOConfig) -> np.ndarray:
        """Strictly-positive large-scale fading ``(L, K)`` spread over ~40 dB."""
        return 10.0 ** rng.uniform(-11.0, -7.0, size=(cfg.L, cfg.K))

    # Centralized: rho is per-user (K,), sums to L*rho_max, and after the global
    # scaling every AP is within budget with the busiest one exactly at rho_max.
    for alloc in ("equal", "fractional"):
        cfg = DMIMOConfig(L=6, M=4, K=4, Q=8, precoding="ZF", operation="centralized",
                          power_alloc=alloc, channel_model="rayleigh")
        H = synthetic_channel(cfg)
        beta = make_beta(cfg)
        Wbar = mh.precoding_directions(cfg, H)
        rho = mh.power_control(cfg, beta, Wbar)
        chk(f"centralized {alloc}: rho shape (K,)", rho.shape == (cfg.K,))
        chk(f"centralized {alloc}: sum rho = L*rho_max",
            np.isclose(rho.sum(), cfg.L * cfg.rho_max))
        W = mh.normalize_precoder(cfg, Wbar, rho)
        Pl = mh.ap_powers(cfg, W)
        chk(f"centralized {alloc}: per-AP budget met",
            np.all(Pl <= cfg.rho_max * tol), f"max {Pl.max():.4f} W <= {cfg.rho_max}")
        chk(f"centralized {alloc}: busiest AP hits rho_max", np.isclose(Pl.max(), cfg.rho_max))

    # Distributed: rho is per-AP-per-user (L,K); the local fractional rule spends
    # each AP's full budget, so per-AP powers equal rho_max after normalization.
    cfg = DMIMOConfig(L=6, M=4, K=4, Q=8, precoding="L-RZF", operation="distributed",
                      power_alloc="fractional", channel_model="rayleigh")
    H = synthetic_channel(cfg)
    beta = make_beta(cfg)
    Wbar = mh.precoding_directions(cfg, H)
    rho = mh.power_control(cfg, beta, Wbar)
    chk("distributed: rho shape (L,K)", rho.shape == (cfg.L, cfg.K))
    chk("distributed: sum_k rho_kl = rho_max per AP", np.allclose(rho.sum(axis=1), cfg.rho_max))
    W = mh.normalize_precoder(cfg, Wbar, rho)
    Pl = mh.ap_powers(cfg, W)
    chk("distributed: per-AP budget met with equality", np.allclose(Pl, cfg.rho_max),
        f"per-AP {Pl.min():.4f}..{Pl.max():.4f} W")

    # v = 0 fractional collapses to equal per-user power.
    ce = DMIMOConfig(L=6, M=4, K=4, Q=8, precoding="ZF", operation="centralized",
                     power_alloc="fractional", v=0.0, channel_model="rayleigh")
    rho0 = mh.power_control(ce, make_beta(ce), None)
    chk("v=0 fractional == equal power", np.allclose(rho0, ce.L * ce.rho_max / ce.K))

    # Equal power is centralized-only: EQUAL + distributed is rejected at build.
    try:
        DMIMOConfig(L=6, M=4, K=4, Q=8, precoding="L-RZF", operation="distributed",
                    power_alloc="equal", channel_model="rayleigh")
        chk("EQUAL + distributed rejected", False, "did not raise")
    except ValueError:
        chk("EQUAL + distributed rejected", True)

    # Fractional rule is exactly rho_k proportional to beta_k^v (centralized) and
    # rho_kl proportional to beta_kl^v per AP (distributed), against hand algebra.
    beta_fix = np.array([[1e-9, 4e-9, 2e-9],
                         [3e-9, 1e-9, 5e-9],
                         [2e-9, 2e-9, 1e-9]])   # (L=3, K=3)
    cf = DMIMOConfig(L=3, M=2, K=3, Q=4, precoding="ZF", operation="centralized",
                     power_alloc="fractional", v=0.6, channel_model="rayleigh")
    beta_k = beta_fix.sum(axis=0)
    ref_c = (cf.L * cf.rho_max) * beta_k ** cf.v / (beta_k ** cf.v).sum()
    chk("centralized fractional = P_tot*beta_k^v / sum_i beta_i^v",
        np.allclose(mh.power_control(cf, beta_fix, None), ref_c))
    df = DMIMOConfig(L=3, M=2, K=3, Q=4, precoding="L-RZF", operation="distributed",
                     power_alloc="fractional", v=0.6, channel_model="rayleigh")
    ref_l = df.rho_max * beta_fix ** df.v / (beta_fix ** df.v).sum(axis=1, keepdims=True)
    chk("local fractional = rho_max*beta_kl^v / sum_i beta_il^v",
        np.allclose(mh.power_control(df, beta_fix, None), ref_l))

    # Sign of v: v>0 favours the stronger user, v<0 the weaker, v=0 is equal.
    beta_mono = np.array([[9e-9, 1e-9]])        # (L=1, K=2): user 0 strong
    def frac_rho(v):
        c = DMIMOConfig(L=1, M=2, K=2, Q=4, precoding="ZF", operation="centralized",
                        power_alloc="fractional", v=v, channel_model="rayleigh")
        return mh.power_control(c, beta_mono, None)
    chk("v>0 favours the stronger user", frac_rho(1.0)[0] > frac_rho(1.0)[1])
    chk("v<0 favours the weaker user", frac_rho(-1.0)[0] < frac_rho(-1.0)[1])
    chk("v=0 gives equal power", np.isclose(*frac_rho(0.0)))

    # Centralized global scaling preserves the per-user power ratios set by rho.
    cr = DMIMOConfig(L=6, M=4, K=4, Q=8, precoding="ZF", operation="centralized",
                     power_alloc="fractional", v=0.7, channel_model="rayleigh")
    Wbar = mh.precoding_directions(cr, synthetic_channel(cr))
    rho = mh.power_control(cr, make_beta(cr), Wbar)
    W = mh.normalize_precoder(cr, Wbar, rho)
    p_user = (np.abs(W) ** 2).sum(axis=(0, 1))  # (K,) realized per-user power
    chk("centralized normalization preserves per-user power ratios",
        np.allclose(p_user / p_user.sum(), rho / rho.sum()))

    # normalize_precoder guards a zero-energy direction (no nan/inf) and rejects
    # a wrongly-shaped rho.
    cg = DMIMOConfig(L=6, M=4, K=4, Q=8, precoding="ZF", operation="centralized",
                     power_alloc="equal", channel_model="rayleigh")
    Wz = mh.precoding_directions(cg, synthetic_channel(cg))
    Wz[:, :, 0] = 0.0                            # user 0 has no direction
    Wn = mh.normalize_precoder(cg, Wz, mh.power_control(cg, make_beta(cg), Wz))
    chk("normalize guards zero-energy direction (all finite)", np.all(np.isfinite(Wn)))
    chk("zero-energy user radiates nothing", np.allclose(Wn[:, :, 0], 0.0))
    chk("budget still met with a dead user", np.all(mh.ap_powers(cg, Wn) <= cg.rho_max * tol))
    try:
        mh.normalize_precoder(cg, Wz, np.zeros((cg.L + 1, cg.K)))
        chk("normalize rejects a bad rho shape", False, "did not raise")
    except ValueError:
        chk("normalize rejects a bad rho shape", True)


def check_channel_and_e2e(chk: Checks) -> None:
    chk.section("Sionna channel model + end-to-end")
    cfg = DMIMOConfig(L=6, M=4, K=4, Q=8, channel_model="sionna-umi",
                      precoding="ZF", n_realizations=1)

    def realize(config):
        rng = np.random.default_rng(config.seed)
        channel = mh.build_channel(config)
        ap, ue = mh.draw_positions(config, rng)
        H, beta = mh.channel_realization(config, channel, ap, ue, rng)
        return ap, ue, H, beta

    try:
        ap, ue, H, beta = realize(cfg)
    except ModuleNotFoundError as exc:
        chk("Sionna available", False, f"skipped: {exc.name} not installed")
        return

    chk("draw_positions shapes", ap.shape == (cfg.L, 2) and ue.shape == (cfg.K, 2))
    chk("positions within coverage area",
        ap.min() >= 0 and ap.max() < cfg.area_size and ue.min() >= 0 and ue.max() < cfg.area_size)
    chk("H shape/dtype", H.shape == (cfg.Q, cfg.K, cfg.M_tot) and H.dtype == np.complex128,
        f"{H.shape} {H.dtype}")
    chk("beta shape", beta.shape == (cfg.L, cfg.K))
    chk("beta strictly positive", np.all(beta > 0))

    beta_dB = 10 * np.log10(beta)
    chk("beta in a physical range (-200..-30 dB)",
        beta_dB.min() > -200 and beta_dB.max() < -30,
        f"{beta_dB.min():.0f}..{beta_dB.max():.0f} dB")

    # beta[l,k] equals the mean per-antenna gain of that AP's H block.
    lk_l, lk_k, M = 3, 2, cfg.M
    blk = H[:, lk_k, lk_l * M:(lk_l + 1) * M]
    chk("beta consistent with H blocks",
        np.isclose(beta[lk_l, lk_k], (np.abs(blk) ** 2).mean()))

    # Reproducibility: same seed -> identical channel.
    _, _, H2, _ = realize(cfg)
    chk("reproducible channel for a fixed seed", np.array_equal(H, H2))

    # End-to-end signal chain through the real power-control pipeline.
    Wbar = mh.precoding_directions(cfg, mh.estimate_channels(cfg, None, H))
    rho = mh.power_control(cfg, beta, Wbar)
    W = mh.normalize_precoder(cfg, Wbar, rho)
    budget_ok = np.all(mh.ap_powers(cfg, W) <= cfg.rho_max * (1 + 1e-9))
    chk("per-AP power budget met after normalization", budget_ok)

    sinr = mh.downlink_sinr(mh.effective_channel(H, W), cfg.noise_power)
    se = mh.spectral_efficiency(sinr, cfg.dl_prelog)
    chk("end-to-end SE finite and positive",
        np.all(np.isfinite(se)) and np.all(se > 0),
        f"sum SE {se.sum():.2f} bit/s/Hz")

    # Driver: simulate_downlink averages the pipeline over realizations and
    # reports the ergodic SE and mean per-AP power.
    from dl_rate import simulate_downlink

    cfg_run = DMIMOConfig(L=6, M=4, K=4, Q=8, channel_model="sionna-umi",
                          precoding="ZF", operation="centralized", n_realizations=3)
    res = simulate_downlink(cfg_run)
    chk("simulate_downlink SE shape (K,)", res.se_per_user.shape == (cfg_run.K,))
    chk("simulate_downlink SE finite and positive",
        np.all(np.isfinite(res.se_per_user)) and np.all(res.se_per_user > 0),
        f"sum SE {res.sum_se:.2f} bit/s/Hz")
    chk("simulate_downlink mean AP power within budget",
        np.all(res.ap_power <= cfg_run.rho_max * (1 + 1e-9)),
        f"max {res.ap_power.max():.4f} W")
    chk("sum_rate = sum_se * B", np.isclose(res.sum_rate, res.sum_se * cfg_run.B))
    chk("se_samples shape (n, K)",
        res.se_samples.shape == (cfg_run.n_realizations, cfg_run.K))
    chk("ergodic SE = mean of se_samples",
        np.allclose(res.se_per_user, res.se_samples.mean(axis=0)))
    chk("se_5pct <= se_median", res.se_5pct <= res.se_median)


def main() -> int:
    chk = Checks()
    check_config(chk)
    check_signal_processing(chk)
    check_precoding(chk)
    check_power_control(chk)
    check_channel_and_e2e(chk)
    return chk.summary()


if __name__ == "__main__":
    sys.exit(main())
