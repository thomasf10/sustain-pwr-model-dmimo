"""Network power consumption of a distributed (cell-free) FR3 deployment.

Implements eq. pnet of ``dmimo_pwr_model.tex``: the per-AP hardware summed over
the network, plus the fronthaul links and the central unit, which the co-located
model has no counterpart for.

    P_net = sum_l ( P_AP,l + Pbar_FH,l / eta_FH_sc ) + Pbar_CPU / eta_CPU_sc

Under the simplifications adopted here every AP is active, so the deep-sleep sum
of eq. pnet is empty, and every AP serves every user, so ``K_l = K`` and all APs
share the network load ``xbar_i``, which collapses the per-AP frame average of
eq. frameavg_ap onto the co-located :func:`fr3_power.frame_average`.

**What is reused and what is new.** A distributed AP is a small fully-digital
array, ``M_RF = M`` and ``M_PS = 0``. Substituting that into the co-located
analog and PA assemblies reduces them *exactly* to eq. ana_dl_ap / ana_ul_ap and
eq. pa_avg_ap, so :func:`ap_analog` and :func:`ap_pa` call
:func:`fr3_power.power_model.analog` and :func:`fr3_power.power_model.pa`
unchanged, the only addition being the PA sizing convention. What is genuinely
new gets its own term instead of being folded into one of theirs: the per-AP
synchronization of :func:`ap_sync`, the fronthaul, and the central unit. The
digital block cannot be reused wholesale:
the encoder and decoder move to the CPU, and the MIMO-processing term depends on
where the precoder is computed and applied, which is the functional split. That
is the one block re-assembled here, from the same :mod:`fr3_power.components`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from fr3_power import components as comp
from fr3_power.config import OperatingPoint
from fr3_power.frame_average import LoadSplit, frame_average
from fr3_power.power_model import _static_per_fpga
from fr3_power import power_model as fr3_pm

from .config import DMIMOPowerParams, PASizing, Split

DL, UL = "DL", "UL"


# ======================================================================
# MIMO processing: operation counts and their placement (eq. gops_split)
# ======================================================================


def xi_precoder_centralized(p: DMIMOPowerParams) -> float:
    """Ops/sample to compute the centralized precoder (Table tab:complexity).

    The CPU inverts the ``K x K`` Gram matrix of the collective channel
    ``H in C^{LM x K}``, so the co-located Cholesky count applies with the RF
    chain count replaced by the total number of distributed antennas ``LM``.
    Amortised over the ``upsilon_coh`` samples of a coherence block. Written as
    a network total rather than per chain.
    """
    K, LM = p.K, p.M_tot
    return (K ** 3 / 3 + 3 * K ** 2 * LM + K * LM) / p.upsilon_coh


def xi_precoder_local(p: DMIMOPowerParams) -> float:
    """Ops/sample for one AP to compute its local precoder (Table tab:complexity).

    AP ``l`` inverts the ``M x M`` matrix of the local regularized precoder,
    which is the same Cholesky accounting with the antenna and user dimensions
    exchanged. ``K_l = K`` under full cell-free service.
    """
    M, K_l = p.M, p.K
    return (M ** 3 / 3 + 3 * M ** 2 * K_l + M * K_l) / p.upsilon_coh


def xi_ap(p: DMIMOPowerParams, direction: str) -> float:
    """Ops/sample of MIMO processing at one AP (eq. gops_split, AP row).

    ``iota_{S2,S3} 2 K_l M + iota_{S3} Xi_comp_loc``. Under S1 the AP does none:
    the CPU applies the precoder and forwards samples. Under S2 and S3 the AP
    applies the precoder or combiner to its ``K_l`` users over its ``M`` chains,
    ``2 K_l M`` per sample. Only S3 also *computes* it, and unlike the
    centralized pair it computes it **in both directions**: the local downlink
    precoder of eq. local-rzf inverts ``E_l E_l^H + lambda I``, whereas the local
    uplink combiner of eq. ul-local-mmse inverts the power-weighted
    ``E_l P E_l^H + lambda I``. The two are different matrices, so no
    factorization is shared and TDD reciprocity buys nothing here, which is
    exactly the asymmetry with the centralized ZF pair in :func:`xi_cpu`.
    """
    if p.split is Split.S1:
        return 0.0
    apply_cost = 2 * p.K * p.M
    if p.split is Split.S3:
        return apply_cost + xi_precoder_local(p)
    return apply_cost


def xi_cpu(p: DMIMOPowerParams, direction: str) -> float:
    """Ops/sample of MIMO processing at the CPU (eq. gops_split, CPU row).

    ``iota_{S1} 2 K LM + iota_{S1,S2} iota_{DL} Xi_comp_cen``. Under S1 the CPU
    applies the precoder over all ``LM`` distributed antennas, ``2 K LM`` per
    sample, and computes it in the downlink. Under S2 it computes but does not
    apply, so only the downlink computation remains. Under S3 it does neither.

    The factorization is charged to the downlink alone because the centralized
    ZF combiner is the same matrix as the ZF precoder and TDD reciprocity lets
    one factorization serve both directions. The local pair of :func:`xi_ap`
    shares no such matrix and is therefore charged twice.

    Every operation appears exactly once across this row and :func:`xi_ap`, so no
    work is counted twice: under S2 in particular the AP is billed only for
    applying the precoder and the CPU only for computing it.
    """
    if p.split is Split.S3:
        return 0.0
    apply_cost = 2 * p.K * p.M_tot if p.split is Split.S1 else 0.0
    if direction == DL:
        return apply_cost + xi_precoder_centralized(p)
    return apply_cost


def _p_mimo(p: DMIMOPowerParams, xi: float, n_fpgas: float, eta: float) -> float:
    """MIMO-processing power from an operation count (eq. pmimo).

    ``Mbar_RF P_s + f_sI Xi / eta_pre``. The FPGA static term is charged only
    when the node actually performs the processing: under S1 an AP has no
    precoder or combiner block at all, so billing it for an idle FPGA would
    overstate the split it is supposed to be cheap for. (The AP still pays one
    FPGA for its baseband filter, which it runs under every split.) The static
    power per FPGA follows the co-located convention in
    ``fr3_power.power_model._static_per_fpga``, so the two models cannot drift.
    """
    if xi <= 0:
        return 0.0
    return n_fpgas + p.f_sI * xi / eta


# ======================================================================
# Per-AP consumption (eq. pap)
# ======================================================================


def ap_operating_point(p: DMIMOPowerParams, P_T_l: float,
                       xbar_DL: float = 1.0, xbar_UL: float = 1.0) -> OperatingPoint:
    """Operating point of one AP: a fully-digital ``M``-antenna array.

    ``R_DL`` and ``R_UL`` are left at zero because no AP-side block depends on
    them: the encoder and decoder are the only rate-driven components and both
    live at the CPU under every split (Table `tab:split`).
    """
    return OperatingPoint(M_ant=p.M, M_RF=p.M, P_T=P_T_l,
                          R_DL=0.0, R_UL=0.0, xbar_DL=xbar_DL, xbar_UL=xbar_UL)


def ap_digital(p: DMIMOPowerParams, op: OperatingPoint) -> LoadSplit:
    """Frame-averaged digital consumption of one AP (eq. dig_ap).

    Every AP performs OFDM (de)modulation, predistortion of its own PAs, and
    baseband filtering whatever the split; only the MIMO term moves. The
    encoder and decoder are absent here by construction, which is the structural
    difference from :func:`fr3_power.power_model.digital`.
    """
    M = op.M_RF
    static = _static_per_fpga(p, M)

    p_ifft = comp.p_ifft(p, M)          # FFT in the uplink, same cost
    p_dpd = comp.p_dpd(p, M)            # downlink only
    p_fil = comp.p_filter_bb(p, M, static)
    p_mimo_dl = _p_mimo(p, xi_ap(p, DL), static, p.eta_precoder)
    p_mimo_ul = _p_mimo(p, xi_ap(p, UL), static, p.eta_combiner)

    P_dig_DL = p_ifft + p_dpd + p_fil + p_mimo_dl
    P_dig_UL = p_ifft + p_fil + p_mimo_ul

    avg_DL = frame_average(p.tau_DL, p.tau_DLsig, op.xbar_DL,
                           P_dig_DL, P_dig_DL, P_dig_DL,
                           p.delta_dig_micro, p.delta_dig_idle)
    avg_UL = frame_average(p.tau_UL, p.tau_ULsig, op.xbar_UL,
                           P_dig_UL, P_dig_UL, P_dig_UL,
                           p.delta_dig_micro, p.delta_dig_idle)
    return (avg_DL + avg_UL).scaled(1 / p.eta_dig_sc)


def ap_analog(p: DMIMOPowerParams, op: OperatingPoint) -> LoadSplit:
    """Frame-averaged analog consumption of one AP (eq. ana_avg_ap).

    With ``M_RF = M_ant = M`` and no phase shifters, the co-located assembly is
    already eq. ana_dl_ap / ana_ul_ap term for term, so it is reused unchanged.
    What changes character in a distributed deployment is the local oscillator:
    it is no longer shared by the whole array, since every AP needs its own, so
    the network pays ``L P_LO``.

    Synchronization is *not* part of this block; see :func:`ap_sync`.
    """
    return fr3_pm.analog(p, op)


def ap_sync(p: DMIMOPowerParams) -> LoadSplit:
    """Synchronization consumption of one AP (eq. pap, fourth term) [W].

    Coherent joint transmission across physically separate nodes requires a
    shared frequency and phase reference and reciprocity-calibrated TDD chains,
    which a co-located array does not pay for at all. It is charged as a
    constant per AP, wholly load-independent, and it is deliberately *not* run
    through :func:`fr3_power.frame_average.frame_average`: the reference has to
    hold whether or not the frame is carrying data, so there is no operating
    mode in which it drops.

    It is a term of its own rather than an addend inside :func:`ap_analog`
    because it shares none of that block's structure. Every other analog term
    scales with ``M_RF`` or is the single shared LO, and every one of them is
    frame-averaged. Keeping it separate also means it does not silently inherit
    ``eta_ana_sc``, and it makes the reduction of the distributed analog block
    to the co-located one exact rather than conditional on ``P_sync = 0``.
    """
    return LoadSplit(p.P_sync / p.eta_sync_sc, 0.0)


def ap_pa(p: DMIMOPowerParams, op: OperatingPoint, rho_max: float) -> LoadSplit:
    """Frame-averaged PA consumption of one AP (eq. pa_avg_ap).

    The per-AP power is shared equally across its ``M`` antennas, so the
    co-located assembly applies with ``P_a,l = P_T,l / M``. Note this is a
    conservative upper bound rather than an identity: the true per-antenna powers
    differ for any non-isotropic precoder, and because ``P_PA`` is concave in its
    output power, Jensen's inequality makes the equal-split value the larger one.
    It is tight when the precoder loads an AP's antennas uniformly.

    ``rho_max`` is the per-AP budget, needed for the sizing convention of
    Remark `rem:pa_sizing`.
    """
    return fr3_pm.pa(replace(p, P_max=p.P_max_for(rho_max)), op)


# ======================================================================
# Fronthaul (eq. fh_basic, fh_duty, fh_avg, fh_rate_avg)
# ======================================================================


def _rate_average(p: DMIMOPowerParams, direction: str, xbar: float,
                  R_data: float, R_sig: float) -> LoadSplit:
    """Frame-average a peak fronthaul rate with a zero base (eq. fh_rate_avg).

    The base level is zero because an idle link carries no traffic, which also
    makes the average linear in its arguments.
    """
    tau = p.tau_DL if direction == DL else p.tau_UL
    tau_sig = p.tau_DLsig if direction == DL else p.tau_ULsig
    return frame_average(tau, tau_sig, xbar, R_data, R_sig, 0.0,
                         p.delta_dig_micro, p.delta_dig_idle)


def fronthaul_rate(p: DMIMOPowerParams, R_DL_ap: float,
                   xbar_DL: float = 1.0, xbar_UL: float = 1.0) -> LoadSplit:
    """Frame-averaged fronthaul rate carried by one link [bit/s].

    What crosses the link is set by the split, and the splits put the frame
    structure and the traffic dependence in opposite roles (Remark
    `rem:no_double`):

    * **S1** forwards ``M`` streams of frequency-domain samples at the
      constellation rate in both directions, ``2 b_FH M f_sI`` (eq. fh_s1). This
      is a hardware constant with no dependence on the delivered rate at all, so
      the frame structure is the only thing that reduces it.
    * **S2** forwards the payload plus the precoding coefficients, an
      ``M x K_l`` complex matrix per subcarrier refreshed once per coherence
      block, ``(f_sI / upsilon_coh) 2 b_FH M K_l`` (eq. fh_s2). Uplink is the
      same per-AP partial sums as S3.
    * **S3** forwards payload in the downlink and per-AP partial sums,
      ``2 b_FH K_l f_sI``, in the uplink (eq. fh_ul).

    The signalling phase separates the centralized splits from the local one.
    Centralized operation needs the CPU to see the raw received pilots in order
    to estimate the collective channel, so under S1 *and* S2 the link carries
    samples during the uplink signalling phase; under S3 the pilots are consumed
    locally and the signalling load is nil. Centralization therefore imposes a
    sample-level fronthaul during pilot transmission whatever the data-phase
    split.

    The delivered-payload terms must *not* be frame-averaged again: the rate
    model already delivers ``R_DL,k`` with the prelog applied, so charging the
    delivered rate directly is exactly equivalent to averaging the peak rate, and
    doing both would count the frame structure twice.

    Args:
        p: Model parameters.
        R_DL_ap: Delivered downlink rate of the users this AP serves [bit/s],
            i.e. ``sum_{k in D_l} R_DL,k``, which is the network ``R_DL`` under
            full cell-free service. Used by the data-sharing splits S2 and S3.
        xbar_DL, xbar_UL: Network loads.

    Returns:
        The frame-averaged fronthaul rate of one link.
    """
    samples = 2 * p.b_FH * p.M * p.f_sI          # M streams of raw samples
    partial_sums = 2 * p.b_FH * p.K * p.f_sI     # K_l streams of per-AP soft estimates

    if p.split is Split.S1:
        return (_rate_average(p, DL, xbar_DL, samples, samples)
                + _rate_average(p, UL, xbar_UL, samples, samples))

    # Data-sharing splits: the delivered payload crosses in the downlink and the
    # prelog is already in it, so it enters unaveraged.
    payload = LoadSplit(0.0, R_DL_ap)

    if p.split is Split.S2:
        # Precoding coefficients: M x K_l complex entries per subcarrier, once
        # per coherence block. A peak rate, so it is frame-averaged.
        coefficients = p.f_sI / p.upsilon_coh * 2 * p.b_FH * p.M * p.K
        return (payload
                + _rate_average(p, DL, xbar_DL, coefficients, coefficients)
                + _rate_average(p, UL, xbar_UL, partial_sums, samples))

    # S3: pilots are consumed locally, so nothing crosses during signalling.
    return payload + _rate_average(p, UL, xbar_UL, partial_sums, 0.0)


def fronthaul(p: DMIMOPowerParams, R_DL_ap: float,
              xbar_DL: float = 1.0, xbar_UL: float = 1.0) -> LoadSplit:
    """Frame-averaged consumption of one fronthaul link (eq. fh_avg) [W].

    A traffic-independent part covering the optical transceivers at both ends,
    charged once per link with a single duty cycle because the link is
    bidirectional and stays up across the whole frame, plus a part proportional
    to the rate carried.

    How much duty cycling actually saves turns entirely on ``delta_FH_micro``,
    which is a placeholder: a transceiver that can be gated deeply between
    bursts approaches the digital reduction factor, whereas one that must hold
    its laser bias and clock recovery to stay linked sits close to one, and the
    static term is then nearly a constant per active link, so switching APs off
    rather than duty cycling is the only lever on it. Report anything that turns
    on this term as a range.
    """
    theta = sum(
        (p.tau_DL if d == DL else p.tau_UL)
        * ((p.tau_DLsig if d == DL else p.tau_ULsig)
           + (1 - (p.tau_DLsig if d == DL else p.tau_ULsig))
           * (xbar_DL if d == DL else xbar_UL))
        for d in (DL, UL)
    )
    static = (theta + (1 - theta) * p.delta_FH_micro) * p.P_FH_0
    traffic = fronthaul_rate(p, R_DL_ap, xbar_DL, xbar_UL).scaled(p.Pi_FH)
    return (LoadSplit(static, 0.0) + traffic).scaled(1 / p.eta_FH_sc)


# ======================================================================
# Central unit (eq. cpu)
# ======================================================================


def cpu(p: DMIMOPowerParams, R_DL: float, R_UL: float,
        xbar_DL: float = 1.0, xbar_UL: float = 1.0) -> LoadSplit:
    """Frame-averaged consumption of the central unit (eq. cpu) [W].

    Channel encoding and decoding stay at the CPU under every split, since that
    is where the payload enters and leaves the network, and are driven by the
    network sum rates. The MIMO term is present only under S1. The averaging
    uses the network-wide load, not a per-AP one.
    """
    p_enc = comp.p_encoder(p, R_DL)
    p_dec = comp.p_decoder(p, R_UL)
    p_mimo_dl = _p_mimo(p, xi_cpu(p, DL), p.cpu_fpgas, p.eta_precoder)
    p_mimo_ul = _p_mimo(p, xi_cpu(p, UL), p.cpu_fpgas, p.eta_combiner)

    avg_DL = frame_average(p.tau_DL, p.tau_DLsig, xbar_DL,
                           p_enc + p_mimo_dl, p_enc + p_mimo_dl, p_enc + p_mimo_dl,
                           p.delta_dig_micro, p.delta_dig_idle)
    avg_UL = frame_average(p.tau_UL, p.tau_ULsig, xbar_UL,
                           p_dec + p_mimo_ul, p_dec + p_mimo_ul, p_dec + p_mimo_ul,
                           p.delta_dig_micro, p.delta_dig_idle)
    always_on = LoadSplit(p.P_CPU_0, 0.0)
    return (avg_DL + avg_UL + always_on).scaled(1 / p.eta_CPU_sc)


# ======================================================================
# Network assembly (eq. pnet)
# ======================================================================


@dataclass
class NetworkBreakdown:
    """Consumption of one distributed operating point [W], by block.

    The AP blocks are already summed over the ``L`` APs, as is the fronthaul
    over the ``L`` links.
    """

    ap_digital: LoadSplit
    ap_analog: LoadSplit
    ap_pa: LoadSplit
    ap_sync: LoadSplit
    fronthaul: LoadSplit
    cpu: LoadSplit
    ap_tx_power: np.ndarray   # per-AP radiated power [W], shape (L,)
    rho_max: float            # per-AP budget [W]

    @property
    def ap_total(self) -> float:
        """Consumption of all APs (eq. pap summed over the network) [W]."""
        return (self.ap_digital.total + self.ap_analog.total
                + self.ap_pa.total + self.ap_sync.total)

    @property
    def total(self) -> float:
        """Total network consumption P_net (eq. pnet) [W]."""
        return self.ap_total + self.fronthaul.total + self.cpu.total

    @property
    def utilisation(self) -> np.ndarray:
        """Per-AP power utilisation u_l = P_T,l / rho_max in [0, 1] (eq. util)."""
        return self.ap_tx_power / self.rho_max

    def summary(self) -> str:
        """Human-readable breakdown."""
        t = self.total
        rows = (("AP digital", self.ap_digital.total),
                ("AP analog", self.ap_analog.total),
                ("AP PA", self.ap_pa.total),
                ("AP sync", self.ap_sync.total),
                ("fronthaul", self.fronthaul.total),
                ("CPU", self.cpu.total))
        lines = [f"P_net = {t:.1f} W"]
        for name, v in rows:
            lines.append(f"  {name:<12} {v:8.2f} W  ({100 * v / t:5.1f} %)")
        u = self.utilisation
        lines.append(f"  mean/max AP utilisation u_l : {u.mean():.3f} / {u.max():.3f}")
        return "\n".join(lines)


def compute_network(p: DMIMOPowerParams, ap_tx_power, rho_max: float,
                    R_DL: float, R_UL: float,
                    xbar_DL: float = 1.0, xbar_UL: float = 1.0) -> NetworkBreakdown:
    """Total consumption of the distributed network (eq. pnet).

    Args:
        p: Model parameters.
        ap_tx_power: Radiated power of each AP ``(L,)`` [W], i.e. ``P_T,l`` of
            eq. ptl. This comes straight from ``DownlinkResult.ap_power``, so
            the power model and the rate model agree by construction on how much
            each AP actually transmits: local operation meets its budget with
            equality, whereas centralized operation only brings the busiest AP
            to ``rho_max``.
        rho_max: Per-AP transmit power budget [W].
        R_DL, R_UL: Delivered network sum rates [bit/s] from the rate model.
        xbar_DL, xbar_UL: Network physical resource loads in [0, 1].

    Returns:
        A :class:`NetworkBreakdown`.
    """
    ap_tx_power = np.asarray(ap_tx_power, dtype=float)
    if ap_tx_power.shape != (p.L,):
        raise ValueError(
            f"ap_tx_power must have shape (L,)=({p.L},), got {ap_tx_power.shape}"
        )
    if np.any(ap_tx_power > rho_max * (1 + 1e-9)):
        raise ValueError(
            f"an AP radiates {ap_tx_power.max():.4f} W, above its budget rho_max="
            f"{rho_max:.4f} W; the rate and power models disagree on the constraint"
        )

    dig = ana = pa_ = syn = LoadSplit(0.0, 0.0)
    for P_T_l in ap_tx_power:
        op = ap_operating_point(p, float(P_T_l), xbar_DL, xbar_UL)
        dig = dig + ap_digital(p, op)
        ana = ana + ap_analog(p, op)
        pa_ = pa_ + ap_pa(p, op, rho_max)
        syn = syn + ap_sync(p)

    # Every AP serves every user, so each link carries the whole downlink
    # payload under a data-sharing split: this duplication is what
    # Remark rem:fh_ceiling identifies as the ceiling on cell-free efficiency.
    link = fronthaul(p, R_DL, xbar_DL, xbar_UL)
    fh = LoadSplit(link.load_ind * p.L, link.load_dep * p.L)

    return NetworkBreakdown(ap_digital=dig, ap_analog=ana, ap_pa=pa_, ap_sync=syn,
                            fronthaul=fh, cpu=cpu(p, R_DL, R_UL, xbar_DL, xbar_UL),
                            ap_tx_power=ap_tx_power, rho_max=rho_max)


def compute_colocated(p: DMIMOPowerParams, P_T: float, R_DL: float, R_UL: float,
                      xbar_DL: float = 1.0, xbar_UL: float = 1.0):
    """Consumption of the co-located baseline, a single ``L*M``-antenna site.

    This is the unmodified co-located model of ``../FR3_power_model``: one site,
    one shared local oscillator, no fronthaul, no central unit, and no
    synchronization power. It is called through the same parameter object so the
    two deployments cannot silently differ in a hardware constant, with only the
    PA sizing convention applied on top.

    The array is fully digital (``M_RF = M_ant``) to match the rate model, which
    has no hybrid beamforming.

    Under the ``PER_AP_BUDGET`` sizing convention each PA is dimensioned for its
    share of the site budget, ``P_max = P_T / (L M)``. At equal total transmit
    power this is the same rating the distributed APs get,
    ``rho_max / M = P_T / (L M)``, so the two deployments are compared with
    identical amplifiers and differ only in how many sites they sit on.

    Returns:
        A :class:`fr3_power.power_model.PowerBreakdown`.
    """
    op = OperatingPoint(M_ant=p.M_tot, M_RF=p.M_tot, P_T=P_T,
                        R_DL=R_DL, R_UL=R_UL, xbar_DL=xbar_DL, xbar_UL=xbar_UL)
    P_max = P_T / p.M_tot if p.pa_sizing is PASizing.PER_AP_BUDGET else p.P_max
    return fr3_pm.compute(replace(p, P_max=P_max), op)


def energy_efficiency(R_DL: float, R_UL: float, total_power: float) -> float:
    """Network energy efficiency (eq. ee_net) [bit/J].

    ``(R_DL + R_UL) / P_net`` with delivered rates, so no prelog is applied here.
    """
    return (R_DL + R_UL) / total_power
