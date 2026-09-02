"""Parameters of the distributed (cell-free) FR3 power model.

:class:`DMIMOPowerParams` extends the co-located :class:`fr3_power.PowerParams`
rather than restating it, so every hardware constant of the single-site model
(PA, converters, mixers, computational efficiencies, reduction factors) is
inherited and a distributed AP is described by the *same* component models
evaluated at ``M_RF = M``. Only the genuinely new quantities are added here: the
fronthaul, the central unit, the per-AP synchronization, and the functional
split.

The class also owns the consistency hazards that ``../README.md`` warns about,
and resolves them in :meth:`DMIMOPowerParams.from_rate_config`:

* the **frame**: both packages now use the fractions ``tau_DL``, ``tau_DLsig``,
  ``tau_ULsig`` of the system model, so the join copies them across and the SE
  prelog of the rate model is the data fraction of the frame averaging here;
* the **coherence block**, which the co-located model derives from Doppler
  (``upsilon_coh``) and the rate model states outright (``tau_c``);
* the **effective bandwidth** ``B_tilde = 0.9 B``, which carries the delivered
  rate in the rate model and divides it again inside the encoder and decoder
  models here.

Simplifications, as instructed for this evaluation:

* every AP is active, so the deep-sleep set of eq. pnet is empty and
  ``delta_ds``, ``P_AP_0``, ``P_FH_0_sleep`` are not modelled;
* full cell-free service, ``D_l = {1..K}``, so ``K_l = K`` and every AP carries
  the same load, which collapses eq. frameavg_ap onto eq. frameavg;
* only splits S1 and S3 are implemented (see :class:`Split`).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from fr3_power.config import PowerParams


class Split(str, Enum):
    """Functional split: where each digital operation runs (Table `tab:split`).

    ``S1`` forwards precoded samples. The CPU encodes, computes the precoder and
    *applies* it, sending each AP the ``M`` streams of frequency-domain samples
    it must radiate; the AP only does OFDM, predistortion, baseband filtering,
    and conversion. The fronthaul carries samples at the constellation rate,
    independently of the delivered traffic.

    ``S2`` forwards coefficients and data. The CPU encodes and computes
    ``W_l[q]``, but AP ``l`` applies it locally, so the fronthaul carries the
    ``K_l`` data streams plus the precoding coefficients, refreshed once per
    coherence block. Because linear combining is distributive over the APs, an
    AP holding its combining coefficients forms its own partial sum, so S2
    attains the *centralized* uplink rate while forwarding ``K_l`` scalar
    streams instead of ``M`` sample streams.

    ``S3`` is fully distributed (local) operation. The CPU only encodes; each AP
    computes and applies its own precoder from local CSI. The fronthaul carries
    payload in the downlink and per-AP partial sums in the uplink.

    S1 and S2 both realize the centralized design and therefore deliver the
    *same rate*; they differ only in where the work is done and what crosses the
    fronthaul. S3 delivers the lower rate of local processing.
    """

    S1 = "S1"  # CPU computes and applies; fronthaul carries samples
    S2 = "S2"  # CPU computes, AP applies; fronthaul carries data + coefficients
    S3 = "S3"  # AP computes and applies; fronthaul carries data


class PASizing(str, Enum):
    """How the maximum PA output power ``P_max`` relates to the AP budget.

    The two conventions give different scaling laws for the network static floor
    and must not be mixed (Remark `rem:pa_sizing`). ``PER_AP_BUDGET`` sizes each
    PA for its share of the AP budget, ``P_max = rho_max / M``, which makes a
    fully loaded AP consume exactly ``rho_max / eta_PAmax`` and the network floor
    ``L xi rho_max / eta_PAmax``: it grows with the number of APs, not with the
    number of antennas. ``FIXED_RATING`` keeps ``P_max`` at the component rating
    already in :class:`fr3_power.PowerParams`, and the floor becomes
    ``L M xi P_max / eta_PAmax``, growing with the total antenna count.

    ``PER_AP_BUDGET`` is the default because it is the only one that keeps the
    comparison fair as the transmit budget is swept: the amplifiers are assumed
    dimensioned for the budget they are actually given.
    """

    PER_AP_BUDGET = "per-ap-budget"  # P_max = rho_max / M
    FIXED_RATING = "fixed-rating"    # P_max as given in PowerParams


@dataclass
class DMIMOPowerParams(PowerParams):
    """Hardware and model constants of the distributed network.

    Inherits every co-located constant from :class:`fr3_power.PowerParams`; the
    fields below are the distributed additions. Values marked UNSOURCED have no
    counterpart in the co-located model and are not yet backed by a reference.
    They are placeholders chosen to be of the right order, and any conclusion
    that turns on one of them should be reported as a sensitivity range rather
    than a point value. Values marked ASSUMED are fixed by a stated argument
    rather than by a measurement, which is weaker than a citation but stronger
    than a placeholder; the field comments give the argument in each case.
    :func:`unsourced_parameters` lists both at runtime.
    """

    # --- Topology ---------------------------------------------------------
    L: int = 16                 # Number of APs
    M: int = 4                  # Antennas per AP (fully digital, so M_RF = M)
    split: Split = Split.S1

    # --- Supply and cooling of the new nodes (eq. pnet) --------------------
    # Set to the base-station value of the co-located model for want of a
    # separate figure for optical transceivers and for central-unit hardware.
    # They keep distinct fields, since a transceiver in a street cabinet and a
    # central unit in a cooled room are not the same enclosure as an AP, so the
    # assumption can be relaxed without touching the equations.
    eta_FH_sc: float = 0.8      # Fronthaul transceiver supply and cooling  [ASSUMED]
    eta_CPU_sc: float = 0.8     # Central unit supply and cooling           [ASSUMED]

    # --- Fronthaul (eq. fh_basic), parametrisation of ngo2018total ---------
    P_FH_0: float = 0.825       # Traffic-independent power per link [W]
    Pi_FH: float = 0.25e-9      # Traffic-dependent coefficient [W per bit/s] (0.25 W/Gbit/s)
    b_FH: int = 8               # Fronthaul bits per real sample            [UNSOURCED]
    delta_FH_micro: float = 0.25  # Idle reduction factor of a transceiver  [UNSOURCED]

    # --- Central unit (eq. cpu) -------------------------------------------
    # Zero because the baseband hardware of the CPU is already charged
    # elsewhere: eq. pmimo gives it the ceil(L*M / 32) FPGAs of the collective
    # array, and the encoder and decoder carry their own static power inside
    # eq. cpu, so nothing is left over. A non-zero value would add the chassis,
    # host and networking of the central unit, which the co-located model does
    # not charge to its site either. Since only the distributed network has a
    # CPU, zero is the best case for D-MIMO: sweep it to bound the effect.
    P_CPU_0: float = 0.0        # Always-on consumption of the central unit [W]  [ASSUMED]
    # Chains per FPGA on the CPU side. eq. pmimo names Mbar_RF only for the AP;
    # the CPU is taken to process the collective L*M-antenna array, so it runs
    # ceil(L*M / cpu_rf_chains_per_fpga) FPGAs.
    cpu_rf_chains_per_fpga: int = 32

    # --- Per-AP synchronization (eq. pap) ----------------------------------
    # Coherent joint transmission across separate nodes needs a shared frequency
    # and phase reference and reciprocity-calibrated TDD chains. It costs power
    # at every AP continuously, which is why it is a constant rather than a
    # frame-averaged block. A co-located array does not pay it.
    #
    # This is the AP-side reference hardware only: the disciplined oscillator and
    # the PLL slaved to GNSS or to a reference delivered over the fronthaul. The
    # shared grandmaster clock is not per-AP and belongs in ``P_CPU_0``; the
    # over-the-air part of reciprocity calibration is frame time and CPU work,
    # not a continuous per-AP power, and is not modelled (see the gap list in
    # ``pwr_model.tex``).
    #
    # It carries its own supply-and-cooling efficiency rather than borrowing the
    # analog one: an always-on oscillator module is not the RF front end, and
    # letting it inherit eta_ana_sc would fix its overhead by an accident of
    # placement.
    #
    # 0.1 W matches the parameter table of the manuscript, which quotes no source
    # for it. The plausible range for the AP-side reference hardware spans about
    # two orders of magnitude, from a temperature-controlled oscillator slaved
    # over the fronthaul to an oven-controlled one with its own GNSS receiver, so
    # this value sits at the cheap end of it: at L = 32 the term is then a few
    # watts of a network total in the hundreds, whereas the expensive end would
    # be a visible share. Sweep it rather than quoting a point value.
    P_sync: float = 0.1         # Per-AP reference distribution + calibration [W]  [UNSOURCED]
    eta_sync_sc: float = 0.8    # Synchronization supply and cooling        [ASSUMED]

    # --- Power amplifier sizing (Remark rem:pa_sizing) ---------------------
    pa_sizing: PASizing = PASizing.PER_AP_BUDGET

    # --- Coherence block ---------------------------------------------------
    # When set, overrides the Doppler-derived PowerParams.upsilon_coh so that the
    # precoder amortisation and the rate model's pilot overhead use one and the
    # same block length. from_rate_config sets it to DMIMOConfig.tau_c.
    upsilon_coh_override: Optional[float] = None

    def __post_init__(self) -> None:
        self.split = Split(self.split)
        self.pa_sizing = PASizing(self.pa_sizing)
        for name in ("L", "M", "K", "b_FH", "cpu_rf_chains_per_fpga"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("eta_FH_sc", "eta_CPU_sc", "eta_sync_sc"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in (0, 1], got {getattr(self, name)}")

    # --- Derived ----------------------------------------------------------
    @property
    def M_tot(self) -> int:
        """Total number of distributed antennas, L * M."""
        return self.L * self.M

    @property
    def upsilon_coh(self) -> float:
        """Samples per coherence block, overriding the Doppler-derived default.

        :class:`fr3_power.PowerParams` derives this from the delay spread and the
        receiver velocity, while ``D_MIMO_rate`` states ``tau_c`` outright. They
        are the same physical quantity, so letting both stand would make the
        combined model assume two different fading rates: the precoder
        amortisation of eq. gops_cen would use one block length and the pilot
        overhead of the rate model another. ``from_rate_config`` sets the
        override from ``tau_c`` so a single number governs both.
        """
        if self.upsilon_coh_override is not None:
            return float(self.upsilon_coh_override)
        return super().upsilon_coh

    @property
    def cpu_fpgas(self) -> float:
        """FPGAs the CPU runs the centralized precoder on, ceil(L*M / chains)."""
        import numpy as np
        return float(np.ceil(self.M_tot / self.cpu_rf_chains_per_fpga))

    def P_max_for(self, rho_max: float) -> float:
        """Maximum PA output power [W] under the configured sizing convention."""
        if self.pa_sizing is PASizing.PER_AP_BUDGET:
            return rho_max / self.M
        return self.P_max

    # --- Construction from a rate-model configuration ----------------------
    @classmethod
    def from_rate_config(cls, cfg, **overrides) -> "DMIMOPowerParams":
        """Build parameters whose frame and topology match a ``DMIMOConfig``.

        This is the join between the two packages. Both now describe the frame
        in the *same* fractions, so the join is a copy rather than a
        translation: ``tau_DL``, ``tau_DLsig`` and ``tau_ULsig`` carry straight
        across, and the data fraction that the rate model applies as the SE
        prelog, ``tau_i (1 - tau_i,sig) xbar_i``, is by construction the
        fraction that the frame averaging of eq. frameavg charges at the data
        power level. The frame spans one coherence block, so ``upsilon_coh`` is
        pinned to ``tau_c``: one fading timescale governs both the precoder
        amortisation and the signalling overhead of the rate.

        The carrier, bandwidth, subcarrier spacing, effective-bandwidth factor,
        user count and topology are copied across as well, so no hardware or
        band constant can differ between the rate and the power of one point.

        Args:
            cfg: A ``config_dmimo.DMIMOConfig``.
            **overrides: Any field of :class:`DMIMOPowerParams`, applied last.

        Returns:
            Parameters consistent with ``cfg``.
        """
        fields = dict(
            K=cfg.K,
            f_c=cfg.f_c,
            B=cfg.B,
            Delta_f=cfg.Delta_f,
            B_tilde_factor=cfg.B_tilde_factor,
            L=cfg.L,
            M=cfg.M,
            tau_DL=cfg.tau_DL,
            tau_DLsig=cfg.tau_DLsig,
            tau_ULsig=cfg.tau_ULsig,
            upsilon_coh_override=float(cfg.tau_c),
        )
        fields.update(overrides)
        p = cls(**fields)

        # The split and the rate model's cooperation level must agree, or the
        # power of one system would be charged against the rate of another.
        from config_dmimo import OperationMode, PrecodingScheme
        # The operation counts of Table tab:complexity are scheme-dependent, but
        # only the (R)ZF/MMSE family is priced here: eq. gops_split carries the
        # Cholesky count whatever cfg.precoding says. MR needs no factorization
        # at all, so pricing it with this model would charge it for work it does
        # not do.
        if cfg.precoding is PrecodingScheme.MR:
            warnings.warn(
                "precoding=MR has Xi_comp = 0 in Table tab:complexity, but the power "
                "model charges the (R)ZF/MMSE factorization count regardless of the "
                "scheme, so the MIMO-processing power is overstated for MR.",
                stacklevel=2,
            )
        centralized = cfg.operation is OperationMode.CENTRALIZED
        expected = {Split.S1, Split.S2} if centralized else {Split.S3}
        if p.split not in expected:
            names = " or ".join(sorted(s.value for s in expected))
            warnings.warn(
                f"split={p.split.value} but the rate configuration is "
                f"{cfg.operation.value} operation, which is {names}. The power "
                "and the rate would then describe different systems.",
                stacklevel=2,
            )
        return p


#: Parameters with no counterpart in the co-located model and no sourced value.
#: Listed so a report can state plainly which numbers are placeholders.
UNSOURCED = (
    ("b_FH", "fronthaul quantiser resolution; sets the S1 fronthaul rate outright"),
    ("delta_FH_micro", "idle reduction factor of an optical transceiver"),
    ("P_sync", "per-AP reference distribution and reciprocity calibration"),
)

#: Parameters fixed by a stated assumption rather than by a reference. These do
#: have a defensible justification (see the field comments), but they are not
#: measurements, so a conclusion that turns on one should still be reported as a
#: range.
ASSUMED = (
    ("eta_FH_sc", "set to the base-station supply-and-cooling value"),
    ("eta_CPU_sc", "set to the base-station supply-and-cooling value"),
    ("eta_sync_sc", "set to the base-station supply-and-cooling value"),
    ("P_CPU_0", "zero; the CPU baseband static is charged through eq. pmimo"),
)


def unsourced_parameters(p: DMIMOPowerParams) -> str:
    """Human-readable list of the placeholder and assumed parameters."""
    lines = ["Parameters with no sourced value (placeholders; treat as a sensitivity range):"]
    for name, what in UNSOURCED:
        lines.append(f"  {name:<16} = {getattr(p, name):<10} {what}")
    lines.append("Parameters fixed by a stated assumption rather than a reference:")
    for name, what in ASSUMED:
        lines.append(f"  {name:<16} = {getattr(p, name):<10} {what}")
    return "\n".join(lines)
