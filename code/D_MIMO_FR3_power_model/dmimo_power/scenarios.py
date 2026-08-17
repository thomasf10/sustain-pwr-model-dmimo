"""The three deployments being compared, and their link to the rate model.

Defines one scenario per MIMO configuration, all covering the same area with the
same number of antennas, the same users, and the same *total* transmit power, so
that the only things that differ are where the antennas sit and how the spatial
processing is split:

    COLOCATED    L = 1 site of L0*M antennas at the area centre, centralized ZF
    CENTRALIZED  L0 APs of M antennas, centralized ZF        -> split S1
    DISTRIBUTED  L0 APs of M antennas, local L-MMSE / L-RZF  -> split S3

Holding the total transmit power fixed means the per-AP budget is
``rho_max = P_budget / L``, so the co-located site and the distributed network
radiate the same total power and, under the default PA sizing, carry amplifiers
of the same rating. The co-located baseline is drawn from the *same* rate code
as the distributed ones, so the comparison is controlled: identical 38.901 UMi
channel model, identical K, B, carrier, coherence block, and rate expression.

Running the Monte Carlo is the slow part, so :func:`rates_for` caches every
(scenario, budget) result in ``data/rates_dmimo.json`` keyed by a hash of the
parameters that affect it. Delete that file to force a recomputation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from config_dmimo import APPlacement, DMIMOConfig
from dl_rate import simulate_downlink
from ul_rate import simulate_uplink

from .config import DMIMOPowerParams, Split

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "rates_dmimo.json"


class RateFamily(str, Enum):
    """Configurations of the *rate* model, which is blind to the functional split.

    A split moves work between the AP and the CPU and changes what crosses the
    fronthaul, but S1 and S2 both realize the centralized precoder and therefore
    deliver the same rate. Keying the Monte Carlo on this rather than on the
    deployment means the expensive part is computed once and shared.
    """

    COLOCATED = "colocated"      # one site holding every antenna
    CENTRALIZED = "centralized"  # distributed APs, joint design from global CSI
    LOCAL = "local"              # distributed APs, per-AP design from local CSI


class Deployment(str, Enum):
    """The MIMO configurations under comparison."""

    COLOCATED = "colocated"              # one site, all antennas together
    CENTRALIZED_S1 = "centralized-s1"    # CPU computes and applies; samples on the fronthaul
    CENTRALIZED_S2 = "centralized-s2"    # CPU computes, AP applies; data + coefficients
    DISTRIBUTED_S3 = "distributed-s3"    # AP computes and applies; data only

    @property
    def label(self) -> str:
        return {
            Deployment.COLOCATED: "co-located MIMO",
            Deployment.CENTRALIZED_S1: "D-MIMO, centralized (S1)",
            Deployment.CENTRALIZED_S2: "D-MIMO, centralized (S2)",
            Deployment.DISTRIBUTED_S3: "D-MIMO, distributed (S3)",
        }[self]

    @property
    def split(self) -> Optional[Split]:
        """Functional split, or ``None`` for the co-located site (no fronthaul)."""
        return {
            Deployment.COLOCATED: None,
            Deployment.CENTRALIZED_S1: Split.S1,
            Deployment.CENTRALIZED_S2: Split.S2,
            Deployment.DISTRIBUTED_S3: Split.S3,
        }[self]

    @property
    def rate_family(self) -> RateFamily:
        """Which rate computation this deployment needs."""
        return {
            Deployment.COLOCATED: RateFamily.COLOCATED,
            Deployment.CENTRALIZED_S1: RateFamily.CENTRALIZED,
            Deployment.CENTRALIZED_S2: RateFamily.CENTRALIZED,
            Deployment.DISTRIBUTED_S3: RateFamily.LOCAL,
        }[self]

    @property
    def is_distributed(self) -> bool:
        """True when the deployment has APs, a fronthaul, and a central unit."""
        return self is not Deployment.COLOCATED


@dataclass
class Scenario:
    """A deployment family, before a transmit budget is chosen.

    Args:
        L: Number of APs of the distributed deployments. The co-located
            baseline uses one site of ``L * M`` antennas, so every deployment
            has the same total antenna count.
        M: Antennas per AP.
        K: Users.
        Q: OFDM subcarriers evaluated.
        area_size: Coverage-area side [m].
        tau_c, tau_p, tau_u: Coherence-block split [samples]. ``tau_u > 0`` is
            required for a non-zero uplink rate.
        n_realizations: Monte Carlo drops per point.
        seed: RNG seed; shared by all deployments so they see the same UE drops.
    """

    L: int = 16
    M: int = 4
    K: int = 10
    Q: int = 16
    area_size: float = 200.0
    tau_c: int = 200
    tau_p: int = 20
    tau_u: int = 90
    n_realizations: int = 20
    seed: int = 0

    @property
    def M_tot(self) -> int:
        return self.L * self.M

    def with_aps(self, L: int) -> "Scenario":
        """Same total antenna count, redistributed over ``L`` APs.

        This is the axis of the fixed-power sweep: chopping one array of
        ``M_tot`` antennas into ``L`` arrays of ``M_tot / L`` holds the radiating
        hardware constant and varies only how finely it is spread over the area.
        ``L`` must divide ``M_tot``, since a fractional antenna has no meaning.
        """
        if self.M_tot % L:
            raise ValueError(
                f"L={L} does not divide the total antenna count M_tot={self.M_tot}; "
                f"choose L from {sorted(d for d in range(1, self.M_tot + 1) if self.M_tot % d == 0)}"
            )
        return replace(self, L=L, M=self.M_tot // L)

    def rate_config(self, deployment: Deployment, P_budget: float) -> DMIMOConfig:
        """Rate-model configuration for one deployment at one transmit budget.

        The total radiated power is ``P_budget`` in every case: the co-located
        site gets it all, and each of the ``L`` APs gets ``P_budget / L``.
        Deployments sharing a :class:`RateFamily` get identical configurations,
        which is what lets S1 and S2 share one Monte Carlo run.
        """
        common = dict(K=self.K, Q=self.Q, area_size=self.area_size,
                      tau_c=self.tau_c, tau_p=self.tau_p, tau_u=self.tau_u,
                      n_realizations=self.n_realizations, seed=self.seed)
        family = deployment.rate_family

        if family is RateFamily.COLOCATED:
            # One site holding every antenna, placed at the area centre: a
            # uniform drop would sometimes park the only site in a corner and
            # penalize it for a position no real single-cell deployment picks.
            return DMIMOConfig(L=1, M=self.M_tot, rho_max=P_budget,
                               ap_placement=APPlacement.CENTER,
                               precoding="ZF", operation="centralized",
                               combining="ZF", **common)

        if family is RateFamily.CENTRALIZED:
            return DMIMOConfig(L=self.L, M=self.M, rho_max=P_budget / self.L,
                               precoding="ZF", operation="centralized",
                               combining="ZF", **common)

        # Fully local: each AP designs from its own CSI only.
        return DMIMOConfig(L=self.L, M=self.M, rho_max=P_budget / self.L,
                           precoding="L-RZF", operation="distributed",
                           combining="L-MMSE", **common)

    def power_params(self, deployment: Deployment, P_budget: float,
                     **overrides) -> DMIMOPowerParams:
        """Power-model parameters whose frame and topology match the rate config.

        Built through :meth:`DMIMOPowerParams.from_rate_config`, which is what
        keeps the frame consistent between the two packages.
        """
        cfg = self.rate_config(deployment, P_budget)
        split = deployment.split or Split.S1  # co-located: unused, no fronthaul
        return DMIMOPowerParams.from_rate_config(cfg, split=split, **overrides)


@dataclass
class RatePoint:
    """Rate-model outcome for one (scenario, deployment, budget) [SI units]."""

    R_DL: float                  # delivered downlink sum rate [bit/s]
    R_UL: float                  # delivered uplink sum rate [bit/s]
    ap_power: np.ndarray         # mean per-AP radiated power [W], shape (L,)
    rho_max: float               # per-AP budget [W]
    se_dl_median: float = 0.0    # median per-user DL SE [bit/s/Hz]
    se_ul_median: float = 0.0    # median per-user UL SE [bit/s/Hz]

    @property
    def R_total(self) -> float:
        return self.R_DL + self.R_UL


def _cache_key(scenario: Scenario, deployment: Deployment, P_budget: float) -> str:
    """Stable key over everything that changes the Monte Carlo outcome.

    Keyed on the *rate family* rather than the deployment, so S1 and S2 (which
    realize the same centralized precoder and therefore deliver the same rate)
    share a single cached run instead of computing it twice.
    """
    payload = json.dumps({"scenario": asdict(scenario),
                          "rate_family": deployment.rate_family.value,
                          "P_budget": round(float(P_budget), 12)},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _load_cache() -> Dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    with open(CACHE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_cache(cache: Dict[str, dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=1, sort_keys=True)


def rates_for(scenario: Scenario, deployment: Deployment, P_budget: float,
              use_cache: bool = True, verbose: bool = True) -> RatePoint:
    """Delivered rates and per-AP powers for one deployment at one budget.

    Runs the downlink and uplink Monte Carlo of ``../D_MIMO_rate`` on the
    matched configuration. Both directions are run with the same seed, so they
    see the same sequence of AP/UE drops and the rate pair describes one network
    rather than two.

    The returned rates are *delivered*: the rate model has already applied the
    ``tau_d / tau_c`` and ``tau_u / tau_c`` prelogs, so nothing downstream may
    apply a prelog again.
    """
    key = _cache_key(scenario, deployment, P_budget)
    cache = _load_cache() if use_cache else {}
    if key in cache:
        entry = cache[key]
        return RatePoint(R_DL=entry["R_DL"], R_UL=entry["R_UL"],
                         ap_power=np.asarray(entry["ap_power"], dtype=float),
                         rho_max=entry["rho_max"],
                         se_dl_median=entry.get("se_dl_median", 0.0),
                         se_ul_median=entry.get("se_ul_median", 0.0))

    cfg = scenario.rate_config(deployment, P_budget)
    if verbose:
        print(f"  simulating {deployment.rate_family.value} rates at "
              f"P_budget = {P_budget:.2f} W "
              f"(L={cfg.L}, M={cfg.M}, rho_max={cfg.rho_max:.4f} W)")
    dl = simulate_downlink(cfg)
    ul = simulate_uplink(cfg)

    point = RatePoint(R_DL=dl.sum_rate, R_UL=ul.sum_rate,
                      ap_power=dl.ap_power, rho_max=cfg.rho_max,
                      se_dl_median=dl.se_median, se_ul_median=ul.se_median)
    if use_cache:
        cache[key] = {"R_DL": point.R_DL, "R_UL": point.R_UL,
                      "ap_power": point.ap_power.tolist(), "rho_max": point.rho_max,
                      "se_dl_median": point.se_dl_median,
                      "se_ul_median": point.se_ul_median,
                      "_scenario": asdict(scenario),
                      "_rate_family": deployment.rate_family.value,
                      "_P_budget": P_budget}
        _save_cache(cache)
    return point


@dataclass
class OperatingResult:
    """Rate and power of one deployment at one point of a sweep."""

    deployment: Deployment
    P_budget: float
    rates: RatePoint
    power: float                      # total consumption [W]
    scenario: Scenario = None         # topology this point was evaluated on
    breakdown: object = field(repr=False, default=None)

    @property
    def R_total(self) -> float:
        return self.rates.R_total

    @property
    def energy_efficiency(self) -> float:
        """Delivered bits per joule (eq. ee_net)."""
        return self.R_total / self.power

    @property
    def L(self) -> int:
        """Number of APs; one for the co-located site whatever the scenario."""
        return 1 if not self.deployment.is_distributed else self.scenario.L

    @property
    def M(self) -> int:
        """Antennas per AP."""
        return self.scenario.M_tot if not self.deployment.is_distributed else self.scenario.M


def evaluate(scenario: Scenario, deployment: Deployment, P_budget: float,
             use_cache: bool = True, verbose: bool = True,
             **param_overrides) -> OperatingResult:
    """Full rate-and-power evaluation of one deployment at one transmit budget."""
    from .network import compute_colocated, compute_network

    rates = rates_for(scenario, deployment, P_budget, use_cache, verbose)
    p = scenario.power_params(deployment, P_budget, **param_overrides)

    if not deployment.is_distributed:
        # One site: the co-located model of ../FR3_power_model, unchanged.
        breakdown = compute_colocated(p, P_budget, rates.R_DL, rates.R_UL)
    else:
        breakdown = compute_network(p, rates.ap_power, rates.rho_max,
                                    rates.R_DL, rates.R_UL)
    return OperatingResult(deployment=deployment, P_budget=P_budget, rates=rates,
                           power=breakdown.total, scenario=scenario,
                           breakdown=breakdown)


def topologies(scenario: Scenario, deployments) -> Dict[tuple, DMIMOConfig]:
    """Distinct network geometries among ``deployments``, keyed by ``(L, M, placement)``.

    Deployments that differ only in the functional split share a geometry: S1,
    S2 and S3 all sit on the same ``L`` APs. Deduplicating avoids drawing the
    same network several times.
    """
    seen = {}
    for deployment in deployments:
        cfg = scenario.rate_config(deployment, 1.0)   # geometry ignores the budget
        seen.setdefault((cfg.L, cfg.M, cfg.ap_placement.value), cfg)
    return seen
