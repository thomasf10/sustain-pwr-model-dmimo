# D-MIMO FR3 power model

Power model of a distributed massive MIMO (cell-free) FR3 deployment, and the
rate-versus-power comparison of three MIMO configurations at equal transmit
budget. Implements the distributed extension of
`../../power_model_description/sections/dmimo_pwr_model.tex`.

```
dmimo_power/config.py       DMIMOPowerParams (extends fr3_power.PowerParams), Split, PASizing
dmimo_power/network.py      per-AP / fronthaul / CPU blocks and the network total (eq. pnet)
dmimo_power/scenarios.py    the four deployments, the link to the rate model, rate cache
dmimo_power/plotting.py     the comparison and network-overview figures
dmimo_power/manifest.py     JSON record of the parameters behind a run
scripts/compare_deployments.py   the end-to-end sweeps, figures and manifest
tests/test_dmimo_power.py        19 checks, including the co-located equivalence
data/rates_dmimo.json            cached Monte Carlo rates (regenerable)
data/runs/*.json                 run manifests (kept: they record what produced a figure)
figures/                         generated figures (git-ignored, regenerable)
```

Run with the repository virtual environment (Python 3.13 with Sionna):

```
python tests/test_dmimo_power.py                            # model invariants, no Sionna needed
python scripts/compare_deployments.py                       # both sweeps + figures + manifest
python scripts/compare_deployments.py --sweep aps --fixed-budget 8
python scripts/compare_deployments.py --no-cache            # recompute the rates
```

## What is reused

Nothing is re-derived. A distributed AP is a small **fully-digital** array,
`M_RF = M` and `M_PS = 0`, and substituting that into the co-located assemblies
reduces them *exactly* to the per-AP equations of the manuscript:

| Block | Source |
|---|---|
| Analog (eq. `ana_dl_ap`, `ana_ul_ap`) | `fr3_power.power_model.analog`, unchanged, plus `P_sync` |
| Power amplifier (eq. `pa_avg_ap`) | `fr3_power.power_model.pa`, unchanged, plus the sizing convention |
| Frame averaging (eq. `frameavg_ap`) | `fr3_power.frame_average.frame_average`, unchanged |
| Subcomponents (converters, mixers, LNA, encoder, IFFT, DPD, filters) | `fr3_power.components`, unchanged |
| Digital (eq. `dig_ap`) | re-assembled from the same components |

The digital block is the only one that cannot be reused wholesale, for two
structural reasons: the encoder and decoder move to the CPU under every split,
and the MIMO-processing term depends on where the precoder is computed and
applied, which is what the functional split decides.

Rates and per-AP transmit powers come from `../D_MIMO_rate/`. Using
`DownlinkResult.ap_power` directly is what makes the two models agree by
construction on how much each AP radiates: local operation meets its budget with
equality, whereas centralized operation only brings the busiest AP to `rho_max`
(eq. `power-normalization`), and the difference in utilisation `u_l` is visible
in the PA term.

## Simplifications

As instructed for this evaluation, and each one narrower than the manuscript:

- **All APs are active.** The deep-sleep set of eq. `pnet` is empty, so
  `delta_ds`, `P_AP_0` and the sleeping fronthaul level are not modelled. This
  matches the rate simulations, where every AP serves.
- **Full cell-free service**, `D_l = {1..K}`. Hence `K_l = K`, every AP carries
  the same load, and the per-AP frame average of eq. `frameavg_ap` collapses
  onto the co-located one. It also means the downlink payload is duplicated to
  all `L` APs under a data-sharing split, which is the worst case for the
  fronthaul.
- **All three splits S1, S2, S3 are implemented.**

## The four deployments

All cover the same 200 m area with the same total antenna count, the same users,
the same 38.901 UMi channel model, and the same **total** radiated power. Only
the antenna geometry and the processing location change.

| | antennas | rate model | split | fronthaul carries |
|---|---|---|---|---|
| co-located MIMO | 1 site x `L*M` | centralized ZF, site at the area centre | none | nothing (no fronthaul, no CPU) |
| D-MIMO, centralized | `L` APs x `M` | centralized ZF | S1 | `M` streams of samples |
| D-MIMO, centralized | `L` APs x `M` | centralized ZF | S2 | payload + precoding coefficients |
| D-MIMO, distributed | `L` APs x `M` | local L-MMSE / L-RZF | S3 | payload + per-AP partial sums |

S1 and S2 realize the *same* centralized precoder, so they deliver the same rate
and differ only in where the work runs and what crosses the fronthaul. That is
why `Deployment.rate_family` exists: the expensive Monte Carlo is keyed on the
rate family rather than the deployment, so the two centralized splits share one
cached run instead of computing an identical result twice.

Equal total power means the per-AP budget is `rho_max = P_budget / L`. Under the
default PA sizing convention this also gives every deployment amplifiers of the
same rating, `P_max = P_budget / (L M)`, so they are not compared across
different hardware.

The co-located baseline is drawn from the *same* rate code as the distributed
ones rather than from `../FR3_power_model/`'s own cached table, which describes a
different scenario (1024 antennas, hybrid beamforming, `K = 8`, 10 GHz) and would
not be comparable. It is placed at the area centre via
`DMIMOConfig.ap_placement`, since a uniform drop would sometimes park the only
site in a corner and penalize it for a position no real single-cell deployment
would choose.

## The two sweeps

Both plot delivered sum rate against network power on the *same axes*, so they
can be read against each other; only the parameter moving along each curve
changes.

**Sweep 1, transmit power** (`--sweep budget`, figure `rate_vs_power_budget.png`)
varies the total radiated power from 0.5 to 32 W at a fixed topology.

**Sweep 2, number of APs** (`--sweep aps`, figure `rate_vs_power_aps.png`) fixes
the transmit power and redistributes the *same* total antenna count over `L`
APs, from a few large arrays to many small ones (`L = 2, 4, 8, 16, 32` with
`M = 32, 16, 8, 4, 2`). Holding `L*M` constant keeps the radiating hardware
fixed and varies only how finely it is spread over the area. The co-located site
is the `L = 1` reference point, drawn as an open marker because it is a single
point rather than a truncated curve.

## Results

`python scripts/compare_deployments.py` at 64 total antennas, `K = 10`, 20 drops
per point.

### Sweep 1: transmit power, at L = 16 x M = 4

| deployment | P_TX [W] | R_DL [Gb/s] | R_UL [Gb/s] | R_tot [Gb/s] | P_net [W] | EE [Mb/J] |
|---|---|---|---|---|---|---|
| co-located | 0.5 | 8.92 | 5.40 | 14.32 | 53.3 | 268.8 |
| co-located | 4.0 | 12.62 | 5.40 | 18.02 | 66.6 | 270.5 |
| co-located | 32.0 | 17.06 | 5.40 | 22.46 | 173.5 | 129.5 |
| D-MIMO centralized (S1) | 0.5 | 12.19 | 14.77 | 26.96 | 319.3 | 84.5 |
| D-MIMO centralized (S1) | 4.0 | 17.45 | 14.77 | 32.22 | 326.0 | 98.9 |
| D-MIMO centralized (S1) | 32.0 | 22.82 | 14.77 | 37.60 | 379.6 | 99.0 |
| D-MIMO centralized (S2) | 0.5 | 12.19 | 14.77 | 26.96 | 446.9 | 60.3 |
| D-MIMO centralized (S2) | 4.0 | 17.45 | 14.77 | 32.22 | 464.5 | 69.4 |
| D-MIMO centralized (S2) | 32.0 | 22.82 | 14.77 | 37.60 | 542.0 | 69.4 |
| D-MIMO distributed (S3) | 0.5 | 9.74 | 7.38 | 17.12 | 404.1 | 42.4 |
| D-MIMO distributed (S3) | 4.0 | 11.42 | 7.38 | 18.79 | 424.9 | 44.2 |
| D-MIMO distributed (S3) | 32.0 | 11.85 | 7.38 | 19.22 | 533.7 | 36.0 |

### Sweep 2: number of APs, at P_TX = 8 W and 64 antennas

| deployment | L | M | R_tot [Gb/s] | P_net [W] | EE [Mb/J] |
|---|---|---|---|---|---|
| co-located | 1 | 64 | 19.42 | 81.9 | 237.2 |
| S1 | 2 | 32 | 21.17 | 297.0 | 71.3 |
| S1 | 8 | 8 | 31.61 | 309.2 | 102.2 |
| S1 | 32 | 2 | 35.33 | 384.6 | 91.9 |
| S2 | 2 | 32 | 21.17 | 193.4 | 109.5 |
| S2 | 8 | 8 | 31.61 | 313.0 | 101.0 |
| S2 | 32 | 2 | 35.33 | 813.3 | 43.4 |
| S3 | 2 | 32 | 20.20 | 178.1 | 113.4 |
| S3 | 8 | 8 | 23.28 | 299.9 | 77.6 |
| S3 | 32 | 2 | 15.43 | 712.1 | 21.7 |

### What to read off

**Distribution buys rate and costs power.** Centralized D-MIMO delivers 1.7x the
co-located sum rate at 4 W, but at four to seven times the network consumption,
so its energy efficiency is lower. The rate gain is real macro diversity, and it
is largest in the uplink (14.8 against 5.4 Gbit/s): a user is always near *some*
AP, which a single central array cannot offer.

**The uplink rate does not move along sweep 1.** This is a correctness signal,
not a bug: the swept budget is the AP downlink budget, while uplink power is set
by the per-user `p_max`, which the sweep does not touch.

**The fronthaul dominates, at 50-61% of network power.** This is the term the
co-located model has no counterpart for, and it decides the comparison on its
own, as Remark `rem:fh_ceiling` predicts. The measured S3 efficiency sits below
the ceiling `1 / (Pi_FH min_k |L_k|)` = 250 Mbit/J that the remark derives for
`L = 16`.

**S1 and S2 cross over, and which one wins depends on `L`.** They deliver
identical rates, so in the figures they differ only horizontally, which makes the
fronthaul cost of the split directly readable. S2 is much cheaper at few large
APs (193 W against 297 W at `L = 2`) and much more expensive at many small ones
(813 W against 385 W at `L = 32`), crossing over near `L = 8`. The reason is that
S1's load `2 b_FH M f_sI` *falls* as the arrays shrink, while S2's uplink partial
sums `2 b_FH K_l f_sI` do not depend on `M` at all and are paid by every one of
the growing number of links. Forwarding `K_l` scalar streams instead of `M`
sample streams is only a saving when `K_l < M`.

**The same effect makes S3 non-monotonic, and it is the most striking curve.**
Its rate peaks at `L = 8` and then *falls* (23.3 to 15.4 Gbit/s) while power
keeps climbing, so the curve doubles back on itself: at `M = 2` a local precoder
has too few degrees of freedom to suppress `K = 10` users, and the fronthaul
grows with the link count regardless. Fully distributed operation therefore has a
genuine optimum in the number of APs, which centralized operation does not.

**Warning: several parameters are placeholders.** `eta_FH_sc`, `eta_CPU_sc`,
`b_FH`, `delta_FH_micro`, `P_CPU_0` and `P_sync` have no counterpart in the
co-located model and no sourced value yet (see the to-do list at the end of
`dmimo_pwr_model.tex`); `unsourced_parameters()` prints them at the top of every
run and every manifest names them. Since the fronthaul is the dominant term,
`b_FH` and `Pi_FH` in particular carry the S1-versus-S2-versus-S3 conclusion, and
it should be reported as a sensitivity range over them rather than as a point
value. The inherited PA constants `xi`, `alpha` and `eta_PAmax` were fitted for
macro-cell amplifiers, not AP-class ones.

## Network overviews

Every run writes `figures/network_L{L}x{M}.png` for each distinct geometry it
touched, drawn with `mimo_helpers.plot_network`: the APs, the users, the central
unit, and the fronthaul links. These are the *first Monte Carlo drop* of the
corresponding run, seeded with `cfg.seed`, so the picture is the layout the
numbers actually came from rather than an unrelated illustration. The seed is
shared across deployments, so the users sit in the same places in every panel and
only the AP layout changes between them. Geometries are deduplicated: the splits
share a layout, and the co-located site is the same at every point of an AP-count
sweep.

## Run manifests

Each run writes `data/runs/<name>.json`, a self-describing record meant to answer
"what exactly produced this figure?" months later. It holds the provenance
(UTC timestamp, git commit and dirty flag, branch, Python/NumPy/platform
versions), the scenario, the sweep definition, **every** power-model parameter
per deployment including the inherited co-located ones, the derived quantities
that are properties rather than fields (`upsilon_coh`, `f_sI`, the three frame
prelogs, ...) and so would otherwise be invisible, the placeholder parameters
named explicitly, the full result table with the per-block power breakdown, and
the figures written.

JSON was chosen over YAML or TOML because it needs no extra dependency, the
repository already caches its rates that way, it diffs line by line under git,
and it round-trips back into Python. TOML would be the better choice for
hand-edited *input* configuration; this is generated *output*, where being
machine-readable matters more than being hand-editable. Manifests are committed
rather than ignored, unlike the figures, because they are the record of what the
figures mean.

## Time budget: the two sets of taus

The two packages describe the same time budget in different vocabularies, and
this is the single most likely source of a silent factor-of-something error when
wiring them together. `DMIMOPowerParams.from_rate_config` is the one place that
resolves it, and it should be the only way these parameters are built.

`../FR3_power_model/` works in the **frame domain**. `PowerParams` carries
`tau_DL` (downlink share of the frame, default `0.75`), `tau_UL = 1 - tau_DL`,
and the signalling fractions `tau_DLsig = tau_ULsig = 1/14`. Its downlink prelog
is `tau_DL * (1 - tau_DLsig)`.

`../D_MIMO_rate/` works in the **coherence-block domain**. `DMIMOConfig` splits a
block of `tau_c = 200` samples into `tau_p` pilots, `tau_u` uplink data and
`tau_d` downlink data, and exposes `dl_prelog = tau_d / tau_c` and
`ul_prelog = tau_u / tau_c`.

Charging the pilots to the uplink signalling phase, as `dmimo_sysmodel.tex`
does, pins the mapping exactly:

```
tau_DL    = tau_d / tau_c
tau_UL    = (tau_p + tau_u) / tau_c        ( = 1 - tau_DL )
tau_ULsig = tau_p / (tau_p + tau_u)
tau_DLsig = 0                              (the downlink phase is all data)
```

so `tau_UL * tau_ULsig` is the pilot share and `tau_UL * (1 - tau_ULsig)` the
uplink data share. The degenerate `tau_u = 0` gives `tau_ULsig = 1`, the
downlink-only frame in which the uplink phase carries nothing but pilots, with
`tau_UL = tau_p/tau_c = 0.1` and `tau_DL = 0.9`. `test_frame_mapping_from_rate_config`
pins all four identities.

Two further consequences:

- **The rates are already delivered rates.** The rate model applies
  `dl_prelog` / `ul_prelog` inside `mimo_helpers.spectral_efficiency`, so
  `DownlinkResult.sum_rate` has the prelog in it. Nothing downstream applies one
  again. This matters most in the fronthaul, where Remark `rem:no_double` shows
  that charging the delivered rate directly is *equivalent* to frame-averaging
  the peak rate, so doing both would count the frame structure twice;
  `test_s3_downlink_prelog_is_not_applied_twice` guards it.
- **`tau_p` is pure bookkeeping.** Under the perfect-CSI assumption of
  `mimo_helpers.estimate_channels` it never touches the SINR, only the prelog. It
  stops being free the moment pilot-based estimation is modelled.

### The coherence block had two values

`fr3_power.PowerParams.upsilon_coh` is *derived* from Doppler and delay spread
(`(c/dist_max) * (lambda_c/(2*vel))`, giving 500 samples at the defaults) and
sets how often the precoder is recomputed; `DMIMOConfig.tau_c` is a flat 200.
They are the same physical quantity, so leaving both would make the combined
model assume the channel changes at two different rates: precoder amortisation
over a block 2.5x longer than the pilot overhead assumes.
`DMIMOPowerParams.upsilon_coh` now overrides the derived value with `tau_c`, so
one number governs both packages.

## What is not modelled

Carried over from `ssec:dmimo_gaps`, and flagged rather than absorbed into the
parameters. Fronthaul quantization is modelled only through its rate, so S1 is
charged for its transport load but not debited for the accuracy `b_FH` costs it.
Channel-estimation processing is unpriced, since the rate model assumes perfect
CSI, although it scales with the deployment at both nodes. `P_sync` is charged as
a continuous power but not as frame time, even though over-the-air calibration
consumes symbols. AP selection and user-centric clustering are absent by the
simplifications above, and are the dominant lever on cell-free energy efficiency.
