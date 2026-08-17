# D-MIMO FR3 power model

**Status: planned. This folder is a placeholder; the work described below still
needs to be done.**

## Goal

Extend the parametric FR3 base-station power model (see `../FR3_power_model/`) to
the distributed massive MIMO (cell-free) setting described in
`../../power_model_description/sections/dmimo_sysmodel.tex`. Instead of a single
base station with a co-located array, the deployment is a set of geographically
distributed access points (APs), each with a few antennas and its own RF chains,
PAs, and digital front end, jointly serving the users and coordinated by a
central processing unit (CPU) over fronthaul links.

The central-MIMO model accounts for digital, analog, and power-amplifier
consumption at one site. Moving to D-MIMO changes the picture in ways the model
must capture:

- **Per-AP hardware, summed over the network.** The digital, analog, and PA
  blocks now apply per AP and are aggregated over all `L` APs, with each AP's
  operating point (antennas, RF chains, transmit power, load) set by the D-MIMO
  system model rather than assumed identical.
- **Fronthaul consumption.** The transport of samples or messages between the
  APs and the CPU is a first-class contributor to the total power and does not
  exist in the single-site model. Its cost scales with the fronthaul rate, which
  depends on the cooperation level and the precoding scheme (centralized schemes
  move more information than local ones).
- **Central processing.** The CPU-side computation for centralized precoding /
  combining (for example ZF, RZF, MMSE and their partial variants) adds a
  processing term that a fully local scheme does not incur.
- **Cooperation-level trade-offs.** Centralized versus distributed processing
  trades achievable rate against fronthaul and computation cost. The extended
  model should make this trade-off explicit so energy efficiency can be compared
  across schemes.

## Approach

Leverage the existing pieces in this repository rather than re-deriving them:

- **Rates from `../D_MIMO_rate/`.** The D-MIMO rate simulation produces the
  ergodic per-user spectral efficiencies, sum rate, and per-AP transmit powers
  for a given `DMIMOConfig` (topology, precoding, power control, propagation).
  These feed the rate-dependent terms of the power model and set each AP's
  operating point.
- **Power blocks from `../FR3_power_model/`.** The modular `fr3_power` package
  already implements the digital / analog / PA breakdown, the operating-mode
  frame averaging, and the energy-efficiency computation for a single site.
  Reuse it per AP and add the network-level fronthaul and CPU terms on top.

## Time budget: the two sets of taus

The two packages describe the same time budget in different vocabularies, and
this is the single most likely source of a silent factor-of-something error when
wiring them together. Read this before passing a rate from one to the other.

`../FR3_power_model/` works in the **frame domain**. `PowerParams` carries
`tau_DL` (downlink share of the frame, default `0.75`), `tau_UL = 1 - tau_DL`,
and the signalling fractions `tau_DLsig = tau_ULsig = 1/14`. Its downlink prelog
is `tau_DL * (1 - tau_DLsig)`.

`../D_MIMO_rate/` works in the **coherence-block domain**. `DMIMOConfig` carries
`tau_c = 200` samples per block, of which `tau_p = 20` carry uplink pilots, and
exposes `dl_prelog = (tau_c - tau_p) / tau_c = 0.9`.

`power_model_description/sections/dmimo_sysmodel.tex` states the correspondence
in frame language: the pilots occupy the uplink signalling phase, so the pilot
fraction of the frame is `tau_UL * tau_ULsig`, and the downlink rate carries the
prelog `tau_DL * (1 - tau_DLsig)`. The downlink-only case that `D_MIMO_rate`
actually simulates is `tau_ULsig = 1` and `tau_DLsig = 0`, which pins the frame:

```
tau_UL    = tau_p / tau_c           = 0.1     (uplink phase is nothing but pilots)
tau_DL    = 1 - tau_p / tau_c       = 0.9     (downlink phase is nothing but data)
tau_DLsig = 0, tau_ULsig = 1
```

Three consequences for the integration:

- **The rates are already delivered rates.** `dl_rate.simulate_downlink` applies
  `cfg.dl_prelog` inside `mimo_helpers.spectral_efficiency`, so
  `DownlinkResult.sum_rate` has the prelog in it, exactly as `rates.json` does on
  the FR3 side. Do not apply a prelog again when feeding it to `fr3_power`.
- **Do not mix the two frames.** The `PowerParams` default `tau_DL = 0.75` and
  the frame implied by `tau_c`/`tau_p` above (`tau_DL = 0.9`) are different
  frames, and their downlink prelogs differ by 29%:

  ```
  FR3    tau_DL * (1 - tau_DLsig) = 0.75 * 13/14   = 0.696
  DMIMO  (tau_c - tau_p) / tau_c  = 180/200        = 0.900
  ```

  Taking a rate computed under one and averaging the hardware over the other is
  not a double count, it is an inconsistency, and nothing in either package will
  complain. Set `tau_DL`, `tau_DLsig`, `tau_ULsig` from the D-MIMO config rather
  than leaving the FR3 defaults.
- **`tau_p` is pure bookkeeping here.** Under the perfect-CSI assumption of
  `mimo_helpers.estimate_channels`, `tau_p` never touches the SINR; it only sets
  the prelog. It stops being free the moment pilot-based estimation is modelled.

There is also a third symbol for the coherence block, still unreconciled.
`fr3_power.PowerParams.upsilon_coh` is *derived* from the Doppler and delay
spread (`(c/dist_max) * (lambda_c/(2*vel))`, giving `500` samples at the
defaults) and sets how often the precoder is recomputed. `DMIMOConfig.tau_c` is a
flat `200`. They are the same physical quantity with two values in two packages,
so the combined model is currently inconsistent about how fast the channel
changes: the precoder amortisation assumes a block 2.5x longer than the pilot
overhead does. Pick one before reporting combined numbers.

## Planned deliverables

- A power model that returns the total D-MIMO network consumption and its
  breakdown: per-AP digital / analog / PA, fronthaul, and CPU processing.
- A fronthaul consumption model tied to the cooperation level and precoding
  scheme, with the fronthaul rate derived from the system model.
- End-to-end energy efficiency (delivered bits per joule) for the distributed
  deployment, using the rates from `../D_MIMO_rate/`.
- Comparison of centralized versus distributed processing on the rate / power /
  energy-efficiency trade-off.

## TODO

- [ ] Implement the per-AP aggregation of the FR3 power blocks.
- [ ] Model fronthaul power as a function of fronthaul rate and cooperation level.
- [ ] Model CPU processing power for the centralized precoding schemes.
- [ ] Wire in rates and per-AP powers from `../D_MIMO_rate/` (needs that model
      completed first).
- [ ] Document the extended model in
      `../../power_model_description/` and cross-reference the equations here.
