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
