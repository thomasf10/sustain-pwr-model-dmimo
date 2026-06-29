# FR3 base-station power consumption model

A clean, modular re-implementation of the parametric power model of upper
mid-band (FR3) base stations from:

> E. Peschiera, S. Yun, Y. Lee, L. Van der Perre and F. Rottenberg,
> "A parametric power model of upper mid-band (FR3) base stations for 6G,"
> 2026 IEEE ICASSP, Barcelona, Spain, 2026, pp. 21476-21480.
> [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11464669)

This reproduces Figures 2a, 2b and 2c. It is a structural refactor of the
original three stand-alone scripts: identical results, but with the parameters,
model, and plotting separated so the code is reusable and testable. The
model equations are documented in `pcons_model.pdf` (Chapter 2), and functions
reference the corresponding equation numbers.

## Layout

```
fr3_power/            # the package
  config.py           # PowerParams + OperatingPoint dataclasses (single source of truth)
  components.py        # subcomponent power models (eqs 2.20-2.39)
  frame_average.py     # operating-mode averaging + load-ind/load-dep split
  power_model.py       # digital() / analog() / pa() -> per-component breakdown
  beamforming.py       # DFT codebooks + subarray beam selection (rate model)
  rate_model.py        # Sionna channel generation + ZF hybrid beamforming
  rates.py             # loader for the cached rate table
  plotting.py          # the figures
scripts/              # thin drivers, one per figure
  fig_2a.py  fig_2b.py  fig_2c.py
data/rates.json        # cached ergodic rates consumed by the power figures
tests/test_equivalence.py # test equivalence between new implementation and the original paper scripts
```

## Design

* **One source of truth for parameters.** All hardware/model constants live in
  `PowerParams` with the FR3 defaults of Section 2.3. The quantities that vary
  along a sweep (antennas, RF chains, transmit power, loads, rates) live in
  `OperatingPoint`. No more copy-pasted parameter blocks.
* **Shared frame averaging.** The averaging over operating modes (data /
  signalling / micro-sleep / idle) and the load-independent vs load-dependent
  split are implemented once in `frame_average.py` and reused by all three
  components, instead of being re-derived by hand for each.
* **Power and rate models decoupled.** The rate model (Fig 2c) is slow and needs
  Sionna + TensorFlow + ideally a GPU. It writes its results to
  `data/rates.json`, which the power figures read. The power figures therefore
  run instantly with only NumPy and Matplotlib.

## Usage

```bash
pip install -r requirements.txt

python scripts/fig_2a.py          # digital + analog vs M_RF (+ prints energy efficiency)
python scripts/fig_2b.py          # power amplifier vs M_ant
python scripts/fig_2c.py          # ergodic rates from the cached table
python scripts/fig_2c.py --recompute   # re-run the Sionna simulation (slow, needs GPU)
```

Programmatic use:

```python
from fr3_power import PowerParams, OperatingPoint, compute, energy_efficiency

params = PowerParams()
op = OperatingPoint(M_ant=1024, M_RF=64, P_T=100.0, R_DL=5.6e9, R_UL=1.3e9)
b = compute(params, op)
print(b.total, b.digital.total, b.analog.total, b.pa.total)
print(energy_efficiency(op, b) * 1e-6, "Mbit/J")
```

## Tests

```bash
python tests/test_equivalence.py        # or: pytest tests/
```

`test_equivalence.py` inlines the original `Fig_2a.py` / `Fig_2b.py` formulas
and asserts the refactored model reproduces them to floating-point precision
(full load and partial load), guaranteeing the refactor is behaviour-preserving.

## Notes / fixes carried over

* The original `Fig_2b.py` set `eta_DPD = 0.2e12` while `Fig_2a.py` used
  `2e12`; the latter matches the paper (ASIC, 0.2e4 GOPS/W) and is used here.
  This had no effect on Fig 2b, which plots only the PA.
* The original analog-UL average used `tau_DLsig` in its first term (a copy-
  paste slip); since `tau_DLsig == tau_ULsig` numerically, results are
  unchanged. The clean version uses `tau_ULsig`.
* All evaluations assume every antenna/RF chain is active for the whole frame
  (`M_a = M_ant`, `M_RF,a = M_RF`, Section 2.3), so the "sleep mode" terms are
  zero and are not modelled.
* **Load `xbar` assumes full spatial load (`K = K_max`).** The model uses
  `xbar` as the *active-subcarrier ratio* (the average fraction of allocated
  data subcarriers, averaged over the `K` served users; eq 1 of the paper).
  The frame averaging actually needs the *active-resource time ratio*
  `N_a,i / N_i` -- the fraction of the frame over which the RF chains/antennas
  carry data, which sets the data-vs-micro-sleep split and is independent of
  the number of users. The two are equal only when every stream is scheduled,
  `K_i = K_max`. So using `xbar` as the data-mode weight is **exact for
  `K = K_max`** (the regime evaluated here) and only **approximate** for fewer
  users. Serving `K_i < K_max` changes the transmit power and the rate, but not
  whether a resource is active. The code does not enforce `K = K_max`; it is an
  assumption on the meaning of the `xbar` you pass in.
* The original scripts fed the power model the Fig 2c rates scaled by a flat
  `0.93`, which the original author identifies as the **pre-log factor** of
  eq (2.40) — specifically its `(1 - tau_sig) = 13/14 ~ 0.93` signalling-overhead
  component (equal for DL and UL since `tau_DLsig = tau_ULsig = 1/14`), with the
  `tau_i`, load and bandwidth factors already inside the rate values. To make
  this robust, the **pre-log is owned in exactly one place**: `rate_model`
  applies the full eq (2.40) pre-log `tau_i*(1 - tau_sig)` when it computes the
  rates, and `rates.json` stores the resulting *delivered* rates (`R_DL_Gbps`,
  `R_UL_Gbps`). The loader and power figures apply no further factor, so
  `rate_model.compute_and_save` can be re-run safely with no risk of
  double-counting. The seed values in `rates.json` are the paper's reference
  rates (original raw values times `0.93`).
