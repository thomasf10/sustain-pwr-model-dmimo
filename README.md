# sustain-pwr-model-dmimo

Sustainability-oriented power modelling for upper mid-band (FR3) and distributed
massive MIMO (D-MIMO, cell-free) base stations. The repository brings together
the manuscript that describes the model, the original reference code from the
FR3 power-model paper, a clean modular re-implementation of that model, the
rate-simulation machinery that feeds the power model with achievable rates, and
the distributed extension that ties the two together.

The work builds on the parametric FR3 power model of E. Peschiera, S. Yun,
Y. Lee, L. Van der Perre and F. Rottenberg, "A parametric power model of upper
mid-band (FR3) base stations for 6G," 2026 IEEE ICASSP, Barcelona,
[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11464669), and
extends it toward cell-free / distributed deployments.

## Repository layout

```
sustain-pwr-model-dmimo/
├── sustain-dmimo-pwr-model-paper/   the manuscript: system model, power model, evaluation
├── power_model_description/         earlier LaTeX write-up, superseded by the manuscript
├── context_papers/                  reference PDFs and the vendored cell-free monograph
└── code/
    ├── FR3_power_consumption_original_from_paper/   original paper scripts
    ├── FR3_power_model/                             modular re-implementation
    ├── D_MIMO_rate/                                 D-MIMO rate simulations (DL and UL)
    ├── D_MIMO_FR3_power_model/                      power model extended to D-MIMO
    ├── power_gui/                                   Streamlit GUI over the models
    ├── Dockerfile                                   container for the GUI
    └── docker-compose.yml                           `docker compose up` wrapper
```

Python work runs on the virtual environment at the repository root, `.venv`,
which is Python 3.13 with Sionna installed. The system Python 3.8 will not do.

## `sustain-dmimo-pwr-model-paper/` — the manuscript

**This is the authoritative description of the model.** The LaTeX source
(`main.tex`, built to `main.pdf`) is the conference paper *"A Power Model for
D-MIMO Systems: Comparing the Energy Consumption of D-MIMO and Co-located
MIMO"*, assembled from `sections/`:

- `introduction.tex`
- `sysmodel.tex` — the D-MIMO system model: deployment and channel, downlink and
  uplink signal models and SINRs, centralized versus distributed operation,
  power control, and the delivered-rate expressions.
- `pwr_model.tex` — the power model: the functional splits S1/S2/S3, MIMO
  processing and its placement, the per-AP digital / analog / amplifier blocks,
  the fronthaul, the central unit, synchronization, and the network total.
- `numerical_eval.tex` — the parameter tables and the four evaluations
  (rate against power, network power against the number of APs, the per-block
  breakdown, and energy efficiency).
- `conclusion.tex`, `appendix.tex`

The figures in `figs/` are pgfplots sources written directly by
`code/D_MIMO_FR3_power_model/scripts/`, one script per evaluation, so the
manuscript and the code cannot drift apart on the numbers. See
[Where the results live](#where-the-results-live) for the figure-by-figure
mapping and the configuration behind each one.

## `power_model_description/` — earlier write-up (superseded)

The LaTeX write-up that preceded the manuscript, covering the same material in
`sections/original_pwr_model.tex` (the co-located FR3 model of Peschiera et al.),
`sections/dmimo_sysmodel.tex` and `sections/dmimo_ul_sysmodel.tex` (the
distributed downlink and uplink system models), and
`sections/dmimo_pwr_model.tex` (the distributed power model), plus a
`dmimo_sysmodel_old.tex` kept from an earlier revision.

It is **superseded by the manuscript** and is retained only because the sub-READMEs
of `code/D_MIMO_rate/` and `code/D_MIMO_FR3_power_model/`, and the module
docstring of `dmimo_power/__init__.py`, still cite its section files as the
source of the equations they implement. Read the manuscript instead; use this
folder only to resolve those citations until they are repointed.

## `code/` — implementations

### `FR3_power_consumption_original_from_paper/`

The unmodified reference scripts released with the FR3 power-model paper. Three
standalone files, `Fig_2a.py`, `Fig_2b.py`, and `Fig_2c.py`, reproduce Figures
2a, 2b, and 2c respectively: digital and analog power versus the number of RF
chains, power-amplifier consumption versus the number of antennas, and the
downlink/uplink ergodic sum rate. Figure 2c generates channels with Sionna,
which is the computationally heavy step. See the folder's own `README.md` for
details. These scripts are the ground truth against which the modular version is
validated.

### `FR3_power_model/`

A structural refactor of the three original scripts into a reusable, testable
`fr3_power` package: identical numerical results, but with parameters, model,
and plotting separated. Highlights:

- `fr3_power/config.py`, a single source of truth for all constants
  (`PowerParams`) and the swept operating point (`OperatingPoint`).
- `fr3_power/components.py`, `frame_average.py`, `power_model.py`, the
  subcomponent models, the shared operating-mode averaging, and the
  per-component digital/analog/PA breakdown.
- `fr3_power/beamforming.py`, `rate_model.py`, `rates.py`, the rate side (DFT
  codebooks, ZF hybrid beamforming, Sionna channel generation); slow rate
  results are cached to `data/rates.json` so the power figures run instantly on
  NumPy and Matplotlib alone.
- `scripts/fig_2a.py`, `fig_2b.py`, `fig_2c.py`, thin drivers, one per figure.
- `tests/test_equivalence.py`, which asserts the refactor reproduces the
  original paper formulas to floating-point precision, and
  `tests/test_beamforming.py`.

Install with `pip install -r requirements.txt`; the folder `README.md` documents
usage, the design rationale, and the small fixes carried over from the
originals.

### `D_MIMO_rate/`

Spectral-efficiency simulation for the distributed massive MIMO system model of
the manuscript, in both directions. This is the machinery that produces the
achievable rates the power model consumes.

- `config_dmimo.py`, `DMIMOConfig`, one dataclass gathering every system-model
  parameter (topology `L`/`M`/`K`/`Q`, RF band, noise derived from bandwidth and
  noise figure, precoding and combining scheme, operation mode, power-allocation
  heuristic and its fractional exponents, path-loss model, coherence
  bookkeeping, Monte Carlo controls), with derived quantities exposed as
  read-only properties so they cannot drift out of sync. The enum fields are
  cross-checked, so a distributed operation mode is paired only with a local
  precoder and the local power-control rule.
- `mimo_helpers.py`, the MIMO building blocks and array-shape conventions: the
  channel-model dispatch, transmit precoding (MR/ZF/RZF/MMSE and local
  L-RZF/L-MMSE), downlink power control with precoder normalization to the
  per-AP budget, the uplink counterparts (uplink power control, local and
  centralized combining, CPU fusion weights), and the signal-processing back end
  (effective channel, SINR including the UATF bound, spectral efficiency,
  per-AP power).
- `sionna_channel.py`, the default channel backend, wrapping Sionna's 3GPP
  TR 38.901 UMi model to produce the collective channel and the large-scale
  fading `beta`.
- `dl_rate.py` and `ul_rate.py`, the downlink and uplink Monte Carlo drivers.
  Under TDD reciprocity both ride on the same channel realization, so run them
  with the same `cfg.seed` and they are paired on geometry.
- `sanity_checks.py`, a pass/fail harness over the config invariants, the exact
  precoding algebra, the power-control formulas and per-AP budget, and one
  Sionna end-to-end run; run `python sanity_checks.py` from the folder.
- `config_cellfree_book.py`, `cellfree_book_channel.py`, `test_cellfree_book.py`,
  a standalone benchmark of the whole pipeline against the published downlink
  results of the cell-free monograph (Demir, Björnson and Sanguinetti, 2021,
  vendored in `context_papers/cell_free_book/`). The first file encodes the
  monograph's running example (Table 5.1) as a `DMIMOConfig`, the second
  implements the monograph's own correlated-Rayleigh channel (wrap-around
  geometry, correlated shadowing, Gaussian local scattering), and the third
  reruns Figures 6.3 and 6.5 and compares. `--sionna` additionally runs the same
  configuration on both channel backends over identical AP/UE layouts, which
  isolates what swapping the monograph's propagation model for 3GPP TR 38.901
  UMi is worth. Run `python test_cellfree_book.py`.

**Status:** both directions run end to end on the Sionna 38.901 UMi channel, and
`sanity_checks.py` and `test_cellfree_book.py` pass. Against the monograph's
Figures 6.3 and 6.5 the simulated per-user SE reproduces every qualitative
conclusion and sits a factor 1.7 (centralized) and 1.1-1.5 (distributed) above
the published curves, which the folder `README.md` attributes item by item,
chiefly to the use of perfect CSI in place of pilot-based estimation with
contamination. What remains stubbed is the analytical correlated-Rayleigh
backend inside `mimo_helpers` (`spatial_correlation` / `generate_channels`), the
scalable partial precoders and combiners (P-MMSE/P-RZF/LP-MMSE), and channel
estimation beyond perfect CSI (pilot-based MMSE with contamination);
`estimate_channels` currently returns the true channel. See the folder
`README.md` for the benchmark results and the full difference list.

### `D_MIMO_FR3_power_model/`

The FR3 power model extended to the distributed MIMO setting, tying the D-MIMO
rate results into the power and energy-efficiency model. This is what produces
the figures in the manuscript.

- `dmimo_power/config.py`, `DMIMOPowerParams` (extending `fr3_power.PowerParams`)
  together with the functional `Split` and the amplifier `PASizing` convention.
- `dmimo_power/network.py`, the per-AP, fronthaul, central-unit and
  synchronization blocks and the network total.
- `dmimo_power/scenarios.py`, the deployments, the link to the rate model, and
  the rate cache; `plotting.py` and `manifest.py` for the figures and the JSON
  record of what produced a run.
- `scripts/`, one driver per evaluation: `eval1_rate_vs_power.py`,
  `eval23_ap_sweep.py`, `eval4_ee_vs_aps.py`, plus the older
  `compare_deployments.py`.
- `tests/test_dmimo_power.py`, the model invariants including the co-located
  equivalence, which needs no Sionna.
- `data/rates_dmimo.json`, cached Monte Carlo rates, and `data/runs/*.json`, the
  committed run manifests. Figures in `figures/` are git-ignored and
  regenerable.

Nothing is re-derived: a distributed AP is a small fully-digital array, and
substituting `M_RF = M`, `M_PS = 0` into the co-located assemblies reduces them
exactly to the per-AP equations. The folder `README.md` documents which blocks
are reused unchanged, the simplifications in force (all APs active, full
cell-free service), and, importantly, the parameters that are still placeholders
rather than sourced values. Since the fronthaul dominates the network power,
`b_FH` and `Pi_FH` in particular carry the comparison between the splits.

Note that the results tables in that folder `README.md` are explicitly marked
stale: they predate the alignment with the cleaned-up system and power models
and describe a different operating point. The manuscript carries the current
numbers.

### `power_gui/`

An interactive Streamlit front end over the models, kept as a thin presentation
layer that imports them as libraries. The **FR3 central power** page exposes the
operating point and hardware parameters as widgets and shows the digital /
analog / PA breakdown, the energy efficiency, and the swept figures; the
**D-MIMO rate** page configures the distributed link over the now-implemented
`D_MIMO_rate` physics. Run with `pip install -r requirements.txt` then
`streamlit run app.py`, or from `code/` with `docker compose up`, which serves
it on port 8501. The container deliberately installs only the light
dependencies, since the GUI reads cached rates and never runs the
Sionna/TensorFlow rate model. See the folder `README.md` for details.

## Where the results live

Every number in the manuscript comes out of `code/D_MIMO_FR3_power_model/`, and
each evaluation is a single script that writes its figure, copies the pgfplots
source into the manuscript's `figs/`, and records a manifest of what produced
it. Paths below are relative to `code/D_MIMO_FR3_power_model/`.

| Manuscript figure | Script | Run manifest | pgfplots source |
|---|---|---|---|
| Fig. 1, `fig:eval1`, rate against network power | `scripts/eval1_rate_vs_power.py` | `data/runs/eval1_rate_vs_power.json` | `figures/eval1_rate_vs_power.tex` |
| Fig. 2, `fig:eval2`, network power against `L` | `scripts/eval23_ap_sweep.py` | `data/runs/eval23_ap_sweep.json` | `figures/eval2_power_vs_aps.tex` |
| Fig. 3, `fig:eval3`, per-block breakdown | `scripts/eval23_ap_sweep.py` | `data/runs/eval23_ap_sweep.json` | `figures/eval3_breakdown_vs_aps.tex` |
| Fig. 4, `fig:eval4`, energy efficiency | `scripts/eval4_ee_vs_aps.py` | `data/runs/eval4_ee_vs_aps.json` | `figures/eval4_ee_vs_aps.tex` |

Figures 2 and 3 share one script and one manifest by construction: they are two
views of the same sweep, so they cannot disagree about a point. Figure 4 reuses
the same AP counts and budget, so it covers identical points.

### Configurations

The scripts were run at their defaults, and the manifests record that. Common to
all four: `LM = 128` total antennas, `K = 20` users, a 200 m square, 20 Monte
Carlo channel realizations per point evaluated on `Q = 64` subcarriers,
`tau_c = 200`, `tau_DL = 0.75`, seed 0, and the 38.901 UMi channel. These are
the values tabulated in `tab:params_sys` of the manuscript.

- **Evaluation 1** sweeps the total transmit budget over
  `1.6, 3.2, 6.4, 12.8, 25.6, 51.2 W` at `L = 32, 64, 128`, with the amplifiers
  re-sized at every point.
- **Evaluations 2, 3 and 4** sweep `L = 2, 4, 8, 16, 32, 64, 128` at the fixed
  budget `12.8 W`, with the co-located baseline evaluated separately at `L = 1`.

Overriding a default is a command-line flag, and the manifest then records the
override rather than the default, so a regenerated figure always carries its own
configuration:

```
python scripts/eval1_rate_vs_power.py                       # the manuscript figure
python scripts/eval23_ap_sweep.py --budget 8 --K 10
python scripts/eval4_ee_vs_aps.py --no-show
python scripts/eval1_rate_vs_power.py --no-cache            # recompute the rates
```

### What is stored where

`data/rates_dmimo.json` caches the expensive part, the Monte Carlo rates coming
out of `D_MIMO_rate/` (Sionna channel generation, precoding, SINR), keyed on
every parameter that affects them. Points already computed are reused across
scripts and across runs, which is why the three evaluations can share one
underlying set of rates. `--no-cache` recomputes instead of reading it.

`data/runs/<name>.json` is the run manifest, and it is committed rather than
ignored, because it is the record of what a figure means. Each holds the
provenance (`written_utc`, git commit, dirty flag and branch, Python, NumPy and
platform versions), the scenario, the sweep definition, every power-model
parameter for every deployment including the derived quantities that are
properties rather than fields, the unsourced and merely assumed parameters named
explicitly, the full result table with the per-block power breakdown, and the
list of figures written.

`figures/` holds the generated PNG and `.tex` pairs. It is git-ignored, since
everything in it is regenerable from the manifest and the rate cache. The `.tex`
half is copied into `sustain-dmimo-pwr-model-paper/figs/`, which *is* committed,
so the manuscript builds without running any Python.

### Provenance of the current figures

The three manifests behind the manuscript were written on 2026-08-31 and
2026-09-01 at commit `7bb36d0`. Note that all three record `dirty: true`, so the
commit hash does not by itself pin the code that produced them.

`data/runs/compare_aps.json`, `compare_budget.json` and `compare_both.json` are
older runs of `scripts/compare_deployments.py` from 2026-08-17, at a different
operating point (64 antennas, `K = 10`, `L = 16` by `M = 4`, budgets from 0.5 to
32 W). They correspond to `figures/rate_vs_power_*.png` and
`power_breakdown.png` and to the results tables in the folder `README.md`, none
of which appear in the manuscript. They are kept as history and should not be
quoted.
