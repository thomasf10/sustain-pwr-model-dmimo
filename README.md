# sustain-pwr-model-dmimo

Sustainability-oriented power modelling for upper mid-band (FR3) and distributed
massive MIMO (cell-free) base stations. The repository brings together the
mathematical description of the power model and its extension to distributed
MIMO (D-MIMO), the original reference code from the FR3 power-model paper, a
clean modular re-implementation of that model, and the rate-simulation
machinery that feeds the model with achievable rates.

The work builds on the parametric FR3 power model of E. Peschiera, S. Yun,
Y. Lee, L. Van der Perre and F. Rottenberg, "A parametric power model of upper
mid-band (FR3) base stations for 6G," 2026 IEEE ICASSP, Barcelona,
[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11464669), and
extends it toward cell-free / distributed deployments.

## Repository layout

```
sustain-pwr-model-dmimo/
├── power_model_description/     LaTeX write-up: FR3 power model + D-MIMO extension
└── code/
    ├── FR3_power_consumption_original_from_paper/   original paper scripts
    ├── FR3_power_model/                             modular re-implementation
    ├── D_MIMO_rate/                                 D-MIMO rate simulations
    ├── D_MIMO_FR3_power_model/                      power model extended to D-MIMO
    └── power_gui/                                   Streamlit GUI over the models
```

## `power_model_description/` — mathematical description

The LaTeX source (`main.tex`, built to `main.pdf`) documents the
model in two parts, assembled from `sections/`:

- `sections/original_pwr_model.tex` — the **Central MIMO Power Model**. It
  reproduces the FR3 model of Peschiera et al.: the three-block split into
  digital, analog, and power-amplifier consumption; the TDD frame structure and
  operating-mode averaging (reference signalling, data, micro-sleep, idle); the
  per-block subcomponent models; and the delivered-rate and energy-efficiency
  expressions.
- `sections/dmimo_sysmodel.tex` — the **Distributed Massive MIMO System Model**.
  It sets up the cell-free downlink (APs, users, subcarriers, channel model),
  the transmit precoding schemes (centralized ZF/RZF/MMSE/P-MMSE/P-RZF and
  distributed MR/L-MMSE/LP-MMSE), and downlink power control, providing the
  system model that the D-MIMO rate code implements and that the power model is
  being extended toward.
  
- TODO: update when content is changes/ is completed

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

- `fr3_power/config.py` — a single source of truth for all constants
  (`PowerParams`) and the swept operating point (`OperatingPoint`).
- `fr3_power/components.py`, `frame_average.py`, `power_model.py` — the
  subcomponent models, the shared operating-mode averaging, and the
  per-component digital/analog/PA breakdown.
- `fr3_power/beamforming.py`, `rate_model.py`, `rates.py` — the rate side (DFT
  codebooks, ZF hybrid beamforming, Sionna channel generation); slow rate
  results are cached to `data/rates.json` so the power figures run instantly on
  NumPy and Matplotlib alone.
- `scripts/fig_2a.py`, `fig_2b.py`, `fig_2c.py` — thin drivers, one per figure.
- `tests/test_equivalence.py` — asserts the refactor reproduces the original
  paper formulas to floating-point precision.

Install with `pip install -r requirements.txt`; the folder `README.md` documents
usage, the design rationale, and the small fixes carried over from the
originals.

### `D_MIMO_rate/`

Rate (spectral-efficiency) simulation for the distributed massive MIMO downlink
of `sections/dmimo_sysmodel.tex`. This is the machinery that produces the
achievable rates the power model consumes.

- `config_dmimo.py` — `DMIMOConfig`, one dataclass gathering every system-model
  parameter (topology `L`/`M`/`K`/`Q`, RF band, noise derived from bandwidth and
  noise figure, precoding scheme and operation mode, power-allocation heuristic
  and its fractional exponent `v`, path-loss model, coherence bookkeeping, Monte
  Carlo controls), with derived quantities exposed as read-only properties so
  they cannot drift out of sync. The enum fields are cross-checked, so a
  distributed operation mode is paired only with a local precoder and the local
  power-control rule.
- `mimo_helpers.py` — the MIMO building blocks and array-shape conventions: the
  channel-model dispatch, transmit precoding (MR/ZF/RZF/MMSE and local
  L-RZF/L-MMSE), downlink power control (equal and fractional allocation) with
  precoder normalization to the per-AP power budget, and the signal-processing
  back end (effective channel, SINR, spectral efficiency, per-AP power).
- `sionna_channel.py` — the default channel backend, wrapping Sionna's 3GPP
  TR 38.901 UMi model to produce the collective downlink channel and the
  large-scale fading `beta`.
- `dl_rate.py` — orchestrates one downlink Monte Carlo experiment (positions →
  channels `(H, beta)` → CSI → precoding → power control → normalization → SINR →
  spectral efficiency) and returns ergodic per-user SEs and per-AP powers.
- `sanity_checks.py` — a pass/fail harness over the config invariants, the exact
  precoding algebra, the power-control formulas and per-AP budget, and one Sionna
  end-to-end run; run `python sanity_checks.py` from the folder.
- `config_cellfree_book.py`, `cellfree_book_channel.py`, `test_cellfree_book.py` —
  a standalone benchmark of the whole pipeline against the published downlink
  results of the cell-free monograph (Demir, Björnson and Sanguinetti, 2021,
  vendored in `context_papers/cell_free_book/`). The first file encodes the
  monograph's running example (Table 5.1) as a `DMIMOConfig`, the second
  implements the monograph's own correlated-Rayleigh channel (wrap-around
  geometry, correlated shadowing, Gaussian local scattering), and the third
  reruns Figures 6.3 and 6.5 and compares. `--sionna` additionally runs the same
  configuration on both channel backends over identical AP/UE layouts, which
  isolates what swapping the monograph's propagation model for 3GPP TR 38.901 UMi
  is worth. Run `python test_cellfree_book.py`.

**Status:** work in progress, but the downlink chain runs end to end on the
Sionna 38.901 UMi channel. The channel model, transmit precoding, power control,
and the signal-processing back end are implemented, and both `sanity_checks.py`
and `test_cellfree_book.py` pass. Against the monograph's Figures 6.3 and 6.5 the
simulated per-user SE reproduces every qualitative conclusion and sits a factor
1.7 (centralized) and 1.1-1.5 (distributed) above the published curves, which the
folder `README.md` attributes item by item, chiefly to the use of perfect CSI in
place of pilot-based estimation with contamination.
What remains stubbed is the analytical correlated-Rayleigh backend inside
`mimo_helpers` (`spatial_correlation`/`generate_channels`), the scalable partial
precoders (P-MMSE/P-RZF/LP-MMSE), and channel estimation beyond perfect CSI
(pilot-based MMSE with contamination); `estimate_channels` currently returns the
true channel. `ul_rate.py` (uplink) is an empty placeholder to be filled in. See
the folder `README.md` for the benchmark results and the full difference list.

### `D_MIMO_FR3_power_model/`

Intended home for the FR3 power model extended to the distributed MIMO setting,
tying the D-MIMO rate results into the power and energy-efficiency model.
**Currently empty (planned work).**

### `power_gui/`

An interactive Streamlit front end over the models, kept as a thin presentation
layer that imports them as libraries. The **FR3 central power** page (fully
working today) exposes the operating point and hardware parameters as widgets
and shows the digital / analog / PA breakdown, the energy efficiency, and the
swept figures; the **D-MIMO rate** page configures the distributed downlink and
becomes fully functional once the `D_MIMO_rate` physics stubs are implemented.
Run with `pip install -r requirements.txt` then `streamlit run app.py`; see the
folder `README.md` for details.

