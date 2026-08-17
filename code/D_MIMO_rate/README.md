# D-MIMO rate model

Downlink spectral-efficiency simulation for the distributed massive MIMO
(cell-free) system model of `power_model_description/sections/dmimo_sysmodel.tex`.
This is the machinery that produces the achievable rates the FR3 power model
consumes.

The system model follows

> Ö. T. Demir, E. Björnson and L. Sanguinetti, "Foundations of User-Centric
> Cell-Free Massive MIMO," *Foundations and Trends in Signal Processing*,
> vol. 14, no. 3-4, pp. 162-472, 2021.
> [arXiv:2108.02541](https://arxiv.org/abs/2108.02541)

whose source is vendored in `context_papers/cell_free_book/`. Equation and
section numbers quoted below refer to it.

## Layout

```
config_dmimo.py             DMIMOConfig: every system-model parameter, one dataclass
mimo_helpers.py             channel dispatch, precoding, power control, SINR/SE back end
sionna_channel.py           default backend: 3GPP TR 38.901 UMi via Sionna
dl_rate.py                  downlink Monte Carlo driver
ul_rate.py                  uplink driver (placeholder)
sanity_checks.py            pass/fail harness over config, precoding algebra, power control

config_cellfree_book.py     the monograph's running example (Table 5.1) as a DMIMOConfig
cellfree_book_channel.py    the monograph's own correlated-Rayleigh channel model
test_cellfree_book.py       standalone benchmark against the published downlink figures
```

`test_cellfree_book.py` writes `cellfree_book_cdf_scenario{A,B}.png` (simulated
CDFs against the published medians) and, with `--sionna`,
`cellfree_book_channel_scenario{A,B}.png` (the two channel backends side by
side). All four are regenerable and are git-ignored.

Run with the project virtual environment at the repository root (Python 3.13
with Sionna installed; the system Python 3.8 will not do):

```
python sanity_checks.py          # unit-level invariants of the building blocks
python test_cellfree_book.py     # end-to-end benchmark against the monograph
```

## Before feeding these rates to the power model

`DownlinkResult.sum_rate` is a **delivered** rate: `simulate_downlink` already
applies `cfg.dl_prelog = (tau_c - tau_p) / tau_c` inside
`mimo_helpers.spectral_efficiency`. Do not apply a prelog again downstream.

More importantly, this package and `../FR3_power_model/` describe the same time
budget in two different vocabularies (`tau_c`/`tau_p` here, `tau_DL`/`tau_DLsig`
there), whose defaults imply downlink prelogs that differ by 29%. Mixing them is
an inconsistency that neither package detects. The correspondence, and what to
set, is written up in `../D_MIMO_FR3_power_model/README.md`.

## Validation against the cell-free monograph

`test_cellfree_book.py` recreates the downlink numerical evaluation of
Section 6.6 on the running example of Section 5.5, and compares the outcome with
the published CDF curves of Figures 6.3 and 6.5. It is a whole-pipeline sanity
check rather than a unit test: it exercises the configuration, the channel model,
the precoders, the power control, and the SE back end together, on a setup whose
answer is known.

Two scenarios are run, matching the two panels of every downlink figure in the
monograph. Scenario A has `L = 400` access points with `N = 1` antenna each;
scenario B has `L = 100` access points with `N = 4` antennas each. Both keep the
total array size at `M = LN = 400` and serve `K = 40` users over a 1 km × 1 km
wrap-around area, with 20 MHz of bandwidth, a −94 dBm receiver noise power,
200 mW of downlink power per AP, and coherence blocks of `tau_c = 200` samples of
which `tau_p = 10` carry pilots.

The script is organised in three parts. It first runs hard pass/fail checks that
the configuration reproduces Table 5.1 and that the channel model satisfies its
defining identities (the wrap-around metric, the shadow covariance of eq. (2.21),
the normalisation `Tr(R_kl)/N = beta_kl` of eq. (2.19), and
`E{h_kl h_kl^H} = R_kl` verified by brute-force Monte Carlo). It then runs the
Monte Carlo itself, and finally compares the resulting SE distributions with the
published ones. Only the first part can fail the run; the comparison is reported
but never fatal, because the differences documented below make an exact match
impossible by construction.

```
python test_cellfree_book.py                  # full run, writes the two CDF figures
python test_cellfree_book.py --deviations     # print the difference list and exit
python test_cellfree_book.py --parameters     # print Table 5.1 side by side and exit
python test_cellfree_book.py --sionna         # add the paired 38.901 UMi comparison
```

### Results

100 drops per curve, seed 0, `python test_cellfree_book.py --realizations 100`:

| Scenario | Scheme | 95%-likely SE (here / book) | Median SE (here / book) | Mean SE | Sum SE | Reference curve |
|---|---|---|---|---|---|---|
| A (400 × 1) | RZF, centralized | 9.90 / 5.30 | **11.46 / 6.90** | 11.61 | 464.2 | Fig. 6.3(a), MMSE (All) |
| A (400 × 1) | L-RZF, distributed | 2.00 / 1.30 | **2.95 / 2.60** | 2.99 | 119.7 | Fig. 6.5(a), L-MMSE (All) |
| A (400 × 1) | MR, distributed | 2.00 / 1.20 | **2.95 / 2.20** | 2.99 | 119.7 | Fig. 6.5(a), MR (DCC) |
| B (100 × 4) | RZF, centralized | 7.97 / 4.00 | **9.58 / 5.70** | 9.77 | 390.7 | Fig. 6.3(b), MMSE (All) |
| B (100 × 4) | L-RZF, distributed | 2.54 / 1.40 | **5.10 / 4.30** | 5.52 | 220.8 | Fig. 6.5(b), L-MMSE (All) |
| B (100 × 4) | MR, distributed | 1.87 / 1.00 | **2.93 / 2.00** | 3.06 | 122.4 | Fig. 6.5(b), MR (DCC) |

All values are in bit/s/Hz. The figures `cellfree_book_cdf_scenarioA.png` and
`cellfree_book_cdf_scenarioB.png` overlay the simulated CDFs with the published
medians.

The monograph reports its downlink results only as plots, so the reference column
was read off the vector figures shipped in
`context_papers/cell_free_book/images/section6/`. Those readings are accurate to
roughly ±0.2 bit/s/Hz and are not values quoted in the text. Because this code
serves every user from every AP, the reference is always the monograph's "(All)"
curve rather than its dynamic-cooperation-clustering counterpart; the only
exception is MR, which the monograph plots only with clustering.

Every qualitative conclusion of Section 6.6 survives. Centralized operation beats
distributed operation, which in turn beats MR. The distributed schemes gain
substantially from concentrating the same 400 antennas into 100 four-antenna APs,
because a local precoder needs several antennas to suppress anything, while the
centralized scheme loses by the same move. MR has a short, bounded upper tail in
both scenarios. The absolute values sit above the published ones by a factor
1.7 for the centralized curves and 1.1 to 1.5 for the distributed ones, for the
reasons below.

### Where the differences come from

The differences are enumerated in machine-readable form in
`config_cellfree_book.DEVIATIONS` and printed by `--deviations`. Ordered by how
much they move the curves:

**Perfect CSI dominates everything else.** `mimo_helpers.estimate_channels`
returns the true channels, whereas the monograph builds MMSE estimates from
`tau_p = 10` orthogonal pilots shared by `K = 40` users. Four users per pilot
means the published SINR carries coherent pilot contamination, on top of ordinary
estimation error, that simply does not exist here. This is the main reason the
centralized curves land about 1.7 times high. The distributed curves are less
affected, at 1.1 to 1.5 times, because a four-antenna AP is interference-limited
long before it is estimation-limited, so removing the estimation error buys it
less.

**The SE expression is the genie-aided one.** `mimo_helpers.downlink_sinr`
computes the instantaneous SINR from the realized effective channel
`h_k^H w_k`, which is exactly the genie-aided bound of Corollary 6.3
(eqs. (6.37)-(6.38)), not the hardening bound of Theorem 6.1 that the published
curves use. The monograph itself shows in Figures 6.4 and 6.6 that the two are
nearly indistinguishable for centralized and LP-MMSE precoding, so this
contributes little for those schemes. It does matter for MR, where the signal and
interference powers have long tails and the hardening bound is loose, which is
part of why the MR gap here (1.3 to 1.5) is wider than the L-MMSE one.

**There is no dynamic cooperation clustering.** Every AP serves every user, so
the pilot assignment and AP selection of Algorithm 4.1 play no role. This is not
a source of error against the "(All)" reference curves, which is why those are
the ones used, but it does rule out reproducing the scalable P-MMSE, P-RZF and
LP-MMSE curves at all: those precoders are defined in terms of the clusters and
raise `NotImplementedError` in `mimo_helpers.precoding_directions`. It also makes
the MR comparison looser than the others, since the monograph only plots MR with
clustering.

**The precoders are RZF and L-RZF, not the `MMSE` enum members.** Under perfect
CSI the estimation-error covariance `C_il` vanishes and the monograph's MMSE and
L-MMSE precoders reduce exactly to a regularized zero-forcing form, so the
schemes themselves are right. The subtlety is the diagonal loading. The `MMSE`
member of `PrecodingScheme` hardcodes a loading of `sigma^2`, which is correct
only for a unit-power dual uplink; with physically scaled channels the dual
uplink transmits at `p_max = 100 mW`, so the loading must be `sigma^2 / p_max`.
`book_config` sets `rzf_reg` accordingly, a factor of 10 dB away from the
built-in default.

**Precoders are normalized per drop rather than in expectation.** The monograph
divides each direction by `sqrt(E{||wbar_k||^2})`, an average over the fading,
while `mimo_helpers.normalize_precoder` divides by the norm realized in that
drop. This removes the per-realization power fluctuation. It has one visible
consequence: in scenario A, where `N = 1` makes every local precoder a scalar per
(AP, user) pair, the per-drop normalization divides that scalar out entirely, so
L-RZF and MR become the same transmitted signal. The two curves coincide exactly
in the table above and in the figure, while Figure 6.5(a) separates them. The
benchmark asserts this equivalence rather than hiding it.

**Centralized power control is approximated; distributed power control is
exact.** The distributed rule of eq. (6.36),
`rho_kl = rho_max sqrt(beta_kl) / sum_i sqrt(beta_il)`, is reproduced exactly by
`mimo_helpers.power_control` with `v = +1/2`, and the benchmark checks it against
the formula. The centralized rule of eq. (6.35) is only approximated by
`v = -1/2`: the `beta_k^(-1/2)` fairness weighting matches, but the `omega_k`
precoder-norm correction is missing, and the per-AP budget is enforced by a
single global scaling that puts the busiest AP at `rho_max` instead of the
monograph's cluster-wise normalization. Both are conservative, so the per-AP
constraint holds either way, but the per-user powers differ by more than a common
factor.

**The channel model is not a source of difference on the default path.**
`cellfree_book_channel.py` implements the monograph's own model rather than
accepting the mismatch that using Sionna would introduce: the toroidal
wrap-around metric, the large-scale fading `beta_kl [dB] = -30.5 - 36.7
log10(d_kl / 1 m) + F_kl` of eq. (2.20), the shadow fading correlated across
users at a common AP as `E{F_kl F_il} = 4^2 2^(-delta_ki / 9 m)` of eq. (2.21),
and the Gaussian local scattering model of eqs. (2.23)-(2.24) at
`sigma_phi = sigma_theta = 15°` on a half-wavelength ULA, evaluated by
Gauss-Hermite quadrature. Scenario A needs none of the angular machinery, since
`N = 1` collapses `R_kl` to the scalar `beta_kl`, so its propagation model is
exact by construction.

The `--sionna` flag quantifies what substituting the repository's default
backend would cost; that comparison has its own section below.

Finally, the running example is frequency-flat, which the benchmark matches
exactly by setting `Q = 1`; the OFDM dimension of the pipeline contributes
nothing here.

### Swapping in the 38.901 UMi channel

The default backend of this folder is Sionna's 3GPP TR 38.901 UMi model, not the
monograph's. `--sionna` runs the identical configuration on both, so the cost of
that substitution can be read off directly. Both runs are fed the same pre-drawn
AP and UE layouts by `draw_position_sequence`, which makes the comparison paired
on geometry: only the propagation model changes. `force_nlos=True` is kept, so
38.901 is held to the NLoS conditions the monograph's correlated Rayleigh model
assumes, and the carrier stays at 2 GHz where the monograph's path-loss model is
calibrated.

```
python test_cellfree_book.py --sionna --sionna-scenario B --sionna-realizations 15
```

The large-scale gains, measured on one shared drop, differ less in level than in
spread:

| `beta_kl` [dB] | median | std | 90th pct | median best-AP |
|---|---|---|---|---|
| book, scenario B (100 × 4) | −125.2 | 9.4 | −111.4 | −85.2 |
| 38.901 UMi, scenario B | −125.2 | 13.0 | −107.1 | −80.8 |
| difference | +0.0 | +3.6 | +4.3 | **+4.4** |
| book, scenario A (400 × 1) | −125.3 | 9.1 | −111.7 | −81.3 |
| 38.901 UMi, scenario A | −125.4 | 13.7 | −107.3 | −75.4 |
| difference | −0.1 | +4.6 | +4.4 | **+5.9** |

and the spectral efficiency follows, 15 drops for scenario B and 8 for
scenario A:

| Scenario | Scheme | Median, book model | Median, 38.901 UMi | Delta | Ratio |
|---|---|---|---|---|---|
| B (100 × 4) | RZF, centralized | 9.56 | 11.46 | +1.90 | 1.20 |
| B (100 × 4) | L-RZF, distributed | 5.20 | 5.59 | +0.39 | 1.08 |
| B (100 × 4) | MR, distributed | 2.97 | 2.99 | +0.02 | 1.01 |
| A (400 × 1) | RZF, centralized | 11.51 | 13.66 | +2.14 | 1.19 |
| A (400 × 1) | L-RZF, distributed | 3.07 | 3.27 | +0.20 | 1.06 |
| A (400 × 1) | MR, distributed | 3.07 | 3.27 | +0.20 | 1.06 |

`cellfree_book_channel_scenarioA.png` and `cellfree_book_channel_scenarioB.png`
plot the corresponding CDF pairs, solid for the monograph's model and dashed for
38.901.

The headline is that the *median* link gain is the same under both models, to
within a tenth of a dB. Textbook path-loss arithmetic suggests otherwise: at 1 km
and 2 GHz, 38.901 UMi NLoS gives 134.7 dB against the monograph's 140.6 dB, so
one would expect UMi to be uniformly the more generous model. What happens
instead is that the two models agree closely on the typical link once the actual
distance distribution of a 1 km × 1 km deployment is taken into account, and
diverge in the *spread*: 38.901 produces a distribution about 4 dB wider.

That spread is what moves the spectral efficiency, because no user rides on a
typical link. Each one rides on its best few APs, and the best-AP gain is 4.4 dB
(scenario B) to 5.9 dB (scenario A) higher under 38.901. Two things widen the
distribution. The larger is the shadowing standard deviation, 7.82 dB for 38.901
UMi NLoS against the monograph's 4 dB, which lengthens the upper tail that a
maximum over 100 or 400 APs then selects from. The smaller is an artefact of the
backend rather than of the model: as `sionna_channel.py` documents, Sionna
exposes no second-order statistic, so `beta` is estimated from a single
realization by averaging `|h|^2` over the `N` antennas and `Q` subcarriers, while
the analytical backend returns the true `beta`. With `Q = 1` that estimate is
noisy, and it is noisiest in scenario A where `N = 1` leaves a single sample.
This is consistent with scenario A showing both the wider spread (+4.6 dB against
+3.6) and the larger best-AP shift.

The consequence is strongly scheme-dependent, and in a direction worth
remembering when reading any result from the default backend. Centralized
precoding gains about 20% in median SE, because coherent joint transmission is
precisely what converts a wider gain distribution into array gain. Distributed
L-RZF gains 6 to 8%, and MR is essentially indifferent at 1 to 6%, since both are
interference-limited rather than gain-limited. The 95%-likely SE barely moves for
the distributed schemes at all, and in scenario B it moves slightly the wrong way
(−0.01 and −0.05 bit/s/Hz), because a wider gain distribution helps the users
with a strong serving AP and does nothing for the ones without.

### What is checked exactly

52 hard checks pass before any comparison is attempted. They cover the parameters
of Table 5.1 (including that the noise power comes out at −94 dBm and the median
channel gain at 1 km at −140.6 dB), the wrap-around metric, the shadow-fading
statistics against eq. (2.21) both analytically and empirically over 4000 draws,
the Hermitian, positive-semidefinite, Toeplitz and trace-normalization properties
of the correlation matrices, the convergence of the empirical channel covariance
to `R_kl`, the 200 mW per-AP power budget under all three precoders, and the two
power-control formulas against their closed forms.
