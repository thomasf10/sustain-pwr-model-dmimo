"""Model parameters and per-point operating state.

All defaults reproduce the FR3 evaluation of Section 2.3 of the manuscript
(1024 antennas, B = 400 MHz, K = 8 users, TDD with tau_DL = 0.75, ...).

Two dataclasses:
    * ``PowerParams``  -- hardware / model constants, shared across all figures.
                          A single source of truth: change a value once here.
    * ``OperatingPoint`` -- the quantities that vary along a sweep (number of
                          antennas / RF chains, transmit power, loads, rates).

The whole evaluation assumes all antennas and RF chains are active for the
full frame, i.e. M_a = M_ant and M_RF,a = M_RF (Section 2.3). Hence the
"sleep mode" terms of eqs (2.23)/(2.26)/(2.33) are identically zero and are
not modelled here.
"""

from __future__ import annotations

from dataclasses import dataclass

SPEED_OF_LIGHT = 3e8  # [m/s]


@dataclass
class PowerParams:
    """Hardware and model constants (FR3 defaults, Section 2.3)."""

    # --- General system ---------------------------------------------------
    K: int = 8                  # Number of users
    f_c: float = 10e9           # Carrier frequency [Hz]
    Delta_f: float = 120e3      # Subcarrier spacing [Hz]
    B: float = 400e6            # Bandwidth [Hz]
    Q_IFFT: int = 4096          # IFFT/FFT size
    mu: float = 0.9             # Effective sampling-frequency factor
    B_tilde_factor: float = 0.9  # Effective-bandwidth factor (B_tilde = factor * B)

    # --- Supply & cooling efficiencies (eq 2.19) --------------------------
    eta_dig_sc: float = 0.81
    eta_ana_sc: float = 0.81
    eta_PA_sc: float = 0.81

    # --- Frame timing (Fig 2.2) -------------------------------------------
    tau_DL: float = 0.75        # DL duration / frame duration
    tau_DLsig: float = 1 / 14   # DL signalling / DL duration
    tau_ULsig: float = 1 / 14   # UL signalling / UL duration
    zeta_DLsig: float = 1 / 12  # Signalling power / maximum transmit power

    # --- Power amplifier (eqs 2.20-2.24) ----------------------------------
    P_max: float = 0.1          # Maximum PA output power [W]
    xi: float = 0.1             # Static-to-dynamic PA consumption weight (xi_PA)
    eta_PAmax: float = 0.15     # Maximum PA efficiency
    alpha: float = 0.75         # PA consumption exponent

    # --- Analog front-end (eqs 2.27-2.31) ---------------------------------
    b_DAC: int = 8              # DAC effective number of bits
    f_DAC: float = 5e9          # DAC sampling frequency [Hz]
    Xi_DAC_1: float = 1.5e-5    # DAC parameter 1 (eq 2.29)
    Xi_DAC_2: float = 1.5e-12   # DAC parameter 2 (eq 2.29)
    b_ADC: int = 8              # ADC effective number of bits
    f_ADC: float = 5e9          # ADC sampling frequency [Hz]
    W_ADC: float = 70e-15       # Walden figure-of-merit of ADC [J/conv. step]
    Xi_LNA: float = 2.7e-11     # LNA figure of merit
    Xi_mix: float = 2.5e-13     # Mixer figure of merit
    Xi_PS: float = 3.5e-11      # Phase-shifter figure of merit
    P_LO: float = 40e-3         # Local-oscillator consumption [W]
    P_filterRF: float = 5e-3    # RF-filter consumption [W]
    P_ana_misc: float = 0.0     # Miscellaneous fixed analog consumption (eq 2.25)

    # --- Digital processing (eqs 2.32-2.39) -------------------------------
    dist_max: float = 300.0     # Max path-length difference TX->RX [m]
    vel: float = 30.0           # Receiver velocity [m/s]
    Xi_DPD: float = 50.0        # GOPS per sample of digital predistortion
    n_filterBB: int = 20        # Baseband-filter taps
    o_filterBB: int = 4         # Baseband-filter oversampling factor
    rf_chains_per_fpga: int = 32  # RF chains sharing one FPGA (static-power step)
    P_encoder_s: float = 0.1    # Encoder static consumption [W]
    P_decoder_s: float = 0.1    # Decoder static consumption [W]
    P_IFFT_s: float = 0.1       # IFFT/FFT static consumption [W]
    P_DPD_s: float = 0.1        # DPD static consumption [W]
    P_dig_link: float = 0.0     # Link-layer fixed consumption (eq 2.32)
    # Computational efficiencies [complex OPS/W] (= GOPS/W * 1e9)
    eta_encoder: float = 2e12   # ASIC, 0.2e4 GOPS/W
    eta_decoder: float = 2e12   # ASIC, 0.2e4 GOPS/W
    eta_precoder: float = 0.2e12  # FPGA, 0.2e3 GOPS/W
    eta_combiner: float = 0.2e12  # FPGA, 0.2e3 GOPS/W
    eta_IFFT: float = 2e12      # ASIC, 0.2e4 GOPS/W
    eta_DPD: float = 2e12       # ASIC, 0.2e4 GOPS/W
    eta_filterBB: float = 0.2e12  # FPGA, 0.2e3 GOPS/W

    # --- Reduction factors (working->micro-sleep / idle) ------------------
    delta_dig_micro: float = 0.5
    delta_dig_idle: float = 0.25
    delta_ana_micro: float = 0.75
    delta_ana_idle: float = 0.5
    delta_PA_micro: float = 0.5
    delta_PA_idle: float = 0.25

    # --- Derived quantities ----------------------------------------------
    @property
    def tau_UL(self) -> float:
        """UL duration / frame duration (TDD complement of tau_DL)."""
        return 1 - self.tau_DL

    @property
    def lda_c(self) -> float:
        """Wavelength at the carrier frequency [m]."""
        return SPEED_OF_LIGHT / self.f_c

    @property
    def B_tilde(self) -> float:
        """Effective bandwidth [Hz]."""
        return self.B_tilde_factor * self.B

    @property
    def f_sI(self) -> float:
        """First sampling frequency (constellation rate) [Hz]."""
        return self.mu * self.B

    @property
    def f_sII(self) -> float:
        """Second sampling frequency (IFFT/FFT/DPD/BB rate) [Hz]."""
        return self.Q_IFFT * self.Delta_f

    @property
    def upsilon_coh(self) -> float:
        """Number of samples per coherence interval (iota_coh, eq 2.39)."""
        return (SPEED_OF_LIGHT / self.dist_max) * (self.lda_c / (2 * self.vel))

    @property
    def Xi_filterBB(self) -> float:
        """GOPS per sample of the baseband filter."""
        return self.n_filterBB * self.o_filterBB


@dataclass
class OperatingPoint:
    """State of one point along a sweep.

    Args:
        M_ant: Number of BS antennas (and PAs in DL / LNAs in UL).
        M_RF: Number of RF chains. ``M_RF == M_ant`` selects fully-digital
            beamforming; otherwise hybrid partially-connected beamforming.
        P_T: Total DL transmit power across all antennas [W].
        R_DL: DL ergodic sum rate [bit/s] (from the rate model).
        R_UL: UL ergodic sum rate [bit/s].
        xbar_DL: Average DL physical resource load in [0, 1].
        xbar_UL: Average UL physical resource load in [0, 1].

    Note on the load ``xbar_i``:
        ``xbar_i`` is the *active-subcarrier ratio*: the average fraction of
        allocated data subcarriers, averaged over the K served users (eq 1 of
        the paper). The frame averaging, however, weights the data mode by the
        *active-resource time ratio* N_a,i / N_i -- the fraction of the frame
        over which the RF chains/antennas actually carry data, which is what
        decides the data-vs-micro-sleep split and is independent of the number
        of users. These two quantities are equal only when the system is fully
        loaded in the spatial/user domain, K_i = K_max (every stream
        scheduled). Hence using ``xbar_i`` as the data-mode weight is exact for
        K_i = K_max -- the regime evaluated here -- and only approximate when
        fewer users are served. Serving K_i < K_max changes the transmit power
        and the rate, but not whether a resource is active.
    """

    M_ant: int
    M_RF: int
    P_T: float
    R_DL: float
    R_UL: float
    xbar_DL: float = 1.0
    xbar_UL: float = 1.0

    @property
    def Pa(self) -> float:
        """Transmit power per power amplifier [W]."""
        return self.P_T / self.M_ant

    @property
    def M_PS(self) -> float:
        """Phase shifters per RF chain.

        Zero for fully-digital beamforming (``M_RF == M_ant``); otherwise
        ``M_ant / M_RF`` for hybrid partially-connected beamforming.
        """
        if self.M_RF == self.M_ant:
            return 0.0
        return self.M_ant / self.M_RF
