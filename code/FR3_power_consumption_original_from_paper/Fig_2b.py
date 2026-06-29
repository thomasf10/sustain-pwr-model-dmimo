import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams.update({'font.size':15, 'text.usetex':True, 'font.family':'serif'})
params = {"xtick.direction": "in", "ytick.direction": "in"}
plt.rcParams.update(params)
plt.close('all')


def P_PA(p,P_max,eta_PAmax):
    # Compute the active-mode PA consumption
    return xi*P_max/eta_PAmax + (1-xi)*P_max**(1-alpha)*p**alpha/eta_PAmax


### Initialize parameters
M_ant_vec = np.array([16,32,64,128,256,512,1024])       # Vector of number of BS antennas
K = 8                                                   # Number of users
xbar_DL = 1                                             # Downlink average physical load
xbar_UL = 1                                             # Uplink average physical load
# xbar_DL = .3                                             # Downlink average physical load
# xbar_UL = .3                                             # Uplink average physical load
f_c = 10e9                                              # Carrier frequency [Hz]
lda_c = 3e8/f_c                                         # Wavelength at carrier frequency [m]
Delta_f = 120e3                                         # Subcarrier spacing [Hz]
B = 400e6                                               # Bandwidth [Hz]
B_tilde = 0.9*B                                         # Effective bandwidth [Hz]
Q_IFFT = 4096                                           # IFFT/FFT size
mu = 0.9                                                # Effective sampling frequency factor
f_sI = mu*B                                             # First sampling frequency [Hz]
f_sII = Q_IFFT*Delta_f                                  # Second sampling frequency [Hz]

eta_dig_sc = 0.81                                       # Supply and cooling efficiency of digital
eta_PA_sc = 0.81                                        # Supply and cooling efficiency of power amplifier
eta_ana_sc = 0.81                                       # Supply and cooling efficiency of analog

tau_DL = 0.75                                           # Ratio of downlink duration to frame duration
tau_DLsig = 1/14                                        # Ratio of downlink signaling duration to downlink duration
zeta_DLsig = 1/12                                       # Ratio of signaling power to maximum transmit power
tau_UL = 1-tau_DL                                       # Ratio of uplink duration to frame duration
tau_ULsig = 1/14                                        # Ratio of uplink signaling duration to uplink duration

P_max = 0.1                                             # Maximum power amplifier output power [W]
xi = 0.1                                                # Weight of static to dynamic power amplifier consumption
eta_PAmax = 0.15                                        # Maximum power amplifier efficiency
alpha = 0.75                                            # Power amplifier consumption exponent

b_DAC = 8                                               # Effective number of bits of digital-to-analog converter
f_DAC = 5e9                                             # Sampling frequency of digital-to-analog converter
b_ADC = 8                                               # Effective number of bits of analog-to-digital converter
f_ADC = 5e9                                             # Sampling frequency of analog-to-digital converter
W_ADC = 70e-15                                          # Walden's figure of merit of analog-to-digital converter [J/cs]
Xi_LNA = 2.7e-11                                        # Figure of merit of low-noise amplifier
Xi_mix = 2.5e-13                                        # Figure of merit of mixer
Xi_PS = 3.5e-11                                         # Figure of merit of phase shifter
P_LO = 40e-3                                            # Power consumption of local oscillator [W]
P_filterRF = 5e-3                                       # Power consumption of radio-frequency filter [W]

R_DL = np.array([1.6,3.64,6.06,8.10,                    # Ergodic sum rate in downlink [bit/s], input from Fig_2c.py
                 10.93,12.10,13.44])*1e9*0.93
R_UL = np.array([1.02,1.25,1.5,1.68,                    # Ergodic sum rate in uplink [bit/s], input from Fig_2c.py
                 1.98,2.18,2.35])*1e9*0.93
dist_max = 300                                          # Maximum length difference between paths from transmitter to receiver [m]
vel = 30                                                # Receiver velocity [m/s]
upsilon_coh = (3e8/dist_max)*(lda_c/(2*vel))            # Number of samples per coherence interval
Xi_DPD = 50                                             # Number of Giga operations per second per sample of digital predistortion
n_filterBB = 20                                         # Number of taps of baseband filter
o_filterBB = 4                                          # Oversampling factor of baseband filter
Xi_filterBB = n_filterBB*o_filterBB                     # Number of Giga operations per second per sample of baseband filter
P_encoder_s = .1                                        # Static power consumption of encoder [W]
P_decoder_s = .1                                        # Static power consumption of decoder [W]
P_IFFT_s = .1                                           # Static power consumption of IFFT/FFT [W]
P_DPD_s = .1                                            # Static power consumption of DPD [W]
eta_encoder = 2e12                                      # Computational efficiency of encoder [GOPS/W]
eta_decoder = 2e12                                      # Computational efficiency of encoder [GOPS/W]
eta_precoder = 0.2e12                                   # Computational efficiency of precoder [GOPS/W]
eta_combiner = 0.2e12                                   # Computational efficiency of combiner [GOPS/W]
eta_IFFT = 2e12                                         # Computational efficiency of IFFT/FFT [GOPS/W]
eta_DPD = 0.2e12                                        # Computational efficiency of digital predistortion [GOPS/W]
eta_filterBB = 0.2e12                                   # Computational efficiency of baseband filter [GOPS/W]

delta_dig_micro = 0.5                                   # Power reduction factor of digital's micro-sleep mode
delta_dig_idle = 0.25                                   # Power reduction factor of digital's idle mode
delta_ana_micro = 0.75                                  # Power reduction factor of analog's micro-sleep mode
delta_ana_idle = 0.5                                    # Power reduction factor of analog's idle mode
delta_PA_micro = 0.5                                    # Power reduction factor of power amplifier's micro-sleep mode
delta_PA_idle = 0.25                                    # Power reduction factor of power amplifier's idle mode

# Vectors containing the results
Pbar_PA_load_ind = np.zeros(len(M_ant_vec))             # Load-independent part of power amplifier's consumption [W]
Pbar_PA_load_dep = np.zeros(len(M_ant_vec))             # Load-dependent part of power amplifier's consumption [W]
Pbar_dig_load_ind = np.zeros(len(M_ant_vec))            # Load-independent part of digital's consumption [W]
Pbar_dig_load_dep = np.zeros(len(M_ant_vec))            # Load-dependent part of digital's consumption [W]
Pbar_ana_load_ind = np.zeros(len(M_ant_vec))            # Load-independent part of analog's consumption [W]
Pbar_ana_load_dep = np.zeros(len(M_ant_vec))            # Load-dependent part of analog's consumption [W]


### Iterate over the number of antennas
for m_ant_index in range(len(M_ant_vec)):
    M_ant = M_ant_vec[m_ant_index]                      # Number of BS antennas
    P_T = 100*M_ant/1024                                # Total transmit power at base station [W]
    Pa = P_T/M_ant                                      # Transmit power per power amplifier [W]

    M_RF = M_ant                                        # Number of RF chains
    M_PS = 0                                            # Number of phase shifters in fully-digital beamforming

    P_precoder_s = 1*np.ceil(M_RF/32)                   # Static power consumption of precoder [W]
    P_combiner_s = 1*np.ceil(M_RF/32)                   # Static power consumption of combiner [W]
    P_filterBB_s = 1*np.ceil(M_RF/32)                   # Static power consumption of baseband filter [W]

    ## Power amplifier
    # Frame-averaged power consumption of one power amplifier
    Pbar_PA1 = (xbar_DL*tau_DL*(1-tau_DLsig)*P_PA(Pa,P_max,eta_PAmax) +
                tau_DL*tau_DLsig*P_PA(zeta_DLsig*P_max,P_max,eta_PAmax) +
                tau_DL*(1-xbar_DL)*(1-tau_DLsig)*P_PA(0,P_max,eta_PAmax)*delta_PA_micro +
                (1-tau_DL)*P_PA(0,P_max,eta_PAmax)*delta_PA_idle)
    # Frame-averaged power consumption of M_ant power amplifiers
    Pbar_PA = M_ant*Pbar_PA1
    # Split between load-independent and load-dependent term
    Pbar_PA_load_ind[m_ant_index] = (Pbar_PA - M_ant*(xbar_DL*tau_DL*(1-tau_DLsig)*P_PA(Pa,P_max,eta_PAmax)-
                                                xbar_DL*tau_DL*(1-tau_DLsig)*P_PA(0,P_max,eta_PAmax)*delta_PA_micro))
    Pbar_PA_load_dep[m_ant_index] = Pbar_PA - Pbar_PA_load_ind[m_ant_index]
    Pbar_PA_load_ind[m_ant_index] = (1/eta_PA_sc)*Pbar_PA_load_ind[m_ant_index]
    Pbar_PA_load_dep[m_ant_index] = (1/eta_PA_sc)*Pbar_PA_load_dep[m_ant_index]

    ## Digital processing
    # Individual subcomponents
    P_enc = P_encoder_s + 1/eta_encoder*(f_sI*14/(3*8)*R_DL[m_ant_index]/B_tilde)
    P_dec = P_decoder_s + 1/eta_decoder*(f_sI*(5*35)/(2*3)*R_UL[m_ant_index]/B_tilde)
    P_pre = P_precoder_s + 1/eta_precoder*M_RF*(f_sI*2*K+f_sI/upsilon_coh*(K**3/(3*M_RF)+3*K**2+K))
    P_com = P_combiner_s + 1/eta_combiner*M_RF*f_sI*2*K
    P_IFFT = P_IFFT_s + 1/eta_IFFT*M_RF*f_sII*3/2*np.log2(Q_IFFT)
    P_DPD = P_DPD_s + 1/eta_DPD*M_RF*f_sII*Xi_DPD
    P_filBB = P_filterBB_s + 1/eta_filterBB*M_RF*f_sII*Xi_filterBB
    # All downlink subcomponents
    P_digDL = P_enc + P_pre + P_IFFT + P_DPD + P_filBB
    # Average over the different operating modes
    Pbar_digDL = (xbar_DL*tau_DL*(1-tau_DLsig)*P_digDL +
                  tau_DL*tau_DLsig*P_digDL +
                  tau_DL*(1-xbar_DL)*(1-tau_DLsig)*P_digDL*delta_dig_micro +
                  (1-tau_DL)*P_digDL*delta_dig_idle)
    # All uplink subcomponents
    P_digUL = P_dec + P_com + P_IFFT + P_filBB
    # Average over the different operating modes
    Pbar_digUL = (xbar_UL*tau_UL*(1-tau_ULsig)*P_digUL +
                  tau_UL*tau_ULsig*P_digUL +
                  tau_UL*(1-xbar_UL)*(1-tau_ULsig)*P_digUL*delta_dig_micro +
                  (1-tau_UL)*P_digUL*delta_dig_idle)
    # Add downlink and uplink
    Pbar_dig = Pbar_digDL+Pbar_digUL
    # Split between load-independent and load-dependent term
    Pbar_dig_load_ind[m_ant_index] = (Pbar_dig - xbar_DL*tau_DL*(1-tau_DLsig)*P_digDL+xbar_DL*tau_DL*(1-tau_DLsig)*P_digDL*delta_dig_micro -
                                      xbar_UL*tau_UL*(1-tau_ULsig)*P_digUL+xbar_UL*tau_UL*(1-tau_ULsig)*P_digUL*delta_dig_micro)
    Pbar_dig_load_dep[m_ant_index] = Pbar_dig - Pbar_dig_load_ind[m_ant_index]
    Pbar_dig_load_ind[m_ant_index] = (1/eta_dig_sc)*Pbar_dig_load_ind[m_ant_index]
    Pbar_dig_load_dep[m_ant_index] = (1/eta_dig_sc)*Pbar_dig_load_dep[m_ant_index]

    ## Analog processing
    # Individual subcomponents
    P_DAC = 1.5e-5*2**(b_DAC)+1.5e-12*b_DAC*f_DAC
    P_ADC = W_ADC*f_ADC*2**b_ADC
    P_mix = Xi_mix*f_c
    P_PS = Xi_PS*B
    P_LNA = Xi_LNA*B
    # All downlink subcomponents in one RF chain
    P_anaDL1 = 2*P_DAC + 2*P_filterRF + 2*P_mix + M_PS*P_PS
    # All downlink RF chains + local oscillator
    P_anaDL = P_LO + M_RF*P_anaDL1
    # Average over the different operating modes
    Pbar_anaDL = (xbar_DL*tau_DL*(1-tau_DLsig)*P_anaDL +
                  tau_DL*tau_DLsig*P_anaDL +
                  tau_DL*(1-xbar_DL)*(1-tau_DLsig)*P_anaDL*delta_ana_micro +
                  (1-tau_DL)*P_anaDL*delta_ana_idle)
    # All uplink subcomponents in one RF chain
    P_anaUL1 = 2*P_ADC + 2*P_filterRF + 2*P_mix + M_PS*P_PS + M_ant/M_RF*P_LNA
    # All uplink RF chains + local oscillator
    P_anaUL = P_LO + M_RF*P_anaUL1
    # Average over the different operating modes
    Pbar_anaUL = (xbar_UL*tau_UL*(1-tau_ULsig)*P_anaUL +
                  tau_UL*tau_ULsig*P_anaUL +
                  tau_UL*(1-xbar_UL)*(1-tau_ULsig)*P_anaUL*delta_ana_micro +
                  (1-tau_UL)*P_anaUL*delta_ana_idle)
    # Add downlink and uplink
    Pbar_ana = Pbar_anaDL + Pbar_anaUL
    # Split between load-independent and load-dependent term
    Pbar_ana_load_ind[m_ant_index] = (Pbar_ana - xbar_DL*tau_DL*(1-tau_DLsig)*P_anaDL+xbar_DL*tau_DL*(1-tau_DLsig)*P_anaDL*delta_ana_micro -
                                      xbar_UL*tau_UL*(1-tau_ULsig)*P_anaUL+xbar_UL*tau_UL*(1-tau_ULsig)*P_anaUL*delta_ana_micro)
    Pbar_ana_load_dep[m_ant_index] = Pbar_ana - Pbar_ana_load_ind[m_ant_index]
    Pbar_ana_load_ind[m_ant_index] = (1/eta_ana_sc)*Pbar_ana_load_ind[m_ant_index]
    Pbar_ana_load_dep[m_ant_index] = (1/eta_ana_sc)*Pbar_ana_load_dep[m_ant_index]


### Plot results
fig, ax = plt.subplots()
xax = []
for i in range(len(M_ant_vec)):
    xax = [i*3]
    yax_li = [Pbar_PA_load_ind[i]]
    yax_ld = [Pbar_PA_load_dep[i]]
    bar_colors_1 = [mcolors.to_rgba(c, alpha=0.9) for c in ['tab:red']]
    bar_colors_2 = [mcolors.to_rgba(c, alpha=0.4) for c in ['tab:red']]
    p1 = ax.bar(xax, yax_li, edgecolor='black', color=bar_colors_1)
    p2 = ax.bar(xax, yax_ld, edgecolor='black', bottom=yax_li, color=bar_colors_2)
ax.set_ylabel('Power consumption, $P_\\mathsf{cons}$ [W]')
ax.set_ylim([0,720])
ax.set_xticks([0,3,6,9,12,15,18],M_ant_vec)     # comment out if creates errors. Matplotlib version used for this code is 3.10.8
ax.set_xlabel('Number of antennas, $M_\\mathsf{ant}$')
plt.grid(linestyle=':',alpha=0.5)
plt.legend((p1[0],p2[0]),
           ('Power amplifier (load-indep.)','Power amplifier (load-dep.)'),
           loc='upper left',fontsize=15)
ax.set_title('$P_\\mathsf{a}\\approx 0.1 = P_\\mathsf{max}$ W'+
             ', $K=$ '+str(K),fontsize=15)
plt.tight_layout()
plt.show()
