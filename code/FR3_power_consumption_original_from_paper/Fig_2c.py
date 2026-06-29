import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy.linalg as la
plt.rcParams.update({'font.size':15, 'text.usetex':True, 'font.family':'serif'})
params = {"xtick.direction": "in", "ytick.direction": "in"}
plt.rcParams.update(params)
plt.close('all')

import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import sionna
except ImportError as e:
    import sys
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print('gpus')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')
sionna.phy.config.seed = 42     # for reproducibility
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.channel.tr38901 import Antenna, AntennaArray, UMi
from sionna.phy.channel import gen_single_sector_topology as gen_topology
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel


def ht(A):
    # Compute the Hermitian transpose of a matrix A
    return np.conjugate(A.T)
def gen_codebook_1D(m):
    # Generate a one-dimensional discrete Fourier transform codebook
    m_range = np.arange(m)
    W = np.exp(-1j*2*np.pi*np.outer(m_range, m_range)/m)
    return W/np.sqrt(m)
def gen_codebook_2D(m_hor, m_ver):
    # Generate a two-dimensional discrete Fourier transform codebook
    W_ver = gen_codebook_1D(m_ver); W_hor = gen_codebook_1D(m_hor)
    codebook = []
    for m in range(m_ver):
        for n in range(m_hor):
            wv = W_ver[:,m]    # vertical beam
            wh = W_hor[:,n]    # horizontal beam
            w_2d = np.kron(wh, wv)
            codebook.append(w_2d)
    return np.array(codebook)
def subarray_beam_sel(m_hor, m_ver, m_rf, hmat, direction):
    # Select the best beam for each subarray in a hybrid partially-connected beamforming architecture
    if np.sqrt(m_hor*m_ver/m_rf) - np.round(np.sqrt(m_hor*m_ver/m_rf)) == 0:
        M_s_hor = M_s_ver = int(np.sqrt(m_hor*m_ver/m_rf))
    else:
        rd = np.random.uniform(0,1)
        if rd > 0.5:
            M_s_hor = int(2**np.ceil(np.log2(np.sqrt(m_hor*m_ver/m_rf)))); M_s_ver = int(np.round(m_hor*m_ver/m_rf)/M_s_hor)
        else:
            M_s_ver = int(2**np.ceil(np.log2(np.sqrt(m_hor*m_ver/m_rf)))); M_s_hor = int(np.round(m_hor*m_ver/m_rf)/M_s_ver)
    codebook = gen_codebook_2D(M_s_hor,M_s_ver)
    map = np.reshape(np.arange(m_hor*m_ver),[m_hor,m_ver]).T
    if direction == "downlink":
        W_RF = np.zeros((m_hor*m_ver,m_rf), dtype=complex)
    else:
        W_RF = np.zeros((m_rf,m_hor*m_ver), dtype=complex)
    l = 0
    for i in range(int(m_hor/M_s_hor)):
        for j in range(int(m_ver/M_s_ver)):
            subarray_idx = map[j*M_s_ver:(j+1)*M_s_ver,i*M_s_hor:(i+1)*M_s_hor].T.reshape(-1)
            if direction == "downlink":
                H_sub = hmat[:,subarray_idx]
                best_idx = np.argmax([np.mean(np.abs(H_sub@codebook[b,:])**2) for b in range(codebook.shape[0])])
                W_RF[subarray_idx,l] = codebook[best_idx,:]
            else:
                H_sub = hmat[subarray_idx,:]
                best_idx = np.argmax([np.mean(np.abs(codebook[b,:]@H_sub)**2) for b in range(codebook.shape[0])])
                W_RF[l,subarray_idx] = codebook[best_idx,:]
            l += 1
    return W_RF
def gen_channel(n_iter, m_hor, m_ver, k, direction):
    # Generate a channel from 3GPP 39.801 Urban Micro channel model
    scenario = "umi"
    ut_array = Antenna(polarization="single",polarization_type="V",antenna_pattern="omni",carrier_frequency=f_c)
    bs_array = AntennaArray(num_rows=m_ver,num_cols=m_hor,polarization="single",polarization_type="V",
                            antenna_pattern="38.901",carrier_frequency=f_c)
    channel_model = UMi(carrier_frequency=f_c,o2i_model="low",ut_array=ut_array,bs_array=bs_array,direction=direction,
                        enable_pathloss=True,enable_shadow_fading=False,always_generate_lsp=False)
    topology = gen_topology(n_iter,k,scenario); channel_model.set_topology(*topology)
    num_streams_per_tx = 1; rx_tx_association = np.zeros([1,k]); rx_tx_association[0,:] = 1
    rg = ResourceGrid(num_ofdm_symbols=14,fft_size=Q_IFFT,subcarrier_spacing=Delta_f,num_tx=int(k),
                      num_streams_per_tx=num_streams_per_tx,cyclic_prefix_length=20,
                      pilot_pattern="kronecker",pilot_ofdm_symbol_indices=[2,11])
    frequencies = subcarrier_frequencies(rg.fft_size,rg.subcarrier_spacing)
    cir = channel_model(1,1); h = cir_to_ofdm_channel(frequencies,*cir,normalize=False)
    h = tf.squeeze(h)
    return h


### Initialize parameters
M_ant = 1024                                            # Number of BS antennas
M_RF_vec = np.array([16,32,64,128,256,512,1024])        # Vector of number of RF chains
K = 8                                                   # Number of users
f_c = 10e9                                              # Carrier frequency [Hz]
Delta_f = 120e3                                         # Subcarrier spacing [Hz]
B = 400e6                                               # Bandwidth [Hz]
B_tilde = 0.9*B                                         # Effective bandwidth [Hz]
Q_IFFT = 4096                                           # IFFT/FFT size
Q = 3000                                                # Number of subcarriers carrying data symbols
# Q_IFFT = 32
P_T_DL = 100                                            # Total transmit power at base station [W]
P_T_UL = 100e-3                                         # Transmit power at each user [W]
tau_DL = 0.75                                           # Ratio of downlink duration to frame duration
tau_DLsig = 1/14                                        # Ratio of downlink signaling duration to downlink duration
tau_UL = 1-tau_DL                                       # Ratio of uplink duration to frame duration
tau_ULsig = 1/14                                        # Ratio of uplink signaling duration to uplink duration
lda_c = 3e8/f_c                                         # Wavelength at carrier frequency [m]
d_ant = lda_c/2                                         # Antenna spacing [m]
k_B = 1.3806491e-23                                     # Boltzmann constant [J/K]
T_n = 290                                               # Thermal noise temperature [K]
F_n = 10**(9/10)                                        # Thermal noise figure
sigma2_n = k_B*T_n*F_n*B                                # Thermal noise variance
N_iter = int(1e3)                                       # Number of channel realizations
if np.sqrt(M_ant)-np.round(np.sqrt(M_ant)) == 0:
    M_hor = M_ver = int(np.sqrt(M_ant))                 # Number of horizontal and vertical elements in uniform planar array
else:
    M_hor = int(2**np.ceil(np.log2(np.sqrt(M_ant))))    # Number of horizontal elements in uniform planar array
    M_ver = int(M_ant/M_hor)                            # Number of vertical elements in uniform planar array

# Vectors containing the results
R_DL = np.zeros(len(M_RF_vec))                          # Ergodic sum rate in downlink [bit/s]
R_UL = np.zeros(len(M_RF_vec))                          # Ergodic sum rate in uplink [bit/s]


### Iterate over number of RF chains
for m_RF_index in range(len(M_RF_vec)):
    M_RF = M_RF_vec[m_RF_index]
    M_s = M_ant//M_RF                                   # Number of antenna elements per subarray

    ## Iterate over number of channel realizations
    for iter_index in range(N_iter):
        print('M_RF =', M_RF,
              ', iter =', iter_index,'of',N_iter)

        H_DL = np.zeros([1,K,M_ant,Q],dtype=complex)    # Downlink channel tensor of dimension K×M_ant×Q
        sionna.phy.config.seed = np.round(np.random.uniform(0,5000))
        tmp = gen_channel(1,M_hor,M_ver,K,"downlink")
        H_DL[0,:,:,:] = tmp[:,:,0:Q]
        H_UL = np.zeros([1,M_ant,K,Q],dtype=complex)    # Uplink channel tensor of dimension M_ant×K×Q
        # sionna.phy.config.seed = np.round(np.random.uniform(0,5000))
        tmp = gen_channel(1,M_hor,M_ver,K,"uplink")
        H_UL[0,:,:,:] = tmp[:,:,0:Q]

        ## Iterate over subcarriers
        for q_index in range(Q):

            ## Downlink
            Hmat = np.zeros([K,M_ant],dtype=complex)    # Channel matrix of dimension K×M_ant
            Hmat[:,:] = np.copy(H_DL[0,:,:,q_index])
            # Normalize channel matrix with respect to noise power
            Hmat = Hmat/np.sqrt(sigma2_n)

            if M_RF == M_ant:
                # Fully-digital beamforming
                # Compute pseudo-inverse of Hmat
                Wmat_dig = ht(Hmat) @ la.inv(Hmat @ ht(Hmat))
                # Normalize digital precoder to have unitary power
                normaliz_factor = la.norm(Wmat_dig,'fro')/np.sqrt(P_T_DL)
                Wmat_dig = Wmat_dig/normaliz_factor
                # Compute sum rate
                for k in range(K):
                    # Desired signal power
                    DS = np.abs((Hmat@Wmat_dig)[k,k])**2
                    # Inter-user interference power
                    IUI = np.sum(np.abs((Hmat@Wmat_dig)[k,:])**2) - np.abs((Hmat@Wmat_dig)[k,k])**2
                    # Spectral efficiency
                    R_DL[m_RF_index] += np.log2(1 + DS/(IUI+1))
            else:
                # Hybrid partially-connected beamforming
                # Compute analog beamformer
                Wmat_ana = subarray_beam_sel(M_hor,M_ver,M_RF,Hmat,"downlink")
                # Compute effective channel matrix after analog beamforming
                Hmat_eff = Hmat @ Wmat_ana
                # Compute pseudo-inverse of Hmat
                Wmat_dig = ht(Hmat_eff) @ la.inv(Hmat_eff @ ht(Hmat_eff))
                # Normalize digital precoder to have unitary power
                normaliz_factor = la.norm(Wmat_ana@Wmat_dig,'fro')/np.sqrt(P_T_DL)
                Wmat_dig = Wmat_dig/normaliz_factor
                # Compute sum rate
                for k in range(K):
                    # Desired signal power
                    DS = np.abs((Hmat_eff@Wmat_dig)[k,k])**2
                    # Inter-user interference power
                    IUI = np.sum(np.abs((Hmat_eff@Wmat_dig)[k,:])**2) - np.abs((Hmat_eff@Wmat_dig)[k,k])**2
                    # Spectral efficiency
                    R_DL[m_RF_index] += np.log2(1 + DS/(IUI+1))

            ## Uplink
            Hmat = np.zeros([M_ant,K],dtype=complex)    # Channel matrix of dimension M_ant×K
            Hmat[:,:] = np.copy(H_UL[0,:,:,q_index])
            # Normalize channel matrix with respect to noise power
            Hmat = Hmat/np.sqrt(sigma2_n)

            if M_RF == M_ant:
                # Fully-digital beamforming
                # Compute pseudo-inverse of Hmat
                Vmat_dig = la.inv(ht(Hmat) @ Hmat) @ ht(Hmat)
                # Compute sum rate
                for k in range(K):
                    # Desired signal power
                    DS = P_T_UL*np.abs((Vmat_dig@Hmat)[k,k])**2
                    # Inter-user interference power
                    IUI = P_T_UL*(np.sum(np.abs((Vmat_dig@Hmat)[k,:])**2) - np.abs((Vmat_dig@Hmat)[k,k])**2)
                    # Spectral efficiency
                    R_UL[m_RF_index] += np.log2(1 + DS/(IUI+1*la.norm(Vmat_dig[k,:])**2))
            else:
                # Hybrid partially-connected beamforming
                # Compute analog beamformer
                Vmat_ana = subarray_beam_sel(M_hor,M_ver,M_RF,Hmat,"uplink")
                # Compute effective channel matrix after analog beamforming
                Hmat_eff = Vmat_ana @ Hmat
                # Compute pseudo-inverse of Hmat
                Vmat_dig = la.inv(ht(Hmat_eff) @ Hmat_eff) @ ht(Hmat_eff)
                # Compute sum rate
                for k in range(K):
                    # Desired signal power
                    DS = P_T_UL*np.abs((Vmat_dig @ Hmat_eff)[k,k])**2
                    # Inter-user interference power
                    IUI = P_T_UL*(np.sum(np.abs((Vmat_dig @ Hmat_eff)[k,:])**2) - np.abs((Vmat_dig @ Hmat_eff)[k,k])**2)
                    # Spectral efficiency
                    R_UL[m_RF_index] += np.log2(1 + DS/(IUI+1*la.norm(Vmat_dig[k,:]@Vmat_ana)**2))

# Compute ergodic rates in bit/s
R_DL = R_DL/N_iter
R_DL = tau_DL*(1-tau_DLsig)*(B_tilde/Q)*R_DL
R_UL = R_UL/N_iter
R_UL = tau_UL*(1-tau_ULsig)*(B_tilde/Q)*R_UL


### Plot results
fig, ax = plt.subplots()
xax = []
for i in range(len(M_RF_vec)):
    xax = [i*3-.4,i*3+.4]
    yax = [R_UL[i]*1e-9, R_DL[i]*1e-9]
    bar_colors_1 = [mcolors.to_rgba(c, alpha=0.7) for c in ['tab:cyan', 'tab:blue']]
    p1 = ax.bar(xax, yax, edgecolor='black', color=bar_colors_1, hatch=[None,'///'])
ax.set_ylabel('Ergodic sum rate, $R_i$ [Gbit/s]')
ax.set_xticks([0,3,6,9,12,15,18],M_RF_vec)      # comment out if creates errors. Matplotlib version used for this code is 3.10.8
# ax.set_xticks([0,3,6,9],M_RF_vec)
ax.set_xlabel('Number of RF chains, $M_\\mathsf{\\scriptstyle{RF}}$')
plt.grid(linestyle=':',alpha=0.5)
plt.legend((p1[0],p1[1]),
           ('Uplink, $i=\\mathsf{UL}$','Downlink, $i=\\mathsf{DL}$'),
           loc='upper left',fontsize=15)
ax.set_title('$M_\\mathsf{ant}=$ '+str(M_ant)+
             ', $P_\\mathsf{\\scriptstyle{T,DL}}=$ '+str(P_T_DL)+' W'+
             ', $P_\\mathsf{\\scriptstyle{T,UL}}=$ '+str(int(P_T_UL*1e3))+' mW'+
             ', $K=$ '+str(K),fontsize=15)
plt.tight_layout()
plt.show()
