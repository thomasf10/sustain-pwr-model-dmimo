"""Figure 2b: power-amplifier consumption vs number of antennas.

Fully-digital beamforming (M_RF = M_ant); M_ant swept from 16 to 1024. The
total transmit power scales with M_ant so that the per-PA power Pa stays at
about P_max = 0.1 W.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fr3_power import OperatingPoint, PowerParams, compute
from fr3_power import plotting, rates as rates_mod

# --- Scenario ------------------------------------------------------------
xbar_DL = xbar_UL = 1.0
params = PowerParams()
rate_table = rates_mod.load()
M_ant_vec = rate_table.M_RF        # same set of values, fully-digital sweep

# --- Sweep ---------------------------------------------------------------
pa = []
for M_ant in M_ant_vec:
    i = rate_table.index_of(M_ant)
    P_T = 100.0 * M_ant / 1024     # keeps Pa ~ P_max
    op = OperatingPoint(M_ant=int(M_ant), M_RF=int(M_ant), P_T=P_T,
                        R_DL=rate_table.R_DL[i], R_UL=rate_table.R_UL[i],
                        xbar_DL=xbar_DL, xbar_UL=xbar_UL)
    pa.append(compute(params, op).pa)

# --- Plot ----------------------------------------------------------------
plotting.setup_style(use_tex=False)
title = rf"$P_\mathrm{{a}} \approx 0.1 = P_\mathrm{{max}}$ W, $K={params.K}$"
plotting.plot_pa(M_ant_vec, pa, title=title)

import matplotlib.pyplot as plt
plt.show()
