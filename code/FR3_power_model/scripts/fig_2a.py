"""Figure 2a: digital and analog consumption vs number of RF chains.

M_ant = 1024 fixed; M_RF swept from 16 to 1024; total transmit power 100 W.
(The PA consumption depends only on M_ant and is shown in Fig 2b.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fr3_power import OperatingPoint, PowerParams, compute, energy_efficiency
from fr3_power import plotting, rates as rates_mod

# --- Scenario ------------------------------------------------------------
M_ant = 1024
P_T = 100.0          # total DL transmit power [W]
xbar_DL = xbar_UL = 1.0

params = PowerParams()
rate_table = rates_mod.load()
M_RF_vec = rate_table.M_RF

# --- Sweep ---------------------------------------------------------------
digital, analog, total_cons, ee = [], [], [], []
for M_RF in M_RF_vec:
    i = rate_table.index_of(M_RF)
    op = OperatingPoint(M_ant=M_ant, M_RF=int(M_RF), P_T=P_T,
                        R_DL=rate_table.R_DL[i], R_UL=rate_table.R_UL[i],
                        xbar_DL=xbar_DL, xbar_UL=xbar_UL)
    b = compute(params, op)
    digital.append(b.digital)
    analog.append(b.analog)
    total_cons.append(b.total)
    ee.append(energy_efficiency(op, b) * 1e-6)   # [Mbit/J]

# --- Report energy efficiency --------------------------------------------
print(f"DL load = {round(xbar_DL, 2)}, UL load = {round(xbar_UL, 2)}")
print("M_RF = {" + ", ".join(str(int(m)) for m in M_RF_vec) + "}")
print("EE   = {" + ", ".join(str(round(x, 2)) for x in ee) + "} Mbit/J")

# --- Plot ----------------------------------------------------------------
plotting.setup_style(use_tex=False)
title = (rf"$M_\mathrm{{ant}}={M_ant}$, $B={int(params.B * 1e-6)}$ MHz, "
         rf"$P_\mathrm{{T,DL}}={int(P_T)}$ W, $K={params.K}$")
plotting.plot_digital_analog(M_RF_vec, digital, analog, title=title)

import matplotlib.pyplot as plt
plt.show()
