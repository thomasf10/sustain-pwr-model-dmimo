"""Figure 2c: ergodic DL/UL sum rate vs number of RF chains.

By default this plots the cached rate table (fast, no Sionna). Pass
``--recompute`` to re-run the Monte-Carlo channel simulation with Sionna and
refresh the cache first (slow; a GPU is strongly recommended).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fr3_power import plotting, rates as rates_mod

# --- Optionally regenerate the cache -------------------------------------
if "--recompute" in sys.argv:
    from fr3_power.rate_model import RateConfig, compute_and_save
    compute_and_save(RateConfig())

# --- Load and plot -------------------------------------------------------
rate_table = rates_mod.load()
params_K = 8
P_T_DL, P_T_UL = 100, 100      # W, mW (for the title only)
M_ant = 1024

plotting.setup_style(use_tex=False)
title = (rf"$M_\mathrm{{ant}}={M_ant}$, $P_\mathrm{{T,DL}}={P_T_DL}$ W, "
         rf"$P_\mathrm{{T,UL}}={P_T_UL}$ mW, $K={params_K}$")
plotting.plot_rates(rate_table.M_RF, rate_table.R_DL, rate_table.R_UL, title=title)

import matplotlib.pyplot as plt
plt.show()
