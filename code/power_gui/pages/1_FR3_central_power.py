"""FR3 central base-station power model — interactive view.

Builds an ``OperatingPoint`` (and optionally overrides ``PowerParams``) from the
sidebar, then shows the per-component breakdown and energy efficiency for that
point, plus the swept manuscript figures (power vs number of RF chains / vs
number of antennas).
"""

from __future__ import annotations

import model_access  # noqa: F401  (bootstraps sys.path)
import streamlit as st

import fr3_power.plotting as plotting
from fr3_power import OperatingPoint, PowerParams, compute, energy_efficiency
from fr3_power import rates as rate_table

import plots
from widgets import edit_dataclass

st.set_page_config(page_title="FR3 central power", page_icon="⚡", layout="wide")
st.title("FR3 central base-station power model")

# Standard array sizes used by the manuscript sweeps.
ARRAY_SIZES = [16, 32, 64, 128, 256, 512, 1024]

# A curated subset of PowerParams surfaced by default in the advanced panel;
# the rest are reachable via "show all parameters".
KEY_PARAMS = [
    "K", "f_c", "B", "tau_DL",
    "P_max", "eta_PAmax", "alpha", "xi",
    "b_DAC", "b_ADC", "P_LO",
    "eta_dig_sc", "eta_ana_sc", "eta_PA_sc",
]


@st.cache_data
def load_rates():
    """Load the cached ergodic-rate table, or ``None`` if unavailable."""
    try:
        return rate_table.load()
    except Exception:  # missing file / bad JSON: fall back to manual rates
        return None


# --- Sidebar: operating point --------------------------------------------
st.sidebar.header("Operating point")

M_ant = st.sidebar.select_slider("Antennas $M_\\mathrm{ant}$", ARRAY_SIZES,
                                 value=1024)
rf_options = [m for m in ARRAY_SIZES if m <= M_ant]
M_RF = st.sidebar.select_slider("RF chains $M_\\mathrm{RF}$", rf_options,
                                value=min(64, M_ant))
P_T = st.sidebar.number_input("Total DL transmit power $P_T$ [W]", value=100.0,
                              min_value=0.0, step=10.0)
xbar_DL = st.sidebar.slider("DL load $\\bar x_\\mathrm{DL}$", 0.0, 1.0, 1.0)
xbar_UL = st.sidebar.slider("UL load $\\bar x_\\mathrm{UL}$", 0.0, 1.0, 1.0)

rt = load_rates()
st.sidebar.subheader("Rates")
rate_source = st.sidebar.radio(
    "Source", ["Cached table", "Manual"],
    index=0 if rt is not None else 1,
    help="The cached table is indexed by M_RF for a 1024-antenna array.")

if rate_source == "Cached table" and rt is not None:
    try:
        idx = rt.index_of(M_RF)
        R_DL, R_UL = float(rt.R_DL[idx]), float(rt.R_UL[idx])
        st.sidebar.caption(
            f"From table: $R_\\mathrm{{DL}}$ = {R_DL/1e9:.2f}, "
            f"$R_\\mathrm{{UL}}$ = {R_UL/1e9:.2f} Gbit/s")
    except KeyError:
        st.sidebar.warning(f"No cached rate for M_RF={M_RF}; enter rates manually.")
        rate_source = "Manual"

if rate_source == "Manual" or rt is None:
    R_DL = st.sidebar.number_input("DL sum rate $R_\\mathrm{DL}$ [Gbit/s]",
                                   value=5.6, min_value=0.0) * 1e9
    R_UL = st.sidebar.number_input("UL sum rate $R_\\mathrm{UL}$ [Gbit/s]",
                                   value=1.3, min_value=0.0) * 1e9

op = OperatingPoint(M_ant=M_ant, M_RF=M_RF, P_T=P_T, R_DL=R_DL, R_UL=R_UL,
                    xbar_DL=xbar_DL, xbar_UL=xbar_UL)

# --- Hardware parameters (advanced) --------------------------------------
with st.sidebar.expander("Hardware parameters"):
    show_all = st.checkbox("Show all parameters", value=False)
    params = edit_dataclass(
        PowerParams(),
        key_prefix="pp_",
        include=None if show_all else KEY_PARAMS,
        columns=1,
    )

# --- Single operating point ----------------------------------------------
b = compute(params, op)
ee = energy_efficiency(op, b)

st.subheader("Operating point")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total power", f"{b.total:.1f} W")
c2.metric("Digital", f"{b.digital.total:.1f} W")
c3.metric("Analog", f"{b.analog.total:.1f} W")
c4.metric("Power amplifier", f"{b.pa.total:.1f} W")
c5.metric("Energy efficiency", f"{ee*1e-6:.2f} Mbit/J")

left, right = st.columns([1, 1])
with left:
    st.pyplot(plots.breakdown_bar(b))
with right:
    st.markdown(
        f"""
**Configuration**

- Beamforming: {"fully-digital" if M_RF == M_ant else "hybrid"}
  ($M_\\mathrm{{ant}}/M_\\mathrm{{RF}}$ = {op.M_PS:g} phase shifters/chain)
- Power per PA: {op.Pa*1e3:.1f} mW
- Delivered rate: {(op.R_DL*op.xbar_DL + op.R_UL*op.xbar_UL)/1e9:.2f} Gbit/s
        """
    )
    st.dataframe({
        "Component": ["Digital", "Analog", "PA"],
        "Load-indep. [W]": [b.digital.load_ind, b.analog.load_ind, b.pa.load_ind],
        "Load-dep. [W]": [b.digital.load_dep, b.analog.load_dep, b.pa.load_dep],
        "Total [W]": [b.digital.total, b.analog.total, b.pa.total],
    })

# --- Sweeps (reproduce Fig 2a / 2b) --------------------------------------
st.subheader("Sweeps")
plotting.setup_style(use_tex=False)
tab_rf, tab_ant = st.tabs(["Digital + analog vs $M_\\mathrm{RF}$",
                           "Power amplifier vs $M_\\mathrm{ant}$"])

with tab_rf:
    sweep = [m for m in ARRAY_SIZES if m <= M_ant]
    dig, ana = [], []
    for m_rf in sweep:
        r_dl, r_ul = R_DL, R_UL
        if rt is not None:
            try:
                j = rt.index_of(m_rf)
                r_dl, r_ul = float(rt.R_DL[j]), float(rt.R_UL[j])
            except KeyError:
                pass
        op_i = OperatingPoint(M_ant=M_ant, M_RF=m_rf, P_T=P_T, R_DL=r_dl,
                              R_UL=r_ul, xbar_DL=xbar_DL, xbar_UL=xbar_UL)
        bi = compute(params, op_i)
        dig.append(bi.digital)
        ana.append(bi.analog)
    fig, _ = plotting.plot_digital_analog(sweep, dig, ana, ylim=None)
    st.pyplot(fig)

with tab_ant:
    pa_splits = []
    for m_ant in ARRAY_SIZES:
        op_i = OperatingPoint(M_ant=m_ant, M_RF=min(M_RF, m_ant), P_T=P_T,
                              R_DL=R_DL, R_UL=R_UL, xbar_DL=xbar_DL,
                              xbar_UL=xbar_UL)
        pa_splits.append(compute(params, op_i).pa)
    fig, _ = plotting.plot_pa(ARRAY_SIZES, pa_splits, ylim=None)
    st.pyplot(fig)
