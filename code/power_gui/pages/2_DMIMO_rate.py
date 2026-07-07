"""Distributed massive MIMO downlink — configuration and rate view.

Builds a ``DMIMOConfig`` from the sidebar and reports the derived quantities
(total array size, noise power, prelog, ...). Running the actual Monte Carlo
rate simulation depends on the physics stubs in ``mimo_helpers`` (channel model,
precoding, power control), which currently raise ``NotImplementedError``; the
"Run" button surfaces that state honestly rather than faking a result.
"""

from __future__ import annotations

import model_access  # noqa: F401  (bootstraps sys.path)
import streamlit as st

from config_dmimo import DMIMOConfig
import dl_rate

from widgets import edit_dataclass

st.set_page_config(page_title="D-MIMO rate", page_icon="📡", layout="wide")
st.title("Distributed massive MIMO downlink")

st.info(
    "The rate simulation uses the physics stubs in `mimo_helpers.py` "
    "(channel model, precoding, power control), which are not implemented yet. "
    "The configuration and its derived quantities below are live; running the "
    "simulation will report the pending implementation.",
    icon="🚧",
)

# --- Sidebar: configuration ----------------------------------------------
st.sidebar.header("System configuration")
with st.sidebar:
    cfg = edit_dataclass(DMIMOConfig(), key_prefix="dm_", columns=1)

# --- Derived quantities ---------------------------------------------------
st.subheader("Derived quantities")
c1, c2, c3 = st.columns(3)
c1.metric("Total antennas $M_\\mathrm{tot}$", f"{cfg.M_tot}")
c2.metric("Noise power $\\sigma^2$", f"{cfg.noise_power_dBm:.1f} dBm")
c3.metric("DL prelog", f"{cfg.dl_prelog:.3f}")

st.code(cfg.summary(), language="text")

# --- Run ------------------------------------------------------------------
st.subheader("Downlink rate simulation")
if st.button("Run downlink simulation", type="primary"):
    with st.spinner("Running Monte Carlo..."):
        try:
            result = dl_rate.simulate_downlink(cfg)
        except NotImplementedError as exc:
            st.error(
                "Simulation not available yet: a `mimo_helpers` stub is "
                f"unimplemented.\n\n`NotImplementedError: {exc}`"
            )
        except Exception as exc:  # surface any other failure plainly
            st.exception(exc)
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("DL sum SE", f"{result.sum_se:.2f} bit/s/Hz")
            m2.metric("DL sum rate", f"{result.sum_rate/1e9:.3f} Gbit/s")
            m3.metric("Max AP power",
                      f"{result.ap_power.max():.3f} W / {cfg.rho_max:.2f} W")
            st.bar_chart({"per-user SE [bit/s/Hz]": result.se_per_user})
