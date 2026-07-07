"""Streamlit GUI for the FR3 / D-MIMO power models.

Entry point. Run with::

    streamlit run app.py

The individual model views live in ``pages/`` and appear in the sidebar. This
landing page only introduces the tool; it holds no model logic.
"""

from __future__ import annotations

import model_access  # noqa: F401  (bootstraps sys.path for the model packages)
import streamlit as st

st.set_page_config(page_title="FR3 / D-MIMO power models", page_icon="⚡",
                   layout="wide")

st.title("FR3 / D-MIMO power models")

st.markdown(
    """
This is an interactive front end over the power and rate models in this
repository. It is a thin presentation layer: every number comes from the model
packages (`fr3_power`, `D_MIMO_rate`), unchanged.

Use the pages in the sidebar:

- **FR3 central power** — the fully implemented single-site FR3 power model.
  Adjust the operating point and hardware parameters and see the digital /
  analog / PA breakdown, the energy efficiency, and the swept figures.
- **D-MIMO rate** — configure the distributed massive MIMO downlink and inspect
  the derived quantities. The rate simulation itself depends on the
  `D_MIMO_rate` physics stubs, which are not implemented yet.
    """
)

with st.expander("Notes on speed and rates"):
    st.markdown(
        """
The FR3 power blocks are pure NumPy and recompute instantly on every change, so
this page is fully reactive. The rate side is not: `fr3_power.rate_model` needs
Sionna and a GPU, so this GUI reads the cached `data/rates.json` and never
recomputes rates live. Regenerate that table from the CLI
(`python scripts/fig_2c.py --recompute`) if you change the rate model.
        """
    )
