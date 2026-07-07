# Power model GUI

An interactive [Streamlit](https://streamlit.io) front end over the power and
rate models in this repository. It is a thin presentation layer: every number
comes from the model packages (`fr3_power`, `D_MIMO_rate`), unchanged. No model
math lives here.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in the browser with two pages in the sidebar.

## Pages

- **FR3 central power** (`pages/1_FR3_central_power.py`) — the fully implemented
  single-site FR3 model. Set the operating point (antennas, RF chains, transmit
  power, loads) and rates in the sidebar, optionally override any hardware
  parameter, and see the digital / analog / PA breakdown, the energy efficiency,
  and the swept manuscript figures (power vs `M_RF`, PA vs `M_ant`). Rates are
  read from the cached `../FR3_power_model/data/rates.json` by default.
- **D-MIMO rate** (`pages/2_DMIMO_rate.py`) — configure the distributed massive
  MIMO downlink (`DMIMOConfig`) and inspect the derived quantities (total array
  size, noise power, prelog). The Monte Carlo rate simulation depends on the
  `mimo_helpers` physics stubs, which are not implemented yet; the Run button
  reports that state rather than faking a result.

## Structure

```
power_gui/
├── app.py            landing page + entry point (streamlit run app.py)
├── model_access.py   puts the two model packages on sys.path
├── widgets.py        dataclass -> Streamlit widgets (edit_dataclass)
├── plots.py          single-point breakdown bar; sweeps reuse fr3_power.plotting
└── pages/            one file per model view (Streamlit multipage)
```

The design keeps the GUI decoupled from the models. `model_access` handles the
import paths (`fr3_power` is a package; `D_MIMO_rate` uses flat imports).
`widgets.edit_dataclass` reflects over a config dataclass and emits one widget
per field, so adding a field to `PowerParams` / `OperatingPoint` / `DMIMOConfig`
surfaces it in the GUI with no changes here. Sweep figures call the manuscript
plotting functions directly so the GUI and the CLI scripts render identically.

## Notes

- The FR3 power blocks are pure NumPy and recompute instantly, so the FR3 page
  is fully reactive. The rate side is not: `fr3_power.rate_model` needs Sionna
  and a GPU, so this GUI never recomputes rates live. Regenerate the cached
  table from the CLI (`cd ../FR3_power_model && python scripts/fig_2c.py
  --recompute`) if the rate model changes.
- The D-MIMO rate page becomes fully functional once the `mimo_helpers` stubs in
  `../D_MIMO_rate` are implemented; no GUI change is needed.
