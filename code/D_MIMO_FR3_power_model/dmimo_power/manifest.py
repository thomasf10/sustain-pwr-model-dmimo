"""Self-describing record of one evaluation run, written as JSON.

JSON is used rather than YAML or TOML because it needs no extra dependency, the
repository already stores its cached rates that way, it diffs line by line under
git, and it round-trips back into Python without a parser of its own. TOML would
be the better choice for hand-edited *input* configuration; this is generated
*output*, where being machine-readable matters more than being hand-editable.

A manifest is meant to answer, months later, "what exactly produced this
figure?". It therefore records everything that went in, not just what was
swept:

* provenance: when, which git commit, whether the tree was dirty, versions;
* the scenario (topology, frame, Monte Carlo settings);
* every power-model parameter per deployment, including the inherited
  co-located ones, plus the derived quantities that are properties rather than
  fields and so would otherwise be invisible;
* which parameters are unsourced placeholders, named explicitly so a reader
  cannot mistake them for measured values;
* the full result table, including the per-block power breakdown;
* the figures the run wrote.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np

from .config import UNSOURCED, DMIMOPowerParams

SCHEMA_VERSION = 1


def _jsonable(obj):
    """Convert dataclasses, enums and numpy types into JSON-native values."""
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(_jsonable(k)): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _git_provenance() -> dict:
    """Commit and dirty flag, so a manifest can be traced to a tree state."""
    def run(*args):
        try:
            return subprocess.run(args, capture_output=True, text=True,
                                  timeout=10, check=True).stdout.strip()
        except Exception:  # noqa: BLE001  (git absent, not a repo, timeout)
            return None

    commit = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {"commit": commit,
            "dirty": None if status is None else bool(status),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD")}


def _derived(p: DMIMOPowerParams) -> dict:
    """Read-only properties of the parameters, which ``asdict`` cannot see.

    These are the quantities the model actually computes with, so a manifest
    that omitted them would not be enough to check a result by hand.
    """
    return {
        "M_tot": p.M_tot,
        "tau_UL": p.tau_UL,
        "dl_data_prelog": p.tau_DL * (1 - p.tau_DLsig),
        "ul_data_prelog": p.tau_UL * (1 - p.tau_ULsig),
        "pilot_share_of_frame": p.tau_UL * p.tau_ULsig,
        "upsilon_coh": p.upsilon_coh,
        "f_sI_Hz": p.f_sI,
        "f_sII_Hz": p.f_sII,
        "B_tilde_Hz": p.B_tilde,
        "lambda_c_m": p.lda_c,
        "cpu_fpgas": p.cpu_fpgas,
        "Xi_filterBB": p.Xi_filterBB,
    }


def _result_record(r) -> dict:
    """One row of the result table, with the per-block breakdown."""
    b = r.breakdown
    blocks = {}
    if hasattr(b, "fronthaul"):        # distributed NetworkBreakdown
        blocks = {"ap_digital_W": b.ap_digital.total,
                  "ap_analog_W": b.ap_analog.total,
                  "ap_pa_W": b.ap_pa.total,
                  "ap_sync_W": b.ap_sync.total,
                  "fronthaul_W": b.fronthaul.total,
                  "cpu_W": b.cpu.total,
                  "ap_utilisation_mean": float(b.utilisation.mean()),
                  "ap_utilisation_max": float(b.utilisation.max())}
    elif b is not None:                # co-located PowerBreakdown
        blocks = {"digital_W": b.digital.total,
                  "analog_W": b.analog.total,
                  "pa_W": b.pa.total}

    return {
        "deployment": r.deployment.value,
        "label": r.deployment.label,
        "split": None if r.deployment.split is None else r.deployment.split.value,
        "L": r.L,
        "M": r.M,
        "P_budget_W": r.P_budget,
        "rho_max_W": r.rates.rho_max,
        "R_DL_bps": r.rates.R_DL,
        "R_UL_bps": r.rates.R_UL,
        "R_total_bps": r.R_total,
        "se_dl_median_bps_hz": r.rates.se_dl_median,
        "se_ul_median_bps_hz": r.rates.se_ul_median,
        "P_net_W": r.power,
        "energy_efficiency_bit_per_J": r.energy_efficiency,
        "power_blocks": blocks,
    }


def build(*, name: str, description: str, scenario, sweep: dict,
          params_by_deployment: dict, results, figures=()) -> dict:
    """Assemble the manifest of one run.

    Args:
        name: Short run identifier, used as the filename.
        description: One line saying what the run varies.
        scenario: The :class:`~dmimo_power.scenarios.Scenario` used.
        sweep: ``{"variable": ..., "values": [...], "unit": ...}``.
        params_by_deployment: ``{Deployment: DMIMOPowerParams}``, the parameters
            actually used (one per deployment, since the split differs).
        results: Flat iterable of ``OperatingResult``.
        figures: Paths written by the run.

    Returns:
        A JSON-serializable dict.
    """
    params_out = {}
    for deployment, p in params_by_deployment.items():
        params_out[deployment.value] = {
            "fields": _jsonable(p),
            "derived": _jsonable(_derived(p)),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "description": description,
        "provenance": {
            "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": _git_provenance(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "scenario": _jsonable(scenario),
        "sweep": _jsonable(sweep),
        "parameters": params_out,
        "unsourced_parameters": [
            {"name": n, "meaning": what,
             "value": _jsonable(getattr(next(iter(params_by_deployment.values())), n))}
            for n, what in UNSOURCED
        ],
        "results": [_result_record(r) for r in results],
        "figures": _jsonable(list(figures)),
    }


def write(manifest: dict, out_dir) -> Path:
    """Write a manifest to ``<out_dir>/<name>.json`` and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{manifest['name']}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return path
