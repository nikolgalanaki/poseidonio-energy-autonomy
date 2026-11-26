"""Μηχανή ωριαίας ετήσιας προσομοίωσης — σύνθεση όλων των υπομοντέλων.

Ένα σενάριο = (κλίμα, μέγεθος Φ/Β, Α/Θ ή λέβητας). Επιστρέφει το ωριαίο
DataFrame και τα ετήσια μεγέθη που τροφοδοτούν την οικονομική αξιολόγηση.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .config import Site
from .climate import synthetic_tmy, morph_to_2040
from .pool_thermal import thermal_demand
from .heat_pump import electricity_demand
from .pv_system import pv_production_kw


@dataclass
class ScenarioResult:
    name: str
    hourly: pd.DataFrame
    annual: dict


def get_climate(site: Site, climate_scenario: str = "current",
                tmy: pd.DataFrame | None = None) -> pd.DataFrame:
    base = tmy if tmy is not None else synthetic_tmy(site.latitude)
    if climate_scenario == "current":
        return base
    return morph_to_2040(base, climate_scenario)


def run_scenario(site: Site, name: str, *, climate_scenario: str = "current",
                 pv_kwp: float | None = None,
                 tmy: pd.DataFrame | None = None) -> ScenarioResult:
    """Προσομοίωση συστήματος Α/Θ + Φ/Β για ένα κλιματικό σενάριο."""
    clim = get_climate(site, climate_scenario, tmy)
    demand = thermal_demand(site, clim)
    hp = electricity_demand(site, demand, clim)

    pv_cfg = site.pv if pv_kwp is None else replace(site.pv, kwp=pv_kwp)
    pv = pv_production_kw(pv_cfg, clim, site.latitude) if pv_cfg.kwp > 0 \
        else pd.Series(0.0, index=clim.index)

    h = pd.concat([demand, hp], axis=1)
    h["pv_kw"] = pv
    h["self_consumed_kw"] = np.minimum(h["pv_kw"], h["p_el_kw"])
    h["grid_import_kw"] = h["p_el_kw"] - h["self_consumed_kw"]
    h["grid_export_kw"] = h["pv_kw"] - h["self_consumed_kw"]

    q_total = float(h["q_total_kw"].sum())
    p_el = float(h["p_el_kw"].sum())
    annual = {
        "climate": climate_scenario,
        "pv_kwp": pv_cfg.kwp,
        "thermal_demand_mwh": q_total / 1000.0,
        "gas_baseline_mwh_fuel": q_total / site.econ.boiler_eff / 1000.0,
        "hp_electricity_mwh": p_el / 1000.0,
        "backup_gas_mwh_fuel": float(h["q_backup_kw"].sum()) / site.heat_pump.backup_gas_eff / 1000.0,
        "scop": q_total and float(h["q_hp_kw"].sum()) / p_el,
        "pv_production_mwh": float(h["pv_kw"].sum()) / 1000.0,
        "self_consumed_mwh": float(h["self_consumed_kw"].sum()) / 1000.0,
        "grid_import_mwh": float(h["grid_import_kw"].sum()) / 1000.0,
        "grid_export_mwh": float(h["grid_export_kw"].sum()) / 1000.0,
    }
    # Δείκτες αυτονομίας/αυτοκατανάλωσης
    annual["self_sufficiency_pct"] = 100.0 * annual["self_consumed_mwh"] / annual["hp_electricity_mwh"] \
        if annual["hp_electricity_mwh"] else 0.0
    annual["self_consumption_pct"] = 100.0 * annual["self_consumed_mwh"] / annual["pv_production_mwh"] \
        if annual["pv_production_mwh"] else 0.0
    annual["peak_thermal_kw"] = float(h["q_total_kw"].max())
    annual["peak_electric_kw"] = float(h["p_el_kw"].max())
    return ScenarioResult(name, h, annual)


def pv_sizing_sweep(site: Site, kwp_values, *, climate_scenario: str = "current",
                    tmy: pd.DataFrame | None = None) -> pd.DataFrame:
    """Διερεύνηση μεγέθους Φ/Β: αυτονομία & οικονομικοί δείκτες ανά kWp."""
    from .economics import evaluate
    rows = []
    clim = get_climate(site, climate_scenario, tmy)
    for kwp in kwp_values:
        r = run_scenario(site, f"pv_{kwp:g}", climate_scenario=climate_scenario,
                         pv_kwp=float(kwp), tmy=clim)
        a = r.annual
        econ = evaluate(site.econ,
                        gas_kwh_baseline=a["gas_baseline_mwh_fuel"] * 1000,
                        grid_import_kwh=a["grid_import_mwh"] * 1000,
                        grid_export_kwh=a["grid_export_mwh"] * 1000,
                        backup_gas_kwh_fuel=a["backup_gas_mwh_fuel"] * 1000,
                        pv_kwp=kwp, hp_kw_th=site.heat_pump.capacity_kw_th)
        rows.append({**a, **econ})
    return pd.DataFrame(rows)
