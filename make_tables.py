"""Παραγωγή αριθμητικών πινάκων για το κείμενο και τα παραρτήματα
(results/tab_*.csv). Όλοι οι αριθμοί του κειμένου προκύπτουν από εδώ.

Χρήση:  python3 make_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from poseidonio import Site, load_pvgis_tmy, run_scenario
from poseidonio.climate import saturation_pressure_kpa, humidity_ratio
from poseidonio.heat_pump import cop
from poseidonio.pool_thermal import pool_losses_kw, LATENT_KJ_KG

ROOT = Path(__file__).parent
R = ROOT / "results"
site = Site()
tmy = load_pvgis_tmy(str(ROOT / "data" / "tmy.csv"))


def monthly_breakdown(h: pd.DataFrame) -> pd.DataFrame:
    cols = {"q_pool_kw": "Δεξαμενές", "q_evap_kw": "εκ των οποίων εξάτμιση",
            "q_vent_kw": "Αερισμός", "q_envelope_kw": "Κέλυφος",
            "q_dhw_kw": "ΖΝΧ", "q_total_kw": "Σύνολο"}
    m = h[list(cols)].groupby(h.index.month).sum() / 1000.0
    m.columns = list(cols.values())
    m.loc["Έτος"] = m.sum()
    return m.round(1)


def electric_balance(h: pd.DataFrame) -> pd.DataFrame:
    cols = {"p_el_kw": "Ζήτηση Α/Θ", "pv_kw": "Παραγωγή Φ/Β",
            "self_consumed_kw": "Αυτοκατανάλωση", "grid_import_kw": "Απορρόφηση",
            "grid_export_kw": "Έγχυση"}
    m = h[list(cols)].groupby(h.index.month).sum() / 1000.0
    m.columns = list(cols.values())
    cop_m = (h["q_hp_kw"].groupby(h.index.month).sum()
             / h["p_el_kw"].groupby(h.index.month).sum())
    m["COP μήνα"] = cop_m
    m.loc["Έτος"] = m.sum()
    m.loc["Έτος", "COP μήνα"] = h["q_hp_kw"].sum() / h["p_el_kw"].sum()
    return m.round(2)


def worked_example_evaporation() -> dict:
    """Αναλυτικό παράδειγμα: Εσωτερική δεξαμενή Α, ώρα λειτουργίας, χειμώνας."""
    pool = site.pools[0]
    t_hall, rh = site.hall.air_temp_winter_c, site.hall.rh_setpoint
    pw = float(saturation_pressure_kpa(pool.water_temp_c))
    pa = rh * float(saturation_pressure_kpa(t_hall))
    v = site.hall.air_speed_ms
    m = pool.area_m2 * (pw - pa) * (0.089 + 0.0782 * v) / LATENT_KJ_KG * pool.activity_factor
    q_evap = m * LATENT_KJ_KG
    from poseidonio.climate import TOTEE_THESSALONIKI
    t_g = float(TOTEE_THESSALONIKI.loc[1, "t_ground_c"])   # Ιανουάριος, ΤΟΤΕΕ Πίν. 6.1
    df = pool_losses_kw(pool, np.array([t_hall]), np.array([rh]), v,
                        np.array([t_g]), np.array([True]))
    return {"A": pool.area_m2, "t_ground": t_g, "pw": pw, "pa": pa, "v": v, "FA": pool.activity_factor,
            "m_kg_s": m, "m_kg_h": m * 3600, "q_evap_kw": q_evap,
            **{k: float(df[k].iloc[0]) for k in df.columns}}


def worked_example_hp_pv() -> pd.DataFrame:
    """Τυπική ημέρα Ιανουαρίου και Ιουλίου: θερμοκρασία, COP, Φ/Β (500 kWp)."""
    r = run_scenario(site, "S2", tmy=tmy)
    h = r.hourly.copy()
    h["t_air_c"] = tmy["t_air_c"]
    h["cop45"] = cop(tmy["t_air_c"].to_numpy(), 45, site.heat_pump.carnot_eff,
                     site.heat_pump.max_cop)
    rows = []
    for day in ["2023-01-15", "2023-07-15"]:
        d = h.loc[day]
        for hr in [3, 9, 12, 15, 21]:
            x = d.iloc[hr]
            rows.append({"Ημέρα": day[5:], "Ώρα": hr, "T_εξ (°C)": x["t_air_c"],
                         "COP 45°C": x["cop45"], "Θερμ. φορτίο (kW)": x["q_total_kw"],
                         "Ηλεκτρ. Α/Θ (kW)": x["p_el_kw"], "Φ/Β (kW)": x["pv_kw"]})
    return pd.DataFrame(rows).round(1)


def cashflows_25y() -> pd.DataFrame:
    r = run_scenario(site, "S2", tmy=tmy)
    a = r.annual
    e = site.econ
    gas = a["gas_baseline_mwh_fuel"] * 1000
    imp, exp = a["grid_import_mwh"] * 1000, a["grid_export_mwh"] * 1000
    capex = a["pv_kwp"] * e.pv_capex_eur_kwp + site.heat_pump.capacity_kw_th \
        * e.hp_capex_eur_kwth + e.balance_capex_eur
    opex = e.opex_pct_capex * capex
    rows = [{"Έτος": 0, "Ταμειακή ροή (€)": -capex}]
    for y in range(1, e.lifetime_years + 1):
        infl = (1 + e.energy_inflation) ** (y - 1)
        degr = (1 - e.pv_degradation) ** (y - 1)
        cf = gas * e.gas_price_eur_kwh * infl - (imp * e.grid_price_eur_kwh * infl
             - exp * degr * e.injection_price_eur_kwh * infl) - opex
        rows.append({"Έτος": y, "Ταμειακή ροή (€)": cf})
    df = pd.DataFrame(rows)
    df["Παρούσα αξία (€)"] = df["Ταμειακή ροή (€)"] / (1 + e.discount_rate) ** df["Έτος"]
    df["Σωρευτική ροή (€)"] = df["Ταμειακή ροή (€)"].cumsum()
    df["Σωρευτική ΠΑ (€)"] = df["Παρούσα αξία (€)"].cumsum()
    return df.round(0)


def main() -> None:
    s0 = run_scenario(site, "S2", tmy=tmy)
    monthly_breakdown(s0.hourly).to_csv(R / "tab_monthly_thermal.csv")
    electric_balance(s0.hourly).to_csv(R / "tab_monthly_electric_S2.csv")
    pd.Series(worked_example_evaporation()).round(4).to_csv(R / "tab_example_evaporation.csv")
    worked_example_hp_pv().to_csv(R / "tab_example_hourly.csv", index=False)
    cashflows_25y().to_csv(R / "tab_cashflows_25y.csv", index=False)
    print(monthly_breakdown(s0.hourly).to_string())
    print(electric_balance(s0.hourly).to_string())
    print(pd.Series(worked_example_evaporation()).round(4).to_string())
    print(worked_example_hp_pv().to_string(index=False))
    print(cashflows_25y().to_string(index=False))


if __name__ == "__main__":
    main()
