"""Ευαισθησία σε μεθοδολογικές παραδοχές — σενάριο S2 (Πίνακας 9.4 της εργασίας).

Εξετάζει πέντε παραδοχές που θα μπορούσαν να μεταβάλουν το συμπέρασμα:
  1. Η υφιστάμενη ηλεκτρική κατανάλωση του κέντρου (σταθερό φορτίο βάσης
     80 kW) απορροφά επίσης παραγωγή Φ/Β (αντί μόνο οι Α/Θ).
  2. Μετρημένο ωριαίο προφίλ ΖΝΧ (με ~25 % τη νύχτα) αντί ομοιόμορφης
     κατανομής στις 07:00–22:00.
  3. CAPEX με ΦΠΑ 24 % (μη εκπιπτόμενος για δημόσιο φορέα).
  4. Κόστος Α/Θ 500 €/kWth αντί 320.
  5. ΦΠΑ 24 % και Α/Θ 500 €/kWth μαζί.
Επιπλέον: κλιμάκωση OPEX 2 %/έτος και εναλλακτικοί συντελεστές εκπομπών (§8.8):
ΚΘΔ/ΑΘΔ του φυσικού αερίου και μέσο μείγμα παραγωγής 2025 (0,225) / 0,04 το 2040.

Χρήση:  python sensitivity_methodology.py [--kwp 500] [--tmy data/tmy.csv]
Έξοδος: results/sensitivity_methodology.csv
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from poseidonio import Site, load_pvgis_tmy, evaluate
from poseidonio.economics import npv
from poseidonio.heat_pump import electricity_demand
from poseidonio.pool_thermal import thermal_demand
from poseidonio.pv_system import pv_production_kw
from dataclasses import replace

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
BASE_LOAD_KW = 80.0
VAT = 0.24


def balance(site: Site, tmy: pd.DataFrame, kwp: float, *, dhw_profile: np.ndarray | None = None,
            base_load_kw: float = 0.0) -> dict:
    """Ετήσιο ενεργειακό ισοζύγιο S2 με τις προαιρετικές παραλλαγές 1 και 2."""
    demand = thermal_demand(site, tmy)
    if dhw_profile is not None:
        # ίδια ημερήσια ενέργεια ΖΝΧ, κατανεμημένη με το μετρημένο ωριαίο προφίλ
        daily = demand["q_dhw_kw"].groupby(demand.index.date).transform("sum").to_numpy()
        share = dhw_profile[demand.index.hour] / dhw_profile.sum()
        demand = demand.copy()
        demand["q_total_kw"] = demand["q_total_kw"] - demand["q_dhw_kw"] + daily * share
        demand["q_dhw_kw"] = daily * share
    hp = electricity_demand(site, demand, tmy)
    pv = pv_production_kw(replace(site.pv, kwp=kwp), tmy, site.latitude).to_numpy()
    p_el = hp["p_el_kw"].to_numpy()
    load = p_el + base_load_kw
    sc = np.minimum(pv, load)
    # Το φορτίο βάσης αγοράζεται ήδη σήμερα από το δίκτυο: μετρά μόνο η διαφορά
    grid_import = load - sc - base_load_kw
    grid_export = pv - sc
    q_total = demand["q_total_kw"].sum()
    return {
        "gas_kwh_baseline": q_total / site.econ.boiler_eff,
        "grid_import_kwh": grid_import.sum(),
        "grid_export_kwh": grid_export.sum(),
        "backup_gas_kwh_fuel": hp["q_backup_kw"].sum() / site.heat_pump.backup_gas_eff,
        "self_sufficiency_pct": 100 * np.minimum(pv, p_el).sum() / p_el.sum(),
        "self_consumption_pct": 100 * sc.sum() / pv.sum(),
    }


def econ(site: Site, b: dict, kwp: float) -> dict:
    return evaluate(site.econ, gas_kwh_baseline=b["gas_kwh_baseline"],
                    grid_import_kwh=b["grid_import_kwh"], grid_export_kwh=b["grid_export_kwh"],
                    backup_gas_kwh_fuel=b["backup_gas_kwh_fuel"], pv_kwp=kwp,
                    hp_kw_th=site.heat_pump.capacity_kw_th)


def with_capex(site: Site, *, vat: bool = False, hp_cost: float | None = None) -> Site:
    s = deepcopy(site)
    if hp_cost is not None:
        s.econ.hp_capex_eur_kwth = hp_cost
    if vat:
        s.econ.pv_capex_eur_kwp *= 1 + VAT
        s.econ.hp_capex_eur_kwth *= 1 + VAT
        s.econ.balance_capex_eur *= 1 + VAT
    return s


def npv_opex_escalation(site: Site, b: dict, kwp: float, rate: float = 0.02) -> float:
    """ΚΠΑ με κλιμάκωση του OPEX κατά `rate` ετησίως (αντί σταθερού ονομαστικά)."""
    e = site.econ
    capex = kwp * e.pv_capex_eur_kwp + site.heat_pump.capacity_kw_th * e.hp_capex_eur_kwth \
        + e.balance_capex_eur
    opex = e.opex_pct_capex * capex
    cf = [-capex]
    for y in range(1, e.lifetime_years + 1):
        infl = (1 + e.energy_inflation) ** (y - 1)
        degr = (1 - e.pv_degradation) ** (y - 1)
        cf.append(b["gas_kwh_baseline"] * e.gas_price_eur_kwh * infl
                  - (b["grid_import_kwh"] * e.grid_price_eur_kwh * infl
                     - b["grid_export_kwh"] * degr * e.injection_price_eur_kwh * infl
                     + b["backup_gas_kwh_fuel"] * e.gas_price_eur_kwh * infl)
                  - opex * (1 + rate) ** (y - 1))
    return npv(e.discount_rate, np.array(cf))


def co2_alternatives(site: Site, tmy: pd.DataFrame, kwp: float) -> None:
    """Εναλλακτικοί συντελεστές εκπομπών (§8.8, §8.9) — ποσοστά έναντι της σημερινής βάσης S0."""
    from poseidonio import run_scenario
    e = site.econ
    s2 = run_scenario(site, "S2", pv_kwp=kwp, tmy=tmy).annual
    s3 = run_scenario(site, "S3", climate_scenario="rcp45", pv_kwp=kwp, tmy=tmy).annual
    base = s2["gas_baseline_mwh_fuel"] * e.ef_gas_fuel          # t (MWh × kg/kWh)
    base_hhv = base / 1.11                                      # αν ο μετρητής δίνει ΑΘΔ
    s2_t, s3_t = s2["grid_import_mwh"] * e.ef_grid_2026, s3["grid_import_mwh"] * e.ef_grid_2040
    print("\nΕναλλακτικοί συντελεστές εκπομπών (§8.8):")
    print(f"  Βάση ΚΘΔ {base:.0f} t· σε ΑΘΔ ≈{base_hhv:.0f} t → μείωση S2 {100*(1-s2_t/base_hhv):.0f} %, "
          f"2040 {100*(1-s3_t/base_hhv):.0f} %")
    s2_low, s3_low = s2["grid_import_mwh"] * 0.225, s3["grid_import_mwh"] * 0.04
    print(f"  Μέσο μείγμα 0,225 (2025) / 0,04 (2040): μείωση S2 {100*(1-s2_low/base):.0f} %, "
          f"2040 {100*(1-s3_low/base):.0f} %")
    chp_gas, chp_el, measured = 6499.0, 2048.0, 3729.643 * e.ef_gas_fuel
    print(f"  ΣΗΘΥΑ με 0,225: {chp_gas*e.ef_gas_fuel - chp_el*0.225:.0f} t έναντι {measured:.0f} t (μετρημένη βάση)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kwp", type=float, default=500.0)
    ap.add_argument("--tmy", default=str(ROOT / "data" / "tmy.csv"))
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    site = Site()
    tmy = load_pvgis_tmy(args.tmy, site.latitude)
    prof = pd.read_csv(ROOT / "data" / "poseidonio_dhw_hourly_profile.csv", comment="#")
    dhw_profile = prof.sort_values("hour")["gas_kwh_fuel_mean"].to_numpy()

    base = balance(site, tmy, args.kwp)
    rows = []

    def add(label, s, b):
        e = econ(s, b, args.kwp)
        rows.append({"Παραδοχή": label, "NPV (M€)": e["npv_eur"] / 1e6, "IRR (%)": 100 * e["irr"],
                     "Επανείσπραξη (έτη)": e["payback_years"],
                     "Αυτονομία (%)": b["self_sufficiency_pct"],
                     "Ιδιοκατανάλωση (%)": b["self_consumption_pct"]})

    add("Βασικό σενάριο S2", site, base)
    add(f"Φορτίο βάσης {BASE_LOAD_KW:.0f} kW απορροφά Φ/Β", site,
        balance(site, tmy, args.kwp, base_load_kw=BASE_LOAD_KW))
    add("Μετρημένο ωριαίο προφίλ ΖΝΧ", site, balance(site, tmy, args.kwp, dhw_profile=dhw_profile))
    add("CAPEX με ΦΠΑ 24 %", with_capex(site, vat=True), base)
    add("Κόστος Α/Θ 500 €/kWth", with_capex(site, hp_cost=500.0), base)
    add("ΦΠΑ 24 % και Α/Θ 500 €/kWth", with_capex(site, vat=True, hp_cost=500.0), base)

    df = pd.DataFrame(rows)
    df.round(2).to_csv(RESULTS / "sensitivity_methodology.csv", index=False)
    print(df.round(2).to_string(index=False))
    d_opex = (npv_opex_escalation(site, base, args.kwp) - econ(site, base, args.kwp)["npv_eur"]) / 1e6
    print(f"\nΚλιμάκωση OPEX 2 %/έτος: μεταβολή ΚΠΑ {d_opex:+.2f} M€")
    co2_alternatives(site, tmy, args.kwp)
    print(f"\nΑποτελέσματα: {RESULTS}/sensitivity_methodology.csv")


if __name__ == "__main__":
    main()
