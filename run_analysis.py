"""Κύρια ανάλυση διπλωματικής — τρέχει όλα τα σενάρια και παράγει
τους πίνακες/διαγράμματα του κεφαλαίου αποτελεσμάτων.

Χρήση:
    python3 run_analysis.py                      # συνθετικό TMY (ΤΟΤΕΕ)
    python3 run_analysis.py --tmy data/tmy.csv   # πραγματικό PVGIS TMY

Σενάρια:
    S0  Υφιστάμενη κατάσταση: λέβητας φυσικού αερίου
    S1  Α/Θ χωρίς Φ/Β (ρεύμα δικτύου), τρέχον κλίμα
    S2  Α/Θ + Φ/Β, τρέχον κλίμα
    S3  Α/Θ + Φ/Β, κλίμα 2040 RCP4.5
    S4  Α/Θ + Φ/Β, κλίμα 2040 RCP8.5
    + διερεύνηση μεγέθους Φ/Β 0 → θεωρητικό όριο στέγης (800 kWp), με
      σήμανση του ρεαλιστικού ορίου (≈700 kWp, μέτρηση §8.4.1)

Οι μεταβολές του κλίματος 2040 διαβάζονται από το data/morphing_2040.csv
(παράγεται με: python3 cds_morphing_2040.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from poseidonio import (Site, synthetic_tmy, load_pvgis_tmy, run_scenario,
                        pv_sizing_sweep, evaluate)
from poseidonio.climate import MORPHING_SOURCE

RESULTS = Path(__file__).parent / "results"


def monthly(series: pd.Series) -> pd.Series:
    return series.resample("ME").sum() / 1000.0  # kW·h → MWh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmy", help="PVGIS TMY CSV (προαιρετικό)")
    ap.add_argument("--kwp", type=float, default=500.0, help="μέγεθος Φ/Β σεναρίων S2–S4")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    site = Site()
    site.pv.kwp = args.kwp
    tmy = load_pvgis_tmy(args.tmy, site.latitude) if args.tmy else synthetic_tmy(site.latitude)
    src = "PVGIS TMY" if args.tmy else "Συνθετικό TMY (ΤΟΤΕΕ 20701-3)"
    print(f"Κλιματικά δεδομένα: {src}")
    print(f"Μεταβολές κλίματος 2040: {MORPHING_SOURCE}")

    # ---- Σενάρια ----------------------------------------------------------
    scen = {
        "S1 Α/Θ (τρέχον κλίμα)": run_scenario(site, "S1", pv_kwp=0.0, tmy=tmy),
        "S2 Α/Θ+Φ/Β (τρέχον κλίμα)": run_scenario(site, "S2", tmy=tmy),
        "S3 Α/Θ+Φ/Β (2040 RCP4.5)": run_scenario(site, "S3", climate_scenario="rcp45", tmy=tmy),
        "S4 Α/Θ+Φ/Β (2040 RCP8.5)": run_scenario(site, "S4", climate_scenario="rcp85", tmy=tmy),
    }

    rows = []
    for name, r in scen.items():
        a = r.annual
        ef_grid = site.econ.ef_grid_2040 if "2040" in name else None
        e = evaluate(site.econ,
                     gas_kwh_baseline=a["gas_baseline_mwh_fuel"] * 1000,
                     grid_import_kwh=a["grid_import_mwh"] * 1000,
                     grid_export_kwh=a["grid_export_mwh"] * 1000,
                     backup_gas_kwh_fuel=a["backup_gas_mwh_fuel"] * 1000,
                     pv_kwp=a["pv_kwp"], hp_kw_th=site.heat_pump.capacity_kw_th,
                     grid_ef=ef_grid)
        rows.append({"scenario": name, **a, **e})
    summary = pd.DataFrame(rows).set_index("scenario")
    summary.to_csv(RESULTS / "summary_scenarios.csv", float_format="%.2f")

    print("\n=== Σύνοψη σεναρίων ===")
    cols = ["thermal_demand_mwh", "hp_electricity_mwh", "scop", "pv_production_mwh",
            "self_sufficiency_pct", "co2_baseline_t", "co2_new_t", "co2_reduction_pct",
            "npv_eur", "irr", "payback_years"]
    print(summary[cols].round(2).to_string())

    # ---- Sweep μεγέθους Φ/Β -----------------------------------------------
    kwp_real = site.pv.available_area_m2 / site.pv.m2_per_kwp     # ρεαλιστικό όριο
    kwp_max = site.pv.theoretical_area_m2 / site.pv.m2_per_kwp    # θεωρητικό όριο
    kwp_grid = np.arange(0, kwp_max + 1, 50)
    sweep = pv_sizing_sweep(site, kwp_grid, tmy=tmy)
    sweep.to_csv(RESULTS / "pv_sizing_sweep.csv", index=False, float_format="%.2f")
    feasible = sweep[sweep["pv_kwp"] <= kwp_real + 1e-6]
    best = feasible.loc[feasible["npv_eur"].idxmax()]
    best_auto = feasible.loc[feasible["self_sufficiency_pct"].idxmax()]
    print(f"\nΒέλτιστο Φ/Β κατά NPV (ρεαλιστικό όριο στέγης {kwp_real:.0f} kWp): "
          f"{best['pv_kwp']:.0f} kWp (NPV {best['npv_eur']/1e6:.2f} M€, "
          f"αυτονομία {best['self_sufficiency_pct']:.1f} %)")
    print(f"Μέγιστη ενεργειακή αυτονομία (ερευνητικό ερώτημα 1): "
          f"{best_auto['self_sufficiency_pct']:.1f} % στα {best_auto['pv_kwp']:.0f} kWp "
          f"(NPV {best_auto['npv_eur']/1e6:.2f} M€)")
    top = sweep.iloc[-1]
    print(f"Θεωρητικό όριο {kwp_max:.0f} kWp: αυτονομία {top['self_sufficiency_pct']:.1f} %, "
          f"NPV {top['npv_eur']/1e6:.2f} M€")

    # ---- Διαγράμματα -------------------------------------------------------
    r2 = scen["S2 Α/Θ+Φ/Β (τρέχον κλίμα)"].hourly
    months = ["Ιαν", "Φεβ", "Μαρ", "Απρ", "Μαϊ", "Ιουν",
              "Ιουλ", "Αυγ", "Σεπ", "Οκτ", "Νοε", "Δεκ"]

    # 1. Μηνιαία θερμική ζήτηση ανά κατηγορία
    fig, ax = plt.subplots(figsize=(9, 5))
    parts = {"Δεξαμενές": "q_pool_kw", "Αερισμός/αφύγρανση": "q_vent_kw",
             "Κέλυφος": "q_envelope_kw", "ΖΝΧ": "q_dhw_kw"}
    bottom = np.zeros(12)
    for label, col in parts.items():
        vals = monthly(r2[col]).to_numpy()
        ax.bar(months, vals, bottom=bottom, label=label)
        bottom += vals
    ax.set_ylabel("MWh θερμικά / μήνα")
    ax.set_title("Μηνιαία θερμική ζήτηση Ποσειδωνίου ανά κατηγορία")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig1_monthly_thermal_demand.png", dpi=150)

    # 2. Φ/Β παραγωγή vs ηλεκτρική ζήτηση Α/Θ
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(months, monthly(r2["p_el_kw"]).to_numpy(), "o-", label="Ζήτηση Α/Θ")
    ax.plot(months, monthly(r2["pv_kw"]).to_numpy(), "s-", label=f"Παραγωγή Φ/Β {site.pv.kwp:.0f} kWp")
    ax.plot(months, monthly(r2["self_consumed_kw"]).to_numpy(), "^--", label="Αυτοκατανάλωση")
    ax.set_ylabel("MWh ηλεκτρικά / μήνα")
    ax.set_title("Μηνιαίο ηλεκτρικό ισοζύγιο (σενάριο S2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig2_electric_balance.png", dpi=150)

    # 3. Sweep: αυτονομία & NPV vs kWp (με το ρεαλιστικό όριο στέγης)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(sweep["pv_kwp"], sweep["self_sufficiency_pct"], "o-", color="tab:blue")
    ax1.set_xlabel("Εγκατεστημένη ισχύς Φ/Β (kWp)")
    ax1.set_ylabel("Ενεργειακή αυτονομία (%)", color="tab:blue")
    ax1.axvline(kwp_real, color="0.4", linestyle="--", linewidth=1)
    ax1.text(kwp_real, ax1.get_ylim()[0] + 1, f" ρεαλιστικό όριο\n στέγης ≈{kwp_real:.0f} kWp",
             color="0.3", fontsize=9, va="bottom")
    ax2 = ax1.twinx()
    ax2.plot(sweep["pv_kwp"], sweep["npv_eur"] / 1e6, "s-", color="tab:green")
    ax2.set_ylabel("NPV (M€)", color="tab:green")
    ax1.set_title("Διαστασιολόγηση Φ/Β: αυτονομία και NPV")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig3_pv_sizing.png", dpi=150)

    # 4. Εκπομπές CO2 ανά σενάριο
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["S0 Λέβητας ΦΑ"] + [n.split(" ")[0] for n in scen]
    co2 = [summary["co2_baseline_t"].iloc[0]] + summary["co2_new_t"].tolist()
    bars = ax.bar(labels, co2, color=["#888"] + ["tab:orange", "tab:green", "tab:cyan", "tab:red"])
    ax.bar_label(bars, fmt="%.0f")
    ax.set_ylabel("t CO₂ / έτος")
    ax.set_title("Ετήσιες εκπομπές CO₂ ανά σενάριο")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig4_co2_scenarios.png", dpi=150)

    # 5. Επίδραση κλιματικής αλλαγής: SCOP & θερμική ζήτηση
    fig, ax = plt.subplots(figsize=(9, 5))
    names = ["Τρέχον", "2040 RCP4.5", "2040 RCP8.5"]
    keys = ["S2 Α/Θ+Φ/Β (τρέχον κλίμα)", "S3 Α/Θ+Φ/Β (2040 RCP4.5)", "S4 Α/Θ+Φ/Β (2040 RCP8.5)"]
    dem = [scen[k].annual["thermal_demand_mwh"] for k in keys]
    scop = [scen[k].annual["scop"] for k in keys]
    ax.bar(names, dem, color="tab:blue", alpha=0.7)
    ax.set_ylabel("Θερμική ζήτηση (MWh/έτος)", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(names, scop, "o-", color="tab:red", linewidth=2)
    ax2.set_ylabel("Εποχιακός βαθμός απόδοσης SCOP", color="tab:red")
    ax.set_title("Επίδραση κλιματικής αλλαγής στο σύστημα (2040)")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig5_climate_impact.png", dpi=150)

    print(f"\nΑποτελέσματα: {RESULTS}/ (2 CSV, 5 PNG)")


if __name__ == "__main__":
    main()
