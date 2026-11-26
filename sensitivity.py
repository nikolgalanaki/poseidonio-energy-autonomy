"""Ανάλυση ευαισθησίας NPV — σενάριο S2 (Α/Θ + Φ/Β, τρέχον κλίμα).

Μεταβάλλει μία-μία τις κρίσιμες παραμέτρους γύρω από τη βασική τιμή και
καταγράφει το NPV (tornado diagram):
  - τιμή ηλεκτρικής ενέργειας ±30 %
  - τιμή φυσικού αερίου ±30 %
  - τιμή έγχυσης Net Billing ±30 %
  - επιτόκιο προεξόφλησης 3–7 %
  - απόδοση Α/Θ (carnot_eff) ±10 %
  - CAPEX ±20 %

Χρήση:
    python3 sensitivity.py [--kwp 500] [--tmy data/tmy.csv]
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from poseidonio import Site, synthetic_tmy, load_pvgis_tmy, run_scenario, evaluate

RESULTS = Path(__file__).parent / "results"


def npv_for(site: Site, tmy, *, kwp: float) -> float:
    r = run_scenario(site, "sens", pv_kwp=kwp, tmy=tmy)
    a = r.annual
    e = evaluate(site.econ,
                 gas_kwh_baseline=a["gas_baseline_mwh_fuel"] * 1000,
                 grid_import_kwh=a["grid_import_mwh"] * 1000,
                 grid_export_kwh=a["grid_export_mwh"] * 1000,
                 backup_gas_kwh_fuel=a["backup_gas_mwh_fuel"] * 1000,
                 pv_kwp=kwp, hp_kw_th=site.heat_pump.capacity_kw_th)
    return e["npv_eur"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kwp", type=float, default=500.0)
    ap.add_argument("--tmy")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    base_site = Site()
    tmy = load_pvgis_tmy(args.tmy, base_site.latitude) if args.tmy \
        else synthetic_tmy(base_site.latitude)
    base_npv = npv_for(base_site, tmy, kwp=args.kwp)

    # (ετικέτα, συνάρτηση που φτιάχνει Site χαμηλής/υψηλής τιμής, περιγραφή εύρους)
    def mod(**changes):
        s = deepcopy(base_site)
        for path, val in changes.items():
            obj, attr = s, path
            while "." in attr:
                head, attr = attr.split(".", 1)
                obj = getattr(obj, head)
            setattr(obj, attr, val)
        return s

    e = base_site.econ
    cases = [
        ("Τιμή ρεύματος ±30 %",
         mod(**{"econ.grid_price_eur_kwh": e.grid_price_eur_kwh * 0.7}),
         mod(**{"econ.grid_price_eur_kwh": e.grid_price_eur_kwh * 1.3})),
        ("Τιμή ΦΑ ±30 %",
         mod(**{"econ.gas_price_eur_kwh": e.gas_price_eur_kwh * 0.7}),
         mod(**{"econ.gas_price_eur_kwh": e.gas_price_eur_kwh * 1.3})),
        ("Τιμή έγχυσης ±30 %",
         mod(**{"econ.injection_price_eur_kwh": e.injection_price_eur_kwh * 0.7}),
         mod(**{"econ.injection_price_eur_kwh": e.injection_price_eur_kwh * 1.3})),
        ("Επιτόκιο 3–7 %",
         mod(**{"econ.discount_rate": 0.07}),
         mod(**{"econ.discount_rate": 0.03})),
        ("Απόδοση Α/Θ ±10 %",
         mod(**{"heat_pump.carnot_eff": base_site.heat_pump.carnot_eff * 0.9}),
         mod(**{"heat_pump.carnot_eff": base_site.heat_pump.carnot_eff * 1.1})),
        ("CAPEX ±20 %",
         mod(**{"econ.pv_capex_eur_kwp": e.pv_capex_eur_kwp * 1.2,
                "econ.hp_capex_eur_kwth": e.hp_capex_eur_kwth * 1.2,
                "econ.balance_capex_eur": e.balance_capex_eur * 1.2}),
         mod(**{"econ.pv_capex_eur_kwp": e.pv_capex_eur_kwp * 0.8,
                "econ.hp_capex_eur_kwth": e.hp_capex_eur_kwth * 0.8,
                "econ.balance_capex_eur": e.balance_capex_eur * 0.8})),
    ]

    rows = []
    for label, s_a, s_b in cases:
        npv_a = npv_for(s_a, tmy, kwp=args.kwp)
        npv_b = npv_for(s_b, tmy, kwp=args.kwp)
        rows.append({"parameter": label,
                     "npv_low_eur": min(npv_a, npv_b),
                     "npv_high_eur": max(npv_a, npv_b)})
    df = pd.DataFrame(rows)
    df["base_npv_eur"] = base_npv
    df["swing_eur"] = (df["npv_high_eur"] - df["npv_low_eur"]).abs()
    df = df.sort_values("swing_eur")
    df.round(0).to_csv(RESULTS / "sensitivity_npv.csv", index=False)

    print(f"Βασικό NPV (S2, {args.kwp:.0f} kWp): {base_npv/1e6:.2f} M€\n")
    print(df.round(0).to_string(index=False))
    worst = df["npv_low_eur"].min()
    print(f"\nΧειρότερη περίπτωση: NPV {worst/1e6:.2f} M€ → "
          + ("παραμένει ΘΕΤΙΚΟ ✓" if worst > 0 else "αρνητικό ✗"))

    fig, ax = plt.subplots(figsize=(9, 5))
    y = range(len(df))
    ax.barh(y, (df["npv_high_eur"] - base_npv) / 1e6, left=base_npv / 1e6,
            color="tab:green", alpha=0.8, label="Ευνοϊκή τιμή")
    ax.barh(y, (df["npv_low_eur"] - base_npv) / 1e6, left=base_npv / 1e6,
            color="tab:red", alpha=0.8, label="Δυσμενής τιμή")
    ax.axvline(base_npv / 1e6, color="k", lw=1)
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_yticks(list(y), df["parameter"])
    ax.set_xlabel("NPV (M€)")
    ax.set_title(f"Ανάλυση ευαισθησίας NPV — σενάριο S2 ({args.kwp:.0f} kWp Φ/Β)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig7_sensitivity.png", dpi=150)
    print(f"\nΑποτελέσματα: {RESULTS}/sensitivity_npv.csv, fig7_sensitivity.png")


if __name__ == "__main__":
    main()
