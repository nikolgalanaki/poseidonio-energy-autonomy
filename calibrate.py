"""Βαθμονόμηση μοντέλου έναντι πραγματικής κατανάλωσης ΦΑ (PTZ μετρητής ΕΔΑ).

Συγκρίνει τη μηνιαία κατανάλωση φυσικού αερίου του σεναρίου S0 (υφιστάμενη
κατάσταση: λέβητες, β.α. 90 %) με τα πραγματικά δεδομένα του έτους αναφοράς
(data/poseidonio_gas_monthly.csv, από Η/Μ μελέτη ΣΗΘΥΑ της ΓΓΑ).

Δείκτες (ASHRAE Guideline 14 για μηνιαία δεδομένα):
  NMBE  (Normalized Mean Bias Error)        — όριο ±5 %
  CV(RMSE) (συντελεστής μεταβλητότητας RMSE) — όριο 15 %
Επιπλέον αναφέρεται η ετήσια απόκλιση (στόχος διπλωματικής: ±10 %).

Χρήση:
    python3 calibrate.py                      # συνθετικό TMY
    python3 calibrate.py --tmy data/tmy.csv   # πραγματικό PVGIS TMY
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

from poseidonio import Site, synthetic_tmy, load_pvgis_tmy
from poseidonio.pool_thermal import thermal_demand

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
MONTHS_GR = ["Ιαν", "Φεβ", "Μαρ", "Απρ", "Μαϊ", "Ιουν",
             "Ιουλ", "Αυγ", "Σεπ", "Οκτ", "Νοε", "Δεκ"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmy", help="PVGIS TMY CSV (προαιρετικό)")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    site = Site()
    tmy = load_pvgis_tmy(args.tmy, site.latitude) if args.tmy else synthetic_tmy(site.latitude)

    real = pd.read_csv(ROOT / "data" / "poseidonio_gas_monthly.csv", comment="#")
    real_kwh = real.set_index("month")["gas_kwh_fuel"]

    demand = thermal_demand(site, tmy)
    model_kwh = (demand["q_total_kw"].groupby(demand.index.month).sum()
                 / site.econ.boiler_eff)

    df = pd.DataFrame({
        "real_mwh": real_kwh / 1000.0,
        "model_mwh": model_kwh / 1000.0,
    })
    df["diff_pct"] = 100.0 * (df["model_mwh"] - df["real_mwh"]) / df["real_mwh"]

    resid = df["model_mwh"] - df["real_mwh"]
    mean_real = df["real_mwh"].mean()
    nmbe = 100.0 * resid.sum() / (len(df) * mean_real)
    cv_rmse = 100.0 * np.sqrt((resid**2).mean()) / mean_real
    annual_dev = 100.0 * (df["model_mwh"].sum() - df["real_mwh"].sum()) / df["real_mwh"].sum()

    print("=== Βαθμονόμηση S0 έναντι PTZ μετρητή ΕΔΑ (kWh καυσίμου ΦΑ) ===\n")
    show = df.copy()
    show.index = MONTHS_GR
    print(show.round(1).to_string())
    print(f"\nΕτήσιο πραγματικό: {df['real_mwh'].sum():.0f} MWh | "
          f"μοντέλο: {df['model_mwh'].sum():.0f} MWh | απόκλιση {annual_dev:+.1f} %")
    print(f"NMBE = {nmbe:+.1f} % (όριο ±5 %) | CV(RMSE) = {cv_rmse:.1f} % (όριο 15 %)")
    ok = abs(nmbe) <= 5.0 and cv_rmse <= 15.0
    print("Κριτήρια ASHRAE Guideline 14:", "ΠΛΗΡΟΥΝΤΑΙ ✓" if ok else "ΔΕΝ πληρούνται — ρύθμισε παραμέτρους [ΒΑΘΜΟΝΟΜΗΣΗ] στο config.py")

    df.round(2).to_csv(RESULTS / "calibration_monthly.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(12)
    ax.bar(x - 0.2, df["real_mwh"], width=0.4, label="Πραγματικά (ΕΔΑ, έτος αναφοράς)")
    ax.bar(x + 0.2, df["model_mwh"], width=0.4, label="Μοντέλο S0 (λέβητας 90 %)")
    ax.set_xticks(x, MONTHS_GR)
    ax.set_ylabel("MWh καυσίμου ΦΑ / μήνα")
    ax.set_title(f"Βαθμονόμηση: NMBE {nmbe:+.1f} %, CV(RMSE) {cv_rmse:.1f} %, "
                 f"ετήσια απόκλιση {annual_dev:+.1f} %")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig6_calibration.png", dpi=150)
    print(f"\nΑποτελέσματα: {RESULTS}/calibration_monthly.csv, fig6_calibration.png")


if __name__ == "__main__":
    main()
