"""Παραγωγή συμπληρωματικών διαγραμμάτων για τα θεωρητικά κεφάλαια και
τα παραρτήματα της διπλωματικής (results/figT*, figD*, figE*).

Όλα τα διαγράμματα παράγονται από το ίδιο το μοντέλο ή από τα πραγματικά
δεδομένα του repository — κανένα δεν είναι «δανεικό» από τρίτες πηγές.

Χρήση:  python3 make_extra_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from poseidonio import Site, load_pvgis_tmy
from poseidonio.climate import TOTEE_THESSALONIKI, MORPHING_2040, saturation_pressure_kpa

ROOT = Path(__file__).parent
R = ROOT / "results"
MONTHS = ["Ιαν", "Φεβ", "Μαρ", "Απρ", "Μαϊ", "Ιουν",
          "Ιουλ", "Αυγ", "Σεπ", "Οκτ", "Νοε", "Δεκ"]
site = Site()


def fig_cop() -> None:
    """COP αντλίας θερμότητας συναρτήσει εξωτερικής θερμοκρασίας (Κεφ. 2)."""
    t_out = np.linspace(-10, 35, 200)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    from poseidonio.heat_pump import cop as cop_model
    hp = site.heat_pump
    for t_sup, label in [(45, "Προσαγωγή 45 °C (πισίνες/χώροι)"),
                         (55, "Προσαγωγή 55 °C (ΖΝΧ)")]:
        ax.plot(t_out, cop_model(t_out, t_sup, hp.carnot_eff, hp.max_cop),
                lw=2, label=label)
    ax.set_xlabel("Εξωτερική θερμοκρασία (°C)")
    ax.set_ylabel("COP")
    ax.set_title("COP αντλίας θερμότητας αέρα–νερού (η_Carnot = 0,45)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(R / "figT_cop.png", dpi=150)


def fig_iv() -> None:
    """Χαρακτηριστική I-V και P-V φωτοβολταϊκού στοιχείου (Κεφ. 3)."""
    isc, voc, n = 11.0, 41.0, 1.6
    v = np.linspace(0, voc, 300)
    i0 = isc / (np.exp(voc / (n * 1.6)) - 1)
    i = np.maximum(isc - i0 * (np.exp(v / (n * 1.6)) - 1), 0)
    p = v * i
    k = int(np.argmax(p))
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(v, i, "b-", lw=2, label="Ρεύμα I")
    ax1.set_xlabel("Τάση V (V)")
    ax1.set_ylabel("Ρεύμα (A)", color="b")
    ax1.axhline(isc, ls=":", c="b", alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(v, p, "r--", lw=2, label="Ισχύς P = V·I")
    ax2.set_ylabel("Ισχύς (W)", color="r")
    ax1.plot(v[k], i[k], "ko")
    ax2.annotate(f"Σημείο μέγιστης ισχύος\n(Vmp={v[k]:.0f} V, Pmp={p[k]:.0f} W)",
                 (v[k], p[k]), textcoords="offset points", xytext=(-150, -40))
    ax1.set_title("Χαρακτηριστική καμπύλη I-V και P-V Φ/Β πλαισίου")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(R / "figT_iv.png", dpi=150)


def fig_solar(tmy: pd.DataFrame) -> None:
    """Μηνιαία ηλιακή ακτινοβολία Θεσσαλονίκης: PVGIS vs ΤΟΤΕΕ (Κεφ. 3)."""
    ghi = tmy.groupby(tmy.index.month)["ghi_w_m2"].sum() / 1000.0
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(12)
    ax.bar(x - 0.2, TOTEE_THESSALONIKI["ghi_kwh_m2"], width=0.4,
           label="ΤΟΤΕΕ 20701-3/2010")
    ax.bar(x + 0.2, ghi.to_numpy(), width=0.4,
           label="PVGIS TMY 2005–2023")
    ax.set_xticks(x, MONTHS)
    ax.set_ylabel("kWh/m² ανά μήνα (οριζόντιο επίπεδο)")
    ax.set_title("Μηνιαία ολική ηλιακή ακτινοβολία — Θεσσαλονίκη")
    ax.legend()
    fig.tight_layout()
    fig.savefig(R / "figT_solar.png", dpi=150)


def fig_psat() -> None:
    """Πίεση κορεσμού υδρατμών vs θερμοκρασία — κινητήρια δύναμη εξάτμισης."""
    t = np.linspace(0, 40, 200)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, saturation_pressure_kpa(t), lw=2)
    for tw, lbl in [(24, "Νερό πισίνας 24 °C"), (31, "Εκμάθησης 31 °C")]:
        pw = float(saturation_pressure_kpa(tw))
        ax.plot(tw, pw, "ro")
        ax.annotate(f"{lbl}\np_w = {pw:.2f} kPa", (tw, pw),
                    textcoords="offset points", xytext=(10, -22))
    pa = 0.65 * float(saturation_pressure_kpa(26))
    ax.axhline(pa, ls="--", c="g", alpha=0.7)
    ax.annotate(f"p_a αέρα χώρου (26 °C, 65 %) = {pa:.2f} kPa",
                (1, pa), textcoords="offset points", xytext=(5, 6), color="g")
    ax.set_xlabel("Θερμοκρασία (°C)")
    ax.set_ylabel("Πίεση κορεσμού (kPa)")
    ax.set_title("Πίεση κορεσμού υδρατμών (τύπος Magnus) — οδηγός της εξάτμισης")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(R / "figT_psat.png", dpi=150)


def fig_grid_ef() -> None:
    """Τροχιά συντελεστή εκπομπών ελληνικής ηλεκτροπαραγωγής (Κεφ. 4, §4.6)."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    # Τιμές που χρησιμοποιεί η μελέτη (συνεχής γραμμή)
    yrs, efs = [2014, 2020, 2026, 2040], [1.12, 0.406, 0.35, 0.12]
    ax.plot(yrs, efs, "o-", lw=2, ms=8, color="tab:blue",
            label="Τιμές της μελέτης (2026: 0,35 · 2040: 0,12)")
    # Αναφορές ΔΑΠΕΕΠ 2025 (σημεία χωρίς γραμμή)
    ax.plot([2025], [0.360], "s", ms=8, color="tab:red",
            label="ΔΑΠΕΕΠ 2025 — υπολειπόμενο μείγμα [27]")
    ax.plot([2025], [0.225], "D", ms=7, color="tab:green",
            label="ΔΑΠΕΕΠ 2025 — μείγμα παραγωγής [27]")
    # Εκτίμηση βάσει ΕΣΕΚ (95,6 % ΑΠΕ το 2035) για το 2040
    ax.fill_between([2038.5, 2041.5], 0.02, 0.04, color="tab:green", alpha=0.25,
                    label="Εκτίμηση βάσει ΕΣΕΚ 2040: 0,02–0,04 [16]")
    ax.axhline(0.202, ls="--", c="brown", alpha=0.8)
    ax.text(2012.5, 0.222, "Καύσιμο ΦΑ: 0,202 kg CO₂/kWh (σταθερό)",
            color="brown", fontsize=9)
    notes = {2014: ("ΔΕΗ 2014 [5]\n1,12", (10, -4)),
             2020: ("Μέσο μείγμα 2020 [5]\n0,406", (8, 8)),
             2026: ("0,35", (8, 6)),
             2040: ("0,12", (8, 6))}
    for x, y in zip(yrs, efs):
        t, off = notes[x]
        ax.annotate(t, (x, y), textcoords="offset points", xytext=off, fontsize=9)
    ax.set_xlabel("Έτος")
    ax.set_ylabel("kg CO₂ / kWh ηλεκτρικής")
    ax.set_title("Απανθρακοποίηση του ελληνικού μείγματος ηλεκτροπαραγωγής", pad=12)
    ax.grid(alpha=0.3)
    ax.set_xlim(2012, 2044)
    ax.set_ylim(0, 1.25)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(R / "figT_grid_ef.png", dpi=150)


def fig_tmy_compare(tmy: pd.DataFrame) -> None:
    """Μέση μηνιαία θερμοκρασία: ΤΟΤΕΕ vs TMY vs 2040 (Κεφ. 4/7)."""
    t_now = tmy.groupby(tmy.index.month)["t_air_c"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(MONTHS, TOTEE_THESSALONIKI["t_mean_c"], "s--",
            label="ΤΟΤΕΕ 20701-3/2010")
    ax.plot(MONTHS, t_now.to_numpy(), "o-", lw=2,
            label="PVGIS TMY 2005–2023")
    t40 = t_now.to_numpy() + np.asarray(MORPHING_2040["rcp85"]["dt_c"])
    ax.plot(MONTHS, t40, "^:", label="2040 RCP8.5 (morphing)")
    ax.set_ylabel("Μέση μηνιαία θερμοκρασία (°C)")
    ax.set_title("Μέση μηνιαία θερμοκρασία Θεσσαλονίκης: ΤΟΤΕΕ, PVGIS TMY και 2040")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(R / "figT_tmy_compare.png", dpi=150)


def fig_real_gas() -> None:
    """Πραγματική μηνιαία κατανάλωση ΦΑ + τυπικά ωριαία προφίλ (Κεφ. 6)."""
    g = pd.read_csv(ROOT / "data" / "poseidonio_gas_monthly.csv", comment="#")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(MONTHS, g["gas_kwh_fuel"] / 1000.0, color="tab:blue")
    ax.set_ylabel("MWh καυσίμου ΦΑ / μήνα")
    ax.set_title("Πραγματική μηνιαία κατανάλωση ΦΑ — PTZ μετρητής ΕΔΑ (έτος αναφοράς)")
    for i, v in enumerate(g["gas_kwh_fuel"] / 1000.0):
        ax.text(i, v + 8, f"{v:.0f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(R / "figD_real_gas_monthly.png", dpi=150)

    h = pd.read_csv(ROOT / "data" / "poseidonio_heating_hourly_typical_day.csv",
                    comment="#")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col, style in [("Jan", "o-"), ("Apr", "s--"), ("Oct", "^:")]:
        ax.plot(h["hour"], h[col], style, label={"Jan": "Ιανουάριος",
                "Apr": "Απρίλιος", "Oct": "Οκτώβριος"}[col])
    ax.set_xlabel("Ώρα ημέρας")
    ax.set_ylabel("kWth (τελική ενέργεια)")
    ax.set_title("Τυπικό ωριαίο προφίλ θέρμανσης (επεξεργασία ωριαίων PTZ [17])")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(R / "figD_hourly_profile.png", dpi=150)


def fig_npv_profile() -> None:
    """Προφίλ ΚΠΑ = f(r) με σημείο IRR (Κεφ. 9) — όπως το υπόδειγμα §8.7."""
    from poseidonio import run_scenario, evaluate
    from copy import deepcopy
    tmy = load_pvgis_tmy(str(ROOT / "data" / "tmy.csv"))
    r0 = run_scenario(site, "S2", tmy=tmy)
    a = r0.annual
    rates = np.linspace(0.0, 0.20, 41)
    npvs = []
    for r in rates:
        s = deepcopy(site)
        s.econ.discount_rate = float(r)
        e = evaluate(s.econ, gas_kwh_baseline=a["gas_baseline_mwh_fuel"] * 1000,
                     grid_import_kwh=a["grid_import_mwh"] * 1000,
                     grid_export_kwh=a["grid_export_mwh"] * 1000,
                     backup_gas_kwh_fuel=a["backup_gas_mwh_fuel"] * 1000,
                     pv_kwp=a["pv_kwp"], hp_kw_th=s.heat_pump.capacity_kw_th)
        npvs.append(e["npv_eur"] / 1e6)
    npvs = np.asarray(npvs)
    irr = float(np.interp(0.0, npvs[::-1], rates[::-1]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rates * 100, npvs, lw=2)
    ax.axhline(0, c="k", lw=0.8)
    ax.plot(irr * 100, 0, "ro", ms=9)
    ax.annotate(f"IRR ≈ {irr*100:.0f} %", (irr * 100, 0),
                textcoords="offset points", xytext=(10, 10), color="r")
    ax.axvline(5, ls="--", c="g", alpha=0.7)
    npv5 = float(np.interp(0.05, rates, npvs))
    ax.annotate(f"r = 5 %\nNPV = {npv5:.2f} M€".replace(".", ","), (5, npv5),
                textcoords="offset points", xytext=(10, 10), color="g")
    ax.set_xlabel("Επιτόκιο προεξόφλησης r (%)")
    ax.set_ylabel("ΚΠΑ (M€)")
    ax.set_title("Προφίλ καθαρής παρούσας αξίας — σενάριο S2 (500 kWp)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(R / "figE_npv_profile.png", dpi=150)


def main() -> None:
    R.mkdir(exist_ok=True)
    tmy = load_pvgis_tmy(str(ROOT / "data" / "tmy.csv"))
    fig_cop()
    fig_iv()
    fig_solar(tmy)
    fig_psat()
    fig_grid_ef()
    fig_tmy_compare(tmy)
    fig_real_gas()
    fig_npv_profile()
    print("OK:", sorted(p.name for p in R.glob("fig[TDE]*.png")))


if __name__ == "__main__":
    main()
