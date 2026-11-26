"""Κλιματικά δεδομένα Θεσσαλονίκης — ωριαίο τυπικό μετεωρολογικό έτος (TMY).

Δύο πηγές:
1. load_pvgis_tmy(path): φορτώνει πραγματικό TMY από το PVGIS
   (https://re.jrc.ec.europa.eu/pvg_tools/en/ → TMY → CSV). ΠΡΟΤΙΜΩΜΕΝΟ
   για την τελική διπλωματική.
2. synthetic_tmy(): συνθετικό έτος από τα μηνιαία κλιματικά δεδομένα της
   ΤΟΤΕΕ 20701-3/2010 για τη Θεσσαλονίκη, ώστε το μοντέλο να τρέχει χωρίς
   εξωτερικά αρχεία. Οι ωριαίες κατανομές παράγονται ντετερμινιστικά
   (ημερήσιο ημιτονοειδές θερμοκρασίας, προφίλ ακτινοβολίας αίθριας ημέρας
   κλιμακωμένο στο μηνιαίο σύνολο).

Κλιματική αλλαγή 2040: μέθοδος "morphing" (Belcher et al., 2005) — μηνιαίες
μεταβολές ΔΤ από προβολές EURO-CORDEX (Copernicus CDS) εφαρμόζονται στο TMY.
Οι τιμές ΔΤ διαβάζονται από το data/morphing_2040.csv, που παράγει το
cds_morphing_2040.py (μέσος όρος μοντέλων, κελί Θεσσαλονίκης, 2031–2050 έναντι
2006–2024). Αν το αρχείο λείπει, χρησιμοποιούνται οι ενδεικτικές τιμές
MORPHING_2040_DEFAULT και εμφανίζεται προειδοποίηση.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# Σημείο του PVGIS TMY (data/tmy.csv). Η εγκατάσταση βρίσκεται στο ≈40,597° Β
# (μέτρηση σε δορυφορική εικόνα)· διαφορά ~2 km, αμελητέα (§7.3).
SITE_LATITUDE_DEG = 40.579
SITE_LONGITUDE_DEG = 22.950

# ΤΟΤΕΕ 20701-3/2010, Θεσσαλονίκη — μηνιαίες τιμές ΑΚΡΙΒΩΣ όπως στους πίνακες:
#   t_mean_c   : Πίνακας 3.1 (μέση μηνιαία θερμοκρασία 24ώρου, °C)
#   rh_mean    : Πίνακας 3.5 (μέση μηνιαία σχετική υγρασία)
#   ghi_kwh_m2 : Πίνακας 4.1 (ηλιακή ακτινοβολία στο οριζόντιο, kWh/m²·μήνα)
#   t_ground_c : Πίνακας 6.1 (θερμοκρασία νερού δικτύου κατά ΕΛΟΤ 1291, °C) —
#                κατά την ΤΟΤΕΕ ≈ θερμοκρασία εδάφους· χρησιμοποιείται και για
#                το νερό αναπλήρωσης/ΖΝΧ και για την αγωγή των δεξαμενών.
TOTEE_THESSALONIKI = pd.DataFrame({
    "t_mean_c":  [5.3, 6.8, 9.8, 14.3, 19.7, 24.5, 26.8, 26.2, 21.9, 16.3, 11.1, 6.9],
    "rh_mean":   [0.760, 0.726, 0.716, 0.675, 0.636, 0.554, 0.528, 0.551, 0.619, 0.701, 0.764, 0.778],
    "ghi_kwh_m2": [52.6, 67.5, 103.2, 140.7, 179.1, 198.6, 209.5, 184.7, 136.7, 91.4, 56.6, 45.5],
    "t_ground_c": [8.2, 7.9, 9.2, 12.8, 16.8, 20.2, 21.5, 22.8, 22.1, 19.4, 15.7, 11.0],
}, index=range(1, 13))

# Χρόνος: όλα τα ωριαία δεδομένα του μοντέλου είναι σε ΤΟΠΙΚΗ ΧΕΙΜΕΡΙΝΗ ώρα
# Ελλάδας (UTC+2, χωρίς θερινή ώρα). Η ηλιακή ώρα προκύπτει με διόρθωση
# γεωγραφικού μήκους ως προς τον μεσημβρινό ζώνης (30° Α) και εξίσωση χρόνου.
TZ_OFFSET_H = 2.0
TZ_MERIDIAN_DEG = 30.0


def _equation_of_time_h(doy: np.ndarray) -> np.ndarray:
    """Εξίσωση του χρόνου (ώρες), τύπος Spencer (1971)."""
    b = 2 * np.pi * (doy - 1) / 365.0
    e_min = 229.18 * (0.000075 + 0.001868 * np.cos(b) - 0.032077 * np.sin(b)
                      - 0.014615 * np.cos(2 * b) - 0.04089 * np.sin(2 * b))
    return e_min / 60.0


def solar_time_h(doy: np.ndarray, hour: np.ndarray, sample_offset_h: float,
                 longitude: float = SITE_LONGITUDE_DEG) -> np.ndarray:
    """Ηλιακή ώρα της δειγματοληψίας κάθε γραμμής (τοπική χειμερινή ώρα)."""
    return (hour + sample_offset_h + (longitude - TZ_MERIDIAN_DEG) / 15.0
            + _equation_of_time_h(doy))


# Ενδεικτικές μηνιαίες αποκλίσεις 2040 — χρησιμοποιούνται ΜΟΝΟ αν δεν υπάρχει
# το data/morphing_2040.csv (εφεδρεία ώστε το μοντέλο να τρέχει πάντα).
MORPHING_2040_DEFAULT = {
    "rcp45": {"dt_c": [0.9, 0.9, 0.8, 0.8, 0.9, 1.0, 1.1, 1.2, 1.0, 0.9, 0.9, 0.9]},
    "rcp85": {"dt_c": [1.2, 1.2, 1.1, 1.1, 1.3, 1.5, 1.7, 1.8, 1.5, 1.3, 1.2, 1.2]},
}
# Μεταβολή σχετικής υγρασίας (ποσοστιαίες μονάδες): ενδεικτική, συμβατή με
# την τάση ξήρανσης της Μεσογείου· δεν λαμβάνεται από το CDS.
DRH_2040 = {"rcp45": -0.01, "rcp85": -0.02}

MORPHING_CSV = Path(__file__).resolve().parents[2] / "data" / "morphing_2040.csv"


def load_morphing_2040(path: Path = MORPHING_CSV) -> tuple[dict, str]:
    """Μηνιαία ΔΤ ανά σενάριο από το CSV του Copernicus CDS (ή τις ενδεικτικές)."""
    if path.exists():
        df = pd.read_csv(path, comment="#")
        out = {}
        for scen in ("rcp45", "rcp85"):
            d = df[df["scenario"] == scen].sort_values("month")
            if len(d) != 12:
                raise ValueError(f"{path.name}: λείπουν μήνες για {scen}")
            out[scen] = {"dt_c": d["dt_c"].tolist()}
        return out, f"EURO-CORDEX / Copernicus CDS ({path.name})"
    warnings.warn("Δεν βρέθηκε το data/morphing_2040.csv — χρησιμοποιούνται ενδεικτικές "
                  "τιμές ΔΤ. Τρέξτε πρώτα: python3 cds_morphing_2040.py")
    return MORPHING_2040_DEFAULT, "ενδεικτικές τιμές (χωρίς CDS)"


MORPHING_2040, MORPHING_SOURCE = load_morphing_2040()
for _k in MORPHING_2040:
    MORPHING_2040[_k]["drh"] = DRH_2040[_k]


def _solar_geometry(lat_deg: float, doy: np.ndarray, solar_hour: np.ndarray):
    """cos(ζενίθιας γωνίας), εξωγήινη ακτινοβολία G0 (W/m²) και απόκλιση.

    solar_hour: η ηλιακή ώρα της δειγματοληψίας (βλ. solar_time_h).
    """
    lat = np.radians(lat_deg)
    decl = np.radians(23.45) * np.sin(2 * np.pi * (284 + doy) / 365)
    ha = np.radians(15.0 * (solar_hour - 12.0))
    cos_z = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    cos_z = np.clip(cos_z, 0.0, 1.0)
    g0 = 1367.0 * (1 + 0.033 * np.cos(2 * np.pi * doy / 365)) * cos_z
    return cos_z, g0, decl


def synthetic_tmy(latitude: float = SITE_LATITUDE_DEG) -> pd.DataFrame:
    """Ωριαίο DataFrame 8760 γραμμών: t_air_c, rh, ghi_w_m2, t_ground_c."""
    idx = pd.date_range("2023-01-01", periods=8760, freq="h")
    doy = idx.dayofyear.to_numpy().astype(float)
    hour = idx.hour.to_numpy().astype(float)
    month = idx.month.to_numpy()

    clim = TOTEE_THESSALONIKI
    t_month = clim["t_mean_c"].to_numpy()[month - 1]
    rh_month = clim["rh_mean"].to_numpy()[month - 1]

    # Ημερήσιο εύρος θερμοκρασίας ~8 °C, ελάχιστο στις 05:00, μέγιστο στις 14:00
    t_air = t_month + 4.0 * np.sin(2 * np.pi * (hour - 9.5) / 24)
    # Η RH κινείται αντίθετα από τη θερμοκρασία μέσα στην ημέρα
    rh = np.clip(rh_month - 0.10 * np.sin(2 * np.pi * (hour - 9.5) / 24), 0.15, 0.98)

    # Ακτινοβολία: προφίλ αίθριας ημέρας κλιμακωμένο στο μηνιαίο σύνολο ΤΟΤΕΕ
    sol_h = solar_time_h(doy, hour, 0.5)        # ωριαίος μέσος όρος: κέντρο hh:30
    cos_z, g0, _ = _solar_geometry(latitude, doy, sol_h)
    ghi_clear = 0.75 * g0
    df = pd.DataFrame({"t_air_c": t_air, "rh": rh, "ghi_w_m2": ghi_clear,
                       "month": month}, index=idx)
    target = clim["ghi_kwh_m2"].to_numpy() * 1000.0  # Wh/m²/μήνα
    month_sum = df.groupby("month")["ghi_w_m2"].transform("sum").to_numpy()
    df["ghi_w_m2"] = df["ghi_w_m2"] * target[month - 1] / np.where(month_sum > 0, month_sum, 1)
    df["t_ground_c"] = clim["t_ground_c"].to_numpy()[month - 1]
    df["cos_zenith"] = cos_z
    df["g0_w_m2"] = g0
    df["solar_hour"] = sol_h
    return df.drop(columns="month")


def load_pvgis_tmy(path: str, latitude: float = SITE_LATITUDE_DEG) -> pd.DataFrame:
    """Φορτώνει PVGIS TMY CSV (v5.2/v5.3) στο ίδιο σχήμα με το synthetic_tmy().

    Εντοπίζει δυναμικά τη γραμμή επικεφαλίδας "time(UTC),..." (το πλήθος των
    γραμμών μεταδεδομένων διαφέρει μεταξύ εκδόσεων PVGIS) και αγνοεί το
    υποσέλιδο με τις περιγραφές των στηλών.
    """
    with open(path, encoding="utf-8-sig") as f:
        lines = f.readlines()
    header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("time(UTC)"))
    from io import StringIO
    data_lines = [lines[header_idx]]
    for ln in lines[header_idx + 1:]:
        if ln.strip() == "" or not ln[:4].isdigit():
            break
        data_lines.append(ln)
    raw = pd.read_csv(StringIO("".join(data_lines)))
    raw.columns = [c.strip() for c in raw.columns]
    # Μετατόπιση από UTC σε τοπική χειμερινή ώρα (UTC+2): η γραμμή 00:00
    # τοπικής ώρας είναι η 22:00 UTC της προηγούμενης ημέρας (το TMY είναι
    # κυκλικό έτος, οπότε οι δύο πρώτες ώρες έρχονται από το τέλος του).
    shift = int(TZ_OFFSET_H)
    roll = lambda col: np.roll(raw[col].to_numpy()[:8760], shift)
    idx = pd.date_range("2023-01-01", periods=8760, freq="h")
    df = pd.DataFrame({
        "t_air_c": roll("T2m"),
        "rh": roll("RH") / 100.0,
        "ghi_w_m2": roll("G(h)"),
    }, index=idx)
    # Το PVGIS δίνει στιγμιαίες τιμές στο hh:mm + «Irradiance Time Offset»
    offset = 0.0
    for ln in lines[:header_idx]:
        if ln.lower().startswith("irradiance time offset"):
            offset = float(ln.split(":")[1].strip())
    month = idx.month.to_numpy()
    df["t_ground_c"] = TOTEE_THESSALONIKI["t_ground_c"].to_numpy()[month - 1]
    doy = idx.dayofyear.to_numpy().astype(float)
    hour = idx.hour.to_numpy().astype(float)
    sol_h = solar_time_h(doy, hour, offset)
    cos_z, g0, _ = _solar_geometry(latitude, doy, sol_h)
    df["cos_zenith"] = cos_z
    df["g0_w_m2"] = g0
    df["solar_hour"] = sol_h
    return df


def morph_to_2040(tmy: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Εφαρμόζει μηνιαία ΔΤ/ΔRH (morphing) στο TMY για το σενάριο RCP."""
    spec = MORPHING_2040[scenario]
    out = tmy.copy()
    month = out.index.month.to_numpy()
    out["t_air_c"] = out["t_air_c"] + np.asarray(spec["dt_c"])[month - 1]
    out["rh"] = np.clip(out["rh"] + spec["drh"], 0.10, 0.98)
    out["t_ground_c"] = out["t_ground_c"] + 0.7 * np.asarray(spec["dt_c"])[month - 1]
    return out


def saturation_pressure_kpa(t_c: np.ndarray) -> np.ndarray:
    """Πίεση κορεσμού υδρατμών (kPa), τύπος Magnus."""
    t_c = np.asarray(t_c, dtype=float)
    return 0.61094 * np.exp(17.625 * t_c / (t_c + 243.04))


def humidity_ratio(t_c: np.ndarray, rh: np.ndarray, p_atm_kpa: float = 101.325) -> np.ndarray:
    """Ειδική υγρασία w (kg υδρατμών / kg ξηρού αέρα)."""
    pv = rh * saturation_pressure_kpa(t_c)
    return 0.622 * pv / (p_atm_kpa - pv)
