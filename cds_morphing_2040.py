#!/usr/bin/env python3
"""Μηνιαίες μεταβολές θερμοκρασίας 2040 (ΔΤ) για τη Θεσσαλονίκη από EURO-CORDEX
(Copernicus Climate Data Store) — είσοδος της μεθόδου morphing (§4.8).

Τι κάνει
--------
Για κάθε ζεύγος μοντέλων GCM/RCM και κάθε σενάριο (RCP4.5, RCP8.5):
  1. κατεβάζει τη μηνιαία μέση θερμοκρασία 2 m (EUR-11, ~12 km) για τα έτη
     2006–2050 από το σύνολο «projections-cordex-domains-single-levels»·
  2. κρατά το κελί του πλέγματος πλησιέστερα στο Ποσειδώνιο (40,597° Β,
     22,950° Α) και τα 8 γειτονικά του (μέσος όρος 3×3 ≈ 36 km × 36 km)·
  3. υπολογίζει για κάθε μήνα τη διαφορά  ΔΤ = Τ(2031–2050) − Τ(2006–2024).
     Η περίοδος αναφοράς 2006–2024 συμπίπτει σχεδόν με την περίοδο του PVGIS
     TMY (2005–2023), οπότε το ΔΤ αντιστοιχεί στη μεταβολή ΠΕΡΑ από το
     κλίμα που ήδη περιέχει το TMY· η περίοδος 2031–2050 έχει κέντρο το 2040·
  4. υπολογίζει τον μέσο όρο των μοντέλων (ensemble mean) και γράφει το
     data/morphing_2040.csv, το οποίο διαβάζει αυτόματα το climate.py.

Προετοιμασία (μία φορά)
-----------------------
  1. Δωρεάν λογαριασμός: https://cds.climate.copernicus.eu  (Register)
  2. Στη σελίδα του συνόλου δεδομένων
     https://cds.climate.copernicus.eu/datasets/projections-cordex-domains-single-levels
     καρτέλα «Download» → στο τέλος αποδοχή των όρων χρήσης (Licence).
  3. Στο προφίλ σας αντιγράψτε το «Personal Access Token» και δημιουργήστε
     το αρχείο  .cdsapirc  στον φάκελο χρήστη (Windows: C:\\Users\\<όνομα>\\.cdsapirc)
     με περιεχόμενο:
         url: https://cds.climate.copernicus.eu/api
         key: <το-token-σας>
  4. pip install -r requirements.txt   (περιλαμβάνει cdsapi, xarray, netCDF4)

Εκτέλεση (από τη ρίζα του repo)
-------------------------------
    python3 cds_morphing_2040.py
    python3 cds_morphing_2040.py --test       # ένα μόνο αίτημα, για έλεγχο

Τα αρχεία που κατεβαίνουν αποθηκεύονται στο data/cordex/ (δεν ξανακατεβαίνουν
σε δεύτερη εκτέλεση). Ζητείται μόνο μικρή περιοχή γύρω από τη Θεσσαλονίκη, οπότε
τα αρχεία είναι μικρά· η αναμονή στην ουρά του CDS μπορεί να διαρκέσει από λίγα λεπτά έως ώρες.
Αν ένα ζεύγος μοντέλων δεν είναι διαθέσιμο, παραλείπεται και ο μέσος όρος
υπολογίζεται από τα υπόλοιπα (απαιτείται τουλάχιστον ένα).
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DL = ROOT / "data" / "cordex"
OUT = ROOT / "data" / "morphing_2040.csv"

SITE_LAT, SITE_LON = 40.597, 22.950
BASE = (2006, 2024)      # περίοδος αναφοράς ≈ περίοδος PVGIS TMY
FUTURE = (2031, 2050)    # κέντρο 2040

# Ζεύγη GCM / RCM / μέλος συνόλου που διατίθενται για RCP4.5 και RCP8.5 (EUR-11)
MODELS = [
    ("ichec_ec_earth", "knmi_racmo22e"),
    ("mpi_m_mpi_esm_lr", "smhi_rca4"),
    ("cnrm_cerfacs_cm5", "clmcom_clm_cclm4_8_17"),
    ("ncc_noresm1_m", "dmi_hirham5"),
]
# Μέλη συνόλου που δοκιμάζονται κατά σειρά (το CDS δέχεται μόνο όσα υπάρχουν
# για κάθε ζεύγος· π.χ. ICHEC-EC-EARTH/KNMI-RACMO22E → r1i1p1)
MEMBERS = ["r1i1p1", "r12i1p1", "r2i1p1", "r3i1p1"]
SCENARIOS = {"rcp45": "rcp_4_5", "rcp85": "rcp_8_5"}

# Από 9/7/2026 το CDS δέχεται τα πεδία "year"/"month" (όχι start/end_year) και
# χωρική υποπεριοχή "area". Ζητείται μικρό πλαίσιο γύρω από τη Θεσσαλονίκη·
# η σειρά των ορίων είναι [Β, Δ, Ν, Α], όπως δίνει η φόρμα λήψης του CDS.
YEARS = [str(y) for y in range(BASE[0], FUTURE[1] + 1)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
BOX_N, BOX_S = round(SITE_LAT + 0.5, 2), round(SITE_LAT - 0.5, 2)
BOX_W, BOX_E = round(SITE_LON - 0.6, 2), round(SITE_LON + 0.6, 2)
AREAS = [[BOX_N, BOX_W, BOX_S, BOX_E]]   # [Β, Δ, Ν, Α] — όπως στη φόρμα του CDS


class AuthError(RuntimeError):
    pass


def download(client, scen: str, gcm: str, rcm: str, res: str) -> Path | None:
    target = DL / f"tas_{scen}_{gcm}_{rcm}_{res}.dat"
    if target.exists() and target.stat().st_size > 0:
        return target
    for member, area in [(m, a) for m in MEMBERS for a in AREAS]:
        req = {
            "domain": "europe",
            "experiment": SCENARIOS[scen],
            "horizontal_resolution": f"0_{res}_degree_x_0_{res}_degree",
            "temporal_resolution": "monthly_mean",
            "variable": "2m_air_temperature",
            "gcm_model": gcm,
            "rcm_model": rcm,
            "ensemble_member": member,
            "year": YEARS,
            "month": MONTHS,
            "area": area,
        }
        try:
            print(f"  αίτημα CDS: {scen} {gcm} / {rcm} / {member}…")
            client.retrieve("projections-cordex-domains-single-levels", req, str(target))
            return target
        except Exception as exc:
            msg = str(exc)
            target.unlink(missing_ok=True)
            if "401 Client Error" in msg:
                raise AuthError("Το κλειδί στο .cdsapirc δεν έγινε δεκτό (401).") from exc
            if "403 Client Error" in msg or "required licences" in msg.lower():
                raise AuthError("Δεν έχουν γίνει αποδεκτοί οι όροι χρήσης του συνόλου (403).\n" + msg[:1500]) from exc
            print("    δεν έγινε δεκτό — πλήρες μήνυμα του Copernicus:")
            print("    " + msg[:1500].replace("\n", "\n    "))
    return None


def _nc_files(path: Path) -> list[Path]:
    """Το CDS επιστρέφει είτε ένα .nc είτε zip με .nc — υποστηρίζονται και τα δύο."""
    if zipfile.is_zipfile(path):
        folder = path.with_suffix("")
        if not folder.exists():
            with zipfile.ZipFile(path) as z:
                z.extractall(folder)
        return sorted(folder.glob("**/*.nc"))
    return [path]


def site_monthly_series(zip_path: Path) -> pd.Series:
    """Μηνιαία χρονοσειρά θερμοκρασίας (°C) του 3×3 κελιού γύρω από το Ποσειδώνιο."""
    import xarray as xr

    files = _nc_files(zip_path)
    if not files:
        raise RuntimeError(f"Δεν βρέθηκαν αρχεία .nc στο {zip_path.name}")
    try:
        coder = {"decode_times": xr.coders.CFDatetimeCoder(use_cftime=True)}
    except AttributeError:                      # παλαιότερες εκδόσεις xarray
        coder = {"use_cftime": True}
    ds = xr.concat([xr.open_dataset(str(f), **coder) for f in files], dim="time")
    ds = ds.sortby("time")
    lat, lon = ds["lat"].values, ds["lon"].values          # 2Δ (περιστραμμένο πλέγμα)
    valid = np.isfinite(ds["tas"].isel(time=0).values)
    d2 = (lat - SITE_LAT) ** 2 + ((lon - SITE_LON) * np.cos(np.radians(SITE_LAT))) ** 2
    d2 = np.where(valid, d2, np.inf)
    iy, ix = np.unravel_index(np.argmin(d2), d2.shape)
    ydim, xdim = ds["tas"].dims[-2], ds["tas"].dims[-1]
    box = ds["tas"].isel({ydim: slice(max(iy - 1, 0), iy + 2),
                          xdim: slice(max(ix - 1, 0), ix + 2)})
    ts = box.mean(dim=[ydim, xdim], skipna=True).load() - 273.15
    idx = pd.to_datetime([f"{t.year}-{t.month:02d}-15" for t in ts["time"].values])
    print(f"    κελί: {lat[iy, ix]:.3f}° Β, {lon[iy, ix]:.3f}° Α")
    return pd.Series(ts.values, index=idx)


def monthly_delta(s: pd.Series) -> np.ndarray:
    base = s[(s.index.year >= BASE[0]) & (s.index.year <= BASE[1])]
    fut = s[(s.index.year >= FUTURE[0]) & (s.index.year <= FUTURE[1])]
    if base.empty or fut.empty:
        raise RuntimeError("Η χρονοσειρά δεν καλύπτει τις περιόδους αναφοράς/μέλλοντος")
    return (fut.groupby(fut.index.month).mean() - base.groupby(base.index.month).mean()).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true",
                    help="ένα μόνο αίτημα (RCP4.5, πρώτο μοντέλο) για έλεγχο")
    ap.add_argument("--res", choices=["11"], default="11",
                    help="ανάλυση πλέγματος: για την Ευρώπη/RCP διατίθεται μόνο το EUR-11 (~12 km)")
    args = ap.parse_args()
    try:
        import cdsapi
    except ImportError:
        sys.exit("Λείπει το cdsapi: pip install -r requirements.txt")
    DL.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    global MODELS
    if args.test:
        MODELS = MODELS[:1]
    rows = []
    for scen in (["rcp45"] if args.test else SCENARIOS):
        print(f"\n=== {scen.upper()} ===")
        for gcm, rcm in MODELS:
            try:
                z = download(client, scen, gcm, rcm, args.res)
            except AuthError as exc:
                sys.exit(f"\n{exc}\nΕλέγξτε το .cdsapirc και την αποδοχή των όρων χρήσης.")
            if z is None:
                print(f"  → παραλείπεται {gcm} / {rcm}")
                continue
            try:
                dt = monthly_delta(site_monthly_series(z))
            except Exception as exc:
                print(f"  → σφάλμα επεξεργασίας {gcm} / {rcm}: {exc}")
                continue
            print("    ΔΤ (°C):", " ".join(f"{x:+.2f}" for x in dt))
            for m, x in enumerate(dt, start=1):
                rows.append({"scenario": scen, "gcm": gcm, "rcm": rcm, "month": m, "dt_c": x})

    if not rows:
        sys.exit("Κανένα μοντέλο δεν κατέβηκε. Ελέγξτε το .cdsapirc και την αποδοχή των όρων χρήσης.")
    df = pd.DataFrame(rows)
    df.to_csv(DL / "morphing_2040_per_model.csv", index=False, float_format="%.3f")
    ens = (df.groupby(["scenario", "month"])
             .agg(dt_c=("dt_c", "mean"), dt_min=("dt_c", "min"), dt_max=("dt_c", "max"),
                  n_models=("dt_c", "size"))
             .reset_index())
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"# EURO-CORDEX EUR-{args.res}, Copernicus CDS — ΔΤ = T({FUTURE[0]}–{FUTURE[1]})"
                f" − T({BASE[0]}–{BASE[1]}), 3×3 κελιά γύρω από {SITE_LAT}° Β, {SITE_LON}° Α\n")
        f.write("# Μοντέλα: " + "; ".join(sorted({f'{g}/{r}' for g, r in zip(df.gcm, df.rcm)})) + "\n")
        ens.to_csv(f, index=False, float_format="%.3f")
    print(f"\nOK → {OUT}")
    print(ens.pivot(index="month", columns="scenario", values="dt_c").round(2).to_string())


if __name__ == "__main__":
    main()
