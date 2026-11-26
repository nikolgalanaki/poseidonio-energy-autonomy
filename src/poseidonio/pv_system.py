"""Μοντέλο φωτοβολταϊκού συστήματος — ωριαία παραγωγή.

Βήματα:
1. Διάσπαση GHI σε άμεση/διάχυτη με τη συσχέτιση Erbs (1982) μέσω του
   δείκτη αιθριότητας kt.
2. Μετατροπή στο κεκλιμένο επίπεδο (POA) με ισοτροπικό μοντέλο Liu–Jordan.
3. Θερμοκρασία κυψέλης με μοντέλο NOCT και διόρθωση ισχύος με γ (%/K).
4. Λοιπές απώλειες συστήματος (inverter, καλωδίωση, ρύπανση) με ενιαίο PR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PVSystem


def _erbs_diffuse_fraction(kt: np.ndarray) -> np.ndarray:
    fd = np.where(kt <= 0.22, 1.0 - 0.09 * kt,
         np.where(kt <= 0.80,
                  0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4,
                  0.165))
    return np.clip(fd, 0.0, 1.0)


def poa_irradiance(tmy: pd.DataFrame, latitude: float, tilt_deg: float,
                   azimuth_deg: float = 0.0, albedo: float = 0.2) -> np.ndarray:
    """Ολική ακτινοβολία στο κεκλιμένο επίπεδο (W/m²)."""
    ghi = tmy["ghi_w_m2"].to_numpy()
    cos_z = tmy["cos_zenith"].to_numpy()
    g0 = tmy["g0_w_m2"].to_numpy()

    kt = np.where(g0 > 1.0, np.clip(ghi / np.maximum(g0, 1.0), 0.0, 1.0), 0.0)
    fd = _erbs_diffuse_fraction(kt)
    dhi = fd * ghi
    dni = np.where(cos_z > 0.05, (ghi - dhi) / np.maximum(cos_z, 0.05), 0.0)

    # Γωνία πρόσπτωσης σε επίπεδο με κλίση β, αζιμούθιο γ (0=Νότος)
    lat = np.radians(latitude)
    beta = np.radians(tilt_deg)
    gamma = np.radians(azimuth_deg)
    doy = tmy.index.dayofyear.to_numpy().astype(float)
    solar_hour = tmy["solar_hour"].to_numpy()
    decl = np.radians(23.45) * np.sin(2 * np.pi * (284 + doy) / 365)
    ha = np.radians(15.0 * (solar_hour - 12.0))
    cos_inc = (np.sin(decl) * np.sin(lat) * np.cos(beta)
               - np.sin(decl) * np.cos(lat) * np.sin(beta) * np.cos(gamma)
               + np.cos(decl) * np.cos(lat) * np.cos(beta) * np.cos(ha)
               + np.cos(decl) * np.sin(lat) * np.sin(beta) * np.cos(gamma) * np.cos(ha)
               + np.cos(decl) * np.sin(beta) * np.sin(gamma) * np.sin(ha))
    cos_inc = np.clip(cos_inc, 0.0, 1.0)

    beam = dni * cos_inc
    diffuse = dhi * (1 + np.cos(beta)) / 2
    ground = ghi * albedo * (1 - np.cos(beta)) / 2
    return beam + diffuse + ground


def pv_production_kw(pv: PVSystem, tmy: pd.DataFrame, latitude: float) -> pd.Series:
    """Ωριαία παραγωγή AC (kW) για το σύστημα pv.kwp."""
    poa = poa_irradiance(tmy, latitude, pv.tilt_deg, pv.azimuth_deg)
    t_cell = tmy["t_air_c"].to_numpy() + (pv.noct_c - 20.0) / 800.0 * poa
    p_dc = pv.kwp * (poa / 1000.0) * (1.0 + pv.gamma_pmp * (t_cell - 25.0))
    p_ac = np.maximum(p_dc * pv.pr_other, 0.0)
    return pd.Series(p_ac, index=tmy.index, name="pv_kw")
