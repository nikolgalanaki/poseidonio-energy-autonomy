"""Θερμικό ισοζύγιο δεξαμενών κολύμβησης — ωριαία ανάλυση.

Μηχανισμοί απωλειών (ASHRAE Handbook—HVAC Applications, Ch. 6 Natatoriums):
  1. Εξάτμιση (κυρίαρχη, ~60–70 %): εξ. Carrier/ASHRAE
       m_evap = A·(p_w − p_a)·(0.089 + 0.0782·V)/Y   [kg/s]
     με p σε kPa, V ταχύτητα αέρα (m/s), Y λανθάνουσα θερμότητα (kJ/kg),
     επί συντελεστή δραστηριότητας FA.
  2. Συναγωγή νερού–αέρα.
  3. Ακτινοβολία προς τις επιφάνειες του χώρου (ή τον ουρανό για την ανοικτή).
  4. Αγωγή προς το έδαφος (μικρή).
  5. Θέρμανση νερού αναπλήρωσης (εξάτμιση + ανανέωση/backwash).

Οι εσωτερικές δεξαμενές βλέπουν τις συνθήκες χώρου (26 °C χειμώνα κατά
ASHRAE/βαθμονόμηση — βλ. config.Hall — και 25 °C θέρος κατά [ΣΗΘΥΑ])· η
ανοικτή δεξαμενή βλέπει τις εξωτερικές συνθήκες του TMY με αυξημένη ταχύτητα
αέρα (έκθεση στον άνεμο, Z=0,25 κατά [ΣΗΘΥΑ]).
Ιούνιο–Σεπτέμβριο η θέρμανση χώρων και πισινών ΔΕΝ λειτουργεί (πραγματική
πρακτική Τεχνικής Υπηρεσίας) — απομένει μόνο το ΖΝΧ.

Επιπλέον φορτία χώρου: θέρμανση νωπού αέρα αφύγρανσης (με ανάκτηση
θερμότητας) και απώλειες κελύφους. ΖΝΧ λουτρών υπολογίζεται χωριστά.

Σημείωση (Πίνακας 7.4): τα μηνιαία σύνολα «Δεξαμενές» και «ΖΝΧ» προκύπτουν
σχεδόν ίσα κατά σύμπτωση — και τα δύο είναι γραμμικά ως προς τη θερμοκρασία
νερού δικτύου με παρόμοια κλίση (~5,7 / ~5,5 kW/K) και η διαφορά των σταθερών
όρων (~95 kW) ισούται περίπου με τον ρυθμό εξάτμισης (επαληθεύτηκε ανεξάρτητα).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .climate import saturation_pressure_kpa, humidity_ratio
from .config import Site, Pool

CP_WATER = 4.186          # kJ/kg·K
CP_AIR = 1.006            # kJ/kg·K
RHO_AIR = 1.2             # kg/m³
LATENT_KJ_KG = 2430.0     # λανθάνουσα θερμότητα εξάτμισης στους ~27 °C
SIGMA = 5.67e-8           # W/m²K⁴
EMISSIVITY_WATER = 0.95
H_CONV = 3.5              # W/m²K συναγωγή ήρεμου εσωτερικού χώρου
U_GROUND = 0.45           # W/m²K τοιχώματα δεξαμενής προς έδαφος
RENEWAL_FRACTION_DAY = 0.02  # ημερήσια ανανέωση νερού (υγειονομική) 2 % όγκου


def pool_losses_kw(pool: Pool, t_air_c: np.ndarray, rh_air: np.ndarray,
                   air_speed: float, t_ground_c: np.ndarray,
                   occupied: np.ndarray) -> pd.DataFrame:
    """Ωριαίες απώλειες μίας δεξαμενής (kW) ανά μηχανισμό.

    t_air_c/rh_air: συνθήκες αέρα πάνω από το νερό (χώρος ή περιβάλλον).
    """
    t_air_c = np.asarray(t_air_c, dtype=float)
    rh_air = np.asarray(rh_air, dtype=float)
    p_w = saturation_pressure_kpa(pool.water_temp_c)
    p_a = rh_air * saturation_pressure_kpa(t_air_c)
    # Συντελεστής δραστηριότητας: πλήρης στις ώρες λειτουργίας, 0.5 κλειστά
    fa = np.where(occupied, pool.activity_factor, 0.5)
    m_evap = pool.area_m2 * np.maximum(p_w - p_a, 0.0) \
        * (0.089 + 0.0782 * air_speed) / LATENT_KJ_KG
    m_evap = m_evap * fa                                    # kg/s
    q_evap = m_evap * LATENT_KJ_KG                          # kW

    q_conv = H_CONV * pool.area_m2 * (pool.water_temp_c - t_air_c) / 1000.0
    t_w_k, t_a_k = pool.water_temp_c + 273.15, t_air_c + 273.15
    q_rad = EMISSIVITY_WATER * SIGMA * pool.area_m2 * (t_w_k**4 - t_a_k**4) / 1000.0
    wall_area = pool.area_m2 + 2 * (pool.length_m + pool.width_m) * pool.depth_m
    q_cond = U_GROUND * wall_area * (pool.water_temp_c - t_ground_c) / 1000.0

    # Νερό αναπλήρωσης: εξατμιζόμενο + ανανέωση, από θερμοκρασία δικτύου
    m_renew = RENEWAL_FRACTION_DAY * pool.volume_m3 * 1000.0 / 86400.0  # kg/s
    q_makeup = (m_evap + m_renew) * CP_WATER * (pool.water_temp_c - t_ground_c)

    n = len(t_ground_c)
    return pd.DataFrame({
        "q_evap_kw": np.broadcast_to(q_evap, (n,)),
        "q_conv_kw": np.maximum(np.broadcast_to(q_conv, (n,)), 0.0),
        "q_rad_kw": np.maximum(np.broadcast_to(q_rad, (n,)), 0.0),
        "q_cond_kw": np.maximum(q_cond, 0.0),
        "q_makeup_kw": np.maximum(q_makeup, 0.0),
        "m_evap_kg_h": np.broadcast_to(m_evap * 3600.0, (n,)),
    })


def hall_ventilation_kw(m_evap_total_kg_h: np.ndarray, hall_t_c: np.ndarray,
                        hall_rh: float, t_out_c: np.ndarray, rh_out: np.ndarray,
                        heat_recovery: float) -> np.ndarray:
    """Αισθητό φορτίο θέρμανσης νωπού αέρα αφύγρανσης (kW).

    Η παροχή νωπού αέρα καθορίζεται από την απαγωγή της εξατμιζόμενης
    υγρασίας: m_air = m_evap / (w_hall − w_out). Ο αέρας προθερμαίνεται
    από εναλλάκτη ανάκτησης και θερμαίνεται ως τη θερμοκρασία χώρου.
    """
    w_hall = humidity_ratio(hall_t_c, hall_rh)
    w_out = humidity_ratio(t_out_c, rh_out)
    dw = np.maximum(w_hall - w_out, 0.002)          # αποφυγή απειρισμού το καλοκαίρι
    m_air = (m_evap_total_kg_h / 3600.0) / dw       # kg ξηρού αέρα / s
    dt = np.maximum(hall_t_c - t_out_c, 0.0) * (1.0 - heat_recovery)
    return m_air * CP_AIR * dt                      # kW


def dhw_load_kw(site: Site, hours: pd.DatetimeIndex, t_ground_c: np.ndarray) -> np.ndarray:
    """Φορτίο ΖΝΧ λουτρών (kW), κατανεμημένο στις ώρες λειτουργίας 07:00–22:00."""
    d = site.dhw
    daily_kwh = d.users_per_day * d.litres_per_user * CP_WATER \
        * np.maximum(d.supply_temp_c - t_ground_c, 0.0) / 3600.0
    open_hours = (hours.hour >= 7) & (hours.hour < 22)
    load = np.where(open_hours, daily_kwh / 15.0, 0.0)
    return load


def thermal_demand(site: Site, tmy: pd.DataFrame) -> pd.DataFrame:
    """Συνολική ωριαία θερμική ζήτηση (kW) ανά κατηγορία."""
    hours = tmy.index
    occupied = ((hours.hour >= 7) & (hours.hour < 22)).astype(bool)
    heating_on = np.isin(hours.month, site.heating_months)
    space_heating_on = np.isin(hours.month, site.space_heating_months)
    t_ground = tmy["t_ground_c"].to_numpy()
    t_out = tmy["t_air_c"].to_numpy()
    rh_out = tmy["rh"].to_numpy()

    # Θερμοκρασία χώρου πάνω από το νερό (καθορίζει την εξάτμιση)
    hall_t = np.where(heating_on, site.hall.air_temp_winter_c,
                      site.hall.air_temp_summer_c)
    hall_rh = np.full(len(tmy), site.hall.rh_setpoint)

    q_pool_total = np.zeros(len(tmy))
    q_evap_total = np.zeros(len(tmy))
    m_evap_indoor = np.zeros(len(tmy))
    for pool in site.pools:
        months = site.heating_months if pool.heated_months is None else pool.heated_months
        heated = np.isin(hours.month, months) if len(months) else np.zeros(len(tmy), bool)
        if pool.indoor:
            df = pool_losses_kw(pool, hall_t, hall_rh, site.hall.air_speed_ms,
                                t_ground, occupied)
            # Η υγρασία απάγεται από την ΚΚΜ ακόμη κι όταν η δεξαμενή δεν
            # θερμαίνεται ενεργά (νερό σε θερμοκρασία ισορροπίας ~ όμοια)
            m_evap_indoor += df["m_evap_kg_h"].to_numpy()
        else:
            speed = pool.air_speed_ms if pool.air_speed_ms is not None else 1.0
            df = pool_losses_kw(pool, t_out, rh_out, speed, t_ground, occupied)
        q_pool = (df["q_evap_kw"] + df["q_conv_kw"] + df["q_rad_kw"]
                  + df["q_cond_kw"] + df["q_makeup_kw"]).to_numpy()
        q_pool_total += np.where(heated, q_pool, 0.0)
        q_evap_total += np.where(heated, df["q_evap_kw"].to_numpy(), 0.0)

    q_vent = hall_ventilation_kw(m_evap_indoor, hall_t, site.hall.rh_setpoint,
                                 t_out, rh_out, site.hall.heat_recovery_eff)
    # Απώλειες κελύφους μειωμένες κατά τα ηλιακά/εσωτερικά κέρδη (απλοποίηση:
    # θερμοκρασία ισορροπίας = θερμοκρασία χώρου)
    q_env = np.maximum(site.hall.envelope_ua_w_k * (hall_t - t_out), 0.0) / 1000.0
    q_dhw = dhw_load_kw(site, hours, t_ground)

    out = pd.DataFrame(index=hours)
    # Θερινή διακοπή: Ιουν–Σεπ μόνο ΖΝΧ· τον Μάιο μόνο πισίνες (όχι χώροι) [ΣΗΘΥΑ]
    out["q_pool_kw"] = q_pool_total
    out["q_evap_kw"] = q_evap_total
    out["q_vent_kw"] = np.where(space_heating_on, q_vent, 0.0)
    out["q_envelope_kw"] = np.where(space_heating_on, q_env, 0.0)
    out["q_dhw_kw"] = q_dhw
    out["q_total_kw"] = out["q_pool_kw"] + out["q_vent_kw"] + out["q_envelope_kw"] + out["q_dhw_kw"]
    return out
