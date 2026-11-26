"""Μοντέλο αντλίας θερμότητας αέρα–νερού.

COP βασισμένο σε κλάσμα του θεωρητικού COP Carnot:
    COP = η_carnot · T_sink / (T_sink − T_source)
με T σε Kelvin, T_sink = θερμοκρασία προσαγωγής + 3 K (συμπυκνωτής),
T_source = θερμοκρασία εξωτερικού αέρα − 5 K (εξατμιστής, προσέγγιση
θερμοκρασιακής διαφοράς εναλλάκτη). Τυπικό η_carnot 0.40–0.50 για
σύγχρονες Α/Θ inverter (validate με δεδομένα κατασκευαστή κατά EN 14511).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Site


def cop(t_air_c: np.ndarray, t_supply_c: float, carnot_eff: float, max_cop: float) -> np.ndarray:
    t_sink = t_supply_c + 3.0 + 273.15
    t_source = np.asarray(t_air_c, dtype=float) - 5.0 + 273.15
    lift = np.maximum(t_sink - t_source, 10.0)
    return np.clip(carnot_eff * t_sink / lift, 1.5, max_cop)


def electricity_demand(site: Site, demand: pd.DataFrame, tmy: pd.DataFrame) -> pd.DataFrame:
    """Ωριαία ηλεκτρική κατανάλωση Α/Θ (kW) για κάλυψη της θερμικής ζήτησης.

    Τα φορτία δεξαμενών/αερισμού/κελύφους καλύπτονται σε χαμηλή θερμοκρασία
    προσαγωγής (45 °C), το ΖΝΧ σε 55 °C. Θερμική ισχύς άνω της ονομαστικής
    καλύπτεται από εφεδρικό λέβητα ΦΑ (καταγράφεται χωριστά).
    """
    hp = site.heat_pump
    t_air = tmy["t_air_c"].to_numpy()
    cop_low = cop(t_air, hp.supply_temp_pool_c, hp.carnot_eff, hp.max_cop)
    cop_dhw = cop(t_air, hp.supply_temp_dhw_c, hp.carnot_eff, hp.max_cop)

    q_low = (demand["q_total_kw"] - demand["q_dhw_kw"]).to_numpy()
    q_dhw = demand["q_dhw_kw"].to_numpy()
    q_req = q_low + q_dhw

    # Περιορισμός ονομαστικής ισχύος: η υπέρβαση πάει στον εφεδρικό λέβητα
    scale = np.minimum(1.0, hp.capacity_kw_th / np.maximum(q_req, 1e-9))
    q_hp_low, q_hp_dhw = q_low * scale, q_dhw * scale
    q_backup = q_req - q_hp_low - q_hp_dhw

    out = pd.DataFrame(index=demand.index)
    out["p_el_kw"] = q_hp_low / cop_low + q_hp_dhw / cop_dhw
    out["q_hp_kw"] = q_hp_low + q_hp_dhw
    out["q_backup_kw"] = q_backup
    out["cop_effective"] = np.where(out["p_el_kw"] > 0, out["q_hp_kw"] / out["p_el_kw"], np.nan)
    return out
