"""Οικονομική & περιβαλλοντική αξιολόγηση — Net Billing, NPV/IRR/Payback, CO2.

Net Billing (ν. 5037/2023): η αυτοκαταναλισκόμενη ενέργεια συμψηφίζεται σε
ωριαία βάση με το ανταγωνιστικό σκέλος + χρεώσεις δικτύου (αποφεύγεται
πλήρες κόστος προμήθειας), ενώ η εγχεόμενη περίσσεια αποζημιώνεται σε τιμή
αναφοράς χονδρεμπορικής. Εδώ μοντελοποιείται συντηρητικά: αποφυγή αγοράς
στην τιμή λιανικής για την αυτοκατανάλωση, πίστωση injection_price για την
έγχυση.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Economics


def npv(rate: float, cashflows: np.ndarray) -> float:
    years = np.arange(len(cashflows))
    return float(np.sum(cashflows / (1 + rate) ** years))


def irr(cashflows: np.ndarray) -> float:
    """IRR με διχοτόμηση στο [−0.5, 1.0]. NaN αν δεν υπάρχει ρίζα."""
    lo, hi = -0.5, 1.0
    f = lambda r: npv(r, cashflows)
    if f(lo) * f(hi) > 0:
        return float("nan")
    for _ in range(100):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def payback_period(cashflows: np.ndarray) -> float:
    cum = np.cumsum(cashflows)
    pos = np.where(cum >= 0)[0]
    if len(pos) == 0:
        return float("nan")
    k = pos[0]
    if k == 0:
        return 0.0
    return k - 1 + (-cum[k - 1]) / (cum[k] - cum[k - 1])


def evaluate(econ: Economics, *,
             gas_kwh_baseline: float,
             grid_import_kwh: float,
             grid_export_kwh: float,
             backup_gas_kwh_fuel: float,
             pv_kwp: float,
             hp_kw_th: float,
             grid_ef: float | None = None) -> dict:
    """Ετήσιο ενεργειακό ισοζύγιο → οικονομικοί & περιβαλλοντικοί δείκτες.

    gas_kwh_baseline: κατανάλωση καυσίμου ΦΑ του σεναρίου αναφοράς (kWh/έτος).
    grid_import/export: απορρόφηση/έγχυση δικτύου του προτεινόμενου συστήματος.
    """
    ef_grid = econ.ef_grid_2026 if grid_ef is None else grid_ef

    capex = pv_kwp * econ.pv_capex_eur_kwp + hp_kw_th * econ.hp_capex_eur_kwth \
        + econ.balance_capex_eur
    opex = econ.opex_pct_capex * capex

    cost_baseline = gas_kwh_baseline * econ.gas_price_eur_kwh
    cost_new = (grid_import_kwh * econ.grid_price_eur_kwh
                - grid_export_kwh * econ.injection_price_eur_kwh
                + backup_gas_kwh_fuel * econ.gas_price_eur_kwh)
    annual_saving = cost_baseline - cost_new - opex

    # Χρηματορροές 25ετίας: ενεργειακός πληθωρισμός στα οφέλη, απομείωση Φ/Β
    years = np.arange(econ.lifetime_years + 1)
    cf = np.zeros(econ.lifetime_years + 1)
    cf[0] = -capex
    for y in years[1:]:
        infl = (1 + econ.energy_inflation) ** (y - 1)
        degr = (1 - econ.pv_degradation) ** (y - 1)
        cf[y] = (cost_baseline * infl
                 - (grid_import_kwh * econ.grid_price_eur_kwh * infl
                    - grid_export_kwh * degr * econ.injection_price_eur_kwh * infl
                    + backup_gas_kwh_fuel * econ.gas_price_eur_kwh * infl)
                 - opex)

    co2_baseline = gas_kwh_baseline * econ.ef_gas_fuel
    co2_new = grid_import_kwh * ef_grid + backup_gas_kwh_fuel * econ.ef_gas_fuel

    return {
        "capex_eur": capex,
        "annual_saving_eur": annual_saving,
        "npv_eur": npv(econ.discount_rate, cf),
        "irr": irr(cf),
        "payback_years": payback_period(cf),
        "co2_baseline_t": co2_baseline / 1000.0,
        "co2_new_t": co2_new / 1000.0,
        "co2_reduction_pct": 100.0 * (1 - co2_new / co2_baseline) if co2_baseline else float("nan"),
        "cost_baseline_eur": cost_baseline,
        "cost_new_eur": cost_new,
    }
