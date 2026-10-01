"""MSIS (pymsis) density, temperature, mean mass and species fractions."""
from datetime import datetime, timezone

import numpy as np


SPECIES_MASS = {
    "He": 4.002602e-3 / 6.02214076e23,
    "O": 15.999e-3 / 6.02214076e23,
    "N2": 28.014e-3 / 6.02214076e23,
    "O2": 31.998e-3 / 6.02214076e23,
    "Ar": 39.948e-3 / 6.02214076e23,
    "H": 1.008e-3 / 6.02214076e23,
    "N": 14.007e-3 / 6.02214076e23,
}


# C++ CLL / Walker mix order: He, O, N2, O2, Ar, H, N
SPECIES_ORDER = ("He", "O", "N2", "O2", "Ar", "H", "N")


# NRLMSISE-00 / MSIS 2.x via pymsis (Picone et al. 2002; Emmert et al. 2021).
def query_msis(alt_km, lat_deg, lon_deg, dt_utc, f107=150.0, ap=4.0, version=2.1,
               return_species=False):
    import pymsis

    kwargs = {}
    if f107 is not None:
        kwargs["f107s"] = np.array([f107])
        kwargs["f107as"] = np.array([f107])
    if ap is not None:
        # MSIS Ap vector (Picone et al. 2002): daily, 3 h now, -3, -6, -9 h, mean -12..-33 h,
        # mean -36..-57 h; a scalar fills all seven (steady-Ap scenario).
        aps = np.atleast_1d(np.asarray(ap, float))
        if aps.size == 1:
            aps = np.repeat(aps, 7)
        elif aps.size == 7:
            kwargs["geomagnetic_activity"] = -1
        else:
            raise ValueError("ap must be a scalar or a length-7 MSIS Ap history")
        kwargs["aps"] = aps.reshape(1, 7)
    result = pymsis.calculate(
        np.array([np.datetime64(dt_utc, "ns")]),
        np.array([lon_deg]),
        np.array([lat_deg]),
        np.array([alt_km]),
        version=version,
        **kwargs,
    )
    data = result[0] if result.ndim == 2 else np.ravel(result)[:11]
    rho, T = float(data[0]), float(data[10])

    def _n(x):
        x = float(x)
        return x if np.isfinite(x) and x > 0.0 else 0.0

    # Fold AO into O (same mass, same Walker O row). NO has no Walker row
    # and is omitted — χ is renormalized over the seven Walker species, not
    # the full MSIS number density (Walker/ADBSat also drop AO from their table).
    n = {
        "N2": _n(data[1]),
        "O2": _n(data[2]),
        "O": _n(data[3]) + (_n(data[8]) if len(data) > 8 else 0.0),
        "He": _n(data[4]),
        "H": _n(data[5]),
        "Ar": _n(data[6]),
        "N": _n(data[7]),
    }
    tot = sum(n.values())
    m_bar = (sum(n[s] * SPECIES_MASS[s] for s in n) / tot) if tot > 0 else SPECIES_MASS["O"]
    if not return_species:
        return rho, T, float(m_bar)
    if tot > 0:
        chi = np.array([float(n[s] / tot) for s in SPECIES_ORDER], float)
    else:
        chi = np.zeros(7, float)
        chi[1] = 1.0
    return rho, T, float(m_bar), chi


def jd_to_datetime(jd):
    unix = (float(jd) - 2440587.5) * 86400.0
    return datetime.fromtimestamp(unix, tz=timezone.utc).replace(tzinfo=None)
