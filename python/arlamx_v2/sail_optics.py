"""Sail optical partitions (ca, cs, cd) for the plate SRP law."""

from __future__ import annotations

# Plate law F = -P A cos(th) [(ca + cd) s + (2 cs cos(th) + 2/3 cd) n], ca + cs + cd = 1
# (Montenbruck & Gill 2000, Sec. 3.4; Vallado 2013, Sec. 8.6.4); partitions after McInnes 1999.
PRESETS = {
    "cannonball_cr18": {"mode": "cannonball", "Cr": 1.8, "ca": 0.0, "cs": 0.0, "cd": 0.0},
    "absorber": {"mode": "optical", "Cr": 1.0, "ca": 1.00, "cs": 0.00, "cd": 0.00},
    "specular": {"mode": "optical", "Cr": 2.0, "ca": 0.00, "cs": 1.00, "cd": 0.00},
    "lambert": {"mode": "optical", "Cr": 1.0, "ca": 0.00, "cs": 0.00, "cd": 1.00},
    "al_mylar": {"mode": "optical", "Cr": 1.8, "ca": 0.08, "cs": 0.88, "cd": 0.04},
    "al_kapton": {"mode": "optical", "Cr": 1.7, "ca": 0.12, "cs": 0.80, "cd": 0.08},
    "black_kapton": {"mode": "optical", "Cr": 1.1, "ca": 0.92, "cs": 0.00, "cd": 0.08},
    "white_paint": {"mode": "optical", "Cr": 1.4, "ca": 0.20, "cs": 0.04, "cd": 0.76},
    "aged_sail": {"mode": "optical", "Cr": 1.5, "ca": 0.35, "cs": 0.40, "cd": 0.25},
}


def apply_optics(params, name):
    spec = PRESETS[name]
    params.use_panel_srp = True
    if spec["mode"] == "cannonball":
        params.srp_optical = False
        params.Cr = spec["Cr"]
    else:
        params.srp_optical = True
        params.srp_ca = spec["ca"]
        params.srp_cs = spec["cs"]
        params.srp_cd = spec["cd"]
        params.Cr = spec["Cr"]
    return params
