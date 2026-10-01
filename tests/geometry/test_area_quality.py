"""Wetted-area budgets 1% / 5% / 10%. Cube and hex must stay well inside 1%."""
import numpy as np
from arlamx_v2.geometry import QUALITY, cube_mesh, projected_area, simplify_sides


def test_quality_keys():
    assert set(QUALITY) == {"1pct", "5pct", "10pct"}
    assert QUALITY["1pct"]["tol"] == 0.01
    assert QUALITY["10pct"]["tol"] == 0.10


def test_cube_wetted_exact_at_1pct():
    V, F = cube_mesh(1.0)
    n, A, c, rep = simplify_sides(V, F, quality="1pct")
    assert rep.area_rel_err < 1e-9
    assert rep.quality == "1pct"
    # Face-on +X matches 1 m²
    assert abs(projected_area(n, A, [1, 0, 0]) - 1.0) < 1e-9


def test_hex_prism_within_1pct():
    r, h = 0.5, 0.2
    ang = np.linspace(0, 2 * np.pi, 7)[:-1]
    top = np.stack([r * np.cos(ang), r * np.sin(ang), np.full(6, h / 2)], 1)
    bot = np.stack([r * np.cos(ang), r * np.sin(ang), np.full(6, -h / 2)], 1)
    V = np.vstack([top, bot])
    F = []
    for i in range(1, 5):
        F.append([0, i, i + 1])
        F.append([6, 6 + i + 1, 6 + i])
    for i in range(6):
        j = (i + 1) % 6
        F.append([i, j, 6 + j])
        F.append([i, 6 + j, 6 + i])
    n, A, c, rep = simplify_sides(V, np.array(F, int), quality="1pct")
    assert rep.area_rel_err < 0.01
