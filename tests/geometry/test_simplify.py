"""10 — Sealed-side simplification. Citations: Cohen-Steiner et al. 2004; Shewchuk."""
import numpy as np
import pytest
from arlamx_v2.geometry import cube_mesh, simplify_sides


def test_unit_cube_six_sides():
    V, F = cube_mesh(1.0)
    n, A, c, rep = simplify_sides(V, F, theta_deg=12.0)
    assert rep.n_sides == 6
    assert abs(rep.area_out - 6.0) < 1e-9
    assert abs(rep.area_in - 6.0) < 1e-9
    assert rep.unmatched_m == 0.0
    # Normals axis-aligned
    for ni in n:
        assert abs(np.linalg.norm(ni) - 1.0) < 1e-12
        assert abs(np.max(np.abs(ni)) - 1.0) < 1e-9


def test_hex_prism_sealed():
    # Regular hex prism, height 0.2, circumradius 0.5
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
    F = np.array(F, int)
    n, A, c, rep = simplify_sides(V, F, theta_deg=15.0)
    assert rep.unmatched_m == 0.0
    # 6 walls + 2 faces
    assert 7 <= rep.n_sides <= 10
    A_hex = 3 * np.sqrt(3) / 2 * r * r
    A_true = 2 * A_hex + 6 * r * h
    assert abs(rep.area_out / A_true - 1.0) < 0.02


def test_checker_flags_interior_crack():
    V, F = cube_mesh(1.0)
    # Duplicate a split so an interior edge is owned by one clustered side only
    # by dropping one triangle — opens a hole.
    F2 = F[1:]  # remove first triangle of -z
    n, A, c, rep = simplify_sides(V, F2, theta_deg=12.0)
    assert rep.unmatched_m > 0.0 or abs(rep.area_out - 6.0) > 0.1


def test_step_box_matches_stl_box(tmp_path):
    """STEP (cascadio/OpenCASCADE, file units) and STL of the same box simplify alike."""
    pytest.importorskip("cascadio")
    import trimesh
    from pathlib import Path
    from arlamx_v2.geometry import load_mesh, simplify_sides, write_geom, load_geom
    step = Path(__file__).parent / "data" / "box_100x200x20mm.step"
    Vs, Fs = load_mesh(step)
    box = trimesh.creation.box(extents=(100.0, 200.0, 20.0))
    box.apply_translation((50.0, 100.0, 10.0))
    stl = tmp_path / "box.stl"
    box.export(stl)
    Vt, Ft = load_mesh(stl, units="mm")
    ns, As, cs, rs = simplify_sides(Vs, Fs)
    nt, At, ct, rt = simplify_sides(Vt, Ft)
    assert rs.n_sides == rt.n_sides == 6
    assert As.sum() == pytest.approx(0.052, rel=1e-6)          # GLB transport is float32
    assert As.sum() == pytest.approx(At.sum(), rel=1e-6)
    key = lambda n, A: sorted((tuple(np.round(ni, 6)), round(ai, 7)) for ni, ai in zip(n, A))
    assert key(*_sum_by_normal(ns, As)) == key(*_sum_by_normal(nt, At))
    g = write_geom(tmp_path / "box.geom", ns, As, cs, header="box")
    n2, A2, c2 = load_geom(g)
    assert np.allclose(A2, As) and np.allclose(c2, cs)
    with pytest.raises(ValueError):
        load_mesh(step, units="mm")


def _sum_by_normal(n, A):
    d = {}
    for ni, ai in zip(np.round(n, 6), A):
        d[tuple(ni)] = d.get(tuple(ni), 0.0) + ai
    return np.array(list(d.keys())), np.array(list(d.values()))
