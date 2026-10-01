"""Mesh (.stl/.obj/.step) -> sealed planar sides and .geom plates."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


UNIT_SCALES = {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "in": 0.0254, "inch": 0.0254}
STEP_EXT = (".step", ".stp")
MESH_EXT = (".stl", ".obj") + STEP_EXT


QUALITY = {
    "1pct": {"tol": 0.01, "theta_deg": 8.0, "outline_ok": 0.005, "label": "1% (preferred)"},
    "5pct": {"tol": 0.05, "theta_deg": 12.0, "outline_ok": 0.03, "label": "5%"},
    "10pct": {"tol": 0.10, "theta_deg": 22.0, "outline_ok": 0.08, "label": "10% (crude only)"},
}


@dataclass
class SimplifyReport:
    n_sides: int
    n_panels: int
    area_in: float
    area_out: float
    unmatched_m: float
    max_gap_m: float
    quality: str = "1pct"
    area_rel_err: float = 0.0
    theta_deg: float = 8.0


def load_mesh(path, units=None):
    import trimesh

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    ext = path.suffix.lower()
    if ext not in MESH_EXT:
        raise ValueError(f"mesh must be one of {MESH_EXT}, got {ext!r}")
    step = ext in STEP_EXT
    if units is None:
        units = "m" if step else "mm"
    if units not in UNIT_SCALES:
        raise ValueError(units)
    if step and units != "m":
        raise ValueError("STEP carries its own length unit; trimesh/cascadio returns metres")
    mesh = trimesh.load(str(path), force="mesh")
    if step:
        mesh.merge_vertices()
    V = np.asarray(mesh.vertices, float) * UNIT_SCALES[units]
    F = np.asarray(mesh.faces, int)
    return V, F


def load_geom(path):
    n, A, c = [], [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            t = s.split()
            if len(t) != 7:
                raise ValueError(f"bad geom line: {s}")
            nn = np.array([float(t[0]), float(t[1]), float(t[2])])
            mag = np.linalg.norm(nn)
            if mag < 1e-15:
                raise ValueError("zero normal")
            n.append(nn / mag)
            A.append(float(t[3]))
            c.append([float(t[4]), float(t[5]), float(t[6])])
    return np.asarray(n), np.asarray(A), np.asarray(c)


def _face_normals_areas(V, F):
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(v1 - v0, v2 - v0)
    twice = np.linalg.norm(cr, axis=1)
    area = 0.5 * twice
    n = np.zeros_like(cr)
    ok = twice > 1e-15
    n[ok] = cr[ok] / twice[ok, None]
    return n, area


def _adjacency(F):
    edge = {}
    adj = [[] for _ in range(len(F))]
    for i, (a, b, c) in enumerate(F):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            if key in edge:
                j = edge[key]
                adj[i].append(j)
                adj[j].append(i)
            else:
                edge[key] = i
    return adj, edge


# Planar proxies by normal-cone region growing (Cohen-Steiner, Alliez & Desbrun 2004).
def _region_grow(n, adj, theta_deg):
    cmin = float(np.cos(np.radians(theta_deg)))
    N = len(n)
    seen = np.zeros(N, dtype=bool)
    sides = []
    for i in range(N):
        if seen[i] or np.linalg.norm(n[i]) < 0.5:
            continue
        stack = [i]
        seen[i] = True
        group = [i]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if seen[v]:
                    continue
                if float(np.dot(n[u], n[v])) >= cmin:
                    seen[v] = True
                    stack.append(v)
                    group.append(v)
        sides.append(group)
    return sides


# Least-squares plane: normal = smallest singular vector of the centred points.
def _fit_plane(pts, n_hint):
    c = pts.mean(axis=0)
    X = pts - c
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    n = vh[-1]
    if np.dot(n, n_hint) < 0:
        n = -n
    n = n / max(np.linalg.norm(n), 1e-15)
    return n, float(np.dot(n, c))


def _basis(n):
    t = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, t)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


# Ear clipping of a simple polygon (Meisters 1975, Amer. Math. Monthly 82(6)).
def _earclip(poly2):
    m = len(poly2)
    if m < 3:
        return []
    if m == 3:
        return [(0, 1, 2)]
    idx = list(range(m))
    tris = []

    def area2(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def inside(p, a, b, c):
        s = np.sign(area2(a, b, c))
        if s == 0:
            return False
        return (
            s * area2(a, b, p) >= -1e-15
            and s * area2(b, c, p) >= -1e-15
            and s * area2(c, a, p) >= -1e-15
        )

    guard = 0
    while len(idx) > 3 and guard < m * m:
        guard += 1
        clipped = False
        L = len(idx)
        for k in range(L):
            i0, i1, i2 = idx[(k - 1) % L], idx[k], idx[(k + 1) % L]
            a, b, c = poly2[i0], poly2[i1], poly2[i2]
            if area2(a, b, c) <= 1e-18:
                continue
            ear = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                if inside(poly2[j], a, b, c):
                    ear = False
                    break
            if ear:
                tris.append((i0, i1, i2))
                idx.pop(k)
                clipped = True
                break
        if not clipped:
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def _side_boundary(F, faces, V):
    from collections import defaultdict

    count = defaultdict(int)
    for fi in faces:
        a, b, c = F[fi]
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            count[key] += 1
    boundary = [e for e, k in count.items() if k == 1]
    if not boundary:
        return np.zeros((0, 3))
    nxt = {}
    for u, v in boundary:
        nxt.setdefault(u, []).append(v)
        nxt.setdefault(v, []).append(u)
    start = boundary[0][0]
    loop = [start]
    prev = None
    cur = start
    for _ in range(len(boundary) + 2):
        cand = [x for x in nxt.get(cur, []) if x != prev]
        if not cand:
            break
        prev, cur = cur, cand[0]
        if cur == start:
            break
        loop.append(cur)
    return V[np.array(loop, dtype=int)]


def _unmatched_length(V, F, sides):
    from collections import defaultdict

    owner = defaultdict(list)
    for si, faces in enumerate(sides):
        for fi in faces:
            a, b, c = F[fi]
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                if si not in owner[key]:
                    owner[key].append(si)
    length = 0.0
    max_gap = 0.0
    for (u, v), ow in owner.items():
        if len(ow) == 1:
            pass
    glob = defaultdict(int)
    for a, b, c in F:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            glob[key] += 1
    for (u, v), ow in owner.items():
        if len(ow) == 1 and glob[(u, v)] == 2:
            g = float(np.linalg.norm(V[u] - V[v]))
            length += g
            max_gap = max(max_gap, g)
    return length, max_gap


def resolve_quality(quality):
    if quality in QUALITY:
        return quality, QUALITY[quality]
    raise ValueError(f"quality must be one of {list(QUALITY)}, got {quality!r}")


def projected_area(n, A, d, two_sided=False):
    d = np.asarray(d, float)
    dn = np.linalg.norm(d)
    if dn < 1e-15:
        return 0.0
    d = d / dn
    c = np.asarray(n, float) @ d
    A = np.asarray(A, float)
    if two_sided:
        return float(np.sum(A * np.abs(c)))
    return float(np.sum(A * np.maximum(c, 0.0)))


def _emit_original_tris(V, F, faces, nrm, area, n_fit, use_fit_normal):
    ns, As, cs, tris = [], [], [], []
    for fi in faces:
        vs = V[F[fi]]
        a = float(area[fi])
        if a < 1e-15:
            continue
        nn = n_fit if use_fit_normal else nrm[fi]
        ns.append(nn)
        As.append(a)
        cs.append(vs.mean(axis=0))
        tris.append(vs)
    return ns, As, cs, tris


def _emit_outline(V, F, faces, n_fit, A_orig, outline_ok):
    loop = _side_boundary(F, faces, V)
    if len(loop) < 3:
        return None
    u, v = _basis(n_fit)
    poly2 = np.stack([loop @ u, loop @ v], axis=1)
    closed = np.vstack([poly2, poly2[0]])
    a2 = np.sum(closed[:-1, 0] * closed[1:, 1] - closed[1:, 0] * closed[:-1, 1])
    if a2 < 0:
        loop = loop[::-1]
        poly2 = poly2[::-1]
    idx = _earclip(poly2)
    if not idx:
        return None
    ns, As, cs, tris = [], [], [], []
    for i0, i1, i2 in idx:
        vs = np.stack([loop[i0], loop[i1], loop[i2]])
        a = 0.5 * float(np.linalg.norm(np.cross(vs[1] - vs[0], vs[2] - vs[0])))
        if a < 1e-15:
            continue
        ns.append(n_fit)
        As.append(a)
        cs.append(vs.mean(axis=0))
        tris.append(vs)
    if not As:
        return None
    A_sum = float(sum(As))
    if A_orig > 1e-15 and abs(A_sum / A_orig - 1.0) > max(outline_ok, 1e-6):
        return None
    if A_sum > 1e-15:
        scale = A_orig / A_sum
        As = [a * scale for a in As]
    return ns, As, cs, tris


def simplify_sides(V, F, theta_deg=None, return_tris=False, quality="1pct"):
    qname, q = resolve_quality(quality)
    if theta_deg is None:
        theta_deg = q["theta_deg"]
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    nrm, area = _face_normals_areas(V, F)
    keep = area > 1e-15
    F = F[keep]
    nrm = nrm[keep]
    area = area[keep]
    adj, _ = _adjacency(F)
    sides = _region_grow(nrm, adj, float(theta_deg))

    planes = []
    for faces in sides:
        pts = np.vstack([V[F[i]] for i in faces])
        hint = nrm[faces].mean(axis=0)
        planes.append(_fit_plane(pts, hint))

    V2 = V.copy()
    from collections import defaultdict

    side_of = {}
    for si, faces in enumerate(sides):
        for fi in faces:
            side_of[fi] = si
    edge_sides = defaultdict(set)
    for fi, (a, b, c) in enumerate(F):
        si = side_of[fi]
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_sides[key].add(si)
    for (u, v), ss in edge_sides.items():
        if len(ss) != 2:
            continue
        s0, s1 = tuple(ss)
        n0, d0 = planes[s0]
        n1, d1 = planes[s1]
        dirn = np.cross(n0, n1)
        dn = np.linalg.norm(dirn)
        if dn < 1e-12:
            continue
        dirn = dirn / dn
        A = np.stack([n0, n1, dirn])
        try:
            p0 = np.linalg.solve(A, np.array([d0, d1, np.dot(dirn, V[u])]))
        except np.linalg.LinAlgError:
            continue
        for idx in (u, v):
            t = np.dot(V2[idx] - p0, dirn)
            V2[idx] = p0 + t * dirn

    ns, As, cs = [], [], []
    out_tris = []
    for si, faces in enumerate(sides):
        n_fit, _d = planes[si]
        A_orig = float(sum(area[fi] for fi in faces))
        outlined = _emit_outline(V2, F, faces, n_fit, A_orig, q["outline_ok"])
        planar = True
        if len(faces) > 1:
            dots = [abs(float(np.dot(nrm[fi], n_fit))) for fi in faces]
            planar = min(dots) > 0.995
        if outlined is not None:
            e_ns, e_As, e_cs, e_tr = outlined
        else:
            e_ns, e_As, e_cs, e_tr = _emit_original_tris(
                V2, F, faces, nrm, area, n_fit, use_fit_normal=planar
            )
        ns.extend(e_ns)
        As.extend(e_As)
        cs.extend(e_cs)
        out_tris.extend(e_tr)

    def _edge_key(a, b):
        a = np.round(a, 9)
        b = np.round(b, 9)
        pair = (tuple(a), tuple(b))
        return pair if pair[0] <= pair[1] else (pair[1], pair[0])

    from collections import Counter

    ec = Counter()
    for vs in out_tris:
        for i in range(3):
            ec[_edge_key(vs[i], vs[(i + 1) % 3])] += 1
    glob = Counter()
    for a, b, c in F:
        for u, v in ((a, b), (b, c), (c, a)):
            glob[_edge_key(V2[u], V2[v])] += 1
    unmatched = 0.0
    max_gap = 0.0
    for e, k in ec.items():
        if k == 1 and glob.get(e, 0) != 1:
            g = float(np.linalg.norm(np.array(e[0]) - np.array(e[1])))
            unmatched += g
            max_gap = max(max_gap, g)

    n = np.asarray(ns, float)
    A = np.asarray(As, float)
    c = np.asarray(cs, float)
    area_in = float(area.sum())
    area_out = float(A.sum()) if len(A) else 0.0
    report = SimplifyReport(
        n_sides=len(sides),
        n_panels=len(A),
        area_in=area_in,
        area_out=area_out,
        unmatched_m=float(unmatched),
        max_gap_m=float(max_gap),
        quality=qname,
        area_rel_err=abs(area_out / area_in - 1.0) if area_in > 0 else 0.0,
        theta_deg=float(theta_deg),
    )
    if return_tris:
        return n, A, c, report, out_tris
    return n, A, c, report


def cube_mesh(s=1.0):
    h = s * 0.5
    V = np.array(
        [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ],
        float,
    )
    F = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [2, 6, 7],
            [2, 7, 3],
            [0, 3, 7],
            [0, 7, 4],
            [1, 5, 6],
            [1, 6, 2],
        ],
        int,
    )
    return V, F


def write_geom(path, n, A, c, header=""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [f"# {h}" for h in header.splitlines()] if header else []
    rows += [" ".join(f"{x:.9g}" for x in (*ni, ai, *ci)) for ni, ai, ci in zip(n, A, c)]
    path.write_text("\n".join(rows) + "\n")
    return path


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="mesh (.stl/.obj/.step) -> sealed plates .geom")
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--units", default=None, help="STL/OBJ length unit (default mm); STEP uses its own")
    ap.add_argument("--quality", choices=list(QUALITY), default="1pct")
    a = ap.parse_args(argv)
    V, F = load_mesh(a.inp, units=a.units)
    n, A, c, rep = simplify_sides(V, F, quality=a.quality)
    out = a.out or Path(__file__).resolve().parents[2] / "outputs" / "results" / "geometry" / (
        a.inp.stem + f"_{rep.quality}.geom")
    write_geom(out, n, A, c, header=(f"{a.inp.name} simplified ({rep.quality})\n"
                                     f"sides {rep.n_sides} panels {rep.n_panels} "
                                     f"area_in {rep.area_in:.6g} area_out {rep.area_out:.6g} m2"))
    print(f"{a.inp.name}: {rep.n_sides} sides, {rep.n_panels} panels, "
          f"area {rep.area_in:.6g} -> {rep.area_out:.6g} m2 ({rep.area_rel_err:.3%}); wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
