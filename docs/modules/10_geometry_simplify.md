# 10 — STL / OBJ side simplification (Python)

**Folder:** `python/geometry/`  
**Status:** implemented (Python; C++ plant consumes the plates)  
**Tests:** `tests/geometry/`

## What it does

Turns a watertight (or nearly watertight) triangle mesh into a **small sealed plate set** for Sentman. This replaces V1.7’s one-panel-per-CAD-facet path. It is **not** generic decimation: it finds spacecraft **sides**, fits planes, and triangulates each side so neighboring plates share edges (no holes).

The C++ plant never sees the raw CAD.

## Algorithm

1. Load STL or OBJ. Scale units to metres. Weld vertices. Drop area \(<10^{-15}\,\mathrm{m}^2\).
2. Build the face-adjacency graph.
3. Region-grow: two adjacent faces join a side if \(\hat{n}_i\cdot\hat{n}_j \ge \cos\theta_{\mathrm{merge}}\). Quality `1pct` uses \(\theta_{\mathrm{merge}}=8^\circ\) (`geometry.py` `QUALITY`).
4. Least-squares plane per side (area-weighted). Flip so the normal points outward (mean of member normals).
5. Boundary loops of the side, projected to that plane.
6. **Seal:** every mesh edge used by two sides is mapped to the **intersection line of the two planes**. Vertices on that edge are replaced by their projection onto that line. This closes gaps that plane-fitting would otherwise open.
7. **Ear-clip** each sealed planar polygon (`_earclip` in `geometry.py`). Not a constrained Delaunay triangulation.
8. Each output triangle is a Sentman panel: \(\hat{n}\), \(A\), \(\mathbf{r}_c\).
9. Reports: \(N_\mathrm{sides}\), \(N_\mathrm{panels}\), \(\Delta A / A_\mathrm{raw}\), max unmatched-boundary length (must be 0).

**Quality toggle** (`quality=` on `simplify_sides`):

| Key | Budget | Use |
|---|---|---|
| `1pct` | 1 % wetted **and** ram area at the six attitudes | preferred, data generation |
| `5pct` | 5 % | intermediate |
| `10pct` | 10 % | crude smoke only |

Each side keeps the **original triangle area**. An outline is used only when its raw area already sits inside that budget; otherwise the source triangles stay (continuity, no silent membrane loss).

Six SolarCat ram-area checks (flow direction in the body frame): side-flat, side-vertex, ±Z max-drag (one check), 45° about X, 45° about Y, and 0→1° AoA growth from min-drag. Runner: `python -m arlamx_v2.validate_attitudes --quality 1pct`.

Not used on the SC_v3 hex `.geom` bake-off (that file is already a plate model).

## Citations

1. Cohen-Steiner, D., Alliez, P. & Desbrun, M. (2004). Variational shape approximation. *ACM SIGGRAPH / ACM TOG* 23(3), 905–914. (planar proxies for “sides”)
2. O’Rourke, J. (1998). *Computational Geometry in C,* 2nd ed. — ear clipping of a simple polygon (what `geometry._earclip` implements).
3. (supporting) Attene, M., Falcidieno, B. & Spagnuolo, M. (2006). Hierarchical mesh segmentation based on fitting primitives. *Visual Computer* 22, 181–193.

Do **not** use Garland–Heckbert QEM as the primary method: it does not know what a spacecraft side is and readily opens cracks.

## Classroom test

**A. Unit cube** (12 CAD triangles). After simplify: **6 sides**, 2 triangles per side, total area 6, no unmatched edges, each \(|\hat{n}|\) axis-aligned.  
**B. Regular hex prism** (sail-like): 8 sides (6 walls + 2 faces), area within 0.1 % of analytic, sealed.  
**C. Two-box “L”:** must produce more than a convex hull (the inner corner stays). Unmatched boundary length = 0.  
**D.** A mesh with a deliberate 1 mm crack must **fail** the seal check (the test proves the checker works).

## Tests run

2026-08-18 — `tests/geometry/test_simplify.py` **PASS** (cube → 6 sides, area 6, sealed; hex prism sealed; open mesh flagged).
2026-09-25 — STEP input: `geometry.load_mesh` reads `.step/.stp` through trimesh + `cascadio` (OpenCASCADE tessellation). The file's own length unit is used and returned in metres, and vertices are float32 via GLB. `python main.py sim geometry --in part.step` writes the simplified `.geom`. `tests/geometry/test_simplify.py::test_step_box_matches_stl_box`: the STEP and STL of the same 100×200×20 mm box give 6 sides and identical per-normal areas (1e-6), and the `.geom` round-trips. **PASS**.

