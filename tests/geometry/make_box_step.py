"""Write a minimal AP214 STEP solid box (mm) for the STEP loader test."""
import sys
from pathlib import Path


def box_step(lx, ly, lz):
    ents = []

    def add(s):
        ents.append(s)
        return f"#{len(ents)}"

    P = [(x, y, z) for z in (0, lz) for y in (0, ly) for x in (0, lx)]
    cp = [add(f"CARTESIAN_POINT('',({x:.6f},{y:.6f},{z:.6f}))") for x, y, z in P]
    vp = [add(f"VERTEX_POINT('',{c})") for c in cp]
    edges = {}

    def edge(a, b):
        if (a, b) in edges:
            return edges[(a, b)], True
        if (b, a) in edges:
            return edges[(b, a)], False
        d = [P[b][k] - P[a][k] for k in range(3)]
        n = sum(x * x for x in d) ** 0.5
        dr = add(f"DIRECTION('',({d[0]/n:.6f},{d[1]/n:.6f},{d[2]/n:.6f}))")
        vec = add(f"VECTOR('',{dr},{n:.6f})")
        ln = add(f"LINE('',{cp[a]},{vec})")
        e = add(f"EDGE_CURVE('',{vp[a]},{vp[b]},{ln},.T.)")
        edges[(a, b)] = e
        return e, True

    # faces as outward CCW vertex loops, with outward normal
    F = [((0, 2, 3, 1), (0, 0, -1)), ((4, 5, 7, 6), (0, 0, 1)), ((0, 1, 5, 4), (0, -1, 0)),
         ((2, 6, 7, 3), (0, 1, 0)), ((0, 4, 6, 2), (-1, 0, 0)), ((1, 3, 7, 5), (1, 0, 0))]
    faces = []
    for loop, nrm in F:
        oes = []
        for i in range(4):
            e, same = edge(loop[i], loop[(i + 1) % 4])
            oes.append(add(f"ORIENTED_EDGE('',*,*,{e},{'.T.' if same else '.F.'})"))
        el = add(f"EDGE_LOOP('',({','.join(oes)}))")
        fb = add(f"FACE_OUTER_BOUND('',{el},.T.)")
        nd = add(f"DIRECTION('',({nrm[0]:.1f},{nrm[1]:.1f},{nrm[2]:.1f}))")
        rd = add("DIRECTION('',(1.0,0.0,0.0))" if nrm[0] == 0 else "DIRECTION('',(0.0,1.0,0.0))")
        ax = add(f"AXIS2_PLACEMENT_3D('',{cp[loop[0]]},{nd},{rd})")
        pl = add(f"PLANE('',{ax})")
        faces.append(add(f"ADVANCED_FACE('',({fb}),{pl},.T.)"))
    sh = add(f"CLOSED_SHELL('',({','.join(faces)}))")
    br = add(f"MANIFOLD_SOLID_BREP('box',{sh})")
    o = add("CARTESIAN_POINT('',(0.0,0.0,0.0))")
    z = add("DIRECTION('',(0.0,0.0,1.0))")
    x = add("DIRECTION('',(1.0,0.0,0.0))")
    ax0 = add(f"AXIS2_PLACEMENT_3D('',{o},{z},{x})")
    lu = add("( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )")
    au = add("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )")
    su = add("( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )")
    unc = add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),{lu},'distance_accuracy_value','')")
    ctx = add(f"( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT(({unc})) "
              f"GLOBAL_UNIT_ASSIGNED_CONTEXT(({lu},{au},{su})) REPRESENTATION_CONTEXT('',''))")
    rep = add(f"ADVANCED_BREP_SHAPE_REPRESENTATION('',({ax0},{br}),{ctx})")
    app = add("APPLICATION_CONTEXT('automotive design')")
    add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,{app})")
    pc = add(f"PRODUCT_CONTEXT('',{app},'mechanical')")
    prod = add(f"PRODUCT('box','box','',({pc}))")
    pdf = add(f"PRODUCT_DEFINITION_FORMATION('','',{prod})")
    pdc = add(f"PRODUCT_DEFINITION_CONTEXT('part definition',{app},'design')")
    pd = add(f"PRODUCT_DEFINITION('design','',{pdf},{pdc})")
    pds = add(f"PRODUCT_DEFINITION_SHAPE('','',{pd})")
    add(f"SHAPE_DEFINITION_REPRESENTATION({pds},{rep})")
    body = "\n".join(f"#{i + 1}={e};" for i, e in enumerate(ents))
    return ("ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('box'),'2;1');\n"
            "FILE_NAME('box.step','2026-09-25T00:00:00',(''),(''),'','','');\n"
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\nENDSEC;\nDATA;\n"
            + body + "\nENDSEC;\nEND-ISO-10303-21;\n")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data" / "box_100x200x20mm.step"
    out.write_text(box_step(100.0, 200.0, 20.0))
    print(out)
