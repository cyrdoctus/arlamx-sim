"""Independent numerical checks of the ARLAMX v2.1 C++ kernels (run after build).

    PYTHONPATH=python python tests/validation/verify_physics.py


Each check re-derives the quantity from a DIFFERENT formulation than the C++
code uses, so a shared algebra slip cannot pass:

  1. Spherical-harmonic gravity vs a finite-difference gradient of the
     geopotential built from scipy's associated Legendre functions
     (fully-normalised, Condon-Shortley stripped).
  2. Sentman Cp/Ctau vs the closed form written directly in terms of
     cos(theta) (the textbook form), and vs numerical integration of the
     Maxwellian flux for the incident pressure.
  3. Panel SRP optical law vs an explicit vector expression.
  4. Tilted dipole vs -grad of the scalar potential (finite difference).
  5. WMM vs finite-difference -grad V.
  6. Torque-free rigid body: |H| and kinetic energy conserved by the plant RK4.
  7. accel_j3 vs finite-difference gradient of the J3 potential.
"""
import math
import sys

import numpy as np
from scipy import special as sp

from arlamx_v2 import cpp

ROOT = sys.argv[1] if len(sys.argv) > 1 else str(__import__("pathlib").Path(__file__).resolve().parents[2])
GGM = f"{ROOT}/data/GGM03S.txt"
WMM = f"{ROOT}/data/WMM.COF"

ok_all = True


def report(name, err, tol):
    global ok_all
    ok = err < tol
    ok_all &= ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:55s} err={err:.3e}  tol={tol:.0e}")


# --------------------------------------------------------------------- 1. SH
def load_ggm(path, N):
    C = np.zeros((N + 1, N + 1)); S = np.zeros((N + 1, N + 1))
    with open(path) as f:
        hdr = f.readline().replace(",", " ").split()
        Re, mu = float(hdr[0]), float(hdr[1])
        for line in f:
            p = line.replace(",", " ").split()
            if len(p) < 4:
                continue
            n, m = int(p[0]), int(p[1])
            if n <= N and m <= n:
                C[n, m] = float(p[2]); S[n, m] = float(p[3])
    return Re, mu, C, S


def potential_sh(r_vec, Re, mu, C, S, N):
    """U = mu/r [1 + sum (Re/r)^n Pbar_nm(sin phi)(Cbar cos m lam + Sbar sin m lam)]"""
    x, y, z = r_vec
    r = math.sqrt(x * x + y * y + z * z)
    phi = math.asin(z / r)
    lam = math.atan2(y, x)
    U = 1.0
    for n in range(2, N + 1):
        for m in range(0, n + 1):
            # fully normalised: sqrt((2-d)(2n+1)(n-m)!/(n+m)!) * P_nm (no CS phase)
            p = sp.lpmv(m, n, math.sin(phi)) * (-1.0) ** m
            norm = math.sqrt((1 if m == 0 else 2) * (2 * n + 1) * math.factorial(n - m) / math.factorial(n + m))
            U += (Re / r) ** n * norm * p * (C[n, m] * math.cos(m * lam) + S[n, m] * math.sin(m * lam))
    return mu / r * U


def grad_fd(f, r, h=0.05):
    g = np.zeros(3)
    for i in range(3):
        e = np.zeros(3); e[i] = h
        g[i] = (f(r + e) - f(r - e)) / (2 * h)
    return g


import os
if os.path.exists(GGM):
    for N in (2, 4, 8, 12):
        Re, mu, C, S = load_ggm(GGM, N)
        g = cpp.GravityHarmonics(); assert g.load_ggm(GGM, N)
        worst = 0.0
        rng = np.random.default_rng(1)
        pts = [np.array([Re + 400e3, 0, 0]), np.array([5.0e6, 1.2e6, 4.0e6]),
               np.array([-3.0e6, -4.5e6, 3.1e6]), np.array([1e5, 2e5, Re + 500e3])]
        for _ in range(6):
            u = rng.normal(size=3); u /= np.linalg.norm(u); pts.append((Re + 350e3) * u)
        for r in pts:
            a_fd = grad_fd(lambda q: potential_sh(q, Re, mu, C, S, N), r, h=20.0)
            a_cpp = np.asarray(g.accel_ecef(r, N))
            # FD truncation error ~ h^2 * U''' ~ 1e-13 relative; ask for 1e-9
            worst = max(worst, np.linalg.norm(a_cpp - a_fd) / np.linalg.norm(a_fd))
        # central differences on a 6e7 J/kg potential bottom out near 1e-10 relative
        report(f"SH gravity degree {N:2d} vs FD grad of scipy potential", worst, 1e-9)
else:
    print("SKIP  GGM03S.txt not found")

# --------------------------------------------------------------------- 7. J3
def U_j3(rv, mu, Re, J3):
    r = np.linalg.norm(rv); s = rv[2] / r
    P3 = 0.5 * (5 * s ** 3 - 3 * s)
    return -mu / r * (Re / r) ** 3 * J3 * P3  # U = -mu/r * J3 (Re/r)^3 P3 (attractive-potential sign convention -> a = grad(-U)?)

# Use the geopotential convention U = mu/r [1 - sum J_n (Re/r)^n P_n(sin phi)], a = grad U
def U_zonal(rv, mu, Re, J, n):
    r = np.linalg.norm(rv); s = rv[2] / r
    Pn = sp.eval_legendre(n, s)
    return mu / r * (-J * (Re / r) ** n * Pn)

r = np.array([4.5e6, 1.0e6, 5.0e6])
a_j3 = np.asarray(cpp.accel_j3(r, cpp.MU_WGS, cpp.RE_WGS, -2.53265649e-6))
a_fd = grad_fd(lambda q: U_zonal(q, cpp.MU_WGS, cpp.RE_WGS, -2.53265649e-6, 3), r, h=0.5)
report("accel_j3 vs FD grad of J3 potential", np.linalg.norm(a_j3 - a_fd) / np.linalg.norm(a_fd), 1e-7)
a_j2 = np.asarray(cpp.accel_j2(r, cpp.MU_WGS, cpp.RE_WGS, cpp.J2_GGM))
a_fd = grad_fd(lambda q: U_zonal(q, cpp.MU_WGS, cpp.RE_WGS, cpp.J2_GGM, 2), r, h=0.5)
report("accel_j2 vs FD grad of J2 potential", np.linalg.norm(a_j2 - a_fd) / np.linalg.norm(a_fd), 1e-7)

# ---------------------------------------------------------------- 2. Sentman
def sentman_ref(theta, s, Tw_Ti, alpha):
    """Textbook form (Sentman 1961 as in Doornbos 2012 eq. 3.53-3.55)."""
    c, sn = math.cos(theta), math.sin(theta)
    g = s * c
    Z = 1 + math.erf(g); E = math.exp(-g * g)
    Cp_i = (c * c + 1 / (2 * s * s)) * Z + c / (s * math.sqrt(math.pi)) * E
    Ct = sn * (c * Z + E / (s * math.sqrt(math.pi)))
    # re-emission: E_w = 2kT flux-weighted; Tr = (1-alpha) m V^2/(4k) + alpha Tw
    Tr_Ti = alpha * Tw_Ti + (1 - alpha) * s * s / 2
    Cp_r = 0.5 * math.sqrt(Tr_Ti) * (math.sqrt(math.pi) / s * c * Z + E / (s * s))
    return Cp_i + Cp_r, Ct


def cp_incident_quadrature(theta, s):
    """Incident normal momentum flux by direct integration of the drifting
    Maxwellian: independent of the erf closed form."""
    # Molecular velocity in units of the most probable speed; drift = s along -n
    # (n outward, flow hits the plate when v_n < 0). Use plate-normal coordinate.
    # normalised by (1/2) rho V^2 -> Cp = (2/(s^2)) * <v_n^2 weighted flux> with
    # v_n in units of v_mp: Cp = (2/s^2) * (1/sqrt(pi)) * int_{u<0} u^2 exp(-(u+s cos)^2) du
    c = math.cos(theta)
    from scipy import integrate
    f = lambda u: u * u * math.exp(-(u + s * c) ** 2)
    val, _ = integrate.quad(f, -np.inf, 0.0, limit=200)
    return 2.0 / (s * s) * val / math.sqrt(math.pi)


worst = 0.0
for theta in (0.0, 0.3, 0.7, 1.2, 1.5):
    for s in (3.0, 8.0, 15.0):
        for Tw in (0.3, 1.0):
            for al in (0.0, 0.93, 1.0):
                Cp, Ct = cpp.sentman(theta, s, Tw, al)
                Cp_r, Ct_r = sentman_ref(theta, s, Tw, al)
                worst = max(worst, abs(Cp - Cp_r) / max(abs(Cp_r), 1e-12), abs(Ct - Ct_r) / max(abs(Ct_r), 1e-12))
report("Sentman Cp/Ctau vs textbook closed form", worst, 1e-12)
worst = 0.0
for theta in (0.0, 0.5, 1.2):
    for s in (4.0, 8.0):
        Cp_i_cpp = cpp.sentman(theta, s, 1.0, 1.0)[0] - (sentman_ref(theta, s, 1.0, 1.0)[0] - (
            (math.cos(theta) ** 2 + 1 / (2 * s * s)) * (1 + math.erf(s * math.cos(theta)))
            + math.cos(theta) / (s * math.sqrt(math.pi)) * math.exp(-(s * math.cos(theta)) ** 2)))
        worst = max(worst, abs(Cp_i_cpp - cp_incident_quadrature(theta, s)) / cp_incident_quadrature(theta, s))
report("Sentman incident Cp vs Maxwellian quadrature", worst, 1e-9)
# hyperthermal limit: Cp cos + Ct sin -> 2 cos as s -> inf (incident only, alpha=1, Tw->0)
theta = 0.4
Cp, Ct = cpp.sentman(theta, 400.0, 1e-6, 1.0)
report("Sentman hyperthermal limit (2 cos theta)", abs(Cp * math.cos(theta) + Ct * math.sin(theta) - 2 * math.cos(theta)), 5e-3)

# panel-level: two-sided plate at angle, check Cd/Cl vs manual sum of sentman()
rng = np.random.default_rng(0)
n = np.array([[0, 0, 1.0], [0, 0, -1.0], [1.0, 0, 0], [-1.0, 0, 0]])
A = np.array([1.0, 1.0, 0.3, 0.3]); c = np.zeros((4, 3)); c[:, 0] = [0.1, -0.1, 0.05, -0.05]
v = np.array([-5000.0, 1000.0, -5500.0])
rho, T, mb, Tw, aE = 3e-12, 900.0, 2.656e-26, 300.0, 0.93
out = cpp.spacecraft_aero(n, A, c, v, rho, T, mb, Tw, aE, True)
s = np.linalg.norm(v) / math.sqrt(2 * 1.380649e-23 * T / mb)
vh = v / np.linalg.norm(v); q = 0.5 * rho * np.dot(v, v)
F = np.zeros(3); tau = np.zeros(3)
for i in range(4):
    ct = -np.dot(vh, n[i])
    if ct <= 0:
        continue
    th = math.acos(ct)
    Cp, Ct = cpp.sentman(th, s, Tw / T, aE)
    t = vh + ct * n[i]; t /= np.linalg.norm(t)
    Fi = A[i] * q * (-Cp * n[i] + Ct * t)
    F += Fi; tau += np.cross(c[i], Fi)
report("spacecraft_aero force vs manual panel sum", np.linalg.norm(F - out["force"]) / np.linalg.norm(F), 1e-12)
report("spacecraft_aero torque vs manual panel sum", np.linalg.norm(tau - out["torque"]) / max(np.linalg.norm(tau), 1e-30), 1e-10)
report("spacecraft_aero Cd vs manual", abs(out["Cd"] - np.dot(F, vh) / (q * 0.5 * A.sum())), 1e-12)

# --------------------------------------------------------------------- 3. SRP
P = cpp.P_SRP_1AU
n = np.array([[0.6, 0.0, 0.8]]); A = np.array([2.0]); c = np.array([[0.0, 0.1, 0.0]])
sun = np.array([0.0, 0.0, 1.0])
for (ca, cs, cd) in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.2, 0.7, 0.1)):
    F = np.asarray(cpp.panel_srp_optical(n, A, c, sun, ca, cs, cd, 1.0, P)[0])
    cth = np.dot(n[0], sun)
    F_ref = -P * A[0] * cth * ((ca + cd) * sun + (2 * cs * cth + 2.0 / 3.0 * cd) * n[0])
    report(f"SRP optical law ca={ca} cs={cs} cd={cd}", np.linalg.norm(F - F_ref) / np.linalg.norm(F_ref), 1e-12)
# perfect specular mirror: force must be along -n with magnitude 2 P A cos^2
F = np.asarray(cpp.panel_srp_optical(n, A, c, sun, 0.0, 1.0, 0.0, 1.0, P)[0])
cth = np.dot(n[0], sun)
report("SRP specular mirror force along -n, 2PAcos^2", np.linalg.norm(F + 2 * P * A[0] * cth * cth * n[0]) / np.linalg.norm(F), 1e-12)

# ------------------------------------------------------------------ 4. dipole
a_ref = 6371200.0
g10, g11, h11 = -29404.8e-9, -1450.9e-9, 4652.5e-9
def V_dip(rv):
    r = np.linalg.norm(rv); th = math.acos(rv[2] / r); lam = math.atan2(rv[1], rv[0])
    return a_ref * (a_ref / r) ** 2 * (g10 * math.cos(th) + (g11 * math.cos(lam) + h11 * math.sin(lam)) * math.sin(th))
worst = 0.0
for _ in range(5):
    u = rng.normal(size=3); u /= np.linalg.norm(u); r = (a_ref + 450e3) * u
    B_fd = -grad_fd(V_dip, r, h=1.0)
    B = np.asarray(cpp.dipole_field_ecef(r))
    worst = max(worst, np.linalg.norm(B - B_fd) / np.linalg.norm(B_fd))
report("tilted dipole vs -grad V (FD)", worst, 1e-6)

# --------------------------------------------------------------------- 5. WMM
if os.path.exists(WMM):
    w = cpp.WMM(); assert w.load(WMM)
    g = np.zeros((13, 13)); h = np.zeros((13, 13)); gt = np.zeros((13, 13)); ht = np.zeros((13, 13))
    with open(WMM) as f:
        epoch = float(f.readline().split()[0])
        for line in f:
            if "9999" in line:
                break
            p = line.split()
            if len(p) < 6:
                continue
            nn, mm = int(p[0]), int(p[1])
            if 1 <= nn <= 12 and 0 <= mm <= nn:
                g[nn][mm], h[nn][mm], gt[nn][mm], ht[nn][mm] = map(float, p[2:6])
    year = 2025.0
    def V_wmm(rv):
        r = np.linalg.norm(rv); th = math.acos(rv[2] / r); lam = math.atan2(rv[1], rv[0])
        V = 0.0
        for nn in range(1, 13):
            for mm in range(0, nn + 1):
                p = sp.lpmv(mm, nn, math.cos(th)) * (-1.0) ** mm
                if mm > 0:
                    p *= math.sqrt(2.0 * math.factorial(nn - mm) / math.factorial(nn + mm))
                gg = g[nn][mm] + gt[nn][mm] * (year - epoch); hh = h[nn][mm] + ht[nn][mm] * (year - epoch)
                V += (a_ref / r) ** (nn + 1) * (gg * math.cos(mm * lam) + hh * math.sin(mm * lam)) * p
        return a_ref * V * 1e-9
    worst = 0.0
    for _ in range(5):
        u = rng.normal(size=3); u /= np.linalg.norm(u); r = (a_ref + 450e3) * u
        B_fd = -grad_fd(V_wmm, r, h=2.0)
        B = np.asarray(w.field_ecef(r, year))
        worst = max(worst, np.linalg.norm(B - B_fd) / np.linalg.norm(B_fd))
    report("WMM vs -grad V (FD, scipy Legendre)", worst, 1e-6)
else:
    print("SKIP  WMM.COF not found")

# ---------------------------------------------------------------- 6. RK4 body
p = cpp.SimParams(); p.dt_s = 2.0; p.advisor_step_s = 300.0; p.rk4_step_s = 2.0
p.use_panel_srp = False; p.sh_degree = 0; p.ggm_path = GGM
p.gravity_gradient = False   # torque-free: no gravity-gradient moment
sim = cpp.Simulator(p)
assert sim.gravity_loaded()
sim.set_mode("point")
I = np.array([0.0125, 0.0125, 0.025]); sim.set_inertia_diag(I)
ctrl = cpp.MRPFeedback(); ctrl.kp = 0.0; ctrl.kd = 0.0; ctrl.set_inertia_diag(I)
ctrl.set_max_torque(np.array([1e-300, 1e-300, 1e-300]))  # torque-free: clip the gyroscopic feedforward away
sim.set_controller(ctrl)
sim.set_atmosphere(0.0, 1000.0, 2.6e-26)
mu = cpp.MU_WGS
r0 = np.array([cpp.RE_WGS + 400e3, 0, 0]); v0 = np.array([0, math.sqrt(mu / (cpp.RE_WGS + 400e3)), 0])
w0 = np.radians([3.0, -2.0, 4.0])
sim.reset(r0, v0, np.array([0.1, 0.2, -0.1]), w0)
H0 = np.linalg.norm(I * w0); T0 = 0.5 * np.dot(w0, I * w0)
for _ in range(10):
    o = sim.step(np.array([1.0, 0, 0, 0]))
w = np.asarray(o["omega"])
# 5 deg/s tumble at dt = 2 s (omega*h = 0.17 rad): classical RK4 truncation, not symplectic
report("torque-free |H| conservation, 3000 s, 5 deg/s tumble", abs(np.linalg.norm(I * w) - H0) / H0, 1e-4)
report("torque-free T conservation, 3000 s, 5 deg/s tumble", abs(0.5 * np.dot(w, I * w) - T0) / T0, 1e-4)
# flight regime: rate cap is 0.125 deg/s
w0 = np.radians([0.12, -0.08, 0.1])
sim.reset(r0, v0, np.array([0.1, 0.2, -0.1]), w0)
H0 = np.linalg.norm(I * w0); T0 = 0.5 * np.dot(w0, I * w0)
for _ in range(10):
    o = sim.step(np.array([1.0, 0, 0, 0]))
w = np.asarray(o["omega"])
report("torque-free |H| conservation, 3000 s, 0.12 deg/s", abs(np.linalg.norm(I * w) - H0) / H0, 1e-9)
report("torque-free T conservation, 3000 s, 0.12 deg/s", abs(0.5 * np.dot(w, I * w) - T0) / T0, 1e-9)
g0 = cpp.GravityHarmonics(); g0.load_ggm(GGM, 2)
report("vacuum two-body SMA drift over 3000 s (m)", abs(o["sma_m"] - cpp.sma_from_rv(r0, v0, g0.mu())), 1e-2)

print("\nALL PASS" if ok_all else "\nSOME CHECKS FAILED")
