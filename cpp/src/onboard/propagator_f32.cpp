// Onboard FP32 propagator (two-body + J2 + exponential drag + held SRP), mirror of propagator.py.
#include "arlamx/onboard/propagator_f32.hpp"

#include <algorithm>
#include <cmath>

namespace arlamx {

namespace {

constexpr float MU = 3.986004418e14f;
constexpr float RE = 6378137.0f;
constexpr float J2 = 1.0826353865e-3f;
constexpr float OMEGA_E = 7.2921150e-5f;
constexpr float FLAT = 1.0f / 298.257223563f;

struct V3f {
    float x, y, z;
};
inline V3f operator+(V3f a, V3f b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3f operator-(V3f a, V3f b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3f operator*(float s, V3f a) { return {s * a.x, s * a.y, s * a.z}; }
inline bool finite(V3f a) { return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z); }

struct Env {
    float bc_inv, rho0, h0, h_scale, a_srp;
    V3f sun;
};

// Two-body + J2 + 1/2 rho(h) Cd A/m |v_rel| v_rel drag, rho = rho0 exp(-(h - h0)/H)
// (Montenbruck & Gill 2000, Sec. 3.5); held along-track SRP.
inline V3f accel(V3f r, V3f v, const Env& e) {
    const float r2 = r.x * r.x + r.y * r.y + r.z * r.z;
    const float rn = std::sqrt(r2);
    if (rn < 1.0f) return {0.0f, 0.0f, 0.0f};
    const float inv_r3 = 1.0f / (r2 * rn);
    V3f a = (-MU * inv_r3) * r;

    // J2 (Vallado eq. 8-30), z-axis aligned with the pole.
    const float zr = r.z / rn;
    const float f = 1.5f * J2 * MU * RE * RE / (r2 * r2);
    const float five_z2 = 5.0f * zr * zr;
    a = a + V3f{f * (five_z2 - 1.0f) * (r.x / rn), f * (five_z2 - 1.0f) * (r.y / rn),
                f * (five_z2 - 3.0f) * (r.z / rn)};

    // Height above the ellipsoid, first order: r_E = a (1 - f sin^2 phi) (Vallado 2013, ch. 3).
    const float alt = rn - RE * (1.0f - FLAT * zr * zr);
    const float rho = e.rho0 * std::exp(-(alt - e.h0) / e.h_scale);
    const V3f v_rel{v.x + OMEGA_E * r.y, v.y - OMEGA_E * r.x, v.z};
    const float vr = std::sqrt(v_rel.x * v_rel.x + v_rel.y * v_rel.y + v_rel.z * v_rel.z);
    a = a - (0.5f * rho * e.bc_inv * vr) * v_rel;

    if (e.a_srp != 0.0f) {
        const float rs = r.x * e.sun.x + r.y * e.sun.y + r.z * e.sun.z;
        bool lit = rs >= 0.0f;
        if (!lit) {
            const V3f p = r - rs * e.sun;
            lit = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z) >= RE;
        }
        const float vn = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        if (lit && vn > 0.0f) a = a + (e.a_srp / vn) * v;
    }
    return a;
}

// Classical RK4 (Montenbruck & Gill 2000, Sec. 4.1).
inline void rk4(V3f& r, V3f& v, float dt, const Env& e) {
    const V3f k1v = accel(r, v, e);
    const V3f k1r = v;
    const V3f k2v = accel(r + (0.5f * dt) * k1r, v + (0.5f * dt) * k1v, e);
    const V3f k2r = v + (0.5f * dt) * k1v;
    const V3f k3v = accel(r + (0.5f * dt) * k2r, v + (0.5f * dt) * k2v, e);
    const V3f k3r = v + (0.5f * dt) * k2v;
    const V3f k4v = accel(r + dt * k3r, v + dt * k3v, e);
    const V3f k4r = v + dt * k3v;
    const float w = dt / 6.0f;
    r = r + w * (k1r + 2.0f * k2r + 2.0f * k3r + k4r);
    v = v + w * (k1v + 2.0f * k2v + 2.0f * k3v + k4v);
}

inline void propagate(V3f& r, V3f& v, double horizon_s, double dt_s, const Env& e) {
    const int n = std::max(1, static_cast<int>(std::lround(horizon_s / std::max(dt_s, 1e-3))));
    const float dt = static_cast<float>(horizon_s / n);
    for (int i = 0; i < n; ++i) {
        rk4(r, v, dt, e);
        if (!finite(r) || !finite(v)) break;
        if (std::sqrt(r.x * r.x + r.y * r.y + r.z * r.z) < RE * 0.9f) break;
    }
}

}

std::vector<StateF32> propagate_f32_samples(const float r0[3], const float v0[3],
                                            const std::vector<double>& offsets_s, double dt_s,
                                            double bc_inv, double rho0, double alt0_m,
                                            double h_scale, double a_srp, const float sun_hat[3]) {
    Env e{static_cast<float>(bc_inv), static_cast<float>(rho0), static_cast<float>(alt0_m),
          static_cast<float>(h_scale), static_cast<float>(a_srp), {0.0f, 0.0f, 0.0f}};
    if (sun_hat) e.sun = {sun_hat[0], sun_hat[1], sun_hat[2]};
    else e.a_srp = 0.0f;
    V3f r{r0[0], r0[1], r0[2]};
    V3f v{v0[0], v0[1], v0[2]};
    std::vector<StateF32> out;
    out.reserve(offsets_s.size());
    double t = 0.0;
    for (double target : offsets_s) {
        const double seg = std::max(0.0, target - t);
        if (seg > 0.0) {
            propagate(r, v, seg, dt_s, e);
            t = target;
        }
        out.push_back({{r.x, r.y, r.z}, {v.x, v.y, v.z}});
    }
    return out;
}

}
