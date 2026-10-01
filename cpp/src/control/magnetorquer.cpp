// Minimum-norm dipole, per-axis coil clip, torque-rod allocation.
#include "arlamx/control/magnetorquer.hpp"

#include <algorithm>
#include <cmath>

namespace arlamx {

namespace {

constexpr double kTinyB2 = 1e-18;

double clip_axis(double v, double lim) {
    const double a = std::fabs(lim);
    if (a <= 0.0) return 0.0;
    return std::max(-a, std::min(a, v));
}

}

Vec3 saturate_dipole(Vec3 m, Vec3 m_max) {
    return {clip_axis(m.x, m_max.x), clip_axis(m.y, m_max.y), clip_axis(m.z, m_max.z)};
}

MagnetorquerOut apply_magnetorquer(Vec3 tau_cmd, Vec3 B_body, Vec3 m_max) {
    MagnetorquerOut o;
    const double b2 = norm2(B_body);
    const double tn = norm(tau_cmd);
    if (b2 < kTinyB2) {
        o.shortfall_frac = (tn < kTinyB2) ? 0.0 : 1.0;
        return o;
    }
    if (tn < kTinyB2) return o;

    o.m = saturate_dipole(cross(B_body, tau_cmd) / b2, m_max);
    o.tau = cross(o.m, B_body);
    const double ta = norm(o.tau);
    o.shortfall_frac = std::max(0.0, std::min(1.0, 1.0 - ta / tn));
    return o;
}

namespace {

// Pseudo-inverse of a symmetric 3x3 by cyclic Jacobi rotations (Golub & Van Loan 2013, Sec. 8.5).
Mat3 pinv_sym(Mat3 A) {
    Mat3 V = Mat3::identity();
    for (int sweep = 0; sweep < 50; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < 3; ++p)
            for (int q = p + 1; q < 3; ++q) off += A(p, q) * A(p, q);
        if (off < 1e-60) break;
        for (int p = 0; p < 3; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                if (std::fabs(A(p, q)) < 1e-300) continue;
                const double th = 0.5 * std::atan2(2.0 * A(p, q), A(q, q) - A(p, p));
                const double c = std::cos(th), s = std::sin(th);
                for (int k = 0; k < 3; ++k) {
                    const double akp = A(k, p), akq = A(k, q);
                    A(k, p) = c * akp - s * akq;
                    A(k, q) = s * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {
                    const double apk = A(p, k), aqk = A(q, k);
                    A(p, k) = c * apk - s * aqk;
                    A(q, k) = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; ++k) {
                    const double vkp = V(k, p), vkq = V(k, q);
                    V(k, p) = c * vkp - s * vkq;
                    V(k, q) = s * vkp + c * vkq;
                }
            }
        }
    }
    const double lmax = std::max({std::fabs(A(0, 0)), std::fabs(A(1, 1)), std::fabs(A(2, 2))});
    Mat3 P = Mat3::zero();
    for (int i = 0; i < 3; ++i) {
        const double l = A(i, i);
        if (!(l > 1e-12 * lmax) || lmax <= 0.0) continue;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) P(r, c) += V(r, i) * V(c, i) / l;
    }
    return P;
}

}

Vec3 rods_allocate(Vec3 m_des, const RodArray& rods, std::vector<double>& d_out) {
    const std::size_t n = rods.size();
    d_out.assign(n, 0.0);
    Mat3 G = Mat3::zero();
    for (std::size_t k = 0; k < n; ++k) {
        if (!rods.on[k]) continue;
        const Vec3 u = rods.axis[k];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) G(r, c) += u[r] * u[c];
    }
    const Vec3 y = mul(pinv_sym(G), m_des);
    double scale = 1.0;
    for (std::size_t k = 0; k < n; ++k) {
        if (!rods.on[k]) continue;
        d_out[k] = dot(rods.axis[k], y);
        const double a = std::fabs(d_out[k]);
        if (a > rods.dmax[k]) scale = std::min(scale, rods.dmax[k] / a);
    }
    Vec3 m = Vec3::zero();
    for (std::size_t k = 0; k < n; ++k) {
        d_out[k] *= scale;
        m += rods.axis[k] * d_out[k];
    }
    return m;
}

MagnetorquerOut apply_rods(Vec3 tau_cmd, Vec3 B_body, const RodArray& rods,
                           std::vector<double>& d_out) {
    MagnetorquerOut o;
    const std::size_t n = rods.size();
    d_out.assign(n, 0.0);
    const double b2 = norm2(B_body);
    const double tn = norm(tau_cmd);
    if (b2 < kTinyB2) {
        o.shortfall_frac = (tn < kTinyB2) ? 0.0 : 1.0;
        return o;
    }
    if (tn < kTinyB2) return o;
    // Torque-space least squares: rod k gives d_k (u_k x B); d = A^T (A A^T)^+ tau, then one
    // uniform scale for |d_k| <= dmax_k (Penrose 1955; Bodson 2002). Optimal for rods that do
    // not span 3-D, unlike projecting the 3-D minimum-norm dipole onto the rod plane.
    std::vector<Vec3> a(n);
    Mat3 G = Mat3::zero();
    for (std::size_t k = 0; k < n; ++k) {
        if (!rods.on[k]) continue;
        a[k] = cross(rods.axis[k], B_body);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) G(r, c) += a[k][r] * a[k][c];
    }
    const Vec3 y = mul(pinv_sym(G), tau_cmd);
    double scale = 1.0;
    for (std::size_t k = 0; k < n; ++k) {
        if (!rods.on[k]) continue;
        d_out[k] = dot(a[k], y);
        const double ad = std::fabs(d_out[k]);
        if (ad > rods.dmax[k]) scale = std::min(scale, rods.dmax[k] / ad);
    }
    for (std::size_t k = 0; k < n; ++k) {
        d_out[k] *= scale;
        o.m += rods.axis[k] * d_out[k];
    }
    o.tau = cross(o.m, B_body);
    o.shortfall_frac = std::max(0.0, std::min(1.0, 1.0 - norm(o.tau) / tn));
    return o;
}

}
