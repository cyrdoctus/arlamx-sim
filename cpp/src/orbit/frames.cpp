// GMST, ECI->ECEF, Bowring geodetic, analytic Sun/Moon, eclipse, quaternion helpers.
#include "arlamx/orbit/frames.hpp"

#include "arlamx/constants.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace arlamx {

// GMST, IAU-1982 linear term (Vallado 2013, ch. 3).
double gmst_rad(double jd) {
    const double gmst_deg = std::fmod(280.46061837 + 360.98564736629 * (jd - 2451545.0), 360.0);
    const double g = gmst_deg < 0.0 ? gmst_deg + 360.0 : gmst_deg;
    return g * DEG;
}

Mat3 dcm_EN(double gmst) {
    const double c = std::cos(gmst);
    const double s = std::sin(gmst);
    Mat3 M = Mat3::zero();
    M(0, 0) = c;
    M(0, 1) = s;
    M(1, 0) = -s;
    M(1, 1) = c;
    M(2, 2) = 1.0;
    return M;
}

// Bowring 1976, Survey Review 23(181), WGS-84.
void ecef_to_geodetic(Vec3 r, double& lat, double& lon, double& alt_m) {
    const double a = RE_WGS;
    const double f = WGS84_F;
    const double b = a * (1.0 - f);
    const double e2 = 1.0 - (b / a) * (b / a);
    const double ep2 = (a * a - b * b) / (b * b);
    const double p = std::hypot(r.x, r.y);
    if (p < 1.0) {
        lat = (r.z >= 0.0) ? 0.5 * PI : -0.5 * PI;
        lon = 0.0;
        alt_m = std::fabs(r.z) - b;
        return;
    }
    const double th = std::atan2(r.z * a, p * b);
    const double sth = std::sin(th);
    const double cth = std::cos(th);
    lat = std::atan2(r.z + ep2 * b * sth * sth * sth, p - e2 * a * cth * cth * cth);
    lon = std::atan2(r.y, r.x);
    const double sl = std::sin(lat);
    const double cl = std::cos(lat);
    const double N = a / std::sqrt(1.0 - e2 * sl * sl);
    if (cl > 1e-3) alt_m = p / cl - N;
    else alt_m = std::fabs(r.z) / std::max(std::fabs(sl), 1e-12) - N * (1.0 - e2);
}

// Low-precision Sun, Astronomical Almanac (Vallado 2013, Alg. 29).
Vec3 sun_unit_analytic(double jd) {
    const double n = jd - 2451545.0;
    auto wrap = [](double d) {
        d = std::fmod(d, 360.0);
        return d < 0.0 ? d + 360.0 : d;
    };
    const double L = wrap(280.460 + 0.9856474 * n) * DEG;
    const double g = wrap(357.528 + 0.9856003 * n) * DEG;
    const double lam = L + 1.915 * DEG * std::sin(g) + 0.020 * DEG * std::sin(2.0 * g);
    const double eps = (23.439 - 4.0e-7 * n) * DEG;
    return unit({std::cos(lam), std::cos(eps) * std::sin(lam), std::sin(eps) * std::sin(lam)});
}

double sun_dist_au(double jd) {
    const double n = jd - 2451545.0;
    double g = std::fmod(357.528 + 0.9856003 * n, 360.0);
    if (g < 0.0) g += 360.0;
    g *= DEG;
    return 1.00014 - 0.01671 * std::cos(g) - 0.00014 * std::cos(2.0 * g);
}

void moon_analytic(double jd, Vec3& rhat, double& rm) {
    // Meeus, Astronomical Algorithms, Ch. 47 (truncated).
    const double T = (jd - 2451545.0) / 36525.0;
    const double Lp = (218.3164477 + 481267.88123421 * T) * DEG;
    const double D = (297.8501921 + 445267.1114034 * T) * DEG;
    const double M = (357.5291092 + 35999.0502909 * T) * DEG;
    const double Mp = (134.9633964 + 477198.8675055 * T) * DEG;
    const double F = (93.2720950 + 483202.0175233 * T) * DEG;
    const double lon = Lp + DEG * (6.289 * std::sin(Mp) + 1.274 * std::sin(2.0 * D - Mp) +
                                   0.658 * std::sin(2.0 * D) + 0.214 * std::sin(2.0 * Mp) -
                                   0.186 * std::sin(M));
    const double lat = DEG * (5.128 * std::sin(F) + 0.281 * std::sin(Mp + F));
    const double par = DEG * (0.9507 + 0.0518 * std::cos(Mp) + 0.0095 * std::cos(2.0 * D - Mp));
    rm = RE_WGS / std::sin(par);
    const double cl = std::cos(lat);
    // Ecliptic -> equatorial by R1(-eps) (Meeus 1998, ch. 13).
    const double xe = cl * std::cos(lon);
    const double ye = cl * std::sin(lon);
    const double ze = std::sin(lat);
    const double eps = (23.439 - 4.0e-7 * (jd - 2451545.0)) * DEG;
    const double ce = std::cos(eps), se = std::sin(eps);
    rhat = unit({xe, ce * ye - se * ze, se * ye + ce * ze});
}

// Cylindrical shadow (Vallado 2013, ch. 5).
double eclipse_cylindrical(Vec3 r_N, Vec3 sun_hat_N, double Re) {
    const Vec3 s = unit(sun_hat_N);
    if (dot(r_N, s) >= 0.0) return 1.0;
    const Vec3 perp = r_N - dot(r_N, s) * s;
    return (norm(perp) < Re) ? 0.0 : 1.0;
}

// Vis-viva a = -mu / (2 eps), eps = v^2/2 - mu/r.
double sma_from_rv(Vec3 r, Vec3 v, double mu) {
    const double rn = norm(r);
    if (!std::isfinite(rn) || !std::isfinite(norm(v)) || !(mu > 0.0) || rn < 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double eps = 0.5 * norm2(v) - mu / rn;
    if (!std::isfinite(eps) || std::fabs(eps) < 1e-12) return 1e300;
    return -mu / (2.0 * eps);
}

Mat3 dcm_LN(Vec3 r_N, Vec3 v_N) {
    const double rn = norm(r_N);
    const Vec3 h = cross(r_N, v_N);
    const double hn = norm(h);
    if (rn < 1.0 || hn < 1e-12) return Mat3::identity();
    const Vec3 z = (-1.0 / rn) * r_N;
    const Vec3 y = (-1.0 / hn) * h;
    Vec3 x = cross(y, z);
    const double xn = norm(x);
    if (xn < 1e-12) return Mat3::identity();
    x = x / xn;
    Mat3 L;
    L(0, 0) = x.x;
    L(0, 1) = x.y;
    L(0, 2) = x.z;
    L(1, 0) = y.x;
    L(1, 1) = y.y;
    L(1, 2) = y.z;
    L(2, 0) = z.x;
    L(2, 1) = z.y;
    L(2, 2) = z.z;
    return L;
}

Quat quat_unit(Quat q) {
    double n = 0.0;
    for (double c : q) n += c * c;
    n = std::sqrt(n);
    if (!(n > 1e-15) || !std::isfinite(n)) return {1.0, 0.0, 0.0, 0.0};
    for (double& c : q) c /= n;
    return q;
}

Quat quat_normalize(Quat q) {
    q = quat_unit(q);
    if (q[0] < 0.0) {
        for (double& c : q) c = -c;
    }
    return q;
}

// Hamilton a⊗b: apply b first, then a. Schaub q_BR = quat_mul(q_BN, q_RN^*).
Quat quat_mul(Quat a, Quat b) {
    return {a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]};
}

Quat quat_conj(Quat q) { return {q[0], -q[1], -q[2], -q[3]}; }

// Slerp (Shoemake 1985, SIGGRAPH), clipped to max_angle.
Quat slerp_clip(Quat q_new, Quat q_held, double max_angle_rad) {
    Quat qn = quat_normalize(q_new);
    const Quat qh = quat_normalize(q_held);
    double ch = qn[0] * qh[0] + qn[1] * qh[1] + qn[2] * qh[2] + qn[3] * qh[3];
    if (ch < 0.0) {
        for (double& c : qn) c = -c;
        ch = -ch;
    }
    ch = std::min(1.0, std::max(-1.0, ch));
    const double half = std::acos(ch);
    const double full = 2.0 * half;
    if (full <= max_angle_rad + 1e-12) return qn;
    const double frac = max_angle_rad / full;
    const double sh = std::sin(half);
    if (sh < 1e-12) return qh;
    const double wa = std::sin((1.0 - frac) * half) / sh;
    const double wb = std::sin(frac * half) / sh;
    Quat o;
    for (int i = 0; i < 4; ++i) o[i] = wa * qh[i] + wb * qn[i];
    return quat_normalize(o);
}

// Sheppard's method, scalar first (Schaub & Junkins 2018, ch. 3).
Quat dcm_to_quat(const Mat3& C) {
    const double tr = C(0, 0) + C(1, 1) + C(2, 2);
    const double b[4] = {0.25 * (1 + tr), 0.25 * (1 + 2 * C(0, 0) - tr), 0.25 * (1 + 2 * C(1, 1) - tr),
                         0.25 * (1 + 2 * C(2, 2) - tr)};
    int i = 0;
    for (int k = 1; k < 4; ++k)
        if (b[k] > b[i]) i = k;
    Quat q{};
    const double s = std::sqrt(std::max(b[i], 0.0));
    q[i] = s;
    const double f = 0.25 / s;
    const double p01 = (C(1, 2) - C(2, 1)) * f, p02 = (C(2, 0) - C(0, 2)) * f, p03 = (C(0, 1) - C(1, 0)) * f;
    const double p12 = (C(0, 1) + C(1, 0)) * f, p13 = (C(2, 0) + C(0, 2)) * f, p23 = (C(1, 2) + C(2, 1)) * f;
    if (i == 0) q = {s, p01, p02, p03};
    else if (i == 1) q = {p01, s, p12, p13};
    else if (i == 2) q = {p02, p12, s, p23};
    else q = {p03, p13, p23, s};
    return quat_normalize(q);
}

// sigma = q_v / (1 + q_0) (Schaub & Junkins 2018, ch. 3).
Vec3 quat_to_mrp(Quat q) {
    q = quat_normalize(q);
    const double d = 1.0 + q[0];
    if (d < 1e-12) return Vec3::zero();
    return {q[1] / d, q[2] / d, q[3] / d};
}

Quat mrp_to_quat(Vec3 sigma) {
    double s2 = norm2(sigma);
    if (s2 > 1.0) {
        sigma = sigma * (-1.0 / s2);
        s2 = norm2(sigma);
    }
    const double den = 1.0 + s2;
    if (!(den > 0.0) || !std::isfinite(den)) return {1.0, 0.0, 0.0, 0.0};
    return {(1.0 - s2) / den, 2.0 * sigma.x / den, 2.0 * sigma.y / den, 2.0 * sigma.z / den};
}

}
