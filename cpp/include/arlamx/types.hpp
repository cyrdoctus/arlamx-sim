// Vec3, Mat3, panel table, state, atmosphere, aero result and step output types.
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace arlamx {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    static Vec3 zero() { return {}; }

    double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    double& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }
inline Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(double s, Vec3 a) { return a * s; }
inline Vec3 operator/(Vec3 a, double s) { return {a.x / s, a.y / s, a.z / s}; }
inline Vec3& operator+=(Vec3& a, Vec3 b) { a = a + b; return a; }
inline Vec3& operator-=(Vec3& a, Vec3 b) { a = a - b; return a; }

inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double norm2(Vec3 a) { return dot(a, a); }
inline double norm(Vec3 a) { return std::sqrt(norm2(a)); }

inline Vec3 unit(Vec3 a, double eps = 1e-15) {
    const double n = norm(a);
    if (n < eps) return Vec3::zero();
    return a / n;
}

struct Mat3 {
    double a[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    static Mat3 identity() { return {}; }
    static Mat3 zero() {
        Mat3 m;
        for (double& x : m.a) x = 0.0;
        return m;
    }
    static Mat3 diag(double xx, double yy, double zz) {
        Mat3 m = zero();
        m.a[0] = xx;
        m.a[4] = yy;
        m.a[8] = zz;
        return m;
    }

    double operator()(int r, int c) const { return a[3 * r + c]; }
    double& operator()(int r, int c) { return a[3 * r + c]; }
};

inline Vec3 mul(const Mat3& M, Vec3 v) {
    return {M(0, 0) * v.x + M(0, 1) * v.y + M(0, 2) * v.z,
            M(1, 0) * v.x + M(1, 1) * v.y + M(1, 2) * v.z,
            M(2, 0) * v.x + M(2, 1) * v.y + M(2, 2) * v.z};
}

inline Mat3 transpose(const Mat3& M) {
    Mat3 T;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) T(c, r) = M(r, c);
    return T;
}

inline Mat3 mul(const Mat3& A, const Mat3& B) {
    Mat3 C = Mat3::zero();
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 3; ++k) C(r, c) += A(r, k) * B(k, c);
    return C;
}

inline Mat3 inverse_sym(const Mat3& M) {
    const double a00 = M(0, 0), a01 = M(0, 1), a02 = M(0, 2);
    const double a10 = M(1, 0), a11 = M(1, 1), a12 = M(1, 2);
    const double a20 = M(2, 0), a21 = M(2, 1), a22 = M(2, 2);
    const double c00 = a11 * a22 - a12 * a21;
    const double c01 = a02 * a21 - a01 * a22;
    const double c02 = a01 * a12 - a02 * a11;
    const double det = a00 * c00 + a10 * c01 + a20 * c02;
    Mat3 I = Mat3::zero();
    if (std::fabs(det) < 1e-30) return I;
    const double inv = 1.0 / det;
    I(0, 0) = c00 * inv;
    I(0, 1) = c01 * inv;
    I(0, 2) = c02 * inv;
    I(1, 0) = (a12 * a20 - a10 * a22) * inv;
    I(1, 1) = (a00 * a22 - a02 * a20) * inv;
    I(1, 2) = (a02 * a10 - a00 * a12) * inv;
    I(2, 0) = (a10 * a21 - a11 * a20) * inv;
    I(2, 1) = (a01 * a20 - a00 * a21) * inv;
    I(2, 2) = (a00 * a11 - a01 * a10) * inv;
    return I;
}

inline Mat3 skew(Vec3 v) {
    Mat3 S = Mat3::zero();
    S(0, 1) = -v.z;
    S(0, 2) = v.y;
    S(1, 0) = v.z;
    S(1, 2) = -v.x;
    S(2, 0) = -v.y;
    S(2, 1) = v.x;
    return S;
}

struct PanelSoA {
    std::vector<double> nx, ny, nz, area, cx, cy, cz;
    std::size_t size() const { return area.size(); }
    void clear() {
        nx.clear();
        ny.clear();
        nz.clear();
        area.clear();
        cx.clear();
        cy.clear();
        cz.clear();
    }
    void push(Vec3 n, double A, Vec3 c) {
        nx.push_back(n.x);
        ny.push_back(n.y);
        nz.push_back(n.z);
        area.push_back(A);
        cx.push_back(c.x);
        cy.push_back(c.y);
        cz.push_back(c.z);
    }
    Vec3 normal(std::size_t i) const { return {nx[i], ny[i], nz[i]}; }
    Vec3 centroid(std::size_t i) const { return {cx[i], cy[i], cz[i]}; }
};

struct State12 {
    Vec3 r, v, sigma, omega;
    double t = 0.0;
};

struct Atmo {
    double rho = 0.0;
    double T = 1000.0;
    double m_bar = 2.656e-26;
    // Mole fractions {He, O, N2, O2, Ar, H, N}. Sentman ignores these.
    // CLL uses them for the Walker mixture; if has_species is false, CLL
    // falls back to atomic-O Walker coefficients and m_bar.
    double chi[7] = {};
    bool has_species = false;
};

struct AeroResult {
    Vec3 F = {};
    Vec3 tau = {};
    double Cd = 0.0;
    double Cl = 0.0;
    double A_ref = 0.0;
};

struct StepOut {
    State12 y;
    double altitude_km = 0.0;
    double sma_m = 0.0;
    double eclipse = 1.0;
    Vec3 sun_B = {};
    Vec3 B_B = {};
    double Cd = 0.0;
    double Cl = 0.0;
    double drag_N = 0.0;
    double dE_actual = 0.0;
    double dE_baseline = 0.0;
    double dE_drag = 0.0;
    double dE_lift = 0.0;
    double tracking_err_rad = 0.0;
    int n_substeps = 0;
    Vec3 tau_ctrl_mean = {};
    Vec3 tau_aero_mean = {};
    Vec3 m_mean = {};
    Vec3 tau_cmd_mean = {};
    double tau_shortfall_frac = 0.0;
    Vec3 tau_demand_mean = {};
    Vec3 tau_demand_absmean = {};
    Vec3 gyro_mean = {};
    Vec3 tau_srp_mean = {};
    Vec3 tau_erp_mean = {};
    Vec3 tau_gg_mean = {};
    Vec3 tau_env_mean = {};
    double a_srp_sun = 0.0;
    int n_sunlit = 0;
    std::vector<double> rod_duty2_mean;
};

}
