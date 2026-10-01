// GGM03S spherical-harmonic gravity, J2/J3 closed forms, two-body.
//
// The field below is the Montenbruck & Gill potential with fully normalised
// Legendre functions (Holmes & Featherstone 2002). The numerical reference
// for this segment is Basilisk, Autonomous Vehicle Systems Laboratory,
// University of Colorado Boulder: SphericalHarmonicsGravityModel::computeField
// (https://github.com/AVSLab/basilisk, ISC license). tests/basilisk_ref/
// compares the two. Cite Basilisk with any reuse of this evaluator.
// ARLAMX does not include or link Basilisk.
#include "arlamx/orbit/gravity.hpp"

#include "arlamx/constants.hpp"
#include "arlamx/orbit/frames.hpp"

#include <cmath>
#include <fstream>
#include <sstream>

namespace arlamx {

// a = -mu r / |r|^3.
Vec3 accel_twobody(Vec3 r, double mu) {
    const double r2 = norm2(r);
    const double r1 = std::sqrt(r2);
    if (r1 < 1.0) return Vec3::zero();
    return r * (-mu / (r2 * r1));
}

// J2 closed form (Vallado 2013, eq. 8-30).
Vec3 accel_j2(Vec3 r, double mu, double Re, double J2) {
    const double r2 = norm2(r);
    const double r1 = std::sqrt(r2);
    if (r1 < 1.0) return Vec3::zero();
    const double zr2 = (r.z * r.z) / r2;
    const double f2 = 1.5 * J2 * mu * Re * Re / (r1 * r1 * r1 * r1 * r1);
    return {f2 * r.x * (5.0 * zr2 - 1.0),
            f2 * r.y * (5.0 * zr2 - 1.0),
            f2 * r.z * (5.0 * zr2 - 3.0)};
}

// J3 closed form, gradient of U3 = (mu/r) (-J3) (R/r)^3 P3(sin phi) (Vallado 2013, ch. 8).
Vec3 accel_j3(Vec3 r, double mu, double Re, double J3) {
    const double r2 = norm2(r);
    const double r1 = std::sqrt(r2);
    if (r1 < 1.0) return Vec3::zero();
    const double f3 = 0.5 * J3 * mu * Re * Re * Re / (r1 * r1 * r1 * r1 * r1 * r1 * r1);
    return {f3 * 5.0 * r.x * (7.0 * r.z * r.z * r.z / r2 - 3.0 * r.z),
            f3 * 5.0 * r.y * (7.0 * r.z * r.z * r.z / r2 - 3.0 * r.z),
            f3 * 3.0 * (35.0 * r.z * r.z * r.z * r.z / (3.0 * r2) - 10.0 * r.z * r.z + r2)};
}

// Kaula normalization sqrt[(2 - d_m0)(2n + 1)(n - m)! / (n + m)!] (Kaula 1966).
static double norm_factor(int n, int m) {
    if (n == 0 && m == 0) return 1.0;
    if (m == 0) return std::sqrt(2.0 * n + 1.0);
    double v = 2.0 * (2.0 * n + 1.0);
    for (int k = n - m + 1; k <= n + m; ++k) v /= static_cast<double>(k);
    return std::sqrt(v);
}

bool GravityHarmonics::load_ggm(const std::string& path, int max_degree) {
    loaded_ = false;
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    if (!std::getline(in, line)) return false;
    std::istringstream hs(line);
    char comma;
    double Re = 0, mu = 0, om = 0;
    int nfile = 0, mfile = 0;
    if (!(hs >> Re)) {
        hs.clear();
        hs.str(line);
        if (!(hs >> Re >> comma >> mu)) return false;
    } else {
    }
    for (char& c : line) {
        if (c == ',') c = ' ';
    }
    std::istringstream h2(line);
    if (!(h2 >> Re >> mu >> om >> nfile >> mfile)) return false;
    mu_ = mu;
    Re_ = Re;
    nmax_ = std::max(0, std::min(std::min(max_degree, nfile), SH_MAX_DEGREE));
    const int N = nmax_;
    C_.assign(N + 2, std::vector<double>(N + 2, 0.0));
    S_.assign(N + 2, std::vector<double>(N + 2, 0.0));
    C_[0][0] = 1.0;

    while (std::getline(in, line)) {
        for (char& c : line) {
            if (c == ',') c = ' ';
        }
        std::istringstream ls(line);
        int n = -1, m = -1;
        double cb = 0, sb = 0, dummy;
        if (!(ls >> n >> m >> cb >> sb)) continue;
        ls >> dummy >> dummy;
        if (n < 0 || m < 0 || n > N || m > n) continue;
        // File values are already fully normalised (GGM03S). They are stored
        // that way and paired with P_bar below. Denormalising them overflows
        // the sectoral product around degree 70.
        C_[n][m] = cb;
        S_[n][m] = sb;
    }
    // J2 = -C20_unnorm = -sqrt(5) C_bar_20.
    J2_ = (N >= 2) ? -norm_factor(2, 0) * C_[2][0] : 0.0;
    loaded_ = true;
    return true;
}

// a = grad U, U = mu/r sum (R/r)^n P_nm(sin phi)(C cos m l + S sin m l) (Montenbruck & Gill 2000, Sec. 3.2);
// coefficients GGM03S (Tapley et al., CSR).
Vec3 GravityHarmonics::accel_ecef(Vec3 r, int degree) const {
    const double mu = mu_ > 0.0 ? mu_ : MU_GGM;
    Vec3 a0 = accel_twobody(r, mu);
    if (!loaded_) return a0;
    int N = (degree < 0) ? nmax_ : std::min(degree, nmax_);
    if (N < 2) return a0;

    const double r1 = norm(r);
    if (r1 < 1.0) return Vec3::zero();
    const double pxy = std::hypot(r.x, r.y);
    const double sinphi = r.z / r1;
    const double cosphi = pxy / r1;
    const double lam = (pxy < 1e-12) ? 0.0 : std::atan2(r.y, r.x);
    const double tanphi = (cosphi < 1e-14) ? 0.0 : (sinphi / cosphi);

    const int Np = N + 1;
    constexpr int STRIDE = SH_MAX_DEGREE + 3;
    double Pbuf[STRIDE * STRIDE];
    for (int i = 0; i <= Np; ++i) {
        for (int j = 0; j <= Np; ++j) Pbuf[i * STRIDE + j] = 0.0;
    }
    auto P = [&Pbuf](int n, int m) -> double& { return Pbuf[n * STRIDE + m]; };

    // Fully normalised associated Legendre functions in geocentric latitude
    // (Holmes & Featherstone 2002). P_bar stays O(1) at degree 70.
    P(0, 0) = 1.0;
    if (Np >= 1) {
        P(1, 0) = std::sqrt(3.0) * sinphi;
        P(1, 1) = std::sqrt(3.0) * cosphi;
    }
    for (int n = 2; n <= Np; ++n) {
        P(n, n) = std::sqrt((2.0 * n + 1.0) / (2.0 * n)) * cosphi * P(n - 1, n - 1);
        P(n, n - 1) = std::sqrt(2.0 * n + 1.0) * sinphi * P(n - 1, n - 1);
        for (int m = 0; m <= n - 2; ++m) {
            const double nm = static_cast<double>(n - m);
            const double np = static_cast<double>(n + m);
            const double anm = std::sqrt((2.0 * n - 1.0) * (2.0 * n + 1.0) / (nm * np));
            const double bnm = std::sqrt((2.0 * n + 1.0) * (nm - 1.0) * (np - 1.0) /
                                         ((2.0 * n - 3.0) * nm * np));
            P(n, m) = anm * sinphi * P(n - 1, m) - bnm * P(n - 2, m);
        }
    }

    // dP_bar/dφ from dP/dφ = P_{n,m+1} - m tanφ P_nm, rescaled by N_nm/N_n,m+1.
    auto dPdphi = [&](int n, int m) {
        const auto u_nm = [n](int m_ord) {
            if (m_ord == 0) return std::sqrt(0.5 * n * (n + 1.0));
            return std::sqrt(static_cast<double>(n - m_ord) * (n + m_ord + 1.0));
        };
        if (cosphi < 1e-14) {
            return (m == 0) ? u_nm(0) * P(n, 1) : 0.0;
        }
        double d = -static_cast<double>(m) * tanphi * P(n, m);
        if (m + 1 <= n) d += u_nm(m) * P(n, m + 1);
        return d;
    };

    double cml[STRIDE];
    double sml[STRIDE];
    for (int m = 0; m <= N; ++m) {
        cml[m] = std::cos(m * lam);
        sml[m] = std::sin(m * lam);
    }

    const double Re_r = Re_ / r1;
    double dUdr_pert = 0.0;
    double dUdphi = 0.0;
    double dUdlam = 0.0;
    double rr = Re_r * Re_r;
    for (int n = 2; n <= N; ++n) {
        for (int m = 0; m <= n; ++m) {
            const double cm = cml[m];
            const double sm = sml[m];
            const double CS = C_[n][m] * cm + S_[n][m] * sm;
            const double SC = -C_[n][m] * sm + S_[n][m] * cm;
            dUdr_pert += -(n + 1.0) * rr * P(n, m) * CS;
            dUdphi += rr * dPdphi(n, m) * CS;
            dUdlam += rr * P(n, m) * static_cast<double>(m) * SC;
        }
        rr *= Re_r;
    }
    dUdr_pert *= mu / (r1 * r1);
    dUdphi *= mu / r1;
    dUdlam *= mu / r1;

    const double cl = std::cos(lam);
    const double sl = std::sin(lam);
    const Vec3 rhat{cosphi * cl, cosphi * sl, sinphi};
    const Vec3 phihat{-sinphi * cl, -sinphi * sl, cosphi};
    const Vec3 lamhat{-sl, cl, 0.0};
    Vec3 a_pert = dUdr_pert * rhat + (dUdphi / r1) * phihat;
    if (cosphi > 1e-14) a_pert += (dUdlam / (r1 * cosphi)) * lamhat;
    return a0 + a_pert;
}

Vec3 accel_gravity_N(const GravityHarmonics& grav, Vec3 r_N, double gmst, int degree) {
    if (!grav.loaded()) return accel_twobody(r_N, MU_WGS);
    const Mat3 EN = dcm_EN(gmst);
    const Vec3 r_E = mul(EN, r_N);
    return mul(transpose(EN), grav.accel_ecef(r_E, degree));
}

}
