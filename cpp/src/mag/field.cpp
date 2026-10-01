// WMM spherical-harmonic field and the tilted dipole.
#include "arlamx/mag/field.hpp"

#include "arlamx/constants.hpp"

#include <cmath>
#include <fstream>
#include <sstream>

namespace arlamx {

// IGRF-13 / WMM2020 (epoch 2020.0).
// B = (a/r)^3 [3 (d.r_hat) r_hat - d] (Wertz 1978, App. H).
Vec3 dipole_field_ecef(Vec3 r) {
    const double r1 = norm(r);
    if (r1 < 1.0) return Vec3::zero();
    constexpr double a = 6371200.0;  // IGRF/WMM reference radius [m]
    constexpr double g10 = -29404.8e-9, g11 = -1450.9e-9, h11 = 4652.5e-9;
    const Vec3 d{g11, h11, g10};
    const Vec3 rh = r / r1;
    const double ar = a / r1;
    return (ar * ar * ar) * (3.0 * dot(d, rh) * rh - d);
}

bool WMM::load(const std::string& cof_path) {
    loaded_ = false;
    std::ifstream in(cof_path);
    if (!in) return false;
    std::string line;
    if (!std::getline(in, line)) return false;
    std::istringstream hs(line);
    hs >> epoch_;
    for (int n = 0; n < 13; ++n)
        for (int m = 0; m < 13; ++m) g_[n][m] = h_[n][m] = gt_[n][m] = ht_[n][m] = 0.0;
    while (std::getline(in, line)) {
        if (line.find("9999") != std::string::npos) break;
        std::istringstream ls(line);
        int n = 0, m = 0;
        double gv = 0, hv = 0, gtv = 0, htv = 0;
        if (!(ls >> n >> m >> gv >> hv >> gtv >> htv)) continue;
        if (n < 1 || n > 12 || m < 0 || m > n) continue;
        g_[n][m] = gv;
        h_[n][m] = hv;
        gt_[n][m] = gtv;
        ht_[n][m] = htv;
    }
    loaded_ = true;
    return true;
}

Vec3 WMM::field_ecef(Vec3 r, double year) const {
    if (!loaded_) return dipole_field_ecef(r);
    const double r1 = norm(r);
    if (r1 < 1.0) return Vec3::zero();
    constexpr double a = 6371200.0;
    const double dt = year - epoch_;
    const double pxy = std::hypot(r.x, r.y);
    const double theta = std::atan2(pxy, r.z);
    const double lam = (pxy < 1.0) ? 0.0 : std::atan2(r.y, r.x);
    const double ct = std::cos(theta);
    const double st = std::max(std::sin(theta), 1e-14);

    // Schmidt semi-normalized P_n^m(cos th) recursion, B = -grad V (Chulliat et al. 2020, WMM2020 report).
    double P[13][13]{};
    double dP[13][13]{};
    P[0][0] = 1.0;
    dP[0][0] = 0.0;
    P[1][0] = ct;
    dP[1][0] = -st;
    P[1][1] = st;
    dP[1][1] = ct;
    for (int n = 2; n <= 12; ++n) {
        for (int m = 0; m <= n; ++m) {
            if (n == m) {
                const double c = std::sqrt(1.0 - 0.5 / n);
                P[n][n] = c * st * P[n - 1][n - 1];
                dP[n][n] = c * (ct * P[n - 1][n - 1] + st * dP[n - 1][n - 1]);
            } else {
                const double nn = static_cast<double>(n);
                const double mm = static_cast<double>(m);
                const double d = std::sqrt(nn * nn - mm * mm);
                const double k = std::sqrt((nn - 1.0) * (nn - 1.0) - mm * mm);
                const double Pm2 = (n - 2 >= m) ? P[n - 2][m] : 0.0;
                const double dPm2 = (n - 2 >= m) ? dP[n - 2][m] : 0.0;
                P[n][m] = ((2.0 * nn - 1.0) * ct * P[n - 1][m] - k * Pm2) / d;
                dP[n][m] = ((2.0 * nn - 1.0) * (ct * dP[n - 1][m] - st * P[n - 1][m]) - k * dPm2) / d;
            }
        }
    }

    double cml[13];
    double sml[13];
    for (int m = 0; m <= 12; ++m) {
        cml[m] = std::cos(m * lam);
        sml[m] = std::sin(m * lam);
    }

    double Br = 0, Bt = 0, Bp = 0;
    const double ar = a / r1;
    double arp = ar * ar;
    for (int n = 1; n <= 12; ++n) {
        arp *= ar;
        for (int m = 0; m <= n; ++m) {
            const double g = g_[n][m] + gt_[n][m] * dt;
            const double h = h_[n][m] + ht_[n][m] * dt;
            const double cm = cml[m];
            const double sm = sml[m];
            const double gh = g * cm + h * sm;
            Br += (n + 1.0) * arp * gh * P[n][m];
            Bt += -arp * gh * dP[n][m];
            Bp += arp * m * (g * sm - h * cm) * P[n][m] / st;
        }
    }
    Br *= 1e-9;
    Bt *= 1e-9;
    Bp *= 1e-9;
    const double cl = std::cos(lam);
    const double sl = std::sin(lam);
    return {Br * st * cl + Bt * ct * cl - Bp * sl,
            Br * st * sl + Bt * ct * sl + Bp * cl,
            Br * ct - Bt * st};
}

}
