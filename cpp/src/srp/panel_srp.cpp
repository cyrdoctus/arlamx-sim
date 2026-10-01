// Panel radiation pressure: cannonball, optical plate, Earth IR and albedo.
#include "arlamx/srp/panel_srp.hpp"

#include "arlamx/constants.hpp"

#include <algorithm>

namespace arlamx {

static void resolve_opt(double& pressure, double& Cr) {
    if (pressure <= 0.0) pressure = P_SRP_1AU;
    if (Cr <= 0.0) Cr = CR_DEFAULT;
}

Vec3 panel_srp_force(const PanelSoA& panels, Vec3 sun_hat_B, double eclipse,
                     double pressure, double Cr) {
    return panel_srp_force_torque(panels, sun_hat_B, eclipse, pressure, Cr).first;
}

// Cannonball plate: F_i = -P C_r A cos(th) s, tau = sum r_c x F_i (Vallado 2013, Sec. 8.6.4).
std::pair<Vec3, Vec3> panel_srp_force_torque(const PanelSoA& panels, Vec3 sun_hat_B,
                                             double eclipse, double pressure, double Cr) {
    resolve_opt(pressure, Cr);
    const double sn = norm(sun_hat_B);
    if (sn < 1e-12 || eclipse <= 0.0) return {Vec3::zero(), Vec3::zero()};
    const Vec3 sh = sun_hat_B / sn;
    const double p = pressure * eclipse;
    Vec3 F = Vec3::zero();
    Vec3 tau = Vec3::zero();
    for (std::size_t i = 0; i < panels.size(); ++i) {
        const Vec3 n = panels.normal(i);
        const double c = dot(n, sh);
        if (c <= 1e-12) continue;
        const Vec3 fi = (p * panels.area[i] * Cr * c) * (-sh);
        F += fi;
        tau += cross(panels.centroid(i), fi);
    }
    return {F, tau};
}

// Optical plate: F_i = -P A cos(th) [(ca + cd) s + (2 cs cos(th) + 2/3 cd) n]
// (Montenbruck & Gill 2000, Sec. 3.4).
std::pair<Vec3, Vec3> panel_srp_optical(const PanelSoA& panels, Vec3 sun_hat_B, double eclipse,
                                        double pressure, SailOptics opt) {
    if (pressure <= 0.0) pressure = P_SRP_1AU;
    double ca = opt.ca;
    double cs = opt.cs;
    double cd = opt.cd;
    const double sum = ca + cs + cd;
    if (sum > 1e-12) {
        ca /= sum;
        cs /= sum;
        cd /= sum;
    } else {
        ca = 1.0;
        cs = 0.0;
        cd = 0.0;
    }
    const double sn = norm(sun_hat_B);
    if (sn < 1e-12 || eclipse <= 0.0) return {Vec3::zero(), Vec3::zero()};
    const Vec3 sh = sun_hat_B / sn;
    const double p = pressure * eclipse;
    Vec3 F = Vec3::zero();
    Vec3 tau = Vec3::zero();
    for (std::size_t i = 0; i < panels.size(); ++i) {
        const Vec3 n = panels.normal(i);
        const double cth = dot(n, sh);
        if (cth <= 1e-12) continue;
        const Vec3 fi = (-p * panels.area[i] * cth) *
                        ((ca + cd) * sh + (2.0 * cs * cth + (2.0 / 3.0) * cd) * n);
        F += fi;
        tau += cross(panels.centroid(i), fi);
    }
    return {F, tau};
}

// Earth IR P = P1 e0/4 (R/r)^2 and albedo P = P_sun a0 (R/r)^2 max(0, r_hat.s) (Knocke et al. 1988);
// (R/r)^2 = configuration factor of a plate facing a sphere (Howell, Catalog of Configuration Factors).
std::pair<Vec3, Vec3> earth_rad(const PanelSoA& panels, Vec3 nadir_B, double r_m,
                                double cos_sun, double p_sun, SailOptics sol, SailOptics ir) {
    if (!(r_m > RE_WGS)) return {Vec3::zero(), Vec3::zero()};
    const double vf = (RE_WGS / r_m) * (RE_WGS / r_m);
    auto out = panel_srp_optical(panels, nadir_B, 1.0, P_SRP_1AU * 0.25 * EARTH_EMISS * vf, ir);
    const double alb = p_sun * EARTH_ALBEDO * vf * std::max(0.0, cos_sun);
    if (alb > 0.0) {
        const auto a = panel_srp_optical(panels, nadir_B, 1.0, alb, sol);
        out.first += a.first;
        out.second += a.second;
    }
    return out;
}

}
