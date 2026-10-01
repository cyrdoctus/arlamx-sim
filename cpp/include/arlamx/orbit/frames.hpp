// Time, frames, geodesy, Sun/Moon, eclipse, quaternion helpers.
#pragma once

#include "arlamx/types.hpp"

#include <array>

namespace arlamx {

// IAU-1982 linear GMST [rad] from Julian date (UTC≈UT1).
double gmst_rad(double jd);

Mat3 dcm_EN(double gmst);

// Bowring WGS-84: geodetic lat [rad], ECEF lon [rad], height [m].
void ecef_to_geodetic(Vec3 r_ecef, double& lat, double& lon, double& alt_m);

// Montenbruck low-precision Sun unit vector (mean equator of date, treated as N).
Vec3 sun_unit_analytic(double jd);

// Earth-Sun distance [AU], same low-precision series (Astronomical Almanac;
// Vallado 2013 Alg. 29): r = 1.00014 - 0.01671 cos g - 0.00014 cos 2g.
double sun_dist_au(double jd);

// Meeus low-precision Moon unit vector + distance [m] (Earth-centered).
void moon_analytic(double jd, Vec3& r_moon_hat, double& r_moon_m);

double eclipse_cylindrical(Vec3 r_N, Vec3 sun_hat_N, double Re);

double sma_from_rv(Vec3 r, Vec3 v, double mu);

Mat3 dcm_LN(Vec3 r_N, Vec3 v_N);

using Quat = std::array<double, 4>;
Quat quat_unit(Quat q);
Quat quat_normalize(Quat q);
Quat quat_mul(Quat a, Quat b);
Quat quat_conj(Quat q);
Quat slerp_clip(Quat q_new, Quat q_held, double max_angle_rad);
Vec3 quat_to_mrp(Quat q);
Quat dcm_to_quat(const Mat3& C);
Quat mrp_to_quat(Vec3 sigma);

}
