// Physical constants (SI 2019, WGS-84, GGM03S header, SRP, Earth radiation).
#pragma once

namespace arlamx {

// 2019 SI
constexpr double K_B = 1.380649e-23;
constexpr double N_A = 6.02214076e23;

constexpr double MU_GGM = 3.986004415e14;
constexpr double RE_GGM = 6378136.3;
constexpr double OMEGA_EARTH = 7.2921150e-5;

// WGS-84 / conventional (Bowring, vis-viva when not using GGM header)
constexpr double MU_WGS = 3.986004418e14;
constexpr double RE_WGS = 6378137.0;
constexpr double WGS84_F = 1.0 / 298.257223563;

constexpr double J2_GGM = 1.0826353865466185e-3;

constexpr double J3_EGM = -2.53265649e-6;

// Solar pressure at 1 AU, 1367 W/m^2 / c (Montenbruck & Gill 2000, Sec. 3.4).
constexpr double P_SRP_1AU = 4.56e-6;

// Earth albedo a0 and IR emissivity e0, zonal means (Knocke, Ries & Tapley 1988, AIAA 88-4292).
constexpr double EARTH_ALBEDO = 0.34;
constexpr double EARTH_EMISS = 0.68;
constexpr double CR_DEFAULT = 1.8;

constexpr double DEG = 0.017453292519943295;
constexpr double PI = 3.14159265358979323846;

}
