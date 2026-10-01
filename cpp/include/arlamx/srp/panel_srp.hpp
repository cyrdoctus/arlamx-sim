// Panel solar and Earth radiation pressure interface.
#pragma once

#include "arlamx/types.hpp"

#include <utility>

namespace arlamx {

struct SailOptics {
    double ca = 1.0;
    double cs = 0.0;
    double cd = 0.0;
    bool optical = false;
};

Vec3 panel_srp_force(const PanelSoA& panels, Vec3 sun_hat_B, double eclipse = 1.0,
                     double pressure = 0.0, double Cr = 0.0);

std::pair<Vec3, Vec3> panel_srp_force_torque(const PanelSoA& panels, Vec3 sun_hat_B,
                                             double eclipse = 1.0, double pressure = 0.0,
                                             double Cr = 0.0);

std::pair<Vec3, Vec3> panel_srp_optical(const PanelSoA& panels, Vec3 sun_hat_B, double eclipse,
                                        double pressure, SailOptics opt);

std::pair<Vec3, Vec3> earth_rad(const PanelSoA& panels, Vec3 nadir_B, double r_m,
                                double cos_sun, double p_sun, SailOptics sol, SailOptics ir);

}
