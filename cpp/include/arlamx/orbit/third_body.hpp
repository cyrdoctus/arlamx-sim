// Third-body acceleration and ephemeris interface.
#pragma once

#include "arlamx/types.hpp"

#include <string>
#include <vector>

namespace arlamx {

constexpr double MU_SUN = 1.32712440018e20;
constexpr double MU_MOON = 4.902800118e12;
constexpr double AU = 1.495978707e11;

Vec3 accel_third_body(Vec3 r_sat, Vec3 r_body, double mu_body);

struct Ephemeris {
    bool load_spice(const std::vector<std::string>& kernels);
    bool spice_ok() const { return spice_ok_; }

    Vec3 sun_pos(double jd) const;
    Vec3 moon_pos(double jd) const;
    Vec3 sun_hat(double jd) const { return unit(sun_pos(jd)); }

private:
    bool spice_ok_ = false;
};

}
