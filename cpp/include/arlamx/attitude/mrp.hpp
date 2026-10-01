// MRP kinematics interface.
#pragma once

#include "arlamx/types.hpp"

namespace arlamx {

Mat3 mrp_to_dcm(Vec3 sigma);
Vec3 mrp_shadow(Vec3 sigma);
Vec3 mrp_rate(Vec3 sigma, Vec3 omega);
Vec3 mrp_inverse(Vec3 sigma, bool proper = true);
Vec3 mrp_compose(Vec3 s1, Vec3 s2);
Vec3 mrp_error(Vec3 sigma_BN, Vec3 sigma_target, bool proper = true);

double mrp_angle(Vec3 sigma_err);

}
