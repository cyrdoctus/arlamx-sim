// Rigid-body rotation and gravity-gradient torque interface.
#pragma once

#include "arlamx/types.hpp"

namespace arlamx {

// Euler: w_dot = I^-1 (tau - w x I w) (Schaub & Junkins 2018, ch. 4).
Vec3 omega_dot(const Mat3& I, const Mat3& Iinv, Vec3 omega, Vec3 tau);

// tau_gg = 3 mu / r^3  u x (I u), u = r_B / |r_B| (Markley & Crassidis 2014, ch. 3).
Vec3 gg_torque(const Mat3& I, Vec3 r_B, double mu);

State12 rk4_step(const State12& y, double h, Vec3 F_B, Vec3 tau_B, double mass,
                 const Mat3& I, const Mat3& Iinv, Vec3 a_grav_N);

}
