// Magnetic actuation: 3-axis coils and torque-rod arrays.
#pragma once

#include "arlamx/types.hpp"

#include <vector>

namespace arlamx {

struct MagnetorquerOut {
    Vec3 m = {};
    Vec3 tau = {};
    double shortfall_frac = 0.0;
};

// m = (B x tau) / |B|^2, per-axis clip, tau = m x B (Stickler & Alfriend 1976).
Vec3 saturate_dipole(Vec3 m, Vec3 m_max);
MagnetorquerOut apply_magnetorquer(Vec3 tau_cmd, Vec3 B_body, Vec3 m_max);

struct RodArray {
    std::vector<Vec3> axis;
    std::vector<double> dmax;
    std::vector<char> on;
    std::size_t size() const { return axis.size(); }
};

// Rods that are on: d = U^T (U U^T)^+ m (minimum norm, Penrose 1955), then one uniform
// scale so |d_k| <= dmax_k (direction-preserving, Bodson 2002, JGCD 25(4)).
Vec3 rods_allocate(Vec3 m_des, const RodArray& rods, std::vector<double>& d_out);
// PD torque onto rods: torque-space least squares over the rods that are on.
MagnetorquerOut apply_rods(Vec3 tau_cmd, Vec3 B_body, const RodArray& rods,
                           std::vector<double>& d_out);

}
