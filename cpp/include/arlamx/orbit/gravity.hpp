// Spherical-harmonic gravity interface.
//
// Independent evaluator of the standard geopotential (Montenbruck & Gill;
// Holmes & Featherstone) on GGM03S. Checked against Basilisk's
// SphericalHarmonicsGravityModel, Autonomous Vehicle Systems Laboratory,
// University of Colorado Boulder (https://github.com/AVSLab/basilisk, ISC).
// Cite that model when reusing these segments. This plant does not link Basilisk.
#pragma once

#include "arlamx/types.hpp"

#include <string>
#include <vector>

namespace arlamx {

Vec3 accel_twobody(Vec3 r, double mu);
Vec3 accel_j2(Vec3 r, double mu, double Re, double J2);
Vec3 accel_j3(Vec3 r, double mu, double Re, double J3);

// Fully normalised ALFs stay O(1) through this degree. The old unnormalised
// recurrence was clamped at 24 because sectoral P_n^n reaches ~1e120 by n=70.
inline constexpr int SH_MAX_DEGREE = 70;

class GravityHarmonics {
public:
    bool load_ggm(const std::string& path, int max_degree);
    bool loaded() const { return loaded_; }
    int max_degree() const { return nmax_; }
    double mu() const { return mu_; }
    double Re() const { return Re_; }

    Vec3 accel_ecef(Vec3 r_ecef, int degree = -1) const;

    double J2() const { return J2_; }

private:
    bool loaded_ = false;
    int nmax_ = 0;
    double mu_ = 0.0;
    double Re_ = 0.0;
    double J2_ = 0.0;
    std::vector<std::vector<double>> C_, S_;
};

Vec3 accel_gravity_N(const GravityHarmonics& grav, Vec3 r_N, double gmst, int degree);

}
