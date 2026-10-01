// MRP PD feedback interface.
#pragma once

#include "arlamx/types.hpp"

namespace arlamx {

struct MRPFeedback {
    double kp = 0.01;
    double kd = 0.005;
    Mat3 inertia = Mat3::diag(0.0125, 0.0125, 0.025);
    Vec3 max_torque = {0, 0, 0};
    double max_body_rate = 0.0;
    bool proper = true;

    // tau = -K sigma_err - P w + w x I w (Schaub & Junkins 2018, ch. 8).
    Vec3 compute(Vec3 sigma, Vec3 omega, Vec3 sigma_tgt, Vec3 omega_tgt,
                 double dt_s, Vec3* unclipped = nullptr) const;
};

}
