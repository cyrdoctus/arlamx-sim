// Quaternion PD feedback interface.
#pragma once

#include "arlamx/orbit/frames.hpp"
#include "arlamx/types.hpp"

namespace arlamx {

struct QuaternionFeedback {
    double kp = 0.01;
    double kd = 0.005;
    Mat3 inertia = Mat3::diag(0.0125, 0.0125, 0.025);
    Vec3 max_torque = {0, 0, 0};
    double max_body_rate = 0.0;

    Vec3 compute(Quat q, Vec3 omega, Quat q_tgt, Vec3 omega_tgt, double dt_s,
                 Vec3* unclipped = nullptr) const;
};

}
