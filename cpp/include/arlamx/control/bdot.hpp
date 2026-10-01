// B-dot detumble interface.
#pragma once

#include "arlamx/types.hpp"

namespace arlamx {

struct BDot {
    double gain = 1.0e6;
    double max_dipole = 0.6;
    double dt = 1.0;
    Vec3 B_prev = {0, 0, 0};
    bool have_prev = false;

    void reset() {
        have_prev = false;
        B_prev = Vec3::zero();
    }

    Vec3 dipole(Vec3 B_body);
    static Vec3 torque(Vec3 m, Vec3 B_body) { return cross(m, B_body); }
    static bool detumbled(Vec3 omega, double thresh = 0.01) { return norm(omega) < thresh; }
};

}
