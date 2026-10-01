// Onboard FP32 propagator interface.
#pragma once

#include <vector>

namespace arlamx {

struct StateF32 {
    float r[3];
    float v[3];
};

std::vector<StateF32> propagate_f32_samples(const float r0[3], const float v0[3],
                                            const std::vector<double>& offsets_s, double dt_s,
                                            double bc_inv, double rho0, double alt0_m,
                                            double h_scale, double a_srp = 0.0,
                                            const float sun_hat[3] = nullptr);

}
