// MRP PD attitude feedback with per-axis torque clip.
#include "arlamx/control/mrp_feedback.hpp"

#include "arlamx/attitude/mrp.hpp"

#include <algorithm>
#include <cmath>

namespace arlamx {

static Vec3 clip_torque(Vec3 t, Vec3 cap) {
    if (cap.x <= 0.0 && cap.y <= 0.0 && cap.z <= 0.0) return t;
    auto clip1 = [](double v, double c) {
        if (c <= 0.0) return v;
        return std::max(-c, std::min(c, v));
    };
    return {clip1(t.x, cap.x), clip1(t.y, cap.y), clip1(t.z, cap.z)};
}

Vec3 MRPFeedback::compute(Vec3 sigma, Vec3 omega, Vec3 sigma_tgt, Vec3 omega_tgt,
                          double dt_s, Vec3* unclipped) const {
    const Vec3 serr = mrp_error(sigma, sigma_tgt, proper);
    const Vec3 werr = omega - omega_tgt;
    const Vec3 Iomega = mul(inertia, omega);
    const Vec3 gyro = cross(omega, Iomega);
    Vec3 tau_raw = (-kp) * serr + (-kd) * werr + gyro;
    Vec3 tau = clip_torque(tau_raw, max_torque);

    if (max_body_rate > 0.0 && dt_s > 0.0) {
        const Vec3 pd = tau - gyro;
        const Mat3 Iinv = inverse_sym(inertia);
        const Vec3 wdot = mul(Iinv, pd);
        const Vec3 wnext = omega + wdot * dt_s;
        const double rp = norm(wnext);
        if (rp > max_body_rate && rp > 1e-12) {
            const Vec3 wdes = wnext * (max_body_rate / rp);
            const Vec3 wd_t = (wdes - omega) / dt_s;
            tau_raw = mul(inertia, wd_t) + gyro;
            tau = clip_torque(tau_raw, max_torque);
        }
    }
    if (unclipped) *unclipped = tau_raw;
    return tau;
}

}
