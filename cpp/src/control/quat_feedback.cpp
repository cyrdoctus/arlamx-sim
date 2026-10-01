// Quaternion PD attitude feedback with per-axis torque clip.
#include "arlamx/control/quat_feedback.hpp"

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

// tau = -kp q_e,v - kd w + w x I w (Wie, Weiss & Arapostathis 1989, JGCD 12(3)).
Vec3 QuaternionFeedback::compute(Quat q, Vec3 omega, Quat q_tgt, Vec3 omega_tgt,
                                 double dt_s, Vec3* unclipped) const {
    q = quat_unit(q);
    q_tgt = quat_unit(q_tgt);
    Quat qe = quat_mul(q, quat_conj(q_tgt));
    if (qe[0] < 0.0) {
        for (double& c : qe) c = -c;
    }
    const Vec3 qv{qe[1], qe[2], qe[3]};
    const Vec3 werr = omega - omega_tgt;
    const Vec3 Iomega = mul(inertia, omega);
    const Vec3 gyro = cross(omega, Iomega);
    Vec3 tau_raw = (-kp) * qv + (-kd) * werr + gyro;
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
