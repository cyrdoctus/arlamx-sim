// B-dot detumble dipole.
#include "arlamx/control/bdot.hpp"

namespace arlamx {

// m = -k dB/dt (Stickler & Alfriend 1976).
Vec3 BDot::dipole(Vec3 B_body) {
    if (!have_prev) {
        B_prev = B_body;
        have_prev = true;
        return Vec3::zero();
    }
    const double h = (dt > 0.0) ? dt : 1.0;
    const Vec3 Bdot = (B_body - B_prev) / h;
    B_prev = B_body;
    Vec3 m = Bdot * (-gain);
    const double mag = norm(m);
    if (mag > max_dipole && mag > 0.0) m = m * (max_dipole / mag);
    return m;
}

}
