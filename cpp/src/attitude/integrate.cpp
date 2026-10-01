// Euler rotational dynamics, gravity-gradient torque, RK4 step.
#include "arlamx/attitude/integrate.hpp"

#include "arlamx/attitude/mrp.hpp"

namespace arlamx {

Vec3 gg_torque(const Mat3& I, Vec3 r_B, double mu) {
    const double r = norm(r_B);
    if (!(r > 1.0)) return Vec3::zero();
    const Vec3 u = r_B / r;
    return (3.0 * mu / (r * r * r)) * cross(u, mul(I, u));
}

Vec3 omega_dot(const Mat3& I, const Mat3& Iinv, Vec3 omega, Vec3 tau) {
    return mul(Iinv, tau - cross(omega, mul(I, omega)));
}

static State12 deriv_fixed_grav(const State12& y, Vec3 F_B, Vec3 tau_B, double mass,
                                const Mat3& I, const Mat3& Iinv, Vec3 a_grav_N) {
    const Mat3 C = mrp_to_dcm(y.sigma);
    const Vec3 a_ext = mul(transpose(C), F_B) / mass;
    State12 d;
    d.r = y.v;
    d.v = a_grav_N + a_ext;
    d.sigma = mrp_rate(y.sigma, y.omega);
    d.omega = omega_dot(I, Iinv, y.omega, tau_B);
    d.t = 1.0;
    return d;
}

static State12 axpy(const State12& y, const State12& k, double s) {
    State12 o;
    o.r = y.r + k.r * s;
    o.v = y.v + k.v * s;
    o.sigma = y.sigma + k.sigma * s;
    o.omega = y.omega + k.omega * s;
    o.t = y.t + s;
    return o;
}

State12 rk4_step(const State12& y, double h, Vec3 F_B, Vec3 tau_B, double mass,
                 const Mat3& I, const Mat3& Iinv, Vec3 a_grav_N) {
    auto f = [&](const State12& s) {
        return deriv_fixed_grav(s, F_B, tau_B, mass, I, Iinv, a_grav_N);
    };
    const State12 k1 = f(y);
    const State12 k2 = f(axpy(y, k1, 0.5 * h));
    const State12 k3 = f(axpy(y, k2, 0.5 * h));
    const State12 k4 = f(axpy(y, k3, h));
    State12 o;
    o.r = y.r + (h / 6.0) * (k1.r + 2.0 * k2.r + 2.0 * k3.r + k4.r);
    o.v = y.v + (h / 6.0) * (k1.v + 2.0 * k2.v + 2.0 * k3.v + k4.v);
    o.sigma = mrp_shadow(y.sigma + (h / 6.0) * (k1.sigma + 2.0 * k2.sigma + 2.0 * k3.sigma + k4.sigma));
    o.omega = y.omega + (h / 6.0) * (k1.omega + 2.0 * k2.omega + 2.0 * k3.omega + k4.omega);
    o.t = y.t + h;
    return o;
}

}
