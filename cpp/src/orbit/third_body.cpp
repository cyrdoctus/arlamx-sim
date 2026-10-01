// Third-body acceleration and Sun/Moon ephemeris (analytic or CSPICE).
#include "arlamx/orbit/third_body.hpp"

#include "arlamx/orbit/frames.hpp"

#include <cmath>

#if defined(ARLAMX_HAVE_CSPICE)
extern "C" {
#include "SpiceUsr.h"
}
#endif

namespace arlamx {

// a = mu_k [(r_k - r)/|r_k - r|^3 - r_k/|r_k|^3] (Montenbruck & Gill 2000, Sec. 3.3).
Vec3 accel_third_body(Vec3 r_sat, Vec3 r_body, double mu_body) {
    const Vec3 d = r_body - r_sat;
    const double d2 = norm2(d);
    const double b2 = norm2(r_body);
    const double d1 = std::sqrt(d2);
    const double b1 = std::sqrt(b2);
    if (d1 < 1.0 || b1 < 1.0) return Vec3::zero();
    return mu_body * (d / (d2 * d1) - r_body / (b2 * b1));
}

bool Ephemeris::load_spice(const std::vector<std::string>& kernels) {
    spice_ok_ = false;
#if defined(ARLAMX_HAVE_CSPICE)
    for (const auto& k : kernels) {
        furnsh_c(k.c_str());
        if (failed_c()) {
            reset_c();
            return false;
        }
    }
    spice_ok_ = !kernels.empty();
    return spice_ok_;
#else
    (void)kernels;
    return false;
#endif
}

#if defined(ARLAMX_HAVE_CSPICE)
static Vec3 spice_pos(const char* target, double jd) {
    const double et = (jd - 2451545.0) * 86400.0;
    SpiceDouble state[6];
    SpiceDouble lt = 0.0;
    spkezr_c(target, et, "J2000", "NONE", "EARTH", state, &lt);
    if (failed_c()) {
        reset_c();
        return Vec3::zero();
    }
    return {state[0] * 1000.0, state[1] * 1000.0, state[2] * 1000.0};
}
#endif

Vec3 Ephemeris::sun_pos(double jd) const {
#if defined(ARLAMX_HAVE_CSPICE)
    if (spice_ok_) {
        Vec3 p = spice_pos("SUN", jd);
        if (norm(p) > 1e6) return p;
    }
#endif
    return sun_unit_analytic(jd) * (sun_dist_au(jd) * AU);
}

Vec3 Ephemeris::moon_pos(double jd) const {
#if defined(ARLAMX_HAVE_CSPICE)
    if (spice_ok_) {
        Vec3 p = spice_pos("MOON", jd);
        if (norm(p) > 1e6) return p;
    }
#endif
    Vec3 hat;
    double rm = 0.0;
    moon_analytic(jd, hat, rm);
    return hat * rm;
}

}
