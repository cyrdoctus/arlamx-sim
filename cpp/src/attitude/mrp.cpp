// MRP kinematics, DCM, shadow set, error and composition.
#include "arlamx/attitude/mrp.hpp"

#include <cmath>

namespace arlamx {

// C = I + [8 S^2 - 4(1 - s^2) S] / (1 + s^2)^2 (Schaub & Junkins 2018, ch. 3).
Mat3 mrp_to_dcm(Vec3 s) {
    const double s2 = norm2(s);
    const Mat3 sx = skew(s);
    const Mat3 sx2 = mul(sx, sx);
    const double den = (1.0 + s2) * (1.0 + s2);
    Mat3 C = Mat3::identity();
    const double f8 = 8.0 / den;
    const double f4 = -4.0 * (1.0 - s2) / den;
    for (int i = 0; i < 9; ++i) C.a[i] += f8 * sx2.a[i] + f4 * sx.a[i];
    return C;
}

// Shadow set -s/|s|^2 for |s| > 1 (Schaub & Junkins 2018, ch. 3).
Vec3 mrp_shadow(Vec3 s) {
    const double s2 = norm2(s);
    if (s2 > 1.0) return s * (-1.0 / s2);
    return s;
}

// s_dot = 1/4 [(1 - s^2) I + 2 S + 2 s s^T] w (Schaub & Junkins 2018, ch. 3).
Vec3 mrp_rate(Vec3 s, Vec3 w) {
    const double s2 = norm2(s);
    Mat3 B = Mat3::zero();
    const double d = 1.0 - s2;
    B(0, 0) = d + 2.0 * s.x * s.x;
    B(1, 1) = d + 2.0 * s.y * s.y;
    B(2, 2) = d + 2.0 * s.z * s.z;
    B(0, 1) = 2.0 * s.x * s.y - 2.0 * s.z;
    B(0, 2) = 2.0 * s.x * s.z + 2.0 * s.y;
    B(1, 0) = 2.0 * s.y * s.x + 2.0 * s.z;
    B(1, 2) = 2.0 * s.y * s.z - 2.0 * s.x;
    B(2, 0) = 2.0 * s.z * s.x - 2.0 * s.y;
    B(2, 1) = 2.0 * s.z * s.y + 2.0 * s.x;
    return mul(B, w) * 0.25;
}

Vec3 mrp_inverse(Vec3 s, bool proper) {
    if (proper) return -s;
    const double s2 = norm2(s);
    if (s2 < 1e-12) return -s;
    return s * (-1.0 / s2);
}

// MRP addition (Schaub & Junkins 2018, ch. 3).
Vec3 mrp_compose(Vec3 s1, Vec3 s2) {
    const double a = norm2(s1);
    const double b = norm2(s2);
    const double den = 1.0 + a * b - 2.0 * dot(s1, s2);
    if (std::fabs(den) < 1e-12) return Vec3::zero();
    Vec3 s = ((1.0 - b) * s1 + (1.0 - a) * s2 - 2.0 * cross(s1, s2)) / den;
    return mrp_shadow(s);
}

Vec3 mrp_error(Vec3 sigma_BN, Vec3 sigma_target, bool proper) {
    return mrp_compose(sigma_BN, mrp_inverse(sigma_target, proper));
}

double mrp_angle(Vec3 sigma_err) { return 4.0 * std::atan(norm(sigma_err)); }

}
