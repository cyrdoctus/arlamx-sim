// Geomagnetic field interface.
#pragma once

#include "arlamx/types.hpp"

#include <string>

namespace arlamx {

// Centered tilted dipole in ECEF, tesla — the degree-1 truncation of IGRF-13 /
// WMM2020 (g10, g11, h11 at epoch 2020.0), tilted ~9.4 deg off the spin axis.
Vec3 dipole_field_ecef(Vec3 r_ecef);

class WMM {
public:
    bool load(const std::string& cof_path);
    bool loaded() const { return loaded_; }
    Vec3 field_ecef(Vec3 r_ecef, double year) const;

private:
    bool loaded_ = false;
    double epoch_ = 2020.0;
    double g_[13][13]{};
    double h_[13][13]{};
    double gt_[13][13]{};
    double ht_[13][13]{};
};

}
