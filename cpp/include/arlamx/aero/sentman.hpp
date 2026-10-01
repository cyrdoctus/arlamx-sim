// Sentman free-molecular panel aerodynamics interface.
#pragma once

#include "arlamx/types.hpp"

#include <string>
#include <utility>

namespace arlamx {

enum class GsiModel { Sentman, Cll };

struct GsiParams {
    GsiModel model = GsiModel::Sentman;
    double alpha_E = 0.93;
    double alpha_n = 1.0;
    double alpha_t = 1.0;
};

GsiModel parse_gsi(const std::string& name);

// Sentman (1961) fully-diffuse GSI. M-01 energy-accommodation closure.
std::pair<double, double> sentman(double theta, double s, double Tw_Ti,
                                  double alpha_E);

AeroResult spacecraft_aero(const PanelSoA& panels, Vec3 v_rel_B, double rho,
                           double T, double m_bar, double T_w, double alpha_E,
                           bool one_sided_ref = true);

AeroResult spacecraft_aero(const PanelSoA& panels, Vec3 v_rel_B, double rho,
                           double T, double m_bar, double T_w, GsiParams gsi,
                           bool one_sided_ref = true, const double* chi = nullptr);

AeroResult coefficients_only(const PanelSoA& panels, Vec3 v_rel_B, double rho,
                             double T, double m_bar, double T_w, double alpha_E,
                             bool one_sided_ref = true);

AeroResult coefficients_only(const PanelSoA& panels, Vec3 v_rel_B, double rho,
                             double T, double m_bar, double T_w, GsiParams gsi,
                             bool one_sided_ref = true, const double* chi = nullptr);

}
