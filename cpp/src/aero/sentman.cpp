// Sentman plate coefficients and the panel force/torque sum.
#include "arlamx/aero/sentman.hpp"

#include "arlamx/aero/cll.hpp"
#include "arlamx/constants.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>

namespace arlamx {

static const double SQRT_PI = std::sqrt(PI);

namespace {

struct SentmanFlow {
    double s = 0.0;
    double inv_s = 0.0;
    double inv_s2 = 0.0;
    double sqrt_Tr_Ti = 0.0;
};

inline SentmanFlow make_flow(double s, double Tw_Ti, double alpha_E) {
    SentmanFlow f;
    f.s = s;
    f.inv_s = 1.0 / s;
    f.inv_s2 = f.inv_s * f.inv_s;
    // M-01: energy-flux definition of alpha_E (Moe & Moe 2005).
    const double Tr_Ti = alpha_E * Tw_Ti + (1.0 - alpha_E) * 0.5 * s * s;
    f.sqrt_Tr_Ti = std::sqrt(std::max(Tr_Ti, 0.0));
    return f;
}

// g = s cos(th), Z = 1 + erf g, E = exp(-g^2):
// C_p,i = [(g^2 + 1/2) Z + g E / sqrt(pi)] / s^2, C_tau = sin(th) (g Z + E / sqrt(pi)) / s,
// C_p,r = sqrt(T_r/T_i) (E + sqrt(pi) g Z) / (2 s^2) (Sentman 1961, LMSC-448514).
inline std::pair<double, double> sentman_cs(double cos_t, const SentmanFlow& f) {
    const double gamma = f.s * cos_t;
    const bool hi = gamma > 6.0;
    const double Z = hi ? 2.0 : 1.0 + std::erf(gamma);
    const double E = hi ? 0.0 : std::exp(-gamma * gamma);
    const double Cp_i = ((gamma * gamma + 0.5) * Z + gamma * E / SQRT_PI) * f.inv_s2;
    const double Ctau_over_sin = (gamma * Z + E / SQRT_PI) * f.inv_s;
    const double Cp_r = f.sqrt_Tr_Ti * (E + SQRT_PI * gamma * Z) * f.inv_s2 * 0.5;
    return {Cp_i + Cp_r, Ctau_over_sin};
}

}

std::pair<double, double> sentman(double theta, double s, double Tw_Ti,
                                  double alpha_E) {
    if (!(s > 1e-12)) return {0.0, 0.0};
    const auto [Cp, Ct_over_sin] = sentman_cs(std::cos(theta), make_flow(s, Tw_Ti, alpha_E));
    return {Cp, std::sin(theta) * Ct_over_sin};
}

GsiModel parse_gsi(const std::string& name) {
    std::string s = name;
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (s == "sentman") return GsiModel::Sentman;
    if (s == "cll") return GsiModel::Cll;
    throw std::invalid_argument("gsi must be 'sentman' or 'cll'");
}

namespace {

// F = q sum A_i (-C_p n_i + C_tau t_i), tau = sum c_i x F_i, s = |v| / sqrt(2 k T / m).
AeroResult sum_panels(const PanelSoA& panels, Vec3 v_rel_B, double rho, double T,
                      double m_bar, double T_w, GsiParams gsi, bool one_sided_ref,
                      bool want_torque, const double* chi) {
    AeroResult out;
    const double vmag = norm(v_rel_B);
    const double A_sum = [&]() {
        double s = 0.0;
        for (double a : panels.area) s += a;
        return s;
    }();
    out.A_ref = one_sided_ref ? 0.5 * A_sum : A_sum;
    if (vmag < 1e-12 || T <= 0.0 || m_bar <= 0.0 || rho < 0.0 || panels.size() == 0) {
        return out;
    }
    const Vec3 vhat = v_rel_B / vmag;
    const double s = vmag / std::sqrt(2.0 * K_B * T / m_bar);
    const double Tw_Ti = T_w / T;
    const double q = 0.5 * rho * vmag * vmag;
    if (!(s > 1e-12)) return out;
    const SentmanFlow flow = make_flow(s, Tw_Ti, gsi.alpha_E);
    const bool use_cll = (gsi.model == GsiModel::Cll);
    const CllMix cll_mix =
        use_cll ? make_cll_mix(s, T, vmag, Tw_Ti, gsi.alpha_n, gsi.alpha_t, chi) : CllMix{};

    Vec3 F = Vec3::zero();
    Vec3 tau = Vec3::zero();
    for (std::size_t i = 0; i < panels.size(); ++i) {
        const Vec3 n = panels.normal(i);
        const double cos_theta = -dot(vhat, n);
        if (!use_cll && flow.s * cos_theta <= -4.0) continue;
        const double ct = std::max(-1.0, std::min(1.0, cos_theta));
        const auto [Cp, Ct_over_sin] =
            use_cll ? cll_mix_cs(ct, cll_mix) : sentman_cs(ct, flow);
        const Vec3 t = vhat + cos_theta * n;
        const Vec3 fdir = (-Cp * n) + (Ct_over_sin * t);
        const Vec3 Fi = (want_torque ? (panels.area[i] * q) : panels.area[i]) * fdir;
        F += Fi;
        if (want_torque) tau += cross(panels.centroid(i), Fi);
    }

    if (want_torque) {
        out.F = F;
        out.tau = tau;
        if (q * out.A_ref > 1e-30) {
            out.Cd = dot(F, vhat) / (q * out.A_ref);
            const Vec3 Fp = F - dot(F, vhat) * vhat;
            out.Cl = norm(Fp) / (q * out.A_ref);
        }
    } else {
        if (out.A_ref > 1e-30) {
            out.Cd = dot(F, vhat) / out.A_ref;
            const Vec3 Fp = F - dot(F, vhat) * vhat;
            out.Cl = norm(Fp) / out.A_ref;
        }
        out.F = F * q;
    }
    return out;
}

}

AeroResult spacecraft_aero(const PanelSoA& panels, Vec3 v_rel_B, double rho,
                           double T, double m_bar, double T_w, GsiParams gsi,
                           bool one_sided_ref, const double* chi) {
    return sum_panels(panels, v_rel_B, rho, T, m_bar, T_w, gsi, one_sided_ref, true, chi);
}

AeroResult spacecraft_aero(const PanelSoA& panels, Vec3 v_rel_B, double rho,
                           double T, double m_bar, double T_w, double alpha_E,
                           bool one_sided_ref) {
    GsiParams gsi;
    gsi.alpha_E = alpha_E;
    return spacecraft_aero(panels, v_rel_B, rho, T, m_bar, T_w, gsi, one_sided_ref, nullptr);
}

AeroResult coefficients_only(const PanelSoA& panels, Vec3 v_rel_B, double rho,
                             double T, double m_bar, double T_w, GsiParams gsi,
                             bool one_sided_ref, const double* chi) {
    return sum_panels(panels, v_rel_B, rho, T, m_bar, T_w, gsi, one_sided_ref, false, chi);
}

AeroResult coefficients_only(const PanelSoA& panels, Vec3 v_rel_B, double rho,
                             double T, double m_bar, double T_w, double alpha_E,
                             bool one_sided_ref) {
    GsiParams gsi;
    gsi.alpha_E = alpha_E;
    return coefficients_only(panels, v_rel_B, rho, T, m_bar, T_w, gsi, one_sided_ref, nullptr);
}

}
