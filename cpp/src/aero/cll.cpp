// Walker-CLL (ADBSat eqs. 11-15) species-mixed plate coefficients.
#include "arlamx/aero/cll.hpp"

#include "arlamx/constants.hpp"

#include <algorithm>
#include <cmath>

namespace arlamx {

namespace {

// 2019 SI molar masses [kg] / N_A — same numbers as python/arlamx_v2/atmosphere.py.
constexpr double M_HE = 4.002602e-3 / N_A;
constexpr double M_O = 15.999e-3 / N_A;
constexpr double M_N2 = 28.014e-3 / N_A;
constexpr double M_O2 = 31.998e-3 / N_A;
constexpr double M_AR = 39.948e-3 / N_A;
constexpr double M_H = 1.008e-3 / N_A;
constexpr double M_N = 14.007e-3 / N_A;

constexpr double MASS[CLL_NSPEC] = {M_HE, M_O, M_N2, M_O2, M_AR, M_H, M_N};

static const double SQRT_PI = std::sqrt(PI);

struct WalkerFit {
    double beta = 0.0;
    double gamma = 0.0;
    double delta = 0.0;
    double zeta = 0.0;
};

// Walker, Mehta & Koller, J. Spacecraft Rockets 51 (2014) 1544–1563.
// Rows from ADBSat coeff_CLL Fitted_Parameters (Sinpetru et al., arXiv:2104.05543); He and H mid-band
// rows are identical there ({3.45, 0.52, 2.4, 0.93}); confirm vs Walker 2014 Table 2. Ar reuses N2.
WalkerFit walker_fit(int species, double alpha_n) {
    const double a = std::max(0.0, std::min(1.0, alpha_n));
    if (species == 0) {
        if (a > 0.95) return {6.2, 0.38, 3.3, 0.74};
        if (a > 0.90) return {3.8, 0.52, 3.4, 1.12};
        if (a > 0.50) return {3.45, 0.52, 2.4, 0.93};
        return {0.08, 0.52, 4.2, 1.1};
    }
    if (species == 1) return {5.85, 0.2, 0.48, 31.0};
    if (species == 2 || species == 4) return {6.6, 0.22, 0.48, 35.0};
    if (species == 3) return {6.3, 0.26, 0.42, 20.5};
    if (species == 5) {
        if (a > 0.95) return {3.9, 0.195, 1.4, 0.3};
        if (a > 0.90) return {3.5, 0.42, 2.0, 0.72};
        if (a > 0.50) return {3.45, 0.52, 2.4, 0.93};
        return {0.095, 0.465, 2.9, 0.92};
    }
    return {4.9, 0.32, 0.42, 8.0};
}

inline double species_s(double vmag, double T, int j) {
    return vmag / std::sqrt(2.0 * K_B * T / MASS[j]);
}

void fill_species(CllSpeciesFlow& sp, int species, double s, double Tw_Ti, double alpha_n,
                  double weight) {
    if (!(s > 1e-12) || weight <= 0.0) return;
    sp.active = true;
    sp.s = s;
    sp.inv_s = 1.0 / s;
    sp.inv_s2 = sp.inv_s * sp.inv_s;
    sp.weight = weight;
    sp.schaaf = !(alpha_n < 1.0);
    if (!sp.schaaf) {
        const WalkerFit f = walker_fit(species, alpha_n);
        const double one_m = 1.0 - alpha_n;
        const double reem = std::exp(-f.beta * std::pow(one_m, f.gamma)) *
                            std::pow(std::max(Tw_Ti, 0.0), f.delta) * (f.zeta * sp.inv_s);
        sp.one_sqrt = 1.0 + std::sqrt(one_m);
        sp.reem_g2 = 0.5 * reem * std::sqrt(std::max(Tw_Ti, 0.0)) * SQRT_PI;
    }
}

}

CllMix make_cll_mix(double s_mix, double T, double vmag, double Tw_Ti,
                    double alpha_n, double alpha_t, const double* chi) {
    CllMix mix;
    mix.Tw_Ti = Tw_Ti;
    mix.alpha_n = std::max(0.0, std::min(1.0, alpha_n));
    mix.alpha_t = std::max(0.0, std::min(1.0, alpha_t));

    auto add = [&](int j, double s, double w) {
        fill_species(mix.sp[j], j, s, Tw_Ti, mix.alpha_n, w);
        if (mix.sp[j].active) mix.Mavg += mix.sp[j].weight;
    };

    if (chi == nullptr) {
        add(1, s_mix, MASS[1]);
        return mix;
    }
    for (int j = 0; j < CLL_NSPEC; ++j) {
        const double x = std::max(0.0, chi[j]);
        if (x <= 0.0) continue;
        const double sj = (T > 0.0 && vmag > 0.0) ? species_s(vmag, T, j) : s_mix;
        add(j, sj, x * MASS[j]);
    }
    if (!(mix.Mavg > 0.0)) {
        add(1, s_mix, MASS[1]);
    }
    return mix;
}

// alpha_N < 1: ADBSat eq. 13; alpha_N = 1: Schaaf-Chambre, ADBSat eq. 9; mix: eq. 15.
std::pair<double, double> cll_mix_cs(double cos_t, const CllMix& mix) {
    if (!(mix.Mavg > 0.0)) return {0.0, 0.0};
    const double ct = std::max(-1.0, std::min(1.0, cos_t));
    const double an = mix.alpha_n;
    const double at = mix.alpha_t;
    const double sq = std::sqrt(std::max(mix.Tw_Ti, 0.0));
    const double two_m = 2.0 - an;
    const double half_an_sq = 0.5 * an * sq;
    const double half_an_sq_s_ct_pref = 0.5 * an * sq * SQRT_PI;

    double Cp = 0.0;
    double Cts = 0.0;
    for (int j = 0; j < CLL_NSPEC; ++j) {
        const CllSpeciesFlow& sp = mix.sp[j];
        if (!sp.active || sp.s * ct <= -4.0) continue;
        const double gamma = sp.s * ct;
        const bool hi = gamma > 6.0;
        const double Z = hi ? 2.0 : 1.0 + std::erf(gamma);
        const double E = hi ? 0.0 : std::exp(-gamma * gamma);
        const double G2 = (E / SQRT_PI) + gamma * Z;
        const double Ctau_over_sin = at * sp.inv_s * G2;
        double Cpj = 0.0;
        if (sp.schaaf) {
            Cpj = sp.inv_s2 * ((two_m * sp.s / SQRT_PI * ct + half_an_sq) * E +
                               (two_m * (0.5 + gamma * gamma) +
                                half_an_sq_s_ct_pref * sp.s * ct) *
                                   Z);
        } else {
            const double G1 = (gamma * E / SQRT_PI) + 0.5 * (1.0 + 2.0 * gamma * gamma) * Z;
            Cpj = sp.inv_s2 * (sp.one_sqrt * G1 + sp.reem_g2 * G2);
        }
        Cp += sp.weight * Cpj;
        Cts += sp.weight * Ctau_over_sin;
    }
    return {Cp / mix.Mavg, Cts / mix.Mavg};
}

std::pair<double, double> cll_cs(double cos_t, double s, double Tw_Ti, double alpha_n,
                                 double alpha_t, int species) {
    if (species < 0 || species >= CLL_NSPEC) species = 1;
    double chi[CLL_NSPEC] = {};
    chi[species] = 1.0;
    return cll_mix_cs(cos_t, make_cll_mix(s, 0.0, 0.0, Tw_Ti, alpha_n, alpha_t, chi));
}

std::pair<double, double> cll(double theta, double s, double Tw_Ti, double alpha_n,
                              double alpha_t) {
    const auto [Cp, Ct_over_sin] =
        cll_cs(std::cos(theta), s, Tw_Ti, alpha_n, alpha_t, 1);
    return {Cp, std::sin(theta) * Ct_over_sin};
}

std::pair<double, double> cll_mix_cs(double cos_t, double s_mix, double T, double vmag,
                                     double Tw_Ti, double alpha_n, double alpha_t,
                                     const double* chi) {
    return cll_mix_cs(cos_t, make_cll_mix(s_mix, T, vmag, Tw_Ti, alpha_n, alpha_t, chi));
}

}
