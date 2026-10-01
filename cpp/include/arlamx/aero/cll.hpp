// Walker-CLL plate coefficients interface.
#pragma once

#include "arlamx/types.hpp"

#include <utility>

namespace arlamx {

// Walker–Mehta–Koller 2014 modified closed-form CLL (ADBSat / arXiv:2104.05543
// eqs. 11–15). Not a Monte-Carlo sample of the Cercignani–Lampis kernel.
// Ar is omitted in ADBSat; we reuse the N2 Walker row (documented fallback).
// α_N = 1 takes the Schaaf–Chambre branch (eq. 9), identical to Sentman α_E = 1.

// Walker eq. 13 exceeds eq. 9 by ~7.6 % only for 1 - alpha_N < ~1e-4 (docs/modules/17_aero_cll.md).
constexpr int CLL_NSPEC = 7;

// Angle-independent Walker / Schaaf factors, hoisted out of the panel loop.
struct CllSpeciesFlow {
    double s = 0.0;
    double inv_s = 0.0;
    double inv_s2 = 0.0;
    double reem_g2 = 0.0;
    double one_sqrt = 0.0;
    double weight = 0.0;
    bool active = false;
    bool schaaf = false;
};

struct CllMix {
    CllSpeciesFlow sp[CLL_NSPEC];
    double Mavg = 0.0;
    double Tw_Ti = 1.0;
    double alpha_n = 1.0;
    double alpha_t = 1.0;
};

CllMix make_cll_mix(double s_mix, double T, double vmag, double Tw_Ti,
                    double alpha_n, double alpha_t, const double* chi);

std::pair<double, double> cll(double theta, double s, double Tw_Ti, double alpha_n,
                              double alpha_t);

std::pair<double, double> cll_cs(double cos_t, double s, double Tw_Ti, double alpha_n,
                                 double alpha_t, int species);

// Mass-fraction mix (ADBSat eq. 15). chi may be null → atomic-O, mixture s.
std::pair<double, double> cll_mix_cs(double cos_t, double s_mix, double T, double vmag,
                                     double Tw_Ti, double alpha_n, double alpha_t,
                                     const double* chi);

std::pair<double, double> cll_mix_cs(double cos_t, const CllMix& mix);

}
