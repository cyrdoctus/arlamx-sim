// Simulator: one advisor step of orbit + attitude with aero, SRP, Earth radiation, gravity gradient and magnetic control.
#include "arlamx/api.hpp"

#include "arlamx/aero/sentman.hpp"
#include "arlamx/constants.hpp"
#include "arlamx/srp/panel_srp.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>

namespace arlamx {

Vec3 min_drag_freestream(double vmag, int axis) {
    Vec3 v{};
    const int a = std::max(0, std::min(2, axis));
    v[a] = vmag;
    return v;
}

static double decimal_year_from_jd(double jd) {
    return 2000.0 + (jd - 2451545.0) / 365.25;
}

static int min_drag_axis(const PanelSoA& p) {
    double best = 1e300;
    int ax = 0;
    for (int k = 0; k < 3; ++k) {
        Vec3 d{};
        d[k] = 1.0;
        double A = 0.0;
        for (std::size_t i = 0; i < p.size(); ++i) {
            A += p.area[i] * std::max(0.0, dot(p.normal(i), d));
        }
        if (A < best) {
            best = A;
            ax = k;
        }
    }
    return ax;
}

static void check_sim_params(SimParams& p) {
    if (!std::isfinite(p.mass) || p.mass <= 0.0) {
        throw std::invalid_argument("mass must be finite and positive");
    }
    if (!std::isfinite(p.dt_s) || p.dt_s <= 0.0 || p.dt_s > 300.0) {
        throw std::invalid_argument("dt_s must be in (0, 300] s");
    }
    if (!std::isfinite(p.rk4_step_s) || p.rk4_step_s <= 0.0 || p.rk4_step_s > 120.0) {
        throw std::invalid_argument("rk4_step_s must be in (0, 120] s");
    }
    if (!std::isfinite(p.advisor_step_s) || p.advisor_step_s <= 0.0 || p.advisor_step_s > 3600.0) {
        throw std::invalid_argument("advisor_step_s must be in (0, 3600] s");
    }
    for (char& c : p.attitude_law) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (p.attitude_law != "mrp" && p.attitude_law != "quaternion") {
        throw std::invalid_argument("attitude_law must be 'mrp' or 'quaternion'");
    }
    for (char& c : p.gsi) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (p.cmd_frame != "inertial" && p.cmd_frame != "flow" && p.cmd_frame != "flowB" && p.cmd_frame != "flowS") {
        throw std::invalid_argument("cmd_frame must be 'inertial', 'flow', 'flowB' or 'flowS'");
    }
    if (!(p.mtq_duty > 0.0 && p.mtq_duty <= 1.0)) {
        throw std::invalid_argument("mtq_duty must be in (0, 1]");
    }
    if (p.gsi != "sentman" && p.gsi != "cll") {
        throw std::invalid_argument("gsi must be 'sentman' or 'cll'");
    }
}

Simulator::Simulator(SimParams p) : p_(std::move(p)) {
    check_sim_params(p_);
    Iinv_ = inverse_sym(p_.inertia);
    ctrl_.inertia = p_.inertia;
    qctrl_.inertia = p_.inertia;
    bdot_.max_dipole = 1e300;
    if (!p_.ggm_path.empty()) {
        const std::string ggm = p_.ggm_path;
        if (!load_gravity(ggm)) {
            throw std::runtime_error("GGM failed to load from '" + ggm + "'");
        }
    }
    if (!p_.wmm_path.empty()) load_wmm(p_.wmm_path);
}

bool Simulator::load_gravity(const std::string& ggm_path) {
    if (!grav_.load_ggm(ggm_path, std::max(p_.sh_degree, 2))) {
        p_.ggm_path.clear();
        return false;
    }
    p_.ggm_path = ggm_path;
    return true;
}

bool Simulator::load_wmm(const std::string& wmm_path) {
    if (!wmm_.load(wmm_path)) {
        p_.wmm_path.clear();
        return false;
    }
    p_.wmm_path = wmm_path;
    return true;
}

bool Simulator::load_spice(const std::vector<std::string>& kernels) { return eph_.load_spice(kernels); }

void Simulator::set_panels(PanelSoA p) { panels_ = std::move(p); }
void Simulator::set_mass(double m) { p_.mass = m; }
void Simulator::set_inertia(Mat3 I) {
    p_.inertia = I;
    Iinv_ = inverse_sym(I);
    ctrl_.inertia = I;
    qctrl_.inertia = I;
}
void Simulator::set_atmosphere(Atmo a) {
    if (!std::isfinite(a.rho) || !std::isfinite(a.T) || !std::isfinite(a.m_bar) ||
        a.rho < 0.0 || a.T <= 0.0 || a.m_bar <= 0.0) {
        throw std::invalid_argument("infeasible atmosphere (need rho>=0, T>0, m_bar>0)");
    }
    atmo_ = a;
    has_end_ = false;
}

void Simulator::set_atmosphere_end(Atmo a) {
    if (!std::isfinite(a.rho) || !std::isfinite(a.T) || !std::isfinite(a.m_bar) ||
        a.rho < 0.0 || a.T <= 0.0 || a.m_bar <= 0.0) {
        throw std::invalid_argument("infeasible atmosphere (need rho>=0, T>0, m_bar>0)");
    }
    atmo_end_ = a;
    has_end_ = true;
}

static Atmo lerp_atmo(const Atmo& a, const Atmo& b, double f) {
    Atmo o = a;
    o.rho = (1.0 - f) * a.rho + f * b.rho;
    o.T = (1.0 - f) * a.T + f * b.T;
    o.m_bar = (1.0 - f) * a.m_bar + f * b.m_bar;
    o.has_species = a.has_species && b.has_species;
    for (int i = 0; i < 7; ++i) o.chi[i] = (1.0 - f) * a.chi[i] + f * b.chi[i];
    return o;
}
void Simulator::set_mode(const std::string& mode) {
    if (mode != "point" && mode != "detumble" && mode != "prescribed") {
        throw std::invalid_argument("mode must be point, detumble, or prescribed");
    }
    if (mode != p_.mode) bdot_.reset();
    p_.mode = mode;
}

void Simulator::set_rods(const std::vector<Vec3>& axis, const std::vector<double>& dmax) {
    if (axis.size() != dmax.size()) throw std::invalid_argument("rod axis/dmax size mismatch");
    rods_.axis.clear();
    for (const Vec3& a : axis) {
        if (!(norm(a) > 0.0)) throw std::invalid_argument("rod axis must be non-zero");
        rods_.axis.push_back(unit(a));
    }
    for (double d : dmax) {
        if (!(d >= 0.0) || !std::isfinite(d)) throw std::invalid_argument("rod dmax must be >= 0");
    }
    rods_.dmax = dmax;
    rods_.on.assign(axis.size(), 1);
}

void Simulator::set_rod_gates(const std::vector<char>& on) {
    if (on.size() != rods_.size()) throw std::invalid_argument("rod gate count mismatch");
    rods_.on = on;
}

void Simulator::set_cmd_frame(const std::string& f) {
    if (f != "inertial" && f != "flow" && f != "flowB" && f != "flowS") {
        throw std::invalid_argument("cmd_frame must be 'inertial', 'flow', 'flowB' or 'flowS'");
    }
    p_.cmd_frame = f;
}

void Simulator::set_nav(Vec3 r, Vec3 v) {
    if (!std::isfinite(norm(r)) || !std::isfinite(norm(v)) || norm(r) < 0.5 * RE_WGS) {
        throw std::invalid_argument("nav state must be finite and outside the Earth");
    }
    nav_r_ = r;
    nav_v_ = v;
    nav_t_ = y_.t;
    nav_on_ = true;
}

void Simulator::set_srp_scale(double s) {
    if (!std::isfinite(s) || s < 0.0 || s > 20.0) {
        throw std::invalid_argument("srp_scale must be in [0, 20]");
    }
    p_.srp_scale = s;
}

State12 Simulator::reset(Vec3 r, Vec3 v, Vec3 sigma, Vec3 omega) {
    if (!std::isfinite(norm(r)) || !std::isfinite(norm(v)) || !std::isfinite(norm(sigma)) ||
        !std::isfinite(norm(omega))) {
        throw std::invalid_argument("reset state contains a non-finite component");
    }
    if (norm(r) < 0.5 * RE_WGS) {
        throw std::invalid_argument("reset |r| is inside the Earth");
    }
    y_.r = r;
    y_.v = v;
    y_.sigma = mrp_shadow(sigma);
    y_.omega = omega;
    y_.t = 0.0;
    q_held_ = {1.0, 0.0, 0.0, 0.0};
    bdot_.reset();
    nav_on_ = false;
    return y_;
}

Vec3 Simulator::accel_gravity_env(Vec3 r_N, double jd) const {
    Vec3 a;
    if (grav_.loaded()) {
        a = accel_gravity_N(grav_, r_N, gmst_rad(jd), p_.sh_degree);
    } else if (p_.sh_degree < 2) {
        a = accel_twobody(r_N, MU_WGS);
    } else {
        a = accel_twobody(r_N, MU_WGS) + accel_j2(r_N, MU_WGS, RE_WGS, J2_GGM) +
            accel_j3(r_N, MU_WGS, RE_WGS, J3_EGM);
    }
    (void)jd;
    if (p_.lunisolar) {
        a += accel_third_body(r_N, sun_r_, MU_SUN);
        a += accel_third_body(r_N, moon_r_, MU_MOON);
    }
    return a;
}

double Simulator::mu() const { return grav_.loaded() ? grav_.mu() : MU_WGS; }

static double geo_alt_km(Vec3 r_N) {
    double lat = 0.0, lon = 0.0, h = 0.0;
    ecef_to_geodetic(r_N, lat, lon, h);
    return h / 1e3;
}

Vec3 Simulator::field_body(const Mat3& C_BN, Vec3 r_N, double jd) const {
    const Mat3 EN = dcm_EN(gmst_rad(jd));
    const Vec3 r_E = mul(EN, r_N);
    const Vec3 B_E = wmm_.loaded() ? wmm_.field_ecef(r_E, decimal_year_from_jd(jd))
                                   : dipole_field_ecef(r_E);
    return mul(C_BN, mul(transpose(EN), B_E));
}

// Per substep: F_B = aero + SRP + Earth radiation, tau_B = the same moments + gravity gradient
// + control; classical RK4 (Montenbruck & Gill 2000, Sec. 4.1); work dE = F.v dt / m.
StepOut Simulator::step(Quat q_cmd) {
    if (!std::isfinite(q_cmd[0]) || !std::isfinite(q_cmd[1]) || !std::isfinite(q_cmd[2]) ||
        !std::isfinite(q_cmd[3])) {
        q_cmd = {1.0, 0.0, 0.0, 0.0};
    }
    const Quat q_clip = slerp_clip(q_cmd, q_held_, p_.max_slew_rad);
    q_held_ = q_clip;
    const Vec3 sigma_cmd = quat_to_mrp(q_clip);
    const bool prescribed = (p_.mode == "prescribed");
    const bool detumble = (p_.mode == "detumble");
    const bool flow = (p_.cmd_frame != "inertial");
    const bool flowB = (p_.cmd_frame == "flowB");
    const bool flowS = (p_.cmd_frame == "flowS");
    // Flow frame F: e1 = v_rel, e3 = orbit normal (orthogonalized), e2 = e3 x e1; C_BN = C_BF C_FN,
    // feed-forward w = C_BN (h / r^2) (orbit rate).
    // Two-body a = -mu r / |r|^3 (Vallado 2013, ch. 1), RK4 in <= 10 s steps, nav clock kept monotonic.
    auto nav_to = [&](double t) {
        auto acc = [](Vec3 r) { const double n = norm(r); return r * (-MU_WGS / (n * n * n)); };
        while (t - nav_t_ > 1e-9) {
            const double h = std::min(10.0, t - nav_t_);
            const Vec3 r = nav_r_, v = nav_v_;
            const Vec3 k1r = v, k1v = acc(r);
            const Vec3 k2r = v + k1v * (0.5 * h), k2v = acc(r + k1r * (0.5 * h));
            const Vec3 k3r = v + k2v * (0.5 * h), k3v = acc(r + k2r * (0.5 * h));
            const Vec3 k4r = v + k3v * h, k4v = acc(r + k3r * h);
            nav_r_ = r + (k1r + k2r * 2.0 + k3r * 2.0 + k4r) * (h / 6.0);
            nav_v_ = v + (k1v + k2v * 2.0 + k3v * 2.0 + k4v) * (h / 6.0);
            nav_t_ += h;
        }
    };
    auto target = [&](const State12& y_true, Vec3& sig, Vec3& om) {
        if (!flow) {
            sig = sigma_cmd;
            om = Vec3::zero();
            return;
        }
        State12 y = y_true;
        if (nav_on_) {
            nav_to(y_true.t);
            y.r = nav_r_;
            y.v = nav_v_;
        }
        Vec3 vr = y.v;
        if (p_.corotating) vr = y.v - cross(Vec3{0.0, 0.0, OMEGA_EARTH}, y.r);
        vr = vr - p_.wind_N;
        const Vec3 h = cross(y.r, y.v);
        const Vec3 e1 = unit(vr);
        // flowB: e3 = field component normal to the flow (membrane normal on B_perp maximizes
        // in-plane rod authority); falls back to the orbit normal when B is along the flow.
        Vec3 ref = h;
        if (flowB) {
            const double jdy = p_.epoch_jd + y.t / 86400.0;
            const Mat3 EN = dcm_EN(gmst_rad(jdy));
            const Vec3 rE = mul(EN, y.r);
            const Vec3 BN = mul(transpose(EN), wmm_.loaded() ? wmm_.field_ecef(rE, decimal_year_from_jd(jdy))
                                                            : dipole_field_ecef(rE));
            const Vec3 bp = BN - dot(BN, e1) * e1;
            if (norm(bp) > 0.2 * norm(BN)) ref = dot(bp, h) >= 0.0 ? bp : bp * (-1.0);
        }
        // flowS: e3 = Sun component normal to the flow (min drag with the cells rolled to the Sun).
        if (flowS) {
            const Vec3 sN = eph_.sun_hat(p_.epoch_jd + y.t / 86400.0);
            const Vec3 sp = sN - dot(sN, e1) * e1;
            if (norm(sp) > 0.2) ref = dot(sp, h) >= 0.0 ? sp : sp * (-1.0);
        }
        const Vec3 e3 = unit(ref - dot(ref, e1) * e1);
        const Vec3 e2 = cross(e3, e1);
        Mat3 CFN;
        for (int k = 0; k < 3; ++k) {
            CFN(0, k) = e1[k];
            CFN(1, k) = e2[k];
            CFN(2, k) = e3[k];
        }
        const Mat3 CBN = mul(mrp_to_dcm(sigma_cmd), CFN);
        sig = mrp_shadow(quat_to_mrp(dcm_to_quat(CBN)));
        om = mul(CBN, h / norm2(y.r));
    };
    Vec3 sigma_tgt, omega_tgt;
    target(y_, sigma_tgt, omega_tgt);
    if (prescribed) {
        y_.sigma = mrp_shadow(sigma_tgt);
        y_.omega = Vec3::zero();
    }

    const int nsub = std::max(1, static_cast<int>(std::llround(p_.advisor_step_s / p_.dt_s)));
    const double h_max = std::max(1e-6, p_.rk4_step_s);

    double dE_a = 0.0, dE_b = 0.0, dE_drag = 0.0, dE_lift = 0.0, drag_sum = 0.0, trk = 0.0;
    Vec3 tau_ctrl_sum = Vec3::zero();
    Vec3 tau_aero_sum = Vec3::zero();
    Vec3 m_sum = Vec3::zero();
    Vec3 tau_cmd_sum = Vec3::zero();
    Vec3 tau_demand_sum = Vec3::zero();
    Vec3 tau_demand_abs_sum = Vec3::zero();
    Vec3 gyro_int = Vec3::zero();
    double shortfall_sum = 0.0;
    Vec3 tau_srp_sum = Vec3::zero();
    Vec3 tau_erp_sum = Vec3::zero();
    Vec3 tau_gg_sum = Vec3::zero();
    double a_srp_sum = 0.0;
    int n_sun = 0;
    const std::size_t n_rod = rods_.size();
    std::vector<double> duty2_sum(n_rod, 0.0), d_rod;
    AeroResult last{};
    int done = 0;
    const int ax_min = min_drag_axis(panels_);
    double cd_base = -1.0, aref_base = 0.0, s_base = 0.0;

    GsiParams gsi;
    gsi.model = parse_gsi(p_.gsi);
    gsi.alpha_E = p_.alpha_E;
    gsi.alpha_n = p_.alpha_n;
    gsi.alpha_t = p_.alpha_t;

    if (p_.lunisolar) {
        const double jd0 = p_.epoch_jd + y_.t / 86400.0;
        sun_r_ = eph_.sun_pos(jd0);
        moon_r_ = eph_.moon_pos(jd0);
    }
    SailOptics sol;
    sol.optical = true;
    sol.ca = p_.srp_ca;
    sol.cs = p_.srp_cs;
    sol.cd = p_.srp_cd;
    SailOptics ir;
    ir.optical = true;
    ir.ca = p_.ir_ca;
    ir.cs = p_.ir_cs;
    ir.cd = p_.ir_cd;

    for (int s = 0; s < nsub; ++s) {
        if (!std::isfinite(norm(y_.r)) || !std::isfinite(norm(y_.v))) break;
        target(y_, sigma_tgt, omega_tgt);
        if (prescribed) {
            y_.sigma = mrp_shadow(sigma_tgt);
            y_.omega = Vec3::zero();
        }
        const double jd = p_.epoch_jd + y_.t / 86400.0;
        const Mat3 C = mrp_to_dcm(y_.sigma);
        const Vec3 sun_pos = eph_.sun_pos(jd);
        const double sun_d = norm(sun_pos);
        const Vec3 sun_N = unit(sun_pos);
        const Vec3 sun_B = mul(C, sun_N);
        const double ecl = eclipse_cylindrical(y_.r, sun_N, RE_WGS);
        // Solar pressure scales with (1 AU / d)^2 (Montenbruck & Gill 2000, Sec. 3.4).
        const double p_sun = P_SRP_1AU * (AU / sun_d) * (AU / sun_d) * p_.srp_scale;

        Vec3 v_rel = y_.v;
        if (p_.corotating) {
            const Vec3 we{0.0, 0.0, OMEGA_EARTH};
            v_rel = y_.v - cross(we, y_.r);
        }
        v_rel = v_rel - p_.wind_N;
        const Vec3 v_B_gas = mul(C, v_rel) * (-1.0);
        const double vrel = norm(v_rel);

        const Atmo at = has_end_ ? lerp_atmo(atmo_, atmo_end_, double(s) / nsub) : atmo_;
        AeroResult aero{};
        double Fbase = 0.0;
        if (at.T >= 1.0 && at.rho > 0.0 && at.m_bar > 0.0 && vrel > 1.0 &&
            panels_.size() > 0) {
            const double* chi = at.has_species ? at.chi : nullptr;
            aero = spacecraft_aero(panels_, v_B_gas, at.rho, at.T, at.m_bar, p_.T_w,
                                   gsi, p_.one_sided_ref, chi);
            // Speed ratio s = |v| / sqrt(2kT/m) (Sentman 1961).
            const double s_now = vrel / std::sqrt(2.0 * K_B * at.T / at.m_bar);
            if (cd_base < 0.0 || std::fabs(s_now - s_base) > 2e-4 * s_base) {
                const Vec3 vbase = min_drag_freestream(vrel, ax_min);
                const AeroResult cf =
                    coefficients_only(panels_, vbase, at.rho, at.T, at.m_bar, p_.T_w,
                                      gsi, p_.one_sided_ref, chi);
                cd_base = cf.Cd;
                aref_base = cf.A_ref;
                s_base = s_now;
            }
            const double qdyn = 0.5 * at.rho * vrel * vrel;
            Fbase = cd_base * qdyn * aref_base;
        }

        Vec3 Fsrp = Vec3::zero();
        Vec3 tau_srp = Vec3::zero();
        if (p_.use_panel_srp) {
            const auto ft = p_.srp_optical
                                ? panel_srp_optical(panels_, sun_B, ecl, p_sun, sol)
                                : panel_srp_force_torque(panels_, sun_B, ecl, p_sun, p_.Cr);
            Fsrp = ft.first;
            tau_srp = ft.second;
        }
        const double rn = norm(y_.r);
        const Vec3 nadir_B = mul(C, y_.r * (-1.0 / rn));
        Vec3 Ferp = Vec3::zero();
        Vec3 tau_erp = Vec3::zero();
        if (p_.earth_rad) {
            const auto ft = earth_rad(panels_, nadir_B, rn, dot(y_.r / rn, sun_N), p_sun, sol, ir);
            Ferp = ft.first;
            tau_erp = ft.second;
        }
        const Vec3 tau_gg =
            p_.gravity_gradient ? gg_torque(p_.inertia, nadir_B * rn, mu()) : Vec3::zero();

        Vec3 tau_ctrl = Vec3::zero();
        Vec3 tau_cmd = Vec3::zero();
        Vec3 tau_demand = Vec3::zero();
        Vec3 m_used = Vec3::zero();
        double shortfall = 0.0;
        if (prescribed) {
            tau_ctrl = Vec3::zero();
        } else if (detumble) {
            const Vec3 B_B = field_body(C, y_.r, jd);
            bdot_.dt = p_.dt_s;
            const Vec3 m_raw = bdot_.dipole(B_B);
            m_used = n_rod ? rods_allocate(m_raw, rods_, d_rod) : saturate_dipole(m_raw, dipole_max_);
            tau_cmd = BDot::torque(m_raw, B_B);
            tau_demand = tau_cmd;
            tau_ctrl = BDot::torque(m_used, B_B) * p_.mtq_duty;
            const double tn = norm(tau_cmd);
            shortfall = (tn < 1e-18) ? 0.0 : std::max(0.0, std::min(1.0, 1.0 - norm(tau_ctrl) / tn));
        } else {
            const Vec3 B_B = field_body(C, y_.r, jd);
            if (p_.attitude_law == "quaternion") {
                const Quat q_body = mrp_to_quat(y_.sigma);
                tau_cmd = qctrl_.compute(q_body, y_.omega, flow ? mrp_to_quat(sigma_tgt) : q_clip, omega_tgt, p_.dt_s,
                                         &tau_demand);
            } else {
                tau_cmd = ctrl_.compute(y_.sigma, y_.omega, sigma_tgt, omega_tgt, p_.dt_s,
                                        &tau_demand);
            }
            const MagnetorquerOut mtq = n_rod ? apply_rods(tau_cmd, B_B, rods_, d_rod)
                                              : apply_magnetorquer(tau_cmd, B_B, dipole_max_);
            m_used = mtq.m;
            shortfall = mtq.shortfall_frac;
            tau_ctrl = p_.ideal_torque ? tau_cmd : mtq.tau * p_.mtq_duty;
        }

        const Vec3 F_B = aero.F + Fsrp + Ferp;
        const Vec3 tau_B = aero.tau + tau_srp + tau_erp + tau_gg + tau_ctrl;
        tau_ctrl_sum += tau_ctrl;
        tau_aero_sum += aero.tau;
        if (n_rod && !prescribed && d_rod.size() == n_rod && !(p_.ideal_torque && !detumble)) {
            for (std::size_t k = 0; k < n_rod; ++k) {
                const double u = rods_.dmax[k] > 0.0 ? d_rod[k] / rods_.dmax[k] : 0.0;
                duty2_sum[k] += p_.mtq_duty * u * u;
            }
        }
        tau_srp_sum += tau_srp;
        tau_erp_sum += tau_erp;
        tau_gg_sum += tau_gg;
        m_sum += m_used;
        tau_cmd_sum += tau_cmd;
        tau_demand_sum += tau_demand;
        tau_demand_abs_sum += Vec3{std::fabs(tau_demand.x), std::fabs(tau_demand.y),
                                   std::fabs(tau_demand.z)};
        shortfall_sum += shortfall;

        const Vec3 serr = mrp_error(y_.sigma, sigma_tgt, ctrl_.proper);
        trk += mrp_angle(serr);

        const Vec3 Fa_N = mul(transpose(C), aero.F);
        const Vec3 Fs_N = mul(transpose(C), Fsrp);
        const Vec3 Fe_N = mul(transpose(C), Ferp);
        dE_a += (dot(Fa_N, y_.v) + dot(Fs_N, y_.v) + dot(Fe_N, y_.v)) * p_.dt_s / p_.mass;
        if (ecl > 0.0 && p_.use_panel_srp) {
            a_srp_sum += dot(Fs_N, unit(y_.v)) / (ecl * p_.mass);
            ++n_sun;
        }
        if (vrel > 1.0) {
            const Vec3 vhat_rel = v_rel / vrel;
            dE_b += -Fbase * dot(vhat_rel, y_.v) * p_.dt_s / p_.mass;
            const double f_along = dot(Fa_N, vhat_rel);
            const Vec3 Fdrag_N = f_along * vhat_rel;
            const Vec3 Flift_N = Fa_N - Fdrag_N;
            dE_drag += dot(Fdrag_N, y_.v) * p_.dt_s / p_.mass;
            dE_lift += dot(Flift_N, y_.v) * p_.dt_s / p_.mass;
            drag_sum += std::max(0.0, -f_along);
        }
        last = aero;

        const int nint = std::max(1, static_cast<int>(std::ceil(p_.dt_s / h_max)));
        const double h = p_.dt_s / nint;
        for (int k = 0; k < nint; ++k) {
            auto deriv = [&](const State12& s) {
                const Mat3 Cs = mrp_to_dcm(s.sigma);
                const double jdk = p_.epoch_jd + s.t / 86400.0;
                const Vec3 ag = accel_gravity_env(s.r, jdk);
                const Vec3 aext = mul(transpose(Cs), F_B) / p_.mass;
                State12 d;
                d.r = s.v;
                d.v = ag + aext;
                if (prescribed) {
                    d.sigma = Vec3::zero();
                    d.omega = Vec3::zero();
                } else {
                    d.sigma = mrp_rate(s.sigma, s.omega);
                    d.omega = omega_dot(p_.inertia, Iinv_, s.omega, tau_B);
                }
                d.t = 1.0;
                return d;
            };
            auto add = [](const State12& y, const State12& k, double s) {
                State12 o;
                o.r = y.r + k.r * s;
                o.v = y.v + k.v * s;
                o.sigma = y.sigma + k.sigma * s;
                o.omega = y.omega + k.omega * s;
                o.t = y.t + s;
                return o;
            };
            const State12 k1 = deriv(y_);
            const State12 s2 = add(y_, k1, 0.5 * h);
            const State12 k2 = deriv(s2);
            const State12 s3 = add(y_, k2, 0.5 * h);
            const State12 k3 = deriv(s3);
            const State12 s4 = add(y_, k3, h);
            const State12 k4 = deriv(s4);
            if (!prescribed) {
                auto gyr = [&](const State12& s) {
                    return cross(s.omega, mul(p_.inertia, s.omega));
                };
                gyro_int += (h / 6.0) * (gyr(y_) + 2.0 * gyr(s2) + 2.0 * gyr(s3) + gyr(s4));
            }
            y_.r = y_.r + (h / 6.0) * (k1.r + 2.0 * k2.r + 2.0 * k3.r + k4.r);
            y_.v = y_.v + (h / 6.0) * (k1.v + 2.0 * k2.v + 2.0 * k3.v + k4.v);
            y_.sigma = mrp_shadow(y_.sigma +
                                  (h / 6.0) * (k1.sigma + 2.0 * k2.sigma + 2.0 * k3.sigma + k4.sigma));
            y_.omega = y_.omega + (h / 6.0) * (k1.omega + 2.0 * k2.omega + 2.0 * k3.omega + k4.omega);
            y_.t += h;
            if (prescribed) {
                y_.sigma = mrp_shadow(sigma_tgt);
                y_.omega = Vec3::zero();
            }
        }
        ++done;
        if (geo_alt_km(y_.r) < 80.0) break;
    }

    if (has_end_) {
        atmo_ = atmo_end_;
        has_end_ = false;
    }
    const double jd = p_.epoch_jd + y_.t / 86400.0;
    const Mat3 C = mrp_to_dcm(y_.sigma);
    StepOut o;
    o.y = y_;
    o.altitude_km = geo_alt_km(y_.r);
    o.sma_m = sma_from_rv(y_.r, y_.v, grav_.loaded() ? grav_.mu() : MU_WGS);
    o.eclipse = eclipse_cylindrical(y_.r, eph_.sun_hat(jd), RE_WGS);
    o.sun_B = mul(C, eph_.sun_hat(jd));
    o.B_B = field_body(C, y_.r, jd);
    o.Cd = last.Cd;
    o.Cl = last.Cl;
    o.drag_N = drag_sum / std::max(1, done);
    o.dE_actual = dE_a;
    o.dE_baseline = dE_b;
    o.dE_drag = dE_drag;
    o.dE_lift = dE_lift;
    o.tracking_err_rad = trk / std::max(1, done);
    o.n_substeps = done;
    const double inv_done = 1.0 / std::max(1, done);
    o.tau_ctrl_mean = tau_ctrl_sum * inv_done;
    o.tau_aero_mean = tau_aero_sum * inv_done;
    o.m_mean = m_sum * inv_done;
    o.tau_cmd_mean = tau_cmd_sum * inv_done;
    o.tau_shortfall_frac = shortfall_sum * inv_done;
    o.tau_demand_mean = tau_demand_sum * inv_done;
    o.tau_demand_absmean = tau_demand_abs_sum * inv_done;
    o.gyro_mean = gyro_int / (std::max(1, done) * p_.dt_s);
    o.tau_srp_mean = tau_srp_sum * inv_done;
    o.tau_erp_mean = tau_erp_sum * inv_done;
    o.tau_gg_mean = tau_gg_sum * inv_done;
    o.tau_env_mean = o.tau_aero_mean + o.tau_srp_mean + o.tau_erp_mean + o.tau_gg_mean;
    o.a_srp_sun = n_sun > 0 ? a_srp_sum / n_sun : 0.0;
    o.n_sunlit = n_sun;
    o.rod_duty2_mean.assign(n_rod, 0.0);
    for (std::size_t k = 0; k < n_rod; ++k) o.rod_duty2_mean[k] = duty2_sum[k] * inv_done;
    return o;
}

}
