// SimParams and the Simulator (plant step) interface.
#pragma once

#include "arlamx/attitude/integrate.hpp"
#include "arlamx/attitude/mrp.hpp"
#include "arlamx/control/bdot.hpp"
#include "arlamx/control/magnetorquer.hpp"
#include "arlamx/control/mrp_feedback.hpp"
#include "arlamx/control/quat_feedback.hpp"
#include "arlamx/mag/field.hpp"
#include "arlamx/orbit/frames.hpp"
#include "arlamx/orbit/gravity.hpp"
#include "arlamx/orbit/third_body.hpp"
#include "arlamx/types.hpp"

#include <string>
#include <vector>

namespace arlamx {

struct SimParams {
    double mass = 0.625;
    Mat3 inertia = Mat3::diag(0.0125, 0.0125, 0.025);
    double dt_s = 2.0;
    double advisor_step_s = 300.0;
    double rk4_step_s = 2.0;
    int sh_degree = 4;
    bool lunisolar = false;
    bool use_panel_srp = true;
    bool corotating = true;
    double alpha_E = 0.93;
    double alpha_n = 1.0;  // CLL normal-energy accommodation (Walker α_N)
    double alpha_t = 1.0;
    std::string gsi = "sentman";
    double T_w = 300.0;
    double Cr = 1.8;
    bool srp_optical = false;
    double srp_ca = 1.0;
    double srp_cs = 0.0;
    double srp_cd = 0.0;
    double srp_scale = 1.0;
    bool earth_rad = false;
    double ir_ca = 0.85;
    double ir_cs = 0.0;
    double ir_cd = 0.15;
    bool gravity_gradient = true;
    double mtq_duty = 1.0;
    std::string cmd_frame = "inertial";  // "inertial" | "flow" | "flowB" | "flowS": command held in the flow frame (ZOH), target
                                         // re-evaluated every substep with orbit-rate feed-forward        // coil on-fraction per substep (rest = magnetometer window)
    Vec3 wind_N = {};
    bool one_sided_ref = true;
    bool ideal_torque = false;
    std::string attitude_law = "mrp";
    double max_slew_rad = 45.0 * 3.141592653589793 / 180.0;
    std::string mode = "point";
    double epoch_jd = 2461060.5;  // 2026-01-15 00:00
    std::string ggm_path;
    std::string wmm_path;
};

class Simulator {
public:
    explicit Simulator(SimParams p = {});

    bool load_gravity(const std::string& ggm_path);
    bool load_wmm(const std::string& wmm_path);
    bool load_spice(const std::vector<std::string>& kernels);

    void set_panels(PanelSoA p);
    void set_mass(double m);
    void set_inertia(Mat3 I);
    void set_atmosphere(Atmo a);
    void set_atmosphere_end(Atmo a);
    void set_mode(const std::string& mode);
    void set_srp_scale(double s);
    void set_wind(Vec3 w) { p_.wind_N = w; }
    void set_cmd_frame(const std::string& f);
    // Onboard nav: flow-frame targets from (r, v) at the current time (two-body RK4) instead of truth.
    void set_nav(Vec3 r, Vec3 v);
    void clear_nav() { nav_on_ = false; }
    void set_controller(const MRPFeedback& c) { ctrl_ = c; }
    void set_quat_controller(const QuaternionFeedback& c) { qctrl_ = c; }
    void set_bdot(const BDot& b) {
        bdot_ = b;
        bdot_.max_dipole = 1e300;
    }
    void set_dipole_max(Vec3 m_max) { dipole_max_ = m_max; }
    void set_rods(const std::vector<Vec3>& axis, const std::vector<double>& dmax);
    void set_rod_gates(const std::vector<char>& on);
    const RodArray& rods() const { return rods_; }
    Vec3 dipole_max() const { return dipole_max_; }

    State12 reset(Vec3 r, Vec3 v, Vec3 sigma, Vec3 omega);
    StepOut step(Quat q_cmd);

    const State12& state() const { return y_; }
    const SimParams& params() const { return p_; }
    bool gravity_loaded() const { return grav_.loaded(); }
    bool wmm_loaded() const { return wmm_.loaded(); }
    Vec3 sun_hat(double jd) const { return eph_.sun_hat(jd); }

    GravityHarmonics& gravity() { return grav_; }
    const GravityHarmonics& gravity() const { return grav_; }

private:
    Vec3 accel_gravity_env(Vec3 r_N, double jd) const;
    double mu() const;
    Vec3 field_body(const Mat3& C_BN, Vec3 r_N, double jd) const;

    SimParams p_;
    PanelSoA panels_;
    Mat3 Iinv_{};
    State12 y_{};
    Atmo atmo_{};
    Atmo atmo_end_{};
    bool has_end_ = false;
    Quat q_held_{1, 0, 0, 0};
    GravityHarmonics grav_{};
    Ephemeris eph_{};
    Vec3 sun_r_{}, moon_r_{};
    WMM wmm_{};
    MRPFeedback ctrl_{};
    QuaternionFeedback qctrl_{};
    BDot bdot_{};
    Vec3 dipole_max_{1.473, 1.473, 0.442};
    RodArray rods_{};
    bool nav_on_ = false;
    Vec3 nav_r_{}, nav_v_{};
    double nav_t_ = 0.0;
};

Vec3 min_drag_freestream(double vmag, int axis);

}
