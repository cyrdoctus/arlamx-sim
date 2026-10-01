// pybind11 module arlamx_cpp.
#include "arlamx/aero/cll.hpp"
#include "arlamx/aero/sentman.hpp"
#include "arlamx/api.hpp"
#include "arlamx/attitude/mrp.hpp"
#include "arlamx/constants.hpp"
#include "arlamx/control/bdot.hpp"
#include "arlamx/control/mrp_feedback.hpp"
#include "arlamx/control/quat_feedback.hpp"
#include "arlamx/mag/field.hpp"
#include "arlamx/onboard/propagator_f32.hpp"
#include "arlamx/orbit/frames.hpp"
#include "arlamx/orbit/gravity.hpp"
#include "arlamx/orbit/third_body.hpp"
#include "arlamx/srp/panel_srp.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

namespace py = pybind11;
using namespace arlamx;

static Vec3 as_vec(const py::array_t<double>& a) {
    auto r = a.unchecked<1>();
    if (r.shape(0) < 3) throw std::runtime_error("expected length-3 vector");
    return {r(0), r(1), r(2)};
}

static Quat as_quat(const py::array_t<double>& a) {
    auto r = a.unchecked<1>();
    if (r.shape(0) < 4) throw std::runtime_error("expected length-4 quaternion");
    return {r(0), r(1), r(2), r(3)};
}

static py::array_t<double> to_np_q(Quat q) {
    py::array_t<double> o(4);
    auto r = o.mutable_unchecked<1>();
    for (int i = 0; i < 4; ++i) r(i) = q[i];
    return o;
}

static py::array_t<double> to_np(Vec3 v) {
    py::array_t<double> o(3);
    auto r = o.mutable_unchecked<1>();
    r(0) = v.x;
    r(1) = v.y;
    r(2) = v.z;
    return o;
}

static py::array_t<double> to_np(const Mat3& M) {
    py::array_t<double> o({3, 3});
    auto r = o.mutable_unchecked<2>();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r(i, j) = M(i, j);
    return o;
}

static PanelSoA panels_from_numpy(py::array_t<double> n, py::array_t<double> A,
                                  py::array_t<double> c) {
    auto nn = n.unchecked<2>();
    auto aa = A.unchecked<1>();
    auto cc = c.unchecked<2>();
    const auto N = static_cast<std::size_t>(aa.shape(0));
    if (nn.shape(0) != static_cast<py::ssize_t>(N) || nn.shape(1) != 3)
        throw std::runtime_error("n must be (N,3)");
    if (cc.shape(0) != static_cast<py::ssize_t>(N) || cc.shape(1) != 3)
        throw std::runtime_error("c must be (N,3)");
    PanelSoA p;
    for (std::size_t i = 0; i < N; ++i) {
        p.push({nn(i, 0), nn(i, 1), nn(i, 2)}, aa(i), {cc(i, 0), cc(i, 1), cc(i, 2)});
    }
    return p;
}

static py::dict aero_to_dict(const AeroResult& a) {
    py::dict d;
    d["force"] = to_np(a.F);
    d["torque"] = to_np(a.tau);
    d["Cd"] = a.Cd;
    d["Cl"] = a.Cl;
    d["A_ref"] = a.A_ref;
    return d;
}

PYBIND11_MODULE(arlamx_cpp, m) {
    m.doc() = "ARLAMX v2.7 C++ plant (no Basilisk dependency)";
    m.attr("__version__") = "2.7.5";
    m.attr("SH_MAX_DEGREE") = SH_MAX_DEGREE;
    m.attr("AU") = AU;
    m.attr("EARTH_ALBEDO") = EARTH_ALBEDO;
    m.attr("EARTH_EMISS") = EARTH_EMISS;
    m.attr("OMEGA_EARTH") = OMEGA_EARTH;

    m.attr("MU_GGM") = MU_GGM;
    m.attr("RE_GGM") = RE_GGM;
    m.attr("MU_WGS") = MU_WGS;
    m.attr("RE_WGS") = RE_WGS;
    m.attr("J2_GGM") = J2_GGM;
    m.attr("K_B") = K_B;
    m.attr("P_SRP_1AU") = P_SRP_1AU;

    m.def("sentman", [](double th, double s, double Tw, double aE) {
        auto p = sentman(th, s, Tw, aE);
        return py::make_tuple(p.first, p.second);
    }, py::arg("theta"), py::arg("s"), py::arg("Tw_Ti"), py::arg("alpha_E") = 1.0);

    m.def("cll", [](double th, double s, double Tw, double an, double at) {
        auto p = cll(th, s, Tw, an, at);
        return py::make_tuple(p.first, p.second);
    }, py::arg("theta"), py::arg("s"), py::arg("Tw_Ti"), py::arg("alpha_n") = 1.0,
       py::arg("alpha_t") = 1.0);

    auto aero_kwargs = [](py::array_t<double> n, py::array_t<double> A, py::array_t<double> c,
                          py::array_t<double> v, double rho, double T, double mb, double Tw,
                          double aE, bool one_sided, const std::string& gsi_name, double an,
                          double at, py::object chi_obj, bool coeffs) {
        GsiParams gsi;
        gsi.model = parse_gsi(gsi_name);
        gsi.alpha_E = aE;
        gsi.alpha_n = an;
        gsi.alpha_t = at;
        double chi[CLL_NSPEC] = {};
        const double* chi_ptr = nullptr;
        if (!chi_obj.is_none()) {
            auto arr = py::cast<py::array_t<double>>(chi_obj);
            auto r = arr.unchecked<1>();
            if (r.shape(0) < CLL_NSPEC) throw std::runtime_error("chi must have length 7");
            for (int i = 0; i < CLL_NSPEC; ++i) chi[i] = r(i);
            chi_ptr = chi;
        }
        const AeroResult a =
            coeffs ? coefficients_only(panels_from_numpy(n, A, c), as_vec(v), rho, T, mb, Tw,
                                       gsi, one_sided, chi_ptr)
                   : spacecraft_aero(panels_from_numpy(n, A, c), as_vec(v), rho, T, mb, Tw, gsi,
                                     one_sided, chi_ptr);
        return aero_to_dict(a);
    };

    m.def("spacecraft_aero",
          [aero_kwargs](py::array_t<double> n, py::array_t<double> A, py::array_t<double> c,
                        py::array_t<double> v, double rho, double T, double mb, double Tw,
                        double aE, bool one_sided, const std::string& gsi, double an, double at,
                        py::object chi) {
              return aero_kwargs(n, A, c, v, rho, T, mb, Tw, aE, one_sided, gsi, an, at, chi,
                                 false);
          },
          py::arg("n"), py::arg("A"), py::arg("c"), py::arg("v_rel_B"), py::arg("rho"),
          py::arg("T"), py::arg("m_bar"), py::arg("T_w"), py::arg("alpha_E") = 1.0,
          py::arg("one_sided_ref") = true, py::arg("gsi") = "sentman",
          py::arg("alpha_n") = 1.0, py::arg("alpha_t") = 1.0, py::arg("chi") = py::none());

    m.def("coefficients_only",
          [aero_kwargs](py::array_t<double> n, py::array_t<double> A, py::array_t<double> c,
                        py::array_t<double> v, double rho, double T, double mb, double Tw,
                        double aE, bool one_sided, const std::string& gsi, double an, double at,
                        py::object chi) {
              return aero_kwargs(n, A, c, v, rho, T, mb, Tw, aE, one_sided, gsi, an, at, chi,
                                 true);
          },
          py::arg("n"), py::arg("A"), py::arg("c"), py::arg("v_rel_B"), py::arg("rho"),
          py::arg("T"), py::arg("m_bar"), py::arg("T_w"), py::arg("alpha_E") = 1.0,
          py::arg("one_sided_ref") = true, py::arg("gsi") = "sentman",
          py::arg("alpha_n") = 1.0, py::arg("alpha_t") = 1.0, py::arg("chi") = py::none());

    m.def("panel_srp_force",
          [](py::array_t<double> n, py::array_t<double> A, py::array_t<double> c,
             py::array_t<double> sun, double ecl, double P, double Cr) {
              return to_np(panel_srp_force(panels_from_numpy(n, A, c), as_vec(sun), ecl, P, Cr));
          },
          py::arg("n"), py::arg("A"), py::arg("c"), py::arg("sun_hat_B"),
          py::arg("eclipse") = 1.0, py::arg("pressure") = 0.0, py::arg("Cr") = 0.0);

    m.def("panel_srp_optical",
          [](py::array_t<double> n, py::array_t<double> A, py::array_t<double> c,
             py::array_t<double> sun, double ca, double cs, double cd, double ecl, double P) {
              SailOptics opt;
              opt.optical = true;
              opt.ca = ca;
              opt.cs = cs;
              opt.cd = cd;
              const auto ft = panel_srp_optical(panels_from_numpy(n, A, c), as_vec(sun), ecl, P, opt);
              return py::make_tuple(to_np(ft.first), to_np(ft.second));
          },
          py::arg("n"), py::arg("A"), py::arg("c"), py::arg("sun_hat_B"), py::arg("ca"),
          py::arg("cs"), py::arg("cd"), py::arg("eclipse") = 1.0, py::arg("pressure") = 0.0);

    m.def("panel_srp_force_torque",
          [](py::array_t<double> n, py::array_t<double> A, py::array_t<double> c,
             py::array_t<double> sun, double ecl, double P, double Cr) {
              const auto ft = panel_srp_force_torque(panels_from_numpy(n, A, c), as_vec(sun), ecl, P, Cr);
              return py::make_tuple(to_np(ft.first), to_np(ft.second));
          },
          py::arg("n"), py::arg("A"), py::arg("c"), py::arg("sun_hat_B"),
          py::arg("eclipse") = 1.0, py::arg("pressure") = 0.0, py::arg("Cr") = 0.0);

    m.def("earth_rad",
          [](py::array_t<double> n, py::array_t<double> A, py::array_t<double> c,
             py::array_t<double> nadir_B, double r_m, double cos_sun, double p_sun,
             std::vector<double> sol, std::vector<double> ir) {
              if (sol.size() != 3 || ir.size() != 3) throw std::runtime_error("sol/ir = (ca, cs, cd)");
              SailOptics so{sol[0], sol[1], sol[2], true};
              SailOptics io{ir[0], ir[1], ir[2], true};
              const auto ft = earth_rad(panels_from_numpy(n, A, c), as_vec(nadir_B), r_m, cos_sun,
                                        p_sun, so, io);
              return py::make_tuple(to_np(ft.first), to_np(ft.second));
          },
          py::arg("n"), py::arg("A"), py::arg("c"), py::arg("nadir_B"), py::arg("r_m"),
          py::arg("cos_sun"), py::arg("p_sun"), py::arg("sol"), py::arg("ir"));

    m.def("gg_torque", [](py::array_t<double> I_diag, py::array_t<double> r_B, double mu) {
        const Vec3 d = as_vec(I_diag);
        return to_np(gg_torque(Mat3::diag(d.x, d.y, d.z), as_vec(r_B), mu));
    }, py::arg("I_diag"), py::arg("r_B"), py::arg("mu"));

    m.def("sun_dist_au", &sun_dist_au);
    m.def("dcm_to_quat", [](py::array_t<double> C) {
        auto r = C.unchecked<2>();
        Mat3 M;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) M(i, j) = r(i, j);
        return to_np_q(dcm_to_quat(M));
    });

    m.def("apply_rods", [](py::array_t<double> tau, py::array_t<double> B, py::array_t<double> axis,
                           py::array_t<double> dmax, std::vector<bool> on) {
        auto a = axis.unchecked<2>();
        auto d = dmax.unchecked<1>();
        RodArray r;
        for (py::ssize_t i = 0; i < a.shape(0); ++i) {
            r.axis.push_back(unit(Vec3{a(i, 0), a(i, 1), a(i, 2)}));
            r.dmax.push_back(d(i));
            r.on.push_back(on.at(i) ? 1 : 0);
        }
        std::vector<double> dk;
        const MagnetorquerOut o = apply_rods(as_vec(tau), as_vec(B), r, dk);
        return py::make_tuple(to_np(o.tau), to_np(o.m), dk, o.shortfall_frac);
    }, py::arg("tau"), py::arg("B"), py::arg("axis"), py::arg("dmax"), py::arg("on"));

    m.def("rods_allocate", [](py::array_t<double> m, py::array_t<double> axis,
                              py::array_t<double> dmax, std::vector<bool> on) {
        auto a = axis.unchecked<2>();
        auto d = dmax.unchecked<1>();
        RodArray r;
        for (py::ssize_t i = 0; i < a.shape(0); ++i) {
            r.axis.push_back(unit(Vec3{a(i, 0), a(i, 1), a(i, 2)}));
            r.dmax.push_back(d(i));
            r.on.push_back(on.at(i) ? 1 : 0);
        }
        std::vector<double> dk;
        const Vec3 mm = rods_allocate(as_vec(m), r, dk);
        return py::make_tuple(to_np(mm), dk);
    }, py::arg("m"), py::arg("axis"), py::arg("dmax"), py::arg("on"));

    m.def("propagate_f32_samples",
          [](py::array_t<double> r0, py::array_t<double> v0, std::vector<double> offsets,
             double dt_s, double bc_inv, double rho0, double alt0_m, double h_scale,
             double a_srp, py::object sun_obj) {
              const Vec3 r = as_vec(r0);
              const Vec3 v = as_vec(v0);
              const float rf[3] = {static_cast<float>(r.x), static_cast<float>(r.y),
                                   static_cast<float>(r.z)};
              const float vf[3] = {static_cast<float>(v.x), static_cast<float>(v.y),
                                   static_cast<float>(v.z)};
              float sf[3] = {0.0f, 0.0f, 0.0f};
              if (!sun_obj.is_none()) {
                  const Vec3 s = unit(as_vec(py::cast<py::array_t<double>>(sun_obj)));
                  sf[0] = static_cast<float>(s.x);
                  sf[1] = static_cast<float>(s.y);
                  sf[2] = static_cast<float>(s.z);
              }
              const auto states = propagate_f32_samples(rf, vf, offsets, dt_s, bc_inv, rho0,
                                                        alt0_m, h_scale, a_srp, sf);
              py::list out;
              for (const auto& s : states) {
                  py::array_t<float> ra(3), va(3);
                  auto rr = ra.mutable_unchecked<1>();
                  auto vv = va.mutable_unchecked<1>();
                  for (int i = 0; i < 3; ++i) {
                      rr(i) = s.r[i];
                      vv(i) = s.v[i];
                  }
                  out.append(py::make_tuple(ra, va));
              }
              return out;
          },
          py::arg("r0"), py::arg("v0"), py::arg("offsets_s"), py::arg("dt_s") = 30.0,
          py::arg("bc_inv") = 0.0, py::arg("rho0") = 3.0e-12, py::arg("alt0_m") = 400e3,
          py::arg("h_scale") = 60.0e3, py::arg("a_srp") = 0.0, py::arg("sun_hat") = py::none());

    m.def("accel_twobody", [](py::array_t<double> r, double mu) {
        return to_np(accel_twobody(as_vec(r), mu));
    });
    m.def("accel_j2", [](py::array_t<double> r, double mu, double Re, double J2) {
        return to_np(accel_j2(as_vec(r), mu, Re, J2));
    });
    m.def("accel_j3", [](py::array_t<double> r, double mu, double Re, double J3) {
        return to_np(accel_j3(as_vec(r), mu, Re, J3));
    });

    py::class_<GravityHarmonics>(m, "GravityHarmonics")
        .def(py::init<>())
        .def("load_ggm", &GravityHarmonics::load_ggm, py::arg("path"), py::arg("max_degree") = 20)
        .def("loaded", &GravityHarmonics::loaded)
        .def("mu", &GravityHarmonics::mu)
        .def("Re", &GravityHarmonics::Re)
        .def("J2", &GravityHarmonics::J2)
        .def("max_degree", &GravityHarmonics::max_degree)
        .def("accel_ecef", [](const GravityHarmonics& g, py::array_t<double> r, int deg) {
            return to_np(g.accel_ecef(as_vec(r), deg));
        }, py::arg("r_ecef"), py::arg("degree") = -1);

    m.def("accel_third_body", [](py::array_t<double> rs, py::array_t<double> rb, double mu) {
        return to_np(accel_third_body(as_vec(rs), as_vec(rb), mu));
    });
    m.def("sun_unit_analytic", [](double jd) { return to_np(sun_unit_analytic(jd)); });
    m.def("moon_analytic", [](double jd) {
        Vec3 rh{};
        double rm = 0.0;
        moon_analytic(jd, rh, rm);
        return py::make_tuple(to_np(rh), rm);
    }, py::arg("jd"));
    m.def("eclipse_cylindrical", [](py::array_t<double> r, py::array_t<double> s, double Re) {
        return eclipse_cylindrical(as_vec(r), as_vec(s), Re);
    });
    m.def("gmst_rad", &gmst_rad);
    m.def("sma_from_rv", [](py::array_t<double> r, py::array_t<double> v, double mu) {
        return sma_from_rv(as_vec(r), as_vec(v), mu);
    });
    m.def("ecef_to_geodetic", [](py::array_t<double> r) {
        double lat, lon, alt;
        ecef_to_geodetic(as_vec(r), lat, lon, alt);
        return py::make_tuple(lat, lon, alt);
    });

    m.def("mrp_to_dcm", [](py::array_t<double> s) { return to_np(mrp_to_dcm(as_vec(s))); });
    m.def("mrp_shadow", [](py::array_t<double> s) { return to_np(mrp_shadow(as_vec(s))); });
    m.def("mrp_rate", [](py::array_t<double> s, py::array_t<double> w) {
        return to_np(mrp_rate(as_vec(s), as_vec(w)));
    });
    m.def("mrp_compose", [](py::array_t<double> a, py::array_t<double> b) {
        return to_np(mrp_compose(as_vec(a), as_vec(b)));
    });
    m.def("mrp_error", [](py::array_t<double> b, py::array_t<double> t, bool proper) {
        return to_np(mrp_error(as_vec(b), as_vec(t), proper));
    }, py::arg("sigma_BN"), py::arg("sigma_target"), py::arg("proper") = true);
    m.def("mrp_angle", [](py::array_t<double> s) { return mrp_angle(as_vec(s)); });
    m.def("quat_conj", [](py::array_t<double> q) { return to_np_q(quat_conj(as_quat(q))); });
    m.def("quat_unit", [](py::array_t<double> q) { return to_np_q(quat_unit(as_quat(q))); });
    m.def("quat_normalize", [](py::array_t<double> q) { return to_np_q(quat_normalize(as_quat(q))); });
    m.def("quat_mul", [](py::array_t<double> a, py::array_t<double> b) {
        return to_np_q(quat_mul(as_quat(a), as_quat(b)));
    });
    m.def("slerp_clip", [](py::array_t<double> qn, py::array_t<double> qh, double max_ang) {
        return to_np_q(slerp_clip(as_quat(qn), as_quat(qh), max_ang));
    });
    m.def("quat_to_mrp", [](py::array_t<double> q) { return to_np(quat_to_mrp(as_quat(q))); });
    m.def("mrp_to_quat", [](py::array_t<double> s) { return to_np_q(mrp_to_quat(as_vec(s))); });

    py::class_<MRPFeedback>(m, "MRPFeedback")
        .def(py::init<>())
        .def_readwrite("kp", &MRPFeedback::kp)
        .def_readwrite("kd", &MRPFeedback::kd)
        .def_readwrite("proper", &MRPFeedback::proper)
        .def_readwrite("max_body_rate", &MRPFeedback::max_body_rate)
        .def("set_inertia_diag", [](MRPFeedback& c, py::array_t<double> d) {
            auto v = as_vec(d);
            c.inertia = Mat3::diag(v.x, v.y, v.z);
        })
        .def("set_max_torque", [](MRPFeedback& c, py::array_t<double> t) { c.max_torque = as_vec(t); })
        .def("compute", [](const MRPFeedback& c, py::array_t<double> s, py::array_t<double> w,
                           py::array_t<double> st, py::array_t<double> wt, double dt) {
            return to_np(c.compute(as_vec(s), as_vec(w), as_vec(st), as_vec(wt), dt));
        }, py::arg("sigma"), py::arg("omega"), py::arg("sigma_tgt"),
           py::arg("omega_tgt"), py::arg("dt_s") = 1.0);

    py::class_<QuaternionFeedback>(m, "QuaternionFeedback")
        .def(py::init<>())
        .def_readwrite("kp", &QuaternionFeedback::kp)
        .def_readwrite("kd", &QuaternionFeedback::kd)
        .def_readwrite("max_body_rate", &QuaternionFeedback::max_body_rate)
        .def("set_inertia_diag", [](QuaternionFeedback& c, py::array_t<double> d) {
            auto v = as_vec(d);
            c.inertia = Mat3::diag(v.x, v.y, v.z);
        })
        .def("set_max_torque", [](QuaternionFeedback& c, py::array_t<double> t) {
            c.max_torque = as_vec(t);
        })
        .def("compute", [](const QuaternionFeedback& c, py::array_t<double> q,
                           py::array_t<double> w, py::array_t<double> qt,
                           py::array_t<double> wt, double dt) {
            return to_np(c.compute(as_quat(q), as_vec(w), as_quat(qt), as_vec(wt), dt));
        }, py::arg("q"), py::arg("omega"), py::arg("q_tgt"), py::arg("omega_tgt"),
           py::arg("dt_s") = 1.0);

    py::class_<BDot>(m, "BDot")
        .def(py::init<>())
        .def_readwrite("gain", &BDot::gain)
        .def_readwrite("max_dipole", &BDot::max_dipole)
        .def_readwrite("dt", &BDot::dt)
        .def("reset", &BDot::reset)
        .def("dipole", [](BDot& b, py::array_t<double> B) { return to_np(b.dipole(as_vec(B))); })
        .def_static("torque", [](py::array_t<double> m, py::array_t<double> B) {
            return to_np(BDot::torque(as_vec(m), as_vec(B)));
        });

    m.def("saturate_dipole", [](py::array_t<double> m, py::array_t<double> m_max) {
        return to_np(saturate_dipole(as_vec(m), as_vec(m_max)));
    }, py::arg("m"), py::arg("m_max"));

    m.def("apply_magnetorquer",
          [](py::array_t<double> tau_cmd, py::array_t<double> B,
             py::array_t<double> m_max) {
              const MagnetorquerOut o =
                  apply_magnetorquer(as_vec(tau_cmd), as_vec(B), as_vec(m_max));
              py::dict d;
              d["m"] = to_np(o.m);
              d["tau"] = to_np(o.tau);
              d["shortfall_frac"] = o.shortfall_frac;
              return d;
          },
          py::arg("tau_cmd"), py::arg("B_body"), py::arg("m_max"));

    m.def("dipole_field_ecef", [](py::array_t<double> r) { return to_np(dipole_field_ecef(as_vec(r))); });

    py::class_<WMM>(m, "WMM")
        .def(py::init<>())
        .def("load", &WMM::load)
        .def("loaded", &WMM::loaded)
        .def("field_ecef", [](const WMM& w, py::array_t<double> r, double year) {
            return to_np(w.field_ecef(as_vec(r), year));
        });

    py::class_<SimParams>(m, "SimParams")
        .def(py::init<>())
        .def_readwrite("mass", &SimParams::mass)
        .def_readwrite("dt_s", &SimParams::dt_s)
        .def_readwrite("advisor_step_s", &SimParams::advisor_step_s)
        .def_readwrite("rk4_step_s", &SimParams::rk4_step_s)
        .def_readwrite("sh_degree", &SimParams::sh_degree)
        .def_readwrite("lunisolar", &SimParams::lunisolar)
        .def_readwrite("use_panel_srp", &SimParams::use_panel_srp)
        .def_readwrite("corotating", &SimParams::corotating)
        .def_readwrite("alpha_E", &SimParams::alpha_E)
        .def_readwrite("alpha_n", &SimParams::alpha_n)
        .def_readwrite("alpha_t", &SimParams::alpha_t)
        .def_readwrite("gsi", &SimParams::gsi)
        .def_readwrite("T_w", &SimParams::T_w)
        .def_readwrite("Cr", &SimParams::Cr)
        .def_readwrite("srp_optical", &SimParams::srp_optical)
        .def_readwrite("srp_ca", &SimParams::srp_ca)
        .def_readwrite("srp_cs", &SimParams::srp_cs)
        .def_readwrite("srp_cd", &SimParams::srp_cd)
        .def_readwrite("srp_scale", &SimParams::srp_scale)
        .def_readwrite("earth_rad", &SimParams::earth_rad)
        .def_readwrite("ir_ca", &SimParams::ir_ca)
        .def_readwrite("ir_cs", &SimParams::ir_cs)
        .def_readwrite("ir_cd", &SimParams::ir_cd)
        .def_readwrite("gravity_gradient", &SimParams::gravity_gradient)
        .def_readwrite("mtq_duty", &SimParams::mtq_duty)
        .def_readwrite("cmd_frame", &SimParams::cmd_frame)
        .def_property("wind_N", [](const SimParams& p) { return to_np(p.wind_N); },
                      [](SimParams& p, py::array_t<double> w) { p.wind_N = as_vec(w); })
        .def_readwrite("one_sided_ref", &SimParams::one_sided_ref)
        .def_readwrite("ideal_torque", &SimParams::ideal_torque)
        .def_readwrite("attitude_law", &SimParams::attitude_law)
        .def_readwrite("mode", &SimParams::mode)
        .def_readwrite("epoch_jd", &SimParams::epoch_jd)
        .def_readwrite("ggm_path", &SimParams::ggm_path)
        .def_readwrite("wmm_path", &SimParams::wmm_path)
        .def_readwrite("max_slew_rad", &SimParams::max_slew_rad);

    py::class_<Simulator>(m, "Simulator")
        .def(py::init<SimParams>(), py::arg("params") = SimParams{})
        .def("load_gravity", &Simulator::load_gravity)
        .def("load_wmm", &Simulator::load_wmm)
        .def("load_spice", &Simulator::load_spice)
        .def("gravity_loaded", &Simulator::gravity_loaded)
        .def("wmm_loaded", &Simulator::wmm_loaded)
        .def("set_panels", [](Simulator& s, py::array_t<double> n, py::array_t<double> A,
                              py::array_t<double> c) {
            s.set_panels(panels_from_numpy(n, A, c));
        })
        .def("set_mass", &Simulator::set_mass)
        .def("set_inertia_diag", [](Simulator& s, py::array_t<double> d) {
            auto v = as_vec(d);
            s.set_inertia(Mat3::diag(v.x, v.y, v.z));
        })
        .def("set_inertia", [](Simulator& s, py::array_t<double> I) {
            auto r = I.unchecked<2>();
            Mat3 M;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) M(i, j) = r(i, j);
            s.set_inertia(M);
        })
        .def("set_atmosphere", [](Simulator& s, double rho, double T, double mb,
                                  py::object chi_obj) {
            Atmo a{rho, T, mb};
            if (!chi_obj.is_none()) {
                auto arr = py::cast<py::array_t<double>>(chi_obj);
                auto r = arr.unchecked<1>();
                if (r.shape(0) < 7) throw std::runtime_error("chi must have length 7");
                for (int i = 0; i < 7; ++i) a.chi[i] = r(i);
                a.has_species = true;
            }
            s.set_atmosphere(a);
        }, py::arg("rho"), py::arg("T"), py::arg("m_bar"), py::arg("chi") = py::none())
        .def("set_atmosphere_end", [](Simulator& s, double rho, double T, double mb,
                                      py::object chi_obj) {
            Atmo a{rho, T, mb};
            if (!chi_obj.is_none()) {
                auto arr = py::cast<py::array_t<double>>(chi_obj);
                auto r = arr.unchecked<1>();
                if (r.shape(0) < 7) throw std::runtime_error("chi must have length 7");
                for (int i = 0; i < 7; ++i) a.chi[i] = r(i);
                a.has_species = true;
            }
            s.set_atmosphere_end(a);
        }, py::arg("rho"), py::arg("T"), py::arg("m_bar"), py::arg("chi") = py::none())
        .def("set_mode", &Simulator::set_mode)
        .def("set_srp_scale", &Simulator::set_srp_scale)
        .def("set_wind", [](Simulator& s, py::array_t<double> w) { s.set_wind(as_vec(w)); })
        .def("set_cmd_frame", &Simulator::set_cmd_frame)
        .def("set_nav", [](Simulator& s, py::array_t<double> r, py::array_t<double> v) { s.set_nav(as_vec(r), as_vec(v)); })
        .def("clear_nav", &Simulator::clear_nav)
        .def("set_controller", &Simulator::set_controller)
        .def("set_quat_controller", &Simulator::set_quat_controller)
        .def("params", &Simulator::params, py::return_value_policy::reference_internal)
        .def("set_bdot", &Simulator::set_bdot)
        .def("set_dipole_max", [](Simulator& s, py::array_t<double> m) { s.set_dipole_max(as_vec(m)); })
        .def("dipole_max", [](const Simulator& s) { return to_np(s.dipole_max()); })
        .def("set_rods", [](Simulator& s, py::array_t<double> axis, py::array_t<double> dmax) {
            auto a = axis.unchecked<2>();
            auto d = dmax.unchecked<1>();
            if (a.shape(1) != 3 || a.shape(0) != d.shape(0))
                throw std::runtime_error("axis must be (N,3), dmax (N,)");
            std::vector<Vec3> ax;
            std::vector<double> dm;
            for (py::ssize_t i = 0; i < a.shape(0); ++i) {
                ax.push_back({a(i, 0), a(i, 1), a(i, 2)});
                dm.push_back(d(i));
            }
            s.set_rods(ax, dm);
        }, py::arg("axis"), py::arg("dmax"))
        .def("set_rod_gates", [](Simulator& s, std::vector<bool> on) {
            std::vector<char> c(on.begin(), on.end());
            s.set_rod_gates(c);
        }, py::arg("on"))
        .def("rod_gates", [](const Simulator& s) {
            return std::vector<bool>(s.rods().on.begin(), s.rods().on.end());
        })
        .def("reset", [](Simulator& s, py::array_t<double> r, py::array_t<double> v,
                         py::array_t<double> sig, py::array_t<double> w) {
            auto y = s.reset(as_vec(r), as_vec(v), as_vec(sig), as_vec(w));
            py::dict d;
            d["r"] = to_np(y.r);
            d["v"] = to_np(y.v);
            d["sigma"] = to_np(y.sigma);
            d["omega"] = to_np(y.omega);
            d["t"] = y.t;
            return d;
        })
        .def("step", [](Simulator& s, py::array_t<double> q) {
            auto r = q.unchecked<1>();
            if (r.shape(0) < 4) throw std::runtime_error("quaternion length 4");
            Quat qq{r(0), r(1), r(2), r(3)};
            const StepOut o = s.step(qq);
            py::dict d;
            d["r"] = to_np(o.y.r);
            d["v"] = to_np(o.y.v);
            d["sigma"] = to_np(o.y.sigma);
            d["omega"] = to_np(o.y.omega);
            d["t"] = o.y.t;
            d["altitude_km"] = o.altitude_km;
            d["sma_m"] = o.sma_m;
            d["eclipse"] = o.eclipse;
            d["sun_B"] = to_np(o.sun_B);
            d["B_B"] = to_np(o.B_B);
            d["Cd"] = o.Cd;
            d["Cl"] = o.Cl;
            d["drag_N"] = o.drag_N;
            d["dE_actual"] = o.dE_actual;
            d["dE_baseline"] = o.dE_baseline;
            d["dE_drag"] = o.dE_drag;
            d["dE_lift"] = o.dE_lift;
            d["tracking_err_rad"] = o.tracking_err_rad;
            d["n_substeps"] = o.n_substeps;
            d["tau_ctrl_mean"] = to_np(o.tau_ctrl_mean);
            d["tau_aero_mean"] = to_np(o.tau_aero_mean);
            d["m_mean"] = to_np(o.m_mean);
            d["tau_cmd_mean"] = to_np(o.tau_cmd_mean);
            d["tau_shortfall_frac"] = o.tau_shortfall_frac;
            d["tau_demand_mean"] = to_np(o.tau_demand_mean);
            d["tau_demand_absmean"] = to_np(o.tau_demand_absmean);
            d["gyro_mean"] = to_np(o.gyro_mean);
            d["tau_srp_mean"] = to_np(o.tau_srp_mean);
            d["tau_erp_mean"] = to_np(o.tau_erp_mean);
            d["tau_gg_mean"] = to_np(o.tau_gg_mean);
            d["tau_env_mean"] = to_np(o.tau_env_mean);
            d["a_srp_sun"] = o.a_srp_sun;
            d["n_sunlit"] = o.n_sunlit;
            d["rod_duty2_mean"] = o.rod_duty2_mean;
            return d;
        })
        .def("get_state", [](const Simulator& s) {
            const auto& y = s.state();
            const auto& p = s.params();
            const double jd = p.epoch_jd + y.t / 86400.0;
            const Mat3 C = mrp_to_dcm(y.sigma);
            const Vec3 sun = s.sun_hat(jd);
            py::dict d;
            d["r"] = to_np(y.r);
            d["v"] = to_np(y.v);
            d["sigma"] = to_np(y.sigma);
            d["omega"] = to_np(y.omega);
            d["t"] = y.t;
            d["C_BN"] = to_np(C);
            d["sun_N"] = to_np(sun);
            d["eclipse"] = eclipse_cylindrical(y.r, sun, RE_WGS);
            double lat = 0.0, lon = 0.0, h = 0.0;
            ecef_to_geodetic(y.r, lat, lon, h);
            d["altitude_km"] = h / 1e3;
            return d;
        });
}
