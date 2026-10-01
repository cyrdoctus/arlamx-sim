# 13 — Glossary and references

## Terms as the code uses them

| term | meaning |
|---|---|
| advisor / advisor step | the 300 s decision (RL policy, MPC or heuristic) that emits one target quaternion; 150 substeps of 2 s |
| plant | the C++ `Simulator`; "ideal couple" vs "coils through B" is the `ideal_torque` flag |
| substep / dt_s | the 2 s inner control and integration step |
| GSI | gas–surface interaction model: Sentman (default, α_E) or Walker–CLL (α_N, σ_T) |
| α_E | Sentman energy accommodation, M-01 closure, 0.93 |
| α_N, σ_T | CLL normal-energy and tangential-momentum accommodation |
| s | speed ratio |v_rel|/√(2kT/m̄), ≈ 8 in LEO |
| A_ref | half the total panel area (`one_sided_ref`) |
| dE_vs_baseline | orbital energy change vs the min-drag counterfactual (the reward's longevity term) |
| min-drag counterfactual | Cd of the panel set with the flow along the smallest-projected-area body axis |
| corotating | atmosphere rotates with Earth; v_rel = v − ω⊕ × r (default on since 2026-08-19) |
| IPC | "interactive perturbation control": v7 per-axis authority gates + torque observer |
| delegation / split s | v8b+: fraction of an axis's turn handed to environmental torque; gate = 1 − s(1 − floor) |
| gate_floor | minimum authority per axis, 0.3 (v8+), 0.15 (v3–v7 default) |
| B, P, S | delegation benefit (signed alignment), power saved fraction, significance (eqs. 3–5) |
| τ_env / τ_dist_est | Kalman-filtered disturbance torque estimate from gyro differencing |
| τ_want / tau_demand | full-authority PD demand before the gate's torque clip |
| comparator | v12+: the MPC's scoring function used to veto a proposed quaternion (H = 2) |
| wrong-way clip | v13+: force s_i = 0 when τ_env,i τ_want,i < 0 |
| regime estimator | v14: classical EWMA on |τ_env| that schedules kd, floor, κ, margin |
| band | fraction of time with SoC in 0.4–0.6 |
| brownout | SoC hits 0 → detumble mode + recovery loop |
| gap index | mean signed normalised distance to the MPC over 5 metrics; 0 = MPC |
| column A / D | MPC v1 truth-state reference (9.28/86.6/23.2) / MPC v2 with sensors (9.19/84.1/18.6) |
| dual gate | a checkpoint passing both quiet and storm evaluation bars |
| ship bar | the 24-draw four-task Monte Carlo vs the MPC |
| U575 | STM32U575 (Cortex-M33, 160 MHz) analytic decision-cost model |
| freeze | a named checkpoint copied to `outputs/v13/ckpt*` with its numbers |
| snapshot.json | resolved YAML config frozen next to a trained model |
| classroom test | fixed-input numeric check a reviewer can verify by hand (`VALIDATION_STANDARD.md`) |
| six-petal / original hexagon | `SolarCat_Assembly.STL` (1.75 kg) / `earthcup_hex_v3.geom` (0.72 kg) |
| 1pct | geometry simplification budget: ≤ 1 % wetted and ram area error at six attitudes |

## Symbols and constants (`cpp/include/arlamx/constants.hpp`)

k_B 1.380649e-23, N_A 6.02214076e23, μ (GGM header) 3.986004415e14,
R_E (GGM) 6378136.3 m, ω⊕ 7.2921150e-5, μ (WGS) 3.986004418e14, R_E (WGS)
6378137.0, J2 1.0826353865e-3 (GGM03S C20·√5), J3 −2.53265649e-6,
P_SRP 4.56e-6 Pa, Cr 1.8, I = diag(0.0125, 0.0125, 0.025) kg m².

## Literature the code cites (by module)

- Sentman, L. H. (1961). *Free molecule flow theory and its application to the determination of aerodynamic forces.* LMSC-448514, DTIC AD0265409.
- Moe, K. & Moe, M. M. (2005). Gas–surface interactions and satellite drag coefficients. *Planet. Space Sci.* 53, 793–801.
- Doornbos, E. (2012). *Thermospheric Density and Wind Determination from Satellite Dynamics.* Springer, Ch. 3.
- Walker, A., Mehta, P. & Koller, J. (2014). Drag coefficient model using the Cercignani–Lampis–Lord GSI model. *J. Spacecraft Rockets* 51(5), 1544–1563.
- Sinpetru, L. et al. (2021). ADBSat: methodology of a novel panel method tool. arXiv:2104.05543 (eqs. 6–15 for Sentman, Schaaf–Chambre, CLL).
- Cercignani, C. & Lampis, M. (1971). *Transp. Theory Stat. Phys.* 1, 101–114; Lord, R. G. (1991). *Phys. Fluids A* 3, 706–710.
- Montenbruck, O. & Gill, E. (2000). *Satellite Orbits.* Springer §3.2–3.4.
- Vallado, D. (2013). *Fundamentals of Astrodynamics and Applications*, 4th ed., §8.
- Tapley, B. et al. (2005). GGM02/GGM03S gravity models (CSR). Acton, C. (1996). SPICE. *Planet. Space Sci.* 44, 65–70.
- Picone, Hedin, Drob & Aikin (2002). NRLMSISE-00. *JGR* 107(A12) 1468; Emmert, J. T. (2015). *Adv. Space Res.* 56, 773–824.
- Schaub, H. & Junkins, J. (2018). *Analytical Mechanics of Space Systems*, 4th ed., Ch. 3–4, 8. Shuster, M. (1993). *J. Astronaut. Sci.* 41, 439–517. Tsiotras, P. (1996). *JGCD* 19(4), 772–779. Wie, B. (2008). *Space Vehicle Dynamics and Control.*
- Hairer, Nørsett & Wanner (1993). *Solving ODEs I*, §II.1.
- Stickler, A. & Alfriend, K. (1976). Elementary magnetic attitude control system. *J. Spacecraft Rockets* 13, 282–287. Avanzini, G. & Giulietti, F. (2012). *JGCD* 35(4), 1326–1334.
- Lovera, M. & Astolfi, A. (2004). Spacecraft attitude control using magnetic actuators. *Automatica* 40, 1405–1414.
- Chulliat, A. et al. (2020). WMM 2020–2025 technical report, NOAA/NCEI. Wertz, J. (1978). *Spacecraft Attitude Determination and Control*, App. H. Finlay et al. (2010). IGRF-11.
- McInnes, C. (1999). *Solar Sailing.* Springer.
- Cohen-Steiner, Alliez & Desbrun (2004). Variational shape approximation. *ACM TOG* 23(3). O'Rourke, J. (1998). *Computational Geometry in C.* Attene, Falcidieno & Spagnuolo (2006). *Visual Computer* 22.
- Schulman et al. (2017). PPO. arXiv:1707.06347. Raffin et al. (2021). Stable-Baselines3. *JMLR* 22(268). Towers et al. (2023). Gymnasium.
- Bar-Shalom, Li & Kirubarajan (2001). *Estimation with Applications to Tracking and Navigation* (CV model, Joseph form). Gelb, A. (1974). *Applied Optimal Estimation.*
- King-Hele, D. *Satellite Orbits in an Atmosphere* (lift from a rotating atmosphere).

Paywalled and reviewed by hand only: Moe & Moe 2005, Walker 2014.
