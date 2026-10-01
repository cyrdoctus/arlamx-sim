time: 2026-08-19T01:00:00Z
agent: grok

# Plan only — Cercignani–Lampis–Lord (CLL / CCL) free-molecular GSI

Do **not** implement this in the plant until the user asks. Sentman M-01 stays
the production GSI. This file is the recipe so a later pass can add CLL as a
named alternative next to Sentman.

The literature name is **CLL** (Cercignani–Lampis–Lord). “CCL” in this project
means the same model.

## Why it is a different model

Sentman (1961) + Moe & Moe (2005) is a *Maxwell* kernel: a fraction α_E of
molecules re-emit diffusely at Tr, the rest bounce with the incident energy.
CLL is a *scattering kernel* with two accommodations:

- α_n — normal-momentum accommodation ∈ [0, 1]
- α_t — tangential-momentum accommodation ∈ [0, 1]

Those two numbers set the joint probability that an incident velocity
(v_n^i, v_t^i) becomes (v_n^r, v_t^r). They are **not** α_E. A later
mapping α_E(α_n, α_t, s, Tw/Ti) exists but is not 1-to-1.

## Equations to implement (do not code yet)

Kernel (Lord 1991, after Cercignani & Lampis 1971), speeds in units of
√(2 kT_w / m):

\[
R_n(v_n^i\to v_n^r)=\frac{2 v_n^r}{\alpha_n}
\exp\Bigl(-\frac{(v_n^r)^2+(1-\alpha_n)(v_n^i)^2}{\alpha_n}\Bigr)
I_0\Bigl(\frac{2\sqrt{1-\alpha_n}\,v_n^i v_n^r}{\alpha_n}\Bigr)
\]

\[
R_t(v_t^i\to v_t^r)=\frac{1}{\sqrt{\pi\alpha_t}}
\exp\Bigl(-\frac{(v_t^r-(1-\alpha_t)v_t^i)^2}{\alpha_t}\Bigr)
\]

Panel coefficients: integrate the kernel against the incident drifting
Maxwellian (speed ratio s, incidence θ) to get the outgoing momentum
flux, then

\[
C_p=\frac{p_n}{q},\qquad C_\tau=\frac{\tau}{q},\qquad q=\tfrac12\rho |v|^2.
\]

Walker, Mehta et al. (2014) and Pilinski / Padilla give closed or
semi-closed hyperthermal and finite-s forms. **Use those published
integrals.** Do not Monte-Carlo the kernel in the plant loop.

Hyperthermal check (same classroom as Sentman): as s→∞, α_n=α_t=1
(fully accommodated) must recover the Sentman α_E=1, Tw=Ti plate.

## Where it plugs in

| Today | After CLL |
|---|---|
| `sentman(θ, s, Tw/Ti, α_E)` | `cll(θ, s, Tw/Ti, α_n, α_t)` same return `(Cp, Cτ)` |
| `SimParams.alpha_E` | add `gsi = "sentman" \| "cll"`, `alpha_n`, `alpha_t` |
| tests/aero/test_sentman.py | add `test_cll.py` with Walker/Mehta table points |

The panel sum, A_ref, corotating v_rel, and energy `F·v` do **not** change.

## Sources (2–4, independent)

1. Cercignani, C. & Lampis, M. (1971). Kinetic models for gas–surface
   interactions. *Transp. Theory Stat. Phys.* 1, 101–114. (paywall — user
   must review)
2. Lord, R. G. (1991). Some extensions to the Cercignani–Lampis
   gas–surface scattering kernel. *Phys. Fluids A* 3, 706–710.
3. Walker, A., Mehta, P. & Koller, J. (2014). Drag coefficient model
   using the Cercignani–Lampis–Lord gas–surface interaction model.
   *J. Spacecraft Rockets* 51(5), 1544–1563. (paywall — user must review)
4. Padilla, J. F. & Boyd, I. D. (2008). Assessment of gas-surface
   interaction models for computation of rarefied hypersonic flow.
   *J. Thermophys. Heat Transfer* / AIAA.

Do not use Wikipedia or DSMC blog posts as sources.

## Suggested later work (not this session)

1. Implement `cll()` from Walker’s integrated coefficients, not from a
   raw kernel sample.
2. Classroom: α_n=α_t=1 vs Sentman α_E=1, Tw=Ti, residual < 1e-3 at s=8.
3. Hex min/max drag at 400 km, F10.7=150: report Cd(CLL)/Cd(Sentman).
4. Keep Sentman the default. CLL is a named switch.

## Explicitly not in this session

No `cll.cpp`. No change to the Sentman path except the already-landed
`cosθ ≤ 1e-12` skip.
