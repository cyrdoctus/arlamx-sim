# CLL classroom test spec

Implemented (partial): `tests/aero/test_cll.py` covers T-CLL-1, T-CLL-4, T-CLL-5.
Config wiring: `tests/physics/test_physics_wiring.py`. Do not copy ADBSat MATLAB.
Bindings: `cpp.cll`, `cpp.sentman`, `cpp.spacecraft_aero(..., gsi="cll", chi=...)`.

Species order for `chi` (length 7): He, O, N2, O2, Ar, H, N.

## T-CLL-1 — fully accommodated identity

\(\alpha_N=\sigma_T=1\), \(T_w=T_i\), \(s=8\), \(\theta=0\):

`cpp.cll` vs `cpp.sentman(..., alpha_E=1)` residual on \(C_p\) \(\lt 10^{-3}\) (implementation target: bit-identical on the face-on Schaaf–Chambre / Sentman \(\alpha_E=1\) plate). Repeat at \(s=8\), \(\theta=\pi/4\): \(|C_p^\mathrm{CLL}-C_p^\mathrm{Sentman}|\) and \(|C_\tau|\) residual \(\lt 10^{-3}\).

## T-CLL-2 — hyperthermal face-on

\(s=80\), \(\theta=0\), \(\alpha_N=\sigma_T=1\), \(T_w=T_i\): \(C_p\cos\theta+C_\tau\sin\theta\) within \(5\times10^{-3}\) of \(2\cos\theta\) (same classroom as Sentman).

## T-CLL-3 — face-on plate band

Two-sided unit square, flow along \(-\hat n\), `one_sided_ref=True`, \(s\approx8\), \(\alpha_N=\sigma_T=1\): \(C_D\) in 1.8–2.4 (published fully-diffuse face-on band).

## T-CLL-4 — mixture vs O-only

Same plate, \(s\) from \(T=900\,\mathrm{K}\), \(v=7500\,\mathrm{m/s}\). Compare `chi = [0,1,0,0,0,0,0]` (pure O) vs a 400 km-class mix (e.g. O 0.8, He 0.15, N2 0.05). Forces must be finite; mix \(C_D\) must differ from pure-O by a documented nonzero amount (not bit-identical). Empty / missing `chi` must match pure-O.

## T-CLL-5 — unknown GSI

`SimParams.gsi = "maxwell"` (or any string other than `sentman`/`cll`) must throw at `Simulator` construction. Same for `spacecraft_aero(..., gsi="foo")`.

## T-CLL-6 — Sentman path unchanged

Existing `tests/aero/test_sentman.py` and `tests/validation/verify_physics.py` Sentman checks must still pass with default `gsi="sentman"`.

## Out of scope

DSMC kernel sampling, onboard FP32 propagator, Schaaf–Chambre as a third named GSI, writing this file’s tests in the v2.6 implementation pass.
