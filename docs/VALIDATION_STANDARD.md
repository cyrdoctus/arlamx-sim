# Validation standard (every V2.0 module)

A module is not done until all five exist.

1. **Spec** — a short markdown in `docs/modules/` (this folder).
2. **Two citations** — textbooks or papers that state the equation, not blog posts.
3. **One classroom test** — a numeric case a reviewer can check with a calculator or the cited closed form. Fixed inputs, no RNG, no full-orbit “looks right.”
4. **Test code** — `tests/<module>/`, pytest (or a small C++ executable invoked by pytest). Writes nothing to `outputs/`.
5. **Recorded result** — when the test has been run, fill **Tests run** in the spec with the number, the date, and pass/fail. Until then that section says `NOT RUN`.

Tolerances (unless a spec says otherwise):

| Kind | Typical gate |
|---|---|
| Dimensionless aero identity | 1e-8 relative |
| Gravity vs analytic / ∇V | 1e-8 relative |
| Attitude kinematics / MRP algebra | 1e-12 absolute |
| Closed-loop angle | 0.01° |
| Geometry seam gap | 1e-9 m (numeric zero) |

Do not retune rewards to hide a failed plant test.
