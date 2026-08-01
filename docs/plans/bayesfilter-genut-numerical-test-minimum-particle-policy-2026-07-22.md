# GenUT Numerical Test Minimum Particle Policy

Date: 2026-07-22
Status: `active_owner_directive`

All new GenUT numerical feasibility, tuning, method-comparison, claim,
leaderboard, and default-readiness tests must use `N > 1000` particles.

For model collections containing both one- and two-dimensional Gaussian GenUT
designs, use `N=1008` as the smallest convenient smoke count: it is greater
than 1000 and divisible by six, so the positive GenUT weights are exactly
representable for both dimensions. Larger runs may use another count only when
it exactly represents the selected design weights.

Small arrays may remain in unit tests that check only local algebra, shapes, or
tangent identities. They are mechanics tests and must not be reported as
numerical feasibility or model-comparison evidence.

All historical `N<=1000` GenUT artifacts remain preserved but are mechanics or
historical diagnostics only. They are ineligible for new feasibility,
comparison, tuning, leaderboard, default, or HMC claims.
