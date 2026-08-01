# Score-Capable SIR Terminology Reset Memo

Date: 2026-07-23

## Owner Clarification

Going forward, unqualified `SIR` in GenUT, filter-score, HMC, and leaderboard
work means the parameterized Austria SIR target for which an analytical/manual
recursive observed-data score can be computed. It does not mean the fixed
source-parameter, value-only Austria fixture.

The implemented score-capable target is
`parameterized_zhao_cui_sir_austria_model()` with:

- `J = 9` spatial regions;
- latent state dimension `d = 18`, consisting of `(S_j, I_j)` for each region;
- observation dimension `9`;
- horizon `T = 20`; and
- score parameter order
  `(log_kappa_scale, log_nu_scale, log_observation_noise_scale)`.

The three inferred parameters do not increase the latent state dimension.
There is currently no implemented Austria SIR route with latent dimension
`d = 19`. A future `d = 19` target would require a separately defined model,
parameterization, and target identity rather than relabeling the existing
route.

## Evidence Boundary

The admitted same-target score-bearing comparator currently available is the
fixed level-2 SGQF route in
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p6/SIR-SGQF/r1b-identity/gpu-attempt-02/result.json`.
At theta `(0, 0, 0)`, its score is
`(28.7394530574, -106.6588565703, 9.4311763926)`.

No same-target GenUT score result has yet been executed for this parameterized
Austria SIR route. The fixed-parameter SIR value row and finite-particle LEDH
diagnostics are different targets and must not be substituted into this score
comparison.

## Nonclaims

This terminology clarification does not establish GenUT correctness, method
ranking, an exact SIR score oracle, HMC readiness, or leaderboard completion.
