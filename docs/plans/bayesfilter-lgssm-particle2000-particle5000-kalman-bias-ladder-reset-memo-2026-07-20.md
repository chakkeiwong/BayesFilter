# Reset Memo: LGSSM Particle Bias Ladder Closed

> Active execution correction, 2026-07-20: `N=5000` seed microbatch size one
> was conservative for memory, but the tested size-eight route failed
> same-seed value/total-score parity. Use the classification from
> `docs/plans/bayesfilter-lgssm-n5000-seed-batch-capacity-result-2026-07-20.md`
> for future exact-scope runs: size eight is memory-feasible but not
> claim-admissible; size one remains the correctness fallback until the batched
> semantic drift is repaired. Root-cause classification is GPU TF32
> batch-shape-dependent numerical amplification, not RNG/prepared-input drift
> or observed seed-row mixing.

Date: 2026-07-20
Status: `READY_FOR_TIME_LOCAL_SCORE_DECOMPOSITION_NO_NONLINEAR_TRANSFER`

## Criterion Amendment

The historical `[-5%,+5%]` score band is not DGP- or application-derived and
is no longer the primary scientific criterion for future LGSSM claims. Use the
simultaneous 95% CI for the mean relative bias and require that it contain zero
for every output. Preserve the old band only as historical policy evidence;
failure to reject zero bias is not an equivalence proof.

The preserved `N=5000` and `N=10000` claims both reject zero bias for value and
`q_scale` under this revised criterion. See
`docs/plans/bayesfilter-lgssm-kalman-zero-bias-ci-criterion-amendment-2026-07-20.md`.

## Follow-Up N=10000 Result, 2026-07-20

The exact independently tuned `T=50,N=10000,K=2500,4 x 4` scope completed
with controls `(sinkhorn_steps=20,balance_steps=8)`. The warm-start `(20,5)`
failed validation; `(20,8)` was the first blind direct-gate pass. The untouched
16-seed claim passed all engineering gates, but the Kalman screen failed:

- value mean relative error `+0.1735%`, simultaneous interval
  `[+0.1502%,+0.1968%]` outside `[-0.1%,+0.1%]`;
- `q_scale` mean relative error `-15.90%`, simultaneous interval
  `[-22.00%,-9.79%]` outside `[-5%,+5%]`;
- `phi1` and `phi2` intervals were contained; `phi3` and `r_scale` were
  inconclusive.

Relative to the independently tuned `N=5000` claim, `N=10000` was
descriptively worse for mean value (`+0.1735%` vs `+0.1482%`) and `q_scale`
(`-15.90%` vs `-9.91%`), although seed dispersion was lower for all six
reported outputs. The cross-N comparison is not a statistically supported
ranking because the seed blocks are independent.

This closes the larger-N repair attempt as a screen failure and reinforces the
predeclared next step: a same-observation/same-stream time-local score
decomposition of stationary, transition/proposal, observation-weight and
normalization, carried-weight, and Contract-E reset contributions, each checked
against its same partial scalar derivative. Do not transfer `(20,8)` to another
scope, do not infer a `1/N` rate, and do not launch nonlinear testing until the
LGSSM prerequisite is repaired or explicitly reclassified.

Result note:
`docs/plans/bayesfilter-lgssm-n10000-tuned-kalman-certification-result-2026-07-20.md`.
Aggregate SHA-256:
`edb67e93523d91b4ececebc9354aa962e3259265fbc180bbc7edacb88173739a`.

## Read First

0. The follow-up singleton `N=10000` diagnostic is recorded in
   `bayesfilter-lgssm-n10000-single-seed-kalman-diagnostic-result-2026-07-20.md`.
   Engineering gates passed, but it did not consistently reduce same-seed
   Kalman error: `q_scale` worsened from `+5.02%` to `+22.63%`. This is one-seed
   diagnostic evidence with cross-scope warm-start controls, not a bias estimate.

1. Read
   `bayesfilter-lgssm-particle2000-particle5000-kalman-bias-ladder-result-2026-07-20.md`.
2. Read the aggregate
   `docs/benchmarks/artifacts/lgssm_particle_bias_ladder_20260720/aggregate_final.json`.
3. Preserve the dirty worktree and all versioned attempts. Do not overwrite,
   reset, restore, or delete campaign artifacts.
4. Before any new GPU run, use trusted `nvidia-smi` and a trusted TensorFlow
   GPU/memory-policy probe.

## Frozen Outcome

The larger-particle experiment answered the immediate question:

- `N=2000` did not reduce bias: `q_scale=-45.75%`, simultaneous interval
  `[-58.79%,-32.70%]`.
- `N=5000` did reduce the descriptive `q_scale` bias to `-9.91%`, interval
  `[-17.72%,-2.11%]`, versus `-31.65%` at `N=1024`.
- The reduction is non-monotone and not a statistically supported cross-`N`
  ranking.
- `N=5000` remains `screen_fail` because the value interval
  `[0.1116%,0.1848%]` is wholly above the allowed region; `q_scale` and `phi3`
  are not contained.

Exact final controls are scope-bound only:

| Scope | Controls | Chunks | Engineering | Kalman |
| --- | --- | --- | --- | --- |
| `T=50,N=2000` | `(20,5)` | `K=2000`, `1 x 1` | PASS | `screen_fail` |
| `T=50,N=5000` | `(20,5)` | `K=2500`, `2 x 2` | PASS | `screen_fail` |

Do not transfer these controls to `T=10`, another particle count, or a
nonlinear model. Do not launch `T=10,N=5000` or nonlinear testing because the
memo's LGSSM success prerequisite did not pass.

## Next Required Phase

Create a fresh serious-campaign plan for the previously declared time-local
score decomposition on identical observations and random streams. Compare:

1. active canonical Contract E;
2. no-reset weighted recursion; and
3. exact differentiated Kalman increments.

For `q_scale`, separate and same-scalar-check:

- stationary initial-covariance contribution;
- transition/proposal contribution;
- observation-weight and likelihood-normalization contribution;
- carried previous-weight contribution; and
- Contract E moment/weight/transport reset contribution.

The plan must state whether each diagnostic is a promotion criterion, veto,
continuation veto, repair trigger, or explanatory diagnostic. It must not tune
Sinkhorn/balance against Kalman because direct marginals already pass.

## Preserved Artifacts

- Plan:
  `docs/plans/bayesfilter-lgssm-particle2000-particle5000-kalman-bias-ladder-plan-2026-07-20.md`
- Result:
  `docs/plans/bayesfilter-lgssm-particle2000-particle5000-kalman-bias-ladder-result-2026-07-20.md`
- Aggregate:
  `docs/benchmarks/artifacts/lgssm_particle_bias_ladder_20260720/aggregate_final.json`
- N=2000 scope:
  `docs/benchmarks/artifacts/lgssm_particle_bias_ladder_20260720/n2000_scope_attempt01/`
- First N=5000 failed holdout:
  `docs/benchmarks/artifacts/lgssm_particle_bias_ladder_20260720/n5000_scope_attempt02/`
- Final N=5000 repair scope:
  `docs/benchmarks/artifacts/lgssm_particle_bias_ladder_20260720/n5000_repair_scope_attempt01/`
- Multi-block resource probes:
  `docs/benchmarks/artifacts/lgssm_particle_bias_ladder_20260720/n5000_resource_probe_attempt01/`
  and
  `docs/benchmarks/artifacts/lgssm_particle_bias_ladder_20260720/n5000_resource_probe_attempt02/`.

The aggregate SHA-256 is
`fab768b961214fb5d962fd05a7d868802a65ae55a7cd0d73451de7f662dc495e`.

## Nonclaims

Do not state that the particle ladder proves a `1/N` rate, Kalman equivalence,
HMC/posterior readiness, nonlinear carryover, parameter-region validity,
method superiority, or a universal LEDH control setting. State directly that
`N=5000` is descriptively closer for `q_scale` but the claim remains wrong
relative to the frozen value-and-score certification target.
