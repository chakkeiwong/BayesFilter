# Tuned Cubature/GenUT LGSSM Comparison Plan

Date: 2026-07-21

## Research Intent

Question: after scope-specific tuning, does the Cubature/GenUT staged reset
produce a materially better value/score error profile than the preserved
Contract E `N=5000` and `N=10000` runs on the same LGSSM target?

This is an LGSSM feasibility study for future high-dimensional nonlinear
filtering. It is not an LGSSM-estimation objective and is not a NAWM test.

## Candidate And Scope

The candidate remains:

```text
transition/current increment -> positive Sinkhorn OT -> barycentric cloud
-> residual injection -> Cholesky restoration
```

The claim scope is `d=3`, `N=1008`, `T=50`, float32/TF32 GPU execution, and
the canonical dataset seed `81100`. Cubature and Gaussian GenUT are identical
for the current Gaussian moments (`s=0`, `k=3`); Cubature is the tuning
representative and both designs are evaluated in the untouched claim.

## Tunable Controls

The tuning mechanism exposes and records:

- Sinkhorn entropic scale `epsilon`;
- Sinkhorn iteration count `sinkhorn_steps`;
- Cholesky/reset ridge `ridge`.

The residual design, reset cadence, cost normalization, stabilizer, particle
count, dtype, backend, and horizon are scope-bound rather than tuned in this
campaign. `FD_EPS` is a derivative diagnostic only and is not an algorithmic
control.

Candidate grid:

```text
epsilon       = {0.5, 1.0, 2.0, 4.0}
sinkhorn_steps= {4, 8, 16}
ridge         = {1e-6, 1e-5}
```

This is 24 configurations. Controls are selected independently of GenUT
because the two designs are algebraically identical for this Gaussian scope.

## Data And Seeds

All arms use the same fixed observations (`DATASET_SEED=81100`) and the same
Kalman oracle. Seed partitions are disjoint:

| Partition | Seeds | Role |
|---|---|---|
| Calibration | `82300..82303` | reject non-finite/invalid candidates and nominate candidates |
| Validation | `82310..82313` | select the frozen control tuple |
| Claim | `82320..82335` | untouched 16-seed final comparison |

The claim seed set is not reused from the prior feasibility run. Seed labels are
not treated as paired common-random-number evidence across different methods;
the final comparison remains descriptive unless a paired design is explicitly
run.

## Selection Rule

For each candidate, run the Cubature representative on calibration and
validation seeds. A candidate is invalid if any value/score is non-finite,
replay fails, reset residual exceeds `5e-4`, or either Sinkhorn marginal
residual exceeds `5e-4`.

Among candidates valid on both partitions, select the minimum validation
objective:

\[
J = \max_{j\in\{value,phi_1,phi_2,phi_3,q,r\}}
\left|\operatorname{mean}_{s\in validation} e_{s,j}\right|.
\]

Tie-break by the validation sum of squared coordinate means, then lower
Sinkhorn iterations, then lower ridge. Calibration is not used after
nomination and the claim data are never used for selection.

## Evidence Contract

| Item | Contract |
|---|---|
| Primary comparison | Six-coordinate HMC-relative-error means, SD, SE, simultaneous CI using `3.036283222821165` |
| Frozen screen | value margin `0.001`, score margin `0.05`; same screen as prior Contract E aggregates |
| Hard vetoes | candidate/claim non-finite, replay mismatch, invalid reset, invalid Sinkhorn marginals, missing/stale tuning artifact, or scope mismatch |
| Tuning evidence | complete candidate table, partition identities, selected tuple, selection objective, source hash |
| Claim evidence | fresh 16-seed Cubature and GenUT result with frozen selected controls |
| Comparison evidence | generated T=50 table against preserved N=5000/N=10000 aggregates |
| Nonclaims | no exact-filtering proof, no superiority claim from descriptive means, no 1/N rate, no nonlinear/NAWM conclusion |

## Skeptical Plan Audit

- Tuning on the claim seeds would make the comparison invalid; the partitions
  above prevent that leakage.
- Selecting on only value would hide score failure; the objective uses all six
  coordinates.
- Selecting by Kalman error changes the purpose relative to the prior internal
  Contract E tuning, so the artifact explicitly labels this as a target-aware
  feasibility tuning study and does not promote a default.
- The grid is finite, not globally optimal. “Optimally tuned” means best in the
  declared grid under the declared validation objective.
- The no-JIT route remains an explicit execution exception because the prior
  Cubature score graph has a reproducible XLA Cholesky-gradient layout failure.
- The new `N=1008` scope is not a controlled particle-count comparison to
  `N=5000/10000`; the final report must retain this noncomparability.

## Budget And Stop Conditions

At most 24 candidates x 8 tuning seeds plus 16 claim seeds x 2 designs are
allowed (bounded, no retries beyond localized infrastructure repair). Stop as
invalid on any hard claim veto. If all candidates fail validation, preserve the
failure table and run no claim. A valid but screen-failing claim is negative
candidate evidence, not evidence against the broader cubature direction.

## Artifacts And Commands

```text
docs/benchmarks/artifacts/lgssm_cubature_genut_tuning_20260721_attempt1/
docs/benchmarks/artifacts/lgssm_cubature_genut_tuned_claim_20260721_attempt1/
docs/benchmarks/artifacts/lgssm_cubature_genut_tuned_comparison_20260721.json
```

The tuning driver writes a selected-control artifact. The claim driver refuses
to run without that artifact and verifies its scope, source hash, and selected
tuple before evaluating the untouched claim.
