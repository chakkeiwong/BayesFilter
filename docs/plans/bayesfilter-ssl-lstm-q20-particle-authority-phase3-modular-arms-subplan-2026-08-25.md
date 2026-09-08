# Phase 3 One-Factor Modular Arms Subplan

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `PASS_GATE_ROLE_LIMITED`  
Budget cap: `18000 s`  
Output root:
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase3`

## Objective

Test ETPF, GenUT, invertible LEDH-PFPF, and (only if its own audit passes) a
full second-order ET-PF as one-factor modifications of the same frozen M0
authority. Preserve the raw weighted cloud and authority estimate separately
from every transformed or quadrature cloud.

## Arm order and contracts

1. **M1 ETPF:** verify `D^T 1 = 1`, `D 1 = Nw`, equal-weight transformed mean
   equals the weighted forecast mean, and equal-weight transformed covariance
   equals the weighted forecast covariance. Measure bridge rows and support;
   never call transformed rows IID or exact posterior draws.
2. **M2 GenUT:** use only as a local sigma-point/proposal or covariance
   preconditioner. Verify selected moments and status. Do not count its `2d+1`
   points as a replay bank or global mode representation.
3. **M3 LEDH-PFPF:** run only with a complete invertible map, inverse, Jacobian
   determinant lifecycle, pre-flow proposal, post-flow transition, observation,
   and covariance state. Apply the repository LEDH canonical route policies;
   a simplified historical lane is diagnostic-only.
4. **M4 ET-PF:** compare as an explicitly approximate filter against a trusted
   reference. It cannot replace M0 even if its moments look better.
5. **M5 combinations:** not in this phase; eligible only after component gates
   pass and a new staged/factorial subplan is refreshed.

## Fair comparison

Bind target, partitions, M0 protocol hash, particle count, seed domain, dtype,
backend, tuning scope, and wall-time budget. Tune controls on calibration data
only; freeze them before the audit partition. No arm may use transformed rows as
an unreported source of extra particles.

## Gates

| Arm | Promotion criterion | Veto |
|---|---|---|
| M1 | moment identities plus target/support audit | density or IID claim, bridge/support failure |
| M2 | selected-moment and finite/status checks plus utility diagnostic | sigma points treated as posterior samples or tail failure |
| M3 | affine identity, invertibility/step condition, determinant lifecycle, target checks | any unaccounted reset/no density term |
| M4 | reference agreement under its approximate target definition | presented as exact authority |

Covariance, ESS, whitening, bridge fraction, loss, and runtime are explanatory
unless the arm's own role contract says otherwise. No short-run ranking is
declared without uncertainty analysis.

## Required artifacts

Per-arm manifests, source/route identity, tuning-scope artifact, raw weighted
and transformed clouds, density receipts, target/reference comparisons,
decision table, and Phase 4 eligibility statement.

## Executed receipt

`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase3-attempt1-n100`
ran on the frozen N=100 M0 cloud. M1 and M2 passed their finite moment/status
contracts; M3 is descriptive affine scaffolding only; M4 is approximate. The
M0 cloud remains the only input eligible for a claim-bearing NeuTra screen.
