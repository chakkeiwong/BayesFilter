# Result: Baseline Leaderboard (pre-generic-squared-TT), 2026-08-15

Plan: `docs/plans/bayesfilter-zhao-cui-generic-highdim-analytic-score-program-plan-2026-08-15.md`
(revision 2). This is the baseline the generic ZC-family squared-TT column
will be compared against; that column is BLOCKED pending the audit-mandated
P1A/P2 artifacts and is absent by design.

## Run manifest

- Commit `18cfe609` (dirty tree, uncommitted program work); TF 2.19.1,
  Python 3.11.14, float64, CPU-only (`CUDA_VISIBLE_DEVICES=-1`, intentional).
- Command: `python docs/benchmarks/run_baseline_leaderboard_20260815.py
  --output docs/benchmarks/artifacts/baseline_leaderboard_20260815/attempt01/result.json ...`
- Wall time 655 s. FD step 1e-5 (centered), single dataset per row.
- Artifacts: `docs/benchmarks/artifacts/baseline_leaderboard_20260815/attempt01/{result.json,result.md}`

## Leaderboard

| Model (n, m, T, p) | Algorithm | log-lik | same-target gap | score-vs-FD | wall s | claim |
|---|---|---|---|---|---|---|
| LGSSM (4,4,frozen,18) | exact Kalman QR | -13.282947 | oracle | 9.6e-09 | 1.3 | EXACT_ORACLE |
| ACTUAL_SV (1,1,20,2) | dense reference o401 | -45.182425 | ref | - | 0.1 | REFINED_NUMERICAL_REFERENCE |
| | fixed SGQF l2 | -45.400097 | 2.18e-01 | 1.0e-10 | 0.04 | CERTIFIED_APPROXIMATION |
| | ZC-family scalar TT (frozen) | -45.182425 | 3.5e-08 | - | 2.3 | CERTIFIED_APPROXIMATION |
| KSC_SV (1,1,20,2) | dense KSC reference o401 | -45.248036 | ref | - | 0.2 | REFINED_NUMERICAL_REFERENCE |
| | mixture Kalman (enum.) | -45.250502 | 2.47e-03 | - | 0.2 | CERTIFIED_APPROXIMATION |
| | mixture cut4 | -45.250502 | 2.47e-03 | - | 0.8 | CERTIFIED_APPROXIMATION |
| | mixture fixed SGQF | -45.250502 | 2.47e-03 | 6.9e-11 | 0.5 | CERTIFIED_APPROXIMATION |
| | mixture UKF | -45.250502 | 2.47e-03 | 6.9e-11 | 0.7 | CERTIFIED_APPROXIMATION |
| PP_T20 (2,2,20,6) | SVD-UKF | -103.296353 | x-alg | 3.7e-10 | 9.5 | SURROGATE_USEFULNESS |
| | fixed SGQF l2 | -103.296398 | 4.5e-05 (x-alg) | 3.1e-10 | 3.0 | SURROGATE_USEFULNESS |
| PP_T40 (2,2,40,6) | SVD-UKF | -200.725260 | x-alg | 1.7e-10 | 19.9 | SURROGATE_USEFULNESS |
| | fixed SGQF l2 | -200.725323 | 6.2e-05 (x-alg) | 1.9e-10 | 6.0 | SURROGATE_USEFULNESS |
| SIR_T20 (18,9,20,3) | SVD-UKF | -687.333706 | x-alg | 7.5e-10 | 4.1 | DIAGNOSTIC_ONLY |
| | fixed SGQF l2-axis | -687.301902 | 3.18e-02 (x-alg) | 8.3e-10 | 1.3 | DIAGNOSTIC_ONLY |
| SIR_T40 (18,9,40,3) | SVD-UKF | -1357.244661 | x-alg | 6.9e-10 | 8.3 | DIAGNOSTIC_ONLY |
| | fixed SGQF l2-axis | -1357.211898 | 3.28e-02 (x-alg) | 8.4e-10 | 2.5 | DIAGNOSTIC_ONLY |
| STRUCT_T20 (2,1,20,5) | structural SVD-UKF | -20.870258 | none | 6.2e-11 | 0.5 | SURROGATE_USEFULNESS |
| STRUCT_T100 (2,1,100,5) | structural SVD-UKF | -124.462628 | none | 2.8e-09 | 2.5 | SURROGATE_USEFULNESS |

x-alg = cross-algorithm gap vs the SVD-UKF cell (descriptive only).

## Are the differences reasonable? (interpretation)

1. **Analytic scores are uniformly healthy.** Every analytic-score cell
   passes FD at 6.9e-11 .. 9.6e-09 across all six model families. This is
   the Method A same-scalar property holding on fresh datasets/horizons,
   including T=40 SIR and T=100 structural — the score machinery the generic
   engine will inherit is in good order. (The LGSSM 9.6e-9 is the largest;
   consistent with an 18-parameter FD at step 1e-5 accumulating more
   cancellation error than 2-6 parameter models — an FD-quality effect, not
   a score defect.)
2. **ACTUAL_SV: SGQF gap 2.2e-1 vs TT gap 3.5e-8 — reasonable and
   informative.** The exact log-chi-square observation density is highly
   skewed; a level-2 sparse-grid Gaussian closure cannot represent that
   posterior update, while the density-based TT route reproduces the dense
   reference to 1e-8. This size ordering (density route ~exact; Gaussian
   closure ~1e-1) is exactly the pattern the generic squared-TT program
   predicts and is the motivating case for it.
3. **KSC_SV: all four Gaussian-family algorithms collapse to the identical
   value (-45.250502), gap 2.5e-3.** Reasonable: for the KSC target the
   observation update is conditionally linear-Gaussian per mixture
   component, so exact enumeration, cut4, SGQF, and UKF agree to machine
   precision with each other; the shared 2.5e-3 gap to the dense reference
   is the GPB1-style Gaussian collapse error, matching the 2026-08-14
   three-route benchmark (2.47e-3, same dataset family). Cross-check
   passed.
4. **Predator-prey: UKF-vs-SGQF spread 4.5e-5 (T20) and 6.2e-5 (T40).**
   Reasonable: both are Gaussian closures over the same additive-Gaussian
   RK4 model; they differ only in cubature rule, and this near-Gaussian
   fixture barely distinguishes them. The mild growth with T is expected
   accumulation. No same-target reference was run here (a dense n=2
   reference is affordable and is a P4 artifact).
5. **Austria SIR d=18: UKF-vs-SGQF spread 3.2e-2, roughly flat from T20 to
   T40 (per-step spread shrinking).** Reasonable for two different sigma-
   point rules on an 18-dimensional nonlinear model; consistent with the
   lane-era observation that closure differences at d=18 are 1e-2-scale.
   With no independent truth at d=18, which one is closer is exactly the
   question the generic squared-TT route is being built to answer —
   DIAGNOSTIC_ONLY stands.
6. **Structural model: score FD 6.2e-11 (T20) / 2.8e-9 (T100), simulation
   completion residual <= 2e-14.** The Ch18b constraint-support gate holds
   (deterministic completion computed, not noised); FD degradation with
   horizon is normal error accumulation in the FD baseline. No reference
   column yet: the dense (x_{t-1}, eps_t) quadrature reference is the P4
   structural-gate artifact.
7. **Wall times.** Everything is sub-20 s eager CPU; the T-scaling is
   roughly linear as expected (PP: 9.5 -> 19.9 s; SIR: 4.1 -> 8.3 s). The
   SV TT cell (2.3 s) is the per-model frozen scalar TT, not the future
   engine; no performance claims are made from these numbers (V12).

## Decision table

| Item | Status |
|---|---|
| Decision | Baseline admitted as the standing comparison surface for the generic squared-TT column |
| Primary criterion | All analytic scores pass FD; all same-target gaps where references exist are consistent with prior artifacts |
| Veto diagnostics | None fired (all finite, no floor/condition flags in status telemetry) |
| Main uncertainty | Single dataset per row; PP and structural rows lack same-target references (planned P4 artifacts) |
| Next action | Program unblock artifacts (UB-1 derivation note) -> P1A; PP/structural dense references in P4 |
| Not concluded | No cross-algorithm ranking (descriptive only, S-1 not invoked); no HMC/posterior claims; no statement about which SIR closure is more accurate |

## Nonclaims

Single dataset per row: all cross-algorithm gaps descriptive. Wall times are
eager CPU float64 on this host. The generic squared-TT column is absent by
design (blocked per Codex audit); its later entry will be gated by the P4
reproduction tests against this baseline.
