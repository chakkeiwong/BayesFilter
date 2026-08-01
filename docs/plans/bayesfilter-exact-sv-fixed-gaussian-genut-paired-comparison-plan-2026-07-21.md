# Exact-SV Fixed Gaussian GenUT Paired Comparison Plan

Date: 2026-07-21

Status: `CORRECTED_SINGLE_DGP_GENUT_SIGNAL_PROMISING_REPLICATION_REQUIRED`

> **Correction, 2026-07-22:** The `original` arm is not SV-DGP data and is
> scientifically ineligible.  Including it in the cross-dataset criterion was
> a planning error; that criterion and its failure are revoked.  Only the
> `fresh_dgp` arm is eligible scientific evidence.  See
> `bayesfilter-exact-sv-nondgp-fixture-demotion-correction-2026-07-22.md` and
> `correction_20260722.json` beside the artifact.

Execution note: `attempt01` stopped before any claim row was preserved because
one evaluator was reused across particle sizes and TensorFlow relaxed the
particle dimension to unknown.  The candidate's static-shape guard correctly
failed closed.  The localized repair compiles a separate evaluator for each
`(N, sinkhorn_steps)` scope and proceeds in fresh `attempt02`; the research
question, target, data, controls, seeds, criteria, and compute budget are
unchanged.

## Research Intent Ledger

| Field | Contract |
|---|---|
| Main question | In scalar exact transformed SV at `T=50`, does replacing the fixed Cubature residual design by fixed Gaussian GenUT reduce the persistent dense-reference score discrepancy while retaining the observed likelihood-value accuracy? |
| Mechanism under test | Cubature and fixed Gaussian GenUT restore the same mean and variance, but scalar Cubature has standardized fourth moment `1` while Gaussian GenUT has standardized fourth moment `3`.  A paired improvement would implicate missing residual kurtosis as one contributor to accumulated score error. |
| Candidate arms | Cubature rows `{-1,+1}` with masses `{1/2,1/2}` and fixed Gaussian GenUT rows `{-sqrt(3),0,+sqrt(3)}` with masses `{1/6,2/3,1/6}`; both use the same finite Sinkhorn transport and Contract E--Chol restore. |
| Exact comparator | Independent float64 dense-grid exact log-chi-square transformed-SV filter and its diagnostic differentiated score increments. |
| Data scopes | Corrected active scope: one sequence generated from the stated stationary SV DGP.  The archived direct-Normal sequence is historical engineering-only data and cannot enter any criterion. |
| Particle counts | `N in {1002,1998}`.  Both are divisible by six, so both designs have exact equal-weight row representations. |
| Controls | Fixed `epsilon=2`, `ridge=1e-5`, crossed `sinkhorn_steps in {4,8}`.  No method-specific tuning or selection is allowed. |
| Claim seeds | Sixteen common particle seeds `3620..3635`.  Initial/process noise is generated at `N=1998` and sliced for `N=1002`. |
| Design coupling | For each `(N,seed)`, one stateless random rank ordering assigns both designs: Cubature divides ranks in halves; GenUT divides the same ranks in proportions `1/6,2/3,1/6`. |
| Expected failure mode | If matching Gaussian kurtosis is insufficient, value remains accurate but score bias persists under both rules; finite transport/reset accumulation or unrepresented skewness/tails remain plausible. |
| Promotion criterion | None.  This is a mechanism diagnostic and cannot promote a default. |
| Mechanism-support criterion | Corrected status: no population mechanism criterion can be met from one DGP sequence.  The fresh-DGP paired reduction is nomination evidence requiring multi-DGP replication. |
| Value-preservation screen | For every data/design/step arm at `N=1998`, the pointwise 95% interval for mean dense-value error includes zero and has half-width at most `0.25`.  Failure blocks a claim that GenUT preserves value accuracy but does not invalidate the harness. |
| Hard veto | Nonfinite output; non-GPU candidate placement; memory-growth failure; XLA or TF32 mismatch; exact design moment failure; row/column/reset residual `>=1e-2`; score increments not summing to total within `1e-3`; recursive-versus-same-scalar FD audit error `>0.05`; dense-reference refinement failure. |
| Continuation veto | Dense refinement differs by more than `5e-5` in value or `2e-4` in either score; corrupt/missing artifact; OOM; seed/data identity mismatch. |
| Explanatory diagnostics | Raw value/score errors and intervals, paired absolute-mean-error changes, regularized reference-OPG score distances, per-seed OPG differences, per-time cumulative score-error localization, runtime, and allocator peak. |
| Nonclaims | No adaptive GenUT result, exact Fisher metric, exact-posterior HMC correction, Contract E ranking, broad SV/DGP generality, MLE/HMC readiness, leaderboard/default readiness, or NAWM result. |

## Mathematical Contrast

For a scalar standardized residual `Z`, the two fixed designs are

```text
Cubature:       P(Z=-1)=1/2, P(Z=+1)=1/2
Gaussian GenUT: P(Z=-sqrt(3))=1/6, P(Z=0)=2/3, P(Z=+sqrt(3))=1/6
```

Both have zero mean, unit variance, and zero third moment.  Cubature has fourth
moment one; Gaussian GenUT has fourth moment three.  Because Contract E restores
the same weighted mean and covariance after each arm, the comparison isolates
the effect of this fixed residual shape conditional on the common transport,
controls, noises, observations, and row-rank coupling.  It does not isolate all
higher-order distributional differences, because neither fixed symmetric rule
adapts to the filtering cloud's realized skewness or tail shape.

The fixed GenUT design is parameter-independent.  Its design tangent is
therefore exactly zero, so the existing recursive score remains the total
derivative of the executed finite scalar on its checked differentiable branch.
This statement would not apply to an adaptive moment-estimated GenUT design.

## Statistical Contract

1. One complete particle-filter seed is the sampling unit.  Particle rows and
   time increments are not independent replications.
2. Raw value and score intervals use `t(15)` pointwise 95% intervals.  The two
   score coordinates additionally receive Bonferroni familywise intervals.
3. Paired percentile bootstraps resample the 16 common seeds and report:
   `abs(mean GenUT error) - abs(mean Cubature error)` for value and each score;
   and the GenUT-minus-Cubature mean per-seed regularized-OPG distance.
4. The regularized predictive-score OPG uses the pre-existing diagnostic
   convention: zero diagonal shrinkage, base ridge one, zero ridge floor, and
   identity ridge scale.  It is a predeclared descriptive metric, not exact
   Fisher information and not an acceptance threshold.
5. Only the fresh-DGP sequence is scientifically eligible.  One sequence does
   not support population-wide SV generalization.

## Data And Dense-Reference Contract

- Fresh-DGP scope: generate a stationary scalar AR(1) latent path and exact
  Gaussian observation shocks from a frozen stateless seed at the realized
  float32-origin `theta=[0.25,-0.15]`, `sigma=1`, then form
  `z_t = x_t + 2 log(beta) + log(e_t^2)`.
- Hash and preserve the transformed observation tensor.
- Refine the dense reference at `(order,radius)` equal to
  `(257,8)`, `(401,8)`, and `(401,10)` and use `(401,10)` as the comparator.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
|---|---|---|---|
| Fixed Gaussian GenUT | Ebeigbe Gaussian specialization; mechanism hypothesis | Gaussian fourth moment may not match the skewed filtering posterior | Compare paired full-horizon value/score and per-time drift; make no adaptive-GenUT claim |
| `N={1002,1998}` | Six-divisible mechanism counts; convenience baseline | Limited particle ladder may miss asymptotic behavior | All causal comparisons are paired within this DGP run |
| `epsilon=2`, ridge `1e-5` | Repeated prior tuned selections; warm-start baseline | Cross-scope transfer may be poor | Cross 4/8 steps; residual gates; no optimum/default claim |
| 4 and 8 Sinkhorn steps | Directly addresses prior `N=2000` tuning confound | Neither count may be adequate | Marginal residuals and paired direction across both counts |
| Random rank row coupling | New comparison-design choice | One row pairing may favor a design | Stateless seed-specific pairing and 16 particle seeds; DGP-valid row-order sensitivity remains future work |
| One fresh DGP sequence | Bounded mechanism check | Dataset variability remains unidentified | Explicit single-dataset nonclaim and required multi-DGP replication |
| OPG settings `(0,1,0,I)` | Existing LGSSM diagnostic convention | Ridge/coordinates can make magnitude look favorable | Emit metric eigenvalues, raw errors, and maximum diagonal statistic; no threshold |

## Skeptical Plan Audit

- **Wrong baseline:** repaired.  The comparator is the exact log-chi-square
  dense filter, not KSC, a Gaussian closure, or an LGSSM transfer.
- **Dataset validity:** repaired after execution by excluding the direct-Normal
  fixture completely.  Only the stationary SV-DGP sequence is eligible.
- **Tuning confound:** repaired.  Cubature and GenUT share controls, and both 4
  and 8 Sinkhorn steps are reported without selecting a winner.
- **Row-order confound:** bounded but not eliminated.  Both designs use the
  same seed-specific quantile rank coupling; DGP-valid sensitivity remains
  future work.
- **Proxy promotion:** repaired.  OPG distance, FD, residuals, and runtime are
  diagnostic or veto quantities, not default/promotion criteria.
- **Score correctness:** fixed Gaussian GenUT is parameter-independent, so a
  zero design tangent is correct.  Same-scalar FD remains an audit, not runtime
  score computation.
- **MH overclaim:** prevented.  Good approximate values here do not prove exact
  posterior targeting or uniform value accuracy over the posterior region.
- **Statistical overreach:** corrected.  Seeds quantify particle-estimator
  uncertainty conditional on one dataset and do not estimate DGP population
  uncertainty.
- **Artifact adequacy:** required artifacts preserve observations, dense
  increments, all per-seed rows, route/design/controls, paired comparisons,
  OPG construction, memory, placement, source hashes, and timings.

Corrected audit decision: `SINGLE_DGP_NOMINATION_ONLY`.  The paired fixed-control
program answers whether GenUT changes the finite estimator on one valid DGP
sequence.  It cannot establish a population mechanism without multi-DGP
replication.

## Phases And Budget

1. Add exact Gaussian-GenUT equal-weight design construction to the paired
   benchmark, with moment/count/order tests.
2. Add paired result aggregation, OPG diagnostics, and artifact-integrity tests.
3. Run focused CPU-hidden tests and syntax/diff checks.
4. Execute one GPU/XLA campaign under
   `docs/benchmarks/artifacts/exact_sv_fixed_gaussian_genut_paired_20260721/attempt01/`.
5. Write the terminal result at
   `docs/plans/bayesfilter-exact-sv-fixed-gaussian-genut-paired-comparison-result-2026-07-21.md`.

The scientifically eligible portion contains 128 primary claim rows
(`1 DGP dataset * 2 N * 2 steps * 2 designs * 16 seeds`) and four
representative same-scalar FD audit rows.  The archived non-DGP rows and
alternating-layout sensitivity are ineligible.  One localized infrastructure
retry was used.

## Execution Outcome

`attempt02` completed its engineering gates in `73.885` seconds after the
localized static-shape repair.  On the only eligible fresh-DGP sequence, fixed
Gaussian GenUT produced a statistically supported gamma-score and OPG
improvement under both Sinkhorn counts, and every GenUT value interval included
the dense value.  This is single-DGP nomination evidence only.

Terminal result:

`docs/plans/bayesfilter-exact-sv-fixed-gaussian-genut-paired-comparison-result-2026-07-21.md`
