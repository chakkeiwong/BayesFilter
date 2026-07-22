# Contract E--TP Phase 8 Progressive-Score LGSSM Repair Plan

metadata_date: 2026-07-15
status: PRE_EXECUTION_AUDIT_PASS_READY_TO_IMPLEMENT
phase: 8A
master_plan: `docs/plans/bayesfilter-contract-e-tp-all-model-gradient-comparison-master-plan-2026-07-15.md`
algorithm_id: `contract_e_tp_experimental_v1`
campaign_budget: inherited 128 CPU core-hours, 64 trusted GPU-hours, at most three full-horizon attempts per model

## Phase Objective

Determine whether the recursive LGSSM score failure is caused by discarding
state-conditioned historical score information at each projection. Implement
and compare two fixed-feature repairs on the frozen `d_x=3,p=5` LGSSM center:

1. a compact progressive-score chart retaining the centered score field and its
   interaction with the next corrected-LEDH predictive contribution; and
2. an oracle diagnostic chart retaining the exact finite-horizon LGSSM
   continuation likelihood from a backward information recursion.

Run the smallest discriminating horizon first, then `T=2,10,50`. The
continuation arm diagnoses information sufficiency for this LGSSM; it is not a
transferable nonlinear feature or promotion oracle.

## Entry Conditions

- Phase 3 engineering checks pass and its candidate failure is preserved.
- Phase 4 structural teacher checks pass.
- The frozen target is dataset seed `81100`, physical parameter center
  `(0.72,0.55,0.35,0.35,0.45)`, transition-first timing, and the differentiated
  Kalman likelihood in `ledh_contract_e_tp_lgssm_tf.py`.
- Existing Phase 3 charts are historical baselines and are not overwritten.
- Contract E--TP remains experimental and cannot enter canonical, leaderboard,
  default, or HMC-facing paths.

## Research Intent And Evidence Contract

| Field | Contract |
| --- | --- |
| Main question | Does retaining state-conditioned historical score information repair the recursive value/score loss observed after `T=2`? |
| Baseline | Existing mass/moment/one-step-predictive Phase 3 chart, plus the exact differentiated Kalman value and score. |
| Candidate mechanism | Compact centered target-model score marks retain a state-dependent summary of historical score information; the LGSSM continuation feature preserves the exact-model remaining-horizon likelihood assigned to teacher/student measures on a fixed chart. The executed finite LEDH scalar remains the only owner of its total derivative. |
| Primary promotion criterion | At `T=2,10,50`, same-scalar AD/FD passes; chart/support invariants pass; and value plus every Kalman score component pass the already frozen center screens without sign reversal. |
| Promotion veto | Nonfinite output; same-scalar derivative failure; target/timing mismatch; feature residual beyond derived float64 solve error; rank loss; nonpositive weight; runtime chart switch; oracle continuation evaluated with a different likelihood program; or test data used to tune a nonlinear transferable feature. |
| Continuation veto | The finite target or Kalman oracle is invalid; no positive fixed chart exists for either repair at the center after the declared capacity ladder; or campaign budget is exhausted. Candidate failure alone triggers the next diagnosis. |
| Repair trigger | Earliest per-time value/score drift, incorrect mark recursion, omitted local corrected-weight term, backward-information mismatch, chart conditioning, or quadrature resolution failure. |
| Explanatory only | Cosine similarity, norm ratios, raw component gaps, chart condition number, weight margins, runtime, and memory. |
| Nonclaims | No off-center validity, online score theorem, exact nonlinear filtering, nonlinear feature transfer, HMC readiness, default readiness, leaderboard admission, or method superiority. |
| Artifact root | `docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase8_progressive_score_lgssm_<run_id>/`; every attempt uses a fresh directory. |

## Mathematical Objects

The progressive mark is not a cumulative scalar and not a derivative of a
normalized particle weight. For each retained state it approximates the
target-model conditional additive score

```text
tau_t(x) = E[complete-data additive score through t | X_t=x, y_1:t]
c_t(x)   = tau_t(x) - E_eta[tau_t]
```

The conceptual compact feature candidate is

```text
[mass, selected state moments, next_predictive,
 c_t[1:p], next_predictive*c_t[1:p]]
```

The marks have an additive gauge: adding one common vector to all parent marks
adds it to all child marks, and teacher re-centering removes it. The selected
marks are explicitly re-centered under the student weights before the next
step. Therefore the five zero-mean `c_t` equations are telemetry invariants
rather than projection rows. The tested chart uses only
`next_predictive*c_t[1:p]`: 16 rows total, not 21. This also removes the exact
first-step rank deficiency where the observation-scale `c_t` row lies in the
quadratic moment span.

The oracle diagnostic replaces the one-step predictive row by

```text
R_t(x) = p_theta(y_(t+1):T | X_t=x),
```

computed by an exact TensorFlow backward information recursion for the same
LGSSM target. Matching `E[R_t]` as a parameter-dependent primal identity
matches the total `p`-vector derivative of that exact-model continuation
expectation; gradients are never written into the primal feature vector. This
does not by itself match the subsequently executed finite LEDH recursion.

For the compact arm, teacher child marks use the target-model backward kernel:

```text
B[k,i] proportional_to parent_weight[i] * f_theta(child[k] | parent[i])
tau_child[k] = observation_score(child[k])
             + sum_i B[k,i] * (tau_parent[i] + transition_score(parent[i], child[k]))
```

The selected teacher anchors carry their centered marks into the next step and
are re-centered under the selected weights. The chart preserves
`next_predictive*c`; zero mean is checked as a separate invariant. These are
diagnostic primal features of the
finite Contract E--TP scalar. They are not asserted to equal the tangent measure
of its moving atomic cloud; correctness of that scalar's derivative is checked
separately by autodiff and same-scalar FD.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Earliest diagnostic |
| --- | --- | --- | --- |
| Center-only `delta_grad=0.05` and value boundary `0.001` | Prior owner-approved Phase 8 center screen; inherited, not a new universal threshold | Mistaken off-center/HMC promotion | Artifact scope and nonclaim check |
| Same-scalar FD `0.05*sqrt(p)` | Owner policy, FD-only | Misused as Kalman/cross-method margin | Schema assertion separating the two fields |
| Order-3 quadrature first | Phase 3 baseline hypothesis | Teacher error mistaken for feature failure | `T=1` and order `3,5,7` convergence control |
| Compact `2p` target-model score rows | Poyiadjis/Del Moral backward statistic plus revised-chapter boundary; diagnostic hypothesis, not a finite-LEDH tangent theorem | Feature span remains insufficient or model-filter marks mismatch the useful finite-program statistic | Compare `T=2`, then earliest drift through `T=10` |
| Exact LGSSM continuation row | Kalman backward information identity; oracle diagnostic | Future-data leakage misrepresented as deployable algorithm | Route identity and explicit diagnostic-only nonclaim |
| Square frozen chart | Existing Phase 1 implementation | No strictly positive basis with enlarged features | Offline LP support audit before evaluation |
| Center preparation | Prior approved scope | Chart invalid away from center | No off-center execution or claim |

## Skeptical Pre-Execution Audit

- **Wrong baseline:** vetoed by recomputing the exact Kalman likelihood from the
  same observation prefix and parameter coordinates in every result.
- **Proxy promotion:** feature/tangent identities are engineering evidence only;
  Kalman value and componentwise score are the scientific criteria.
- **Hidden future information:** the continuation feature is explicitly an
  LGSSM oracle diagnostic and cannot nominate nonlinear features or enter a
  leaderboard/default path.
- **Higher-order derivative trap:** do not construct primal features from an
  autodiff score vector. Use analytic/local-score formulas for marks and use the
  parameter-dependent continuation scalar directly.
- **Cost mismatch:** record actual parent, innovation, child, feature, and
  pairwise backward-kernel counts. Do not label an `O(N^3)` teacher operation
  `O(N^2)`.
- **Unfair success:** a continuation arm that uses the whole evaluation horizon
  establishes a finite-horizon certificate only. It does not establish an
  online or nonlinear method.
- **Misleading successful command:** artifacts must contain per-time increments,
  cumulative scores, feature residuals, mark centering, minimum weight, chart
  condition, and oracle gaps; a completion marker alone is insufficient.

Pre-execution disposition after audit: pass. The chapter now distinguishes the
dominated model-filter tangent theorem from the moving atomic finite-LEDH
program. The compact mark arm is a diagnostic feature hypothesis, the
continuation arm is an LGSSM exact-model oracle diagnostic, and neither replaces
same-scalar differentiation. Claude Opus returned `CLAUDE_PROBE_OK` to a health
probe but no content to three bounded substantive file/line reviews; this is
recorded as an advisory-review limitation rather than mathematical approval.
Local review checked the cited Del Moral--Doucet--Singh Section 2 equations
directly and found no remaining execution-blocking mismatch.

## Required Artifacts

1. Revised LaTeX section with definitions, propositions, proofs, complexity,
   and limitations.
2. TensorFlow progressive-score/backward-information implementation with no
   NumPy algorithmic backend.
3. Tests for backward information against direct Kalman conditional likelihood,
   mark centering, score-covariance identity, fixed-chart derivative identity,
   and fail-closed chart behavior.
4. Unique preparation artifacts for every horizon/feature/capacity attempt.
5. Unique evaluation artifacts for `T=2,10,50`, including exact command,
   commit, CPU/GPU status, seeds, wall time, and source artifact hashes.
6. Phase result with decision and inference-status tables plus post-run
   mathematical red-team review.

## Execution Ladder

1. Prove and unit-test the backward-information scalar on `T=2` against a
   direct conditional Kalman calculation at multiple states and parameter
   perturbations.
2. Unit-test the target-model progressive mark recursion on a tiny teacher;
   verify `E[c]=0`, analytic local scores against fixed-state autodiff, and the
   target-model covariance representation against the exact Kalman tangent.
3. Prepare positive fixed charts at `T=2`. Run projected-versus-uncompressed
   value/score identity and same-scalar FD.
4. Prepare and run `T=10`. Record the first per-time deviation from Kalman.
5. Only if `T=10` passes the original center criteria, prepare and run `T=50`.
6. If order 3 fails and the teacher/reference control indicates quadrature
   error, run orders 5 then 7. Do not vary feature family and quadrature order in
   the same diagnostic.
7. If the compact score arm fails while the exact continuation arm passes,
   classify the compact span as insufficient and design the next continuation
   basis. If both fail with valid charts, trace timing, normalization, mark
   update, and backward-information wiring before changing features.

## Required Checks And Reviews

- CPU-hidden focused pytest suite for Phase 1--5 plus new Phase 8 tests.
- TensorFlow autodiff versus centered FD of the same finite scalar.
- Teacher/student feature-value and tangent identity.
- Direct conditional-Kalman check of every continuation feature used in a chart.
- Python compilation and `git diff --check` on touched paths.
- LaTeX build or, if the full book has unrelated failures, a focused log proving
  the edited chapter introduced no new undefined control sequence/reference.
- Pre-run mathematical review of this plan and post-run mathematical review of
  the result. Review is advisory; a material mathematical finding is blocking.

## Forbidden Claims And Actions

- Do not call the global cumulative score a particle feature.
- Do not equate a normalized-weight derivative with the conditional additive
  score without a derivation for the exact finite program.
- Do not insert stopped or externally overwritten gradients into the value path.
- Do not loosen `delta_grad`, the value boundary, or the FD policy after seeing
  results.
- Do not use runtime active-set selection, clipping, silent feature dropping, or
  overwrite an existing artifact.
- Do not proceed to nonlinear promotion from an LGSSM continuation-oracle pass.
- Do not claim canonical, leaderboard, default, HMC, or NAWM readiness.

## Exact Handoff Conditions

Phase 8A may hand off to nonlinear chart preparation only if:

1. the progressive implementation passes its own mathematical and derivative
   checks;
2. `T=2,10,50` all pass the frozen center value and componentwise Kalman-score
   screens with no sign reversal at the accepted refinement rung;
3. all charts are strictly positive, fixed, finite, and full rank;
4. the result clearly distinguishes a general progressive-score result from an
   LGSSM future-continuation oracle result; and
5. terminal review finds no target, differentiation, or evidence-contract error.

If only the continuation oracle passes, the handoff is a diagnosis that missing
future-score information caused the failure. It is not sufficient to claim the
general compact progressive method succeeded; Phase 8 must continue with a
non-oracle continuation-feature repair.

## Stop Conditions

Stop this phase for human direction only if the target or mathematical claim
must change, the campaign compute budget must expand, a destructive/environment
mutation is required, or all declared positive-chart/capacity repairs fail.
Localized code, chart-preparation, serialization, XLA, or numerical failures
are repair-and-retry events within the existing campaign.
