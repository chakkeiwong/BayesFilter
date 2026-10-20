# N5 modern HMC mechanics bridge: active plan

Updated 8 October 2026. The parent is
[the modern-method implementation program](bayesfilter-modern-hmc-implementation-plan-2026-10-07.md).
Results and the next unfinished phase live in
[the execution record](bayesfilter-modern-hmc-execution-2026-10-07.md).
The previous cumulative plan is preserved in
`artifacts/modern-hmc-implementation-2026-10-07/N5/plan-history/`.

## Research intent and evidence contract

Question: can each current nonlinear adapter execute the public candidate-set
tuner with a frozen modern transition, independent measurement/verification,
complete attempted-work accounting, and retention/reload of every verified
member? This is real tuning under a mechanics-only acceptance policy. It is
not yet the posterior or efficiency comparison required by N5.

The exact LGSSM fixture is the plumbing control. The actual core nonlinear
adapters are accumulation, predator–prey UKF, predator–prey sparse quadrature,
SIR sparse quadrature and structural UKF. Each approximate filter defines its
own numerical posterior; this work does not establish its agreement with the
underlying nonlinear generative posterior. No historical learned maps or old
filter qualification labels confer new authority.

| Diagnostic or outcome | Role and action |
| --- | --- |
| Two verified candidate IDs, all declared fresh trials, raw evidence/charge reconciliation, retained export/reload | Primary engineering pass criterion |
| Replicated lower acceptance bound | Tuning qualification criterion under this explicit mechanics policy |
| Finite state/value/score, filter status, all-chain movement, correction and stream identity | Candidate validity checks; failure prevents membership and triggers localization |
| Missing/corrupt evidence, wrong source or target, silent legacy fallback, invalid comparator | Harness continuation veto until repaired |
| Overall compute/attempt cap | Continuation veto for further launches; preserve partial evidence |
| R-hat, ESS, MCSE | Reporting/posterior assessment only; never tuning admission, ranking, repair or termination |
| Other acceptance summaries, elapsed time, first-call timing, realized lengths | Explanatory/pricing only; no method ranking |

A failed candidate does not reject the method. A numerical failure can require
new geometry or a smaller epsilon; an adapter/status failure requires its own
repair. Neither is permission to lower the evidence standard. In this narrow
bridge, success requires both declared members; a completed scheduler with
one or no verified members is explicitly incomplete.

No posterior convergence, sufficient burn-in, posterior precision, sampler
superiority, production default or release claim follows. The cheap adversaries
for the engineering mechanism are existing fixed-HMC wiring and analytic
Gaussian correction/identity/cycle controls in the tests. Their role is to
catch broken dispatch and false mixing authority, not to rank nonlinear methods.

## Frozen design and numerical provenance

Use `tune_hmc_kernel` with repository-issued fixed identity geometry and
`HMCTransitionProfile("hmc", "uniform_duration")`. Rounding, correction and
refresh policy remain part of candidate identity. The current target capability
scope must survive the affine wrapper; the search has its own identifier.

| Choice | Provenance, justification, risk and earliest check |
| --- | --- |
| Four fixture starts | Required by the replicated policy. Data-independent mechanics centers; not a prepared posterior geometry. Check finite value/score/status before tuning. |
| Maximum lengths 2 and 3, epsilon 0.1 | Inherited narrow bridge hypotheses, chosen for cheap dispatch coverage. Can be unstable or mix poorly; do not transfer to scientific fits. SIR already rejected this epsilon. |
| 64 transitions, zero discarded prefix | Mechanics-only design. No burn-in/convergence assertion. Actual posterior fits need separate warm-up and precision criteria. |
| 16 independent measurement and 16 fresh verification trials per candidate | Revised after one/four trials proved underpowered. With the declared betting allocation the capital cannot cross the threshold at four trials even for ideal scores. Sixteen is feasible, not a guarantee of qualification. |
| Lower-bound region `(0.1, 1.0)`, family alpha 0.05, temporal tolerance 0.1, displacement threshold zero | Inherited explicit engineering policy. A low floor checks the mechanism; it does not establish scientific usefulness or a new default. Preserve all policy fields in the manifest. |
| Ten scheduler units, 150,000 conservative work units, no within-cell repairs/refinement | Bounded two-candidate mechanics design. A complete cell costs `16 * 64 * 4 * 2 * (6 + 8) = 114,688` units; reserves cover every prefix and verification call. |
| `2 * (max_steps + 1)` work per transition | Derived bound for value/score plus separate status evaluations, including inactive batch rows. Actual call counts must be reported separately. |
| Seeds `(20261008, 31 + target_index)` | Reproducibility choices, not robustness evidence. Candidate/stage/trial streams are independently derived and checked. |
| CPU reference or GPU/XLA | Explicit execution mode. CPU hides CUDA. GPU uses trusted execution, fresh inventory and memory growth configured before any target import/device work. |

No nontrivial choice above is a recommended scientific tuning default.

## Executable sequence and observed repairs

1. Preserve existing results, repair source/data provenance, failure records,
   wall supervision and the distinction between scheduler completion and
   verified membership. Run focused negative tests before extending the bridge.
2. Complete the remaining nonlinear CPU mechanics cells. Attempt 07 verified
   both members for predator–prey sparse quadrature and structural UKF. SIR
   exposed a missing condition estimate in its native status. Preserve its
   already computed innovation spectrum and report the maximum over time of
   `lambda_max(S_t) / lambda_min(S_t)`; for SPD covariance this is the spectral
   condition number. Do not change value/score, clipping, floors or recursion.
   Existing finite-difference/value-only checks and full-wrapper status tests
   must pass. Claiming condition one would be wrong for nine observations.
3. Run a short trusted GPU diagnostic on LGSSM and accumulation, for randomized
   HMC and MAMS. Eight transitions, epsilon 0.01, maximum length 3 and seed
   `(804,331)` are localization hypotheses, not tuned choices. Require finite
   health, exact lengths/Metropolis decisions and same-device replay; compare
   samples, log densities, momentum and correction with `atol=rtol=1e-9`
   (explicit float64 diagnostic tolerance). Attempt 08 found mismatched
   non-XLA/XLA random streams and a reference adapter missing its XLA flag.
   The small saved RNG diagnostic established matching CPU/XLA and GPU/XLA
   streams. Attempt 09 uses that valid comparator and passed all four cells.
   This does not promise identical seeds across non-XLA and XLA execution.
4. Exercise the GPU public tuner with LGSSM, source hashes, 64 native trials,
   all-member export/reload and exact attempted-work audit. Attempt 10 found
   that the launcher replaced the base target scope, correctly losing XLA
   authority. Preserve the base scope without relaxing readiness checks.
   The focused scope regression and attempt 12 passed.
5. Rerun SIR after the status repair. Attempt 11 passed preparation but epsilon
   0.1 caused a filter-covariance eigensolver failure in the first trial. The
   bounded local curvature diagnostic uses central differences of the analytic
   score at the fixed start bank, with `1e-4` and `1e-5` steps to expose
   truncation/cancellation. Its measured maximum absolute eigenvalue is about
   22,083.4. On a quadratic target leapfrog needs
   `epsilon * sqrt(lambda_max) < 2`. The explicit conservative nomination
   `0.25 / sqrt(max_abs_eigenvalue) = 0.001682313009806192` is one eighth of
   that local quadratic boundary, not a global nonlinear stability guarantee.
   The factor 0.25 is a convenience safety fraction. Attempt 13 uses this
   epsilon with unchanged fresh trial requirements; it is capped at 780
   seconds and may return partial pricing evidence. Preserve the failed parent.
6. Audit terminal records and refresh the comparison matrix, execution record
   and next phase from measured prices. Every scientifically unfinished cell
   stays pending/inconclusive even if its engineering checks pass.

## Budget, commands and supervision

The original allowance remains cumulative: at most two hours of local-test
wall time and fifteen GPU-device minutes for engineering checks. It is not
renewed by each continuation. Historical campaign budgets are not reused.
The completed N0 test logs account for about 1,354 seconds; the subsequent
54-test and 139-test runs add about 479 seconds, plus the preceding failed
fixture run. N5 CPU attempts 02–07 add about 2,324 seconds; same-source
export/reload adds 228 seconds. New regression/localization/failure runs are
also charged. Attempt 13's cap leaves these documented checks within the
original engineering allowance; unpriced full fits remain excluded.

For the current continuation reserve no more than 3,600 additional CPU-test
seconds including attempt 07, replay, repairs and SIR, and no more than the
original 900 GPU seconds. GPU parity attempts consumed 38.14 and 73.50 seconds;
RNG localization about 7 seconds including startup; public attempts 10 and 12
consumed 9.50 and 40.56 seconds. Charge occupied wall time conservatively, not
only reported kernel time. These caps are convenience limits, not performance
predictions. Each worker has an outer subprocess deadline so a native call
cannot outlive the attempt cap; live checkpoints show progress. A budget
exhaustion is not a sampler failure and cannot be upgraded to evidence against
the method.

The launcher defaults to GPU/XLA. An example explicit CPU command is:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/run_modern_hmc_bridge.py \
  --targets NL-SIR --execution-mode cpu_reference \
  --epsilon 0.001682313009806192 --max-wall-time-seconds 780 \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/attempt-13-sir-local-step
```

Use a new output directory for every launch; the example path already exists.
GPU commands must use trusted execution. The launcher saves a fresh probe,
sets `TF_FORCE_GPU_ALLOW_GROWTH=true` before import, verifies growth before
fixture tensors, and records devices, compilation mode, TF32, allocator usage,
source hashes, data hash, seeds, command, environment, time and result paths.
The parent records timeout or child failure and preserves partial checkpoints.

`modern_bridge_audit.py` checks every observed trial's seed, start bank, stage,
transition profile, native device, realized lengths, value/score/status row
counts and corresponding charge. Failed calls without returned chunks remain
charged and explicitly unobserved. Static import closure includes actual
factory/adapter modules and transitive numerical dependencies; it is a
conservative source set, not a dynamic execution profiler. The runtime records
first invocation, not a separate compilation timer; cold invocation time cannot
be reported as compile-only time. Replaying old artifacts after source edits
may intentionally fail; preserve them with their original scope.

## Skeptical review and next phase

Proceed with the bounded engineering sequence: it checks actual public call
chains and native artifacts, preserves all members, tests missing/forged evidence,
and separates failed candidates from invalid harnesses. It has corrected the
wrong parity baseline, stale GPU assumptions, missing source closure and
unenforced wall cap. Posterior diagnostics remain outside tuning decisions.
The remaining risk is that short identity-geometry chains can be mechanically
healthy while scientifically unusable; this explicitly blocks release claims.

The next scientific phase must price preparation through posterior stopping,
not extrapolate these narrow bridge times. Start with the declared fixed-HMC
versus randomized-HMC comparison on LGSSM and accumulation; then compare
SNAPER proposals with simple random lengths and qualify MAMS separately.
Before launching, declare model-specific prepared geometry, independent
posterior references/estimands, meaningful warm-up and precision stopping,
replication/uncertainty, regimes, fair work budgets and a priced total campaign.
The October 10 owner correction reduces the required matrix to five targets
by four methods, excluding NUTS. Preserve the earlier inventory as history.
LAPS stays conditional on qualified MAMS and ensemble-aware validation.

### Measured SIR continuation

The smaller-step CPU retry's first two trials had finite states and scores,
255 accepted moves out of 256 each, and native times 169.56 and 159.48
seconds. These are descriptive mechanics observations, not convergence or
tuning qualification. The second invocation shows that startup alone does
not explain the cost: a 64-trial cohort at this price exceeds the remaining
CPU allowance. Keep attempt 13 bounded and retain its partial evidence.

Within the original fifteen-minute GPU allocation, add one SIR GPU/XLA cell
at the same nominated epsilon, full observation horizon, four starts and
unchanged 64-transition/16-repetition design. Cap it at 600 seconds under the
existing supervisor and fresh inventory. Earlier GPU attempts used less than
180 seconds including localization startup, so this is within the 900-second
total even at its cap. The comparator is the finite CPU mechanism; the question
is compilation/public-lifecycle feasibility, not relative speed or a posterior
verdict. The adapter already declares full-chain XLA capability, and that
guard stays enforced. If compilation or execution cannot fit, preserve the
price and remaining gap; do not shorten trials, alter thresholds or issue a
qualification. No further unpriced retry follows this bounded cell.

Terminal refresh: both SIR attempts reached their declared hard caps and are
finished. The CPU attempt returned four finite trials; the GPU attempt returned
19 and passed the first candidate's measurement cohort. Each has one additional
charged interrupted call. Neither completed verification, so neither retains
a candidate. The offline terminal audit reconciled their raw evidence and
charges. All four CPU/XLA--GPU/XLA parity cells and the GPU LGSSM public
verification/export/reload passed. The final focused suite passed 22 tests.
The active 20-cell comparison inventory remains scientifically incomplete. The next
action is the priced complete-fit design in the execution record; no run is
left active and no evidence requirement has been reduced to fit the budget.
