Active result, 2026-09-24: the owner accepts the M13 saved-run discrepancy as
a warning and directs FP32 as the default for speed and memory economy.
The implemented [policy](surrogate-hmc-fp32-warning-policy-2026-09-24.md)
retains the original 3e-5 comparison as an explanatory diagnostic. It keeps
FP32 validity/finiteness and FP64 derivative, trace and recorded repeatability
checks mandatory. The saved repaired replay passes these engineering checks
with a warning. Numerical dependency hashes still match; no filtering was rerun.
The policy review passed 125 focused CPU tests. Decision and receipts:
[policy results](artifacts/surrogate-hmc-closure-20260924-fp32-policy-01/assessment/report.json).

The goal remains testing surrogate HMC with a separately declared endpoint
posterior. This engineering decision does not qualify M13 posterior sampling.
The force target defaults to FP32/TF32/XLA; FP64 remains an explicit reference,
derivative-check, and declared small endpoint/integrator role. No speedup or
memory benchmark is claimed.

NEXT within the existing master: M13 scope-specific controls and posterior
endpoint/reference preparation. The prepared four-process FP32 repeatability
batch is optional, deferred, and no longer a prerequisite or approval blocker.
LG1 stays closed; Claude review is deferred. AR2 and remaining model work keep
their existing roadmap. The master is not done.

Current charged global remaining allocation: 11,559.064705 CPU-worker
seconds and 193,195.452245850 GPU-wall seconds. This includes the new
70.064439 CPU seconds and the prior materiality analysis's
0.018487292 CPU seconds; this policy change used zero GPU seconds.
The chronological entries below preserve earlier states; the active checkpoint
and newest receipts govern current execution.

# Surrogate-HMC closure master: execution result

Current continuation, 2026-09-23: AR2 now has a qualified structural canonical
consumer, independent Kalman/direct-Gaussian reference, refreshed scoped controls
and an affordable primary posterior-test forecast. The shared small-matrix
repair passed 84 checks plus GPU full-consumer parity and compiler inspection.
Control requalification passed; cost requalification passed 21 checks. The
conservative B allocation forecast is 20517.539 CPU-worker / 21407.652 GPU-wall
seconds. No AR2 posterior chains have run yet. The next existing-master stage
is its prospective uncertainty qualification, followed by B/R testing.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Continue AR2 | Kernel parity and scoped controls pass; primary B cost fits | No numerical veto in checked probes | Actual chain mixing and uncertainty | Scoped calibration, then public tuning/retained B | AR2 posterior success or master completion |

| Inference status | Evidence |
|---|---|
| Hard veto screen | No non-harm, source or scoped-control failure in repaired candidate |
| Statistically supported ranking | None |
| Descriptive-only differences | Four-position and five-transition timing; no matched-precision comparison |
| Default readiness | Not established |
| Next evidence needed | Qualified diagnostics and actual endpoint-corrected posterior samples |

Post-run red team: local cost probes can underestimate difficult trajectories;
the 1.5 cost multiplier, measured L8 bound and runtime budget guard address
allocation risk, not convergence. Passing reference/parity checks still cannot
substitute for retained posterior evidence. Current global charges through
ar2-validation-02 are 30419.199769 CPU / 27685.140215 GPU seconds. The sections
below preserve the first closure pass as historical results.

The master now executes the first closure repairs, and its final regression
review passed all **70 tests**. The wrapped observation-law repair passed its
declared checks. The uncertainty candidate passed the scoped diagnostic
calibration after a fresh, larger confirmation. **The all-model surrogate-HMC
campaign remains incomplete:** AR2's canonical structural consumer and other
model-specific consumers, references and prospective experiments still need
implementation. No new retained HMC chains were launched in this pass. LG1
remains an accepted, closed practical pilot.

The scientific goal is to test whether the full canonical analytical DCTR force
supports reliable parameter inference when a separately declared endpoint
posterior determines Metropolis acceptance. Matching that endpoint's gradient
is not a prerequisite. We preserve model-endpoint sampling accuracy (B/R),
finite-endpoint sampling accuracy (C/D), finite-posterior distortion (D/R), and
their combined effect (C/R) as separate questions.

## Delivered implementation and review

The existing entry point is `scripts/run_surrogate_hmc_program.py --stage closure`.
Its [closure driver](../../bayesfilter/inference/surrogate_hmc_closure_program.py)
runs a required regression stage followed by selected independent numerical
stages. It saves commands, source snapshots, hashes, logs, stage findings and
incremental charges in fresh directories. It separates successful execution
from a positive scientific result and blocks dependent jobs after an execution
failure. Historical formal verdicts and practical pilot judgments remain
separate. The report lists all 17 registered families without counting their
unimplemented variants as tested.

The [implementation review](surrogate-hmc-closure-master-review-2026-09-22.md)
checks the mathematical targets, actual consumer call chains, archive handling,
failure propagation, resource limits, diagnostic scope and remaining omissions.
Repairs found during review included stale SV20 test assumptions, parent CPU
limit inheritance, attempt counting, shared budget discovery and tests reading
live campaign state. No numerical threshold was relaxed. Claude review was
deferred as requested.

## Numerical findings

The M13 range/bearing observation callback now uses the wrapped Gaussian alias
sum, including the conditional angular mean and variance for correlated noise.
For the declared conditional angular standard-deviation range up to pi, four
aliases on either side give an omitted-mass bound of 2.2572e-19. GPU/XLA checks
through the actual callback gave a maximum unit-mass error of 5.6e-16 across
four noise scales. Its analytical tangent agreed with finite differences to
3.1e-11. This qualifies the local observation-law repair; M13's integrated
likelihood, parameter-posterior reference and canonical recursive score remain
unqualified. See [the wrapped-law report](artifacts/surrogate-hmc-closure-20260922-01/wrapped/report.json).

The uncertainty experiment uses a finite law defined by normalized weights on
the qualified KSC T2 reference table. Its 13 function expectations are known for
that finite law. Four-chain IID and stationary refresh simulations test the
same mean, moment, marginal-CDF and predictive-probability function types used
in the intended comparisons. They are controlled simulations, not HMC chains.
Continuous-reference integration error is outside this calibration.

The first pass ran 1,280 replications. IID false rejection was 4/512 (0.781%),
with a 95% Wilson upper bound of 1.991%. All three failed-mixing controls
(excessive persistence, chain offsets and drift) were rejected at both looks in
128/128 replications each; each group's lower confidence bound was 97.086%.
The two dependent groups had interval-failure point estimates of 2.344%, but
their upper confidence bounds exceeded the declared 5% limit. Those results
remain archived. They triggered a larger confirmation, not a changed rule.

The confirmation used 2,048 fresh replications, disjoint seeds and unchanged
functions, margins, interval construction and criteria. It did not pool the
earlier dependent replications into a favorable final denominator.

| Dependence regime | Replications | Any-look simultaneous interval failures, unconditional | 95% upper bound | Declared result |
|---|---:|---:|---:|---|
| Refresh retention 0.5 | 1,024 | 22/1,024 = 2.148% | 3.232% | Passed |
| Refresh retention 0.9 | 1,024 | 33/1,024 = 3.223% | 4.491% | Passed |

Among admitted replications the corresponding failure counts were 22/1,024
and 28/1,014; their upper bounds were 3.232% and 3.962%. No false equivalence
was observed in either confirmation group under the two declared wrong-reference
controls; the per-group upper bound was 0.374%. Strong dependence still caused
at least one look to be rejected in 41.309% of replications, with both looks
rejected in 0.977%. The passed screen therefore does not mean the diagnostic
never rejects a stationary chain or that short chains are sufficient.
See [initial calibration](artifacts/surrogate-hmc-closure-20260922-01/uncertainty/report.json)
and [fresh confirmation](artifacts/surrogate-hmc-closure-20260922-04/uncertainty-confirmation/report.json).

The AR2 probe reached the actual `canonical_batch_fused_value_score` consumer
with Q = diag(0.09, 0). It rejected the singular process covariance before time
updates. This confirms an implementation gap for a valid structural model;
it is not a negative posterior result for surrogate HMC. A complete repair
must use the innovation-space proposal and its probability measure, deterministic
lag completion, canonical flow/reset and analytical recursive score. Adding
artificial lag noise or fixing only the Cholesky would not implement that target.
See [the structural consumer report](artifacts/surrogate-hmc-closure-20260922-01/structural/report.json).

## Execution and resources

The environment was `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, at Git
commit `8a4cffa2fd085f43ab5710c24d6e7b48770aea0d`, with relevant dirty sources
preserved in each run's snapshot. Numerical GPU jobs used trusted device access,
TensorFlow/XLA and verified memory growth. CPU review and structural diagnostics
deliberately hid GPUs. Rank-based diagnostics retain the recorded non-XLA
exception. Exact commands, seeds, reference hashes, hardware and timing are in
each run manifest and numerical report.

| Attempt | Result | CPU-worker seconds | GPU-wall seconds |
|---|---|---:|---:|
| 01 | Initial numerical pass completed; dependent calibration unresolved | 282.045906 | 279.682331 |
| 02 | Review failed on stale test assumptions; no numerical launch | 50.319734 | 0 |
| 03 | Review failed on stale status-schema assertions; no numerical launch | 50.311909 | 0 |
| 04 | 68 tests passed; fresh calibration passed | 755.357142 | 1,022.937129 |
| 05 | Review exposed live-directory test contamination; no numerical launch | 25.037262 | 0 |
| 06 | Final 70 tests passed | 49.127807 | 0 |
| Total measured closure work | Includes every failed attempt | 1,212.199760 | 1,302.619460 |

The pass stays within its 2,400 CPU / 3,600 GPU-second reservation. Both masters
now agree on cumulative budget charges of **28,401.432413 CPU** and
**26,622.426856 GPU** seconds. These include conservative unresolved opening
bookkeeping reserves of 214.712703 CPU and 79.294813 GPU seconds; those reserves
are explicitly not measured new computation. The remaining campaign balances
are **25,598.567587 CPU** and **200,177.573144 GPU** seconds. No family allocation
was enlarged. See [reconciliation](artifacts/surrogate-hmc-all-models-20260922-accounting-reconciliation-01/result.md)
and [final accounting](artifacts/surrogate-hmc-closure-20260922-06/accounting.json).

## Decisions and remaining work

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep LG1 closed | Owner's practical acceptance preserved | No new pilot failure found | Archived formal outcomes unchanged | Advance other scopes | Retrospective formal equivalence |
| Retain uncertainty candidate for its tested diagnostic scope | Declared calibration passed | Bad-mixing controls rejected | Transfer to real HMC, other functions and reference error | Bind the next prospective comparison and validate its uncertainty assumptions | Universal MCSE validity or automatic runtime-default promotion |
| Accept local wrapped-law repair | Normalization, periodicity, covariance and tangent checks passed | Invalid inputs still rejected | Integrated M13 likelihood/reference and canonical score | Qualify complete M13 consumer and same-law references | M13 posterior correctness |
| Keep AR2 inference blocked | Structural canonical implementation absent | Actual consumer rejects valid singular Q | General canonical derivation and implementation | Implement Chapter 18b structural measure through the actual DCTR path | Failure of surrogate HMC as a method |
| Continue remaining roadmap | Several consumers and references absent | New inference prerequisites unresolved | Scope-specific precision and cost | Complete structural/generative support, concrete scopes and independent references | All-model coverage or computational advantage |

| Inference status | Finding |
|---|---|
| Hard veto screen | AR2 consumer incompatibility remains. Missing model/reference evidence blocks affected new inference stages. Review failures were repaired and charged. |
| Statistically supported ranking | No sampler or efficiency ranking established. |
| Descriptive-only differences | Runtime, earlier pilot differences and comparisons with the old diagnostic are not sampler rankings. |
| Default-readiness | Local wrapped observation repair is qualified; no HMC or uncertainty default promoted. |
| Next evidence needed | Full canonical structural/generative integration; scoped continuous/finite posterior references; prospective precision/cost designs; downstream chains; conditional, replicated efficiency comparisons if claiming usefulness. |

The next implementation step is the general structural canonical consumer for
AR2, using Chapter 18b and the existing Kalman fixtures as independent checks.
M09 needs the related generative/shared-noise capability. Larger scopes and
most remaining family variants still need concrete definitions and references.
LG2's existing pilot forecast still exceeds its family allocation. Diagnostic
success cannot supply these missing prerequisites, so this pass did not launch
unqualified model-level chains.

Post-run skeptical review: the strongest alternative explanation for diagnostic
success is that the finite KSC law and simple refresh dependence omit real-chain
difficulties. A failure under a separately declared target/function/dependence
scope would overturn transfer to that scope. The wrapped-law evidence is local,
and the structural probe establishes a blocker rather than a repair. These are
the material limits of the completed work.
