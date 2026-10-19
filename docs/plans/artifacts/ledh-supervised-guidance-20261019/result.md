# Simulation-supervised guidance: executed SIR pilot

19 October 2026. Branch `ledh-refactor-with-policy-fix`, commit
`ea1018de1da55ed02a29a8d60f08a6827fce8a42`, with recorded dirty source hashes.
Plan: [ledh-supervised-guidance-20261019.md](../../ledh-supervised-guidance-20261019.md).

**Unguided LEDH has the lowest T5 validation loss. No guide is promoted.**
The new supervised objective and its full-executor implementation pass their
bounded checks. Guided test means are descriptively lower, but four test
sequences, a large unguided outlier, and an opposing validation comparison do
not establish better score estimation. This is a completed implementation
pilot, not a rejection of the research direction.

## Target and computed quantity

The target is the mean squared error of the analytical finite-LEDH score
against the observed-data model score. We compute squared errors against
independently simulated, fixed-path complete-data scores. These individual
losses are different quantities. Under the manuscript's conditional
independence, regularity and integrability assumptions, their population
expectations differ by a candidate-independent conditional-variance term.
Paired population loss differences therefore equal true score-MSE differences.
This is an exact identity, not evidence that a finite selected candidate has
lower true risk. No observed-score oracle enters fitting or selection.

The consumer is the shared analytical LEDH executor with UKF covariance
propagation, flow, Contract E and diagonal/pairwise GenUT correction. Guide
coefficients are frozen during parameter differentiation. This is not a
derivative through a parameter-dependent refitting operation. The optional
kernel consumer samples its moment-matched Gaussian and uses that same
Gaussian density in the importance denominator. The fitted RBF message is
KDM-inspired; the run does not retrain the original neural KDM. Selection
uses score loss. First/second-moment losses are implemented and reported as
diagnostics; moment-only training and continuous coefficient optimization
have not executed.

## Execution, repairs and validity

All launches use the tftwogpu environment, trusted RTX 4080 SUPER access,
TensorFlow/XLA, FP64 and TF32 off as a numerical diagnostic exception. Memory
growth is verified. OT is fixed at epsilon 1 and five iterations, with 32
terminal balancing updates held fixed as declared in the plan. No transport
control was swept or selected. N=64 and K=N. The model is
the continuous, unclipped shared SIR law at theta=(0,0,0). Each horizon has
separate 2/2/4 calibration/validation/test sequences and two particle streams.

| Attempt | Outcome | Process seconds |
|---|---|---:|
| pilot-01 | Memory-growth guard caught premature initialization; import order repaired before retry | 60.00 conservative charge |
| pilot-02, T2 | 96 rows; 80 valid, 16 raw-guide implementation failures | 261.32 measured |
| pilot-03, T5, fresh seeds | 96/96 valid rows; six frozen-program directional checks pass | 249.51 measured |

The T2 raw guide was rejected by a legacy requirement that guide information
itself be positive semidefinite. The proposition only requires positive total
precision. An explicit optional flag repairs that mismatch; the default check
is preserved. Regression covers admissible negative information and a rejected
nonnormalizable total precision. All T5 raw-guide rows pass after repair.
T2 raw rows remain invalid historical rows; they were not retroactively repaired.

The directional checks use the predeclared tolerance
`abs(FD-analytical) <= 1e-4*(1+abs(analytical))`. They check the frozen finite
program, not closeness to the unknown model score. The largest T5 absolute
difference is 0.5372 for the unguided derivative of approximately 7314.3;
this passes that tolerance, not a machine-precision agreement claim.

Thirty-four distinct CPU tests pass: the 30-test regression, one additional
Gaussian first/second-moment risk identity test (run with its three neighbors),
and three reporting tests. They cover fixed-path score labels, Gaussian
draw/density tangents, default parity, full-executor wiring, invalid proposal
rejection, split separation, and averaging particle replicates within datasets.

Terminal audit verifies 96 unique expected keys per completed pilot, all loss
arithmetic against stored labels, calibration-only scaling, fit flags and
derivative artifacts. The T5 source hashes match the executed source. The T2
manifest correctly differs in the two subsequently repaired source files.
See [terminal-audit.json](terminal-audit.json) and [audit.md](audit.md).

## Descriptive results

Loss is `sum_j ((score_j-label_j)/scale_j)^2`, with coordinate scales frozen
from calibration RMS labels and a unit floor. The T5 scales are
618.4048083, 142.7231312 and 5.5752442. There is no cross-horizon loss ranking:
each horizon has its own data and calibration scales.

| T5 arm | Validation mean | Test mean | Mean paired test difference vs unguided |
|---|---:|---:|---:|
| Unguided | 1653.53 | 65902.87 | 0.00 |
| Half-strength cSMC | 4515.01 | 23443.86 | -42459.00 |
| Projected cSMC | 3136.57 | 9575.05 | -56327.82 |
| Unprojected cSMC | 3151.98 | 9454.93 | -56447.93 |
| Positive-information kernel | 3328.31 | 15098.74 | -50804.13 |
| Product-moment kernel | 3592.32 | 10847.56 | -55055.31 |

Validation nominates projected cSMC within its family and the
positive-information kernel within its family. These are family nominations,
not approvals over unguided. A selector permitted to retain the baseline would
choose unguided on these validation means. No test-selected replacement is
adopted.

| Independent T5 test sequence | Half cSMC | Projected cSMC | Raw cSMC | Kernel information | Kernel product |
|---|---:|---:|---:|---:|---:|
| 261029505 | -206853.76 | -212376.69 | -212797.09 | -214456.30 | -214969.83 |
| 261029506 | -811.24 | -4029.53 | -4053.42 | -5863.76 | -8168.99 |
| 261029507 | -16001.52 | -14132.57 | -14029.21 | -10175.82 | -10736.76 |
| 261029508 | +53830.50 | +5227.52 | +5087.98 | +27279.37 | +13654.34 |

Entries are paired scaled loss differences, averaged over particle streams;
negative means descriptively lower loss. All guides lose to unguided on the
fourth test sequence. The first sequence and observation-noise coordinate
dominate the pooled means. Half strength is a fitted bank member, so it cannot
also be claimed as an independent post-selection adversary. The required
promotion-level heuristic comparison is not established.

At T2, the mean paired differences are +0.3443 for half cSMC, +0.5808 for
projected cSMC, -0.1794 for the information kernel and -0.1642 for product
moments. The kernel changes occur on only one of the four test sequences;
the other three exported messages are inactive. These observations do not
provide a cross-horizon ranking or isolate horizon as the cause of a change.
Full component and moment diagnostics are in each pilot's paired-analysis.json.

## Decisions and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept bounded implementation checks | Mathematical identity, labels, tangents and actual consumer checks pass | T5 has no numerical failure; T2 raw failure repaired | Nonzero finite-difference tolerance and limited tested states | Preserve optional implementation and test coverage | Canonical admission or all-scope correctness |
| Retain unguided baseline; promote no guide | Statistical gain criterion not met | Heuristic dominance not established; validation favors unguided | Four independent test datasets, large outlier, one parameter point | Fresh scope-specific calibration and larger untouched dataset study | Better SIR score accuracy, default readiness or HMC readiness |
| Continue research direction | Learning objective remains mathematically valid | No target-law or harness-invalidating finding in final pilot | Guide bank and fixed numerical controls are narrow hypotheses | Separate representation/consumer, numerical-control and fitting-objective effects | Failure of the underlying supervision idea |

| Inference status | Finding |
|---|---|
| Hard veto screen | T2 raw rows invalid from implementation; all T5 rows pass declared numerical flags |
| Statistically supported ranking | None |
| Descriptive-only differences | Tables above, terminal moment losses, runtime and all per-seed differences |
| Default-readiness | No new default or tuning artifact; OT unchanged |
| Next evidence needed | New calibration/validation/test partitions, target-specific control justification, more independent datasets, parameter coverage and frozen independent heuristic comparisons |

Engineering correctness, numerical validity and scientific benefit remain
separate: successful implementation checks do not move this experiment into
the statistical-evidence category. No confidence interval was predeclared for
this four-sequence mechanics pilot, and none is used to claim a ranking.

Post-run red team: the strongest alternative explanation is an unstable or
poorly calibrated unguided finite-program derivative combined with noisy
complete-data labels, rather than a consistently useful learned guide. Fresh
replication with calibrated numerical controls could overturn the descriptive
means. The weakest evidence is the two-sequence validation selection and the
four-sequence test, not the conditional-risk algebra.

## Audit and resource closeout

The manuscript contains four new propositions with proofs and an executable
algorithm. Direct review and independent numerical identities pass. MathDevMCP
certifies scalar squared-loss expansion and completion of the square only:
its document tool failed internally, and its router could not encode the full
probability statement or the separate Gaussian-score derivative expression.
Claude's read-only reviewer could not authenticate. No full external theorem
certification or independent review approval is claimed. See audit.md.

This stage charged 570.83 GPU-process seconds, including the conservative
initialization reserve. Together with the recorded prior 9325.730914790
seconds, the campaign charge is 9896.560914790 of 43200 seconds, leaving
33303.439085210 seconds. Measured pytest time is 130.75 seconds across initial,
repair and terminal checks; 150 CPU seconds are conservatively reserved for
this stage's checks and reporting, within the 600-second allocation.

Exact commands, environments, seeds, controls, source hashes, memory policy,
data identity, internal wall time and artifact paths are in the two manifests;
launch.log contains external process timings. Prior evidence is preserved.
This closes the seven steps of the bounded implementation plan. The next
research allocation requires its own recorded scope and early diagnostic;
no new procedural permission gate is created.
