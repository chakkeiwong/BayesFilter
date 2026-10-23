# Guided SIR slowdown and merge repair

**DISQUALIFIED, 23 October 2026:** owner rejects the N64 workload. All numerical
and speed conclusions below are historical only; do not use them as evidence or
baselines. Replacement: `docs/plans/ledh-sir-n1008-runtime-20261023.md`.

22 October 2026. Branch `ledh-refactor-with-policy-fix`, HEAD `90a3acebd`.
Plan: [ledh-sir-merge-repair-20261022.md](../../ledh-sir-merge-repair-20261022.md).
Status: final timing and full frozen replay pass against the previous repair.
Direct pre-merge comparison is complete and still shows slower calls and failed
plain-arm numerical equivalence. The determinant-cost diagnostic is complete:
it confirms material added work, while all runtime validity checks are retained.
The subsequent Cholesky-ratio candidate passes numerical checks but fails its
runtime acceptance rule; candidate02 remains the final runtime.

## Findings and repairs

The merge made determinant calculations necessary in the actual plain and
conditional SIR consumers: new validity flags depend on them, so XLA can no
longer discard them. The physical mixture already required them. This explains
why speedups measured in another execution lane did not carry over. The guided
consumer calls the full canonical analytical recurrence; it does not call the
new native batch implementation, which has a different recurrence and supported
features. Substituting that implementation would change the experiment.
The small-matrix specialization also does not accelerate SIR's 9/18-dimensional
covariance operations: it specializes dimensions one and two. SIR's two-variable
moment-correction solve does use that specialization, so its effect must not be
described as absent from the entire consumer.

The 21 October repair replaced QR with checked elimination. This repair uses
the exact identity det(I - epsilon K H/2) = det(I - epsilon H K/2), reducing the
SIR factorization from 18 by 18 to 9 by 9. The analytical derivative includes
both dH and dK. Full-state finiteness and separate value/force validity checks
remain. The [derivation](repair-derivation.md) and tests cover nonlinear
observation maps and smaller, equal and larger observation dimensions.

Two actual merge defects were also repaired. The correction dispatch once
again runs when safety is enabled with zero moment-correction iterations.
Contract E reset once again checks its original row residual, column total
variation and residual-tangent finiteness. The shared helper's factor-validity
flag did not replace those conditions. Fault injection demonstrated the missing
checks before repair and rejection after repair. The safety-only option is off
in this campaign; it did not cause the measured slowdown.

The old/new plain-arm discrepancy is numerically sensitive. The first traced
state difference is 5.68e-14 immediately after moment correction. By step two,
the covariance condition number is about 1.20e9, ESS is about 2, and corrected
tangents and standardized moments fail the existing tolerance. Old and new
correction implementations agree within tolerance when supplied identical
inputs; changing only the intermediate inputs reproduces the amplification.
This supports sensitivity as a mechanism but does not uniquely identify the
initial floating-point operation. Pre-merge full-program equivalence remains
failed. Neither runtime improvement nor algebraic equivalence waives that result.
Smaller differences already occur in the first observation weights (4.16e-17),
so the first differing state is not the first differing arithmetic quantity.

The [call-chain audit](audit-findings.md) records checked modules, excluded
explanations and limitations. It found no physical-mixture tuple error, reset
validity-index error, missing UKF covariance lifecycle, or active FP32 precision
branch in this FP64 experiment. These findings apply to the inspected consumers,
not every execution lane in the repository.

## Final candidate against the previous runtime repair

Attempt 06 uses the final frozen source, four exposed inputs spanning the
selected regimes, eight alternating blocks, and adjacent paired calls. Ratios
are candidate time divided by the 21 October checked-elimination time. Intervals
are approximate 95% Student-t intervals over eight block means; they assume
block independence. Zero direction still evaluates the endpoint used by the
campaign and is reported separately from a nonzero score direction.

| Consumer | Direction | Time ratio | Interval | Predeclared criterion |
| --- | --- | ---: | --- | --- |
| Plain | Zero | 0.692 | [0.538, 0.890] | Pass: upper bound below 1 |
| Plain | Nonzero | 0.722 | [0.583, 0.893] | Pass: upper bound below 1 |
| Conditional | Zero | 0.822 | [0.734, 0.919] | Pass: upper bound below 1 |
| Conditional | Nonzero | 0.860 | [0.790, 0.937] | Pass: upper bound below 1 |
| Physical mixture | Zero | 0.859 | [0.726, 1.016] | Pass: upper bound below 1.10 |
| Physical mixture | Nonzero | 0.798 | [0.693, 0.919] | Pass: upper bound below 1.10 |

All 216 paired warm/timed outputs agree exactly across seven reported fields,
including raw moments, values and directional scores; validity flags match.
There are 384 timed calls. Each worker also reproduces its own archived outputs,
traces each endpoint once, and checks source stability. All three final compiled
endpoints contain no QR. Physical zero-direction speed superiority is not
established by its interval; the declared non-harm criterion passes.

GPU contention is material. Attempt 04 was inconclusive and remains preserved.
Attempt 05 repaired measurement order; its candidate preceded the final guard
restoration and is not used as final-source evidence. Independent GPU utilization
was approximately 81% after our workers exited. CPU-only consumer tests also
overlapped the first part of attempt 06. Pairing reduces confounding but does not
establish exclusive-device performance or performance in another scope.

## Correctness and tests

The final focused suite passes 52 tests; the reporting suite passes 20 tests.
Broader canonical score, recursion, per-particle flow, physical proposal and
neural KDM consumers produced 40 passes and one stale exception assertion.
The assertion was migrated to the intentional structured-status API and both
value-failure and force-only-failure cases pass. No runtime behavior was weakened
to satisfy the test. These initially give114 distinct passes. The final retained
determinant suite adds nine conditioning/failure cases and passes31 tests, giving
123 distinct passes overall. Repeated overlapping checks are not added again.

Full 64-dataset replay passes: all2496 holdout rows and432 nonzero score rows,
zero invalid rows, exact validity flags, and unchanged abs1e-8 + rel1e-8
tolerances. Maximum value difference is6.14e-12 and maximum directional-score
difference is1.49e-10. Raw moments also pass: their largest absolute difference
is1.21e-5 on large-scale quantities, only0.000175 of the allowed tolerance.
All134 frozen inputs match after path resolution; all six endpoints trace once.
The report's first attempt stopped on absolute/relative path spelling and the
metadata-only --plan driver change. The [reporting repair](reporting-repair.md)
verifies actual hashes and parsed computation while accepting those metadata
differences; no archived inputs, manifests, tolerances or results were edited.
The full output is [comparison.json](comparison.json).

The accepted settings remain OT (1,5), terminal balance 32, reset ridge 1e-5,
denominator regularization 1e-7 and the frozen correction controls, with 64
particles, five observations, FP64, TF32 off and XLA on. No retraining, retuning,
changed data, changed random design, tolerance relaxation or scientific default
change is included. This is a scoped engineering comparison, not a production
FP32/TF32 admission run.

## Direct comparison with the pre-merge program

Attempt08 repeats the four-input, eight-block, adjacent-call design against the
preserved pre-merge source. Ratios are final repaired time divided by pre-merge
time, with the same approximate95% intervals and shared-device limitations.

| Consumer | Direction | Time ratio | Interval | Interpretation |
| --- | --- | ---: | --- | --- |
| Plain | Zero | 1.244 | [0.999, 1.549] | Ranking unresolved |
| Plain | Nonzero | 1.418 | [1.045, 1.923] | Final program slower |
| Conditional | Zero | 1.693 | [1.503, 1.907] | Final program slower |
| Conditional | Nonzero | 1.382 | [1.193, 1.602] | Final program slower |
| Physical mixture | Zero | 0.854 | [0.663, 1.101] | Ranking unresolved |
| Physical mixture | Nonzero | 0.839 | [0.582, 1.211] | Ranking unresolved |

These measurements do not establish complete recovery of pre-merge speed.
Plain and conditional were the endpoints in which determinant validity made
previously unused work live. The physical endpoint already used it before the
merge. Other safe-factorization checks and masks also remain live. Static HLO
operation counts do not measure their execution cost, and the profiler returned
no usable GPU kernel events, so their individual runtime shares are not known.

Both workers reproduce their own archived outputs in all216 warm/timed pairs,
with384 timed calls, one trace per endpoint and unchanged sources. The direct
numerical comparison passes for conditional and physical. Plain remains outside
the existing tolerance: maximum directional-score difference9.74046e-4, about
60.47 times its allowed absolute-plus-relative bound on these four inputs.
The value difference1.36221e-6 passes its relative bound, while moments and loss
also contain failing entries. This smaller diagnostic cannot replace the full
pre/merged replay, whose largest plain score difference was about0.02395.

All common validity checks agree and pass. The merged program adds ten passing
monitor fields, so exact flag-dictionary equality across the merge is neither
expected nor asserted. Reporting preserves those added fields, checks for lost
validity fields, and requires unchanged common fields. The stricter repaired-
versus-merged comparison still requires exact flags. Initial strict reporting
outputs remain archived beside the corrected analysis.
See [attempt08 analysis](attempt-08-premerge-timing/analysis.json).

## Causal determinant-cost diagnostic

Attempt09 removes only the determinant dependencies from validity flags in a
temporary diagnostic source copy. Healthy numerical formulas and reported
quantities stay unchanged. This copy intentionally loses singular-flow detection
and is ineligible for runtime use, regardless of its timing. The actual runtime
keeps every determinant check. The source difference and hashes are preserved;
the temporary source copy is deleted at stage close.

All216 paired outputs, including validity flags on these healthy inputs, are
bitwise equal. Both workers reproduce their archived numerical outputs and
trace each endpoint once. HLO inspection verifies that the determinant pivot
loop disappears from plain and conditional, but remains in physical, which
also needs its numerical result. Unrelated ArgMax operations locate failed
particles; counting every ArgMax as determinant work would be wrong.

| Consumer | Direction | Diagnostic / guarded time | Approximate95% interval |
| --- | --- | ---: | --- |
| Plain | Zero | 0.787 | [0.662, 0.936] |
| Plain | Nonzero | 0.826 | [0.667, 1.022] |
| Conditional | Zero | 0.717 | [0.605, 0.850] |
| Conditional | Nonzero | 0.688 | [0.503, 0.941] |
| Physical mixture | Zero | 0.952 | [0.711, 1.274] |
| Physical mixture | Nonzero | 0.984 | [0.808, 1.198] |

The paired model supports a runtime reduction in three conditions, with point
estimates of21–31%. This isolates a material consequence of the new validation
dependencies. It does not measure an exclusive kernel-time share: removing a
dependency can alter compilation and scheduling. The other three intervals do
not establish a ranking. Individual costs of the remaining safe-Cholesky checks,
masks and compiler changes are still unquantified. No weaker guard is adopted.
See [attempt09 analysis](attempt-09-determinant-cost/analysis.json) and
[diagnostic disposition](determinant-cost-diagnostic-disposition.md).

## Decision table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Unsupported conclusion |
| --- | --- | --- | --- | --- | --- |
| Final optimization passes scoped repair checks | All six timing criteria and all2928 replay rows pass against the previous repair | Invalid-input, call-chain and full replay checks pass | Shared-device timing; residual pre-merge slowdown | Retain the guarded repair; any further speed candidate must preserve validation | Complete recovery of pre-merge speed or score-accuracy improvement |
| Restore both dropped merge checks | Fault-injected failures now reject; healthy cases pass | Separate value/force rejection retained | Other repository lanes are outside scope | Retain regression tests | Every merge in every lane is correct |
| Preserve failed pre-merge equivalence verdict | Plain-arm full-program tolerance fails again in direct final-source comparison | No threshold waiver | Initial perturbation operation not uniquely isolated | Investigate numerical conditioning under an explicit numerical-change contract | All old/new scores are equivalent |
| Close determinant ablation as explanatory only | Exact healthy outputs; three paired intervals below one | Known loss of singular-flow rejection prohibits runtime use | Compiler scheduling and remaining check costs | Preserve evidence and delete the diagnostic source copy | Removing a guard is a valid repair |
| Do not retain candidate03 | Consumer parity passes; only three of six timing criteria pass | No numerical failure observed | Broad shared-device intervals | Preserve candidate03 evidence; restore verified candidate02 | Candidate03 is mathematically wrong or generally slower |

## Inference status

| Evidence class | Finding |
| --- | --- |
| Hard veto screen | Focused and full replay parity, validity, source stability and coverage pass. Pre-merge plain equivalence remains failed; the existing heuristic promotion veto remains. |
| Statistically supported ranking | Under the eight-block paired model, plain/conditional and physical nonzero comparisons favor the final source over the previous repair. Against pre-merge, plain nonzero and both conditional directions remain slower. The diagnostic ablation reduces runtime in plain zero and both conditional directions. Candidate03 has supported reductions only in conditional nonzero and both physical directions; this is insufficient for overall retention. |
| Descriptive-only differences | Physical zero-direction improvement versus the previous repair; plain zero and both physical directions versus pre-merge; plain nonzero and both physical directions in the ablation; candidate03 plain zero/nonzero and conditional zero; per-input extremes and comparisons across separate runs. |
| Default-readiness | Not sought or established; this is the frozen FP64 debugging campaign. |
| Next evidence needed | Better precision under shared-device timing for further equivalent optimization of retained validation work; numerical-sensitivity repair with a checked reference; fresh per-scope tuning and untouched downstream evaluation for broader claims. |

## Red-team review and provenance

The strongest alternative timing explanation is time-varying contention; the
pairing and intervals reduce but cannot eliminate it. Reversed intervals under
controlled repeated loads would overturn the runtime ranking. The weakest
numerical evidence is attribution of the initial roundoff perturbation: identical
input checks establish neither XLA bitwise identity nor full-program equivalence.
The earlier guide-quality and heuristic-dominance failure remains unchanged;
faster evaluation does not rescue it.

The causal diagnostic prompted one further exact candidate: a Cholesky determinant
ratio reusing the innovation factor. Its derivation, reviewed protocol and complete
disposition are in [cholesky-determinant-candidate.md](cholesky-determinant-candidate.md).
Attempt10 passes216 paired numerical checks, but three timing intervals cross one,
so it fails the declared retention rule. The frozen candidate and patch are kept;
only its runtime delta is reverted. Candidate02's exact source identity is checked
again. Its full replay need not be rerun for an identical restored runtime.
Nine new general conditioning/failure regressions are retained separately.

Terminal code review: PASS for the scoped engineering repair. The formula
retains both terms in the analytical derivative and the time-step multiplier;
dimension dispatch also retains the original calculation when observations are
not narrower than the state. Actual-consumer wiring and curved-observation
derivative tests establish reachability. The two restored checks are traced to
the guidance merge parent; invalid-input regressions verify their behavior.
All980 package files and414 auxiliary files match the frozen tested source in
[final-source-verification.json](final-source-verification.json). Reporting
changes are separately syntax checked, hashed and covered by20 passing tests.
Independent review was not used for this stage; this is executor review backed
by derivation, fault injection, numerical tests and frozen consumer replays.
The direct pre-merge comparison prevents any claim that all slowdown or numerical
differences are repaired. The completed determinant-cost diagnostic explains a
material portion of the added work and cannot promote weaker validity checks.
The bounded investigation is complete; complete pre-merge recovery is not.
No target change, corrupted input or invalid repaired result was found. The
remaining plain discrepancy is a numerical repair trigger, not evidence against
controlled SMC or KDM. Changing its conditioning requires a separately specified
numerical experiment, with an independent reference and unchanged-target or
explicitly declared changed-target criterion; this report does not authorize a
silent ridge, clipping or tolerance change.

Exact commands are in [final-validation-commands.md](final-validation-commands.md).
Attempts preserve source hashes, commands, Git identity, data seeds, environment,
hardware, memory-growth settings, output rows and elapsed times. Package snapshots
include auxiliary imports. The task allocation is12,800 GPU-process seconds within
the unchanged parent campaign; [budget.json](budget.json) records8375.548099 charged
and4424.451901 unused GPU-process seconds. The one-further-candidate continuation is
complete; no task GPU process remains active. Unused budget does not establish that
an inconclusive candidate passes, and the declared candidate limit is retained.
Prior dirty research changes are preserved. No commit or push is included.
