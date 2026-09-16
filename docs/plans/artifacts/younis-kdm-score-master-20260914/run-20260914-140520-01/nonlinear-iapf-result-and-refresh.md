# Nonlinear iAPF: implementation result and phase refresh

The twenty-row GPU study completes the scalar nonlinear iAPF implementation
prerequisite. The selected mechanics procedure has larger observed score errors
than EKF and UKF in both curvature regimes. This vetoes promotion from this
study; two final replicates on one dataset per regime do not establish a
statistical ranking or reject the iAPF research direction. Comprehensive control
calibration remains necessary.

Plan: [nonlinear iAPF](../../../younis-score-nonlinear-iapf-2026-09-16.md).
Frozen source `f5a4d411a4a0197fe9c1d25c24573337a633c96a`, predecessor
`1c12eefa2ed55b81c4336fca4c006b9877c2df5f`. The frozen worktree is
`.localresources/worktrees/younis-score-nonlinear-iapf-20260916`; it remains
unchanged. Nine source/test files were integrated after all main-checkout files
matched the predecessor. Backups and exact file hashes are in
`main-before-nonlinear-iapf-01/` and `nonlinear-iapf-main-integration.json`.

## What was implemented and checked

The shared conditional-mean functions propagate sine-transition and quadratic-
observation values and analytical tangents through the final filter and both
recursive fitting procedures. The Gaussian-plus-floor normalizer and sampler
remain exact conditional on the previous state. The nonlinear consumer now
uses the same kernels as the Gaussian consumer and accepts a selected iAPF
procedure with adaptive particle counts. It checks model curvature, physical
FP64 observations, their executed FP32 cast, fitting observations, and independent
offline/final streams. It rejects mismatched scope or data evidence.

Thirty focused frozen-source checks pass, including independent quadrature of
the twist, mixture-density correction, six-parameter finite differences,
nonlinear backward fitting targets, actual consumer wiring, and physical
bootstrap recovery. Zero-curvature fitting diagnostics, coefficients, values,
six-component scores and clouds agree exactly with the predecessor. Three
mandatory commit oracle checks pass. See `nonlinear-iapf-tests-01.log`,
`nonlinear-iapf-bootstrap-tests-01.log`, `nonlinear-iapf-zero-parity.json` and
`nonlinear-iapf-commit-01.log`.

The main-checkout run passed fourteen checks and rejected one consumer artifact
after another task changed `bayesfilter/highdim/ledh_canonical_reset_score_tf.py`
within the recorded dependency closure. Numerical evaluation had succeeded;
the source-change veto correctly prevented reuse. The affected consumer passed
on a fresh retry. Both attempts, timings and the classification are retained in
`nonlinear-iapf-main-tests-{01,02}.*` and
`nonlinear-iapf-main-tests-01-failure.json`. No source guard was relaxed.

The manuscript now derives the exact conditional mixture, telescoping path
correction, particle approximation to the initial normalizer, and complete
branchwise frozen-fit derivative in Section 7.3 (Propositions 4--6). The updated
57-page PDF compiles without warnings; pages 23--25 were rendered and inspected.
The prior manuscript is protected in `manuscript-before-nonlinear-iapf-01/`.
Human reading feedback is pending. MathDevMCP proves the completing-square
identity; two symbolic derivative requests were inconclusive because that tool
did not encode the derivative syntax. This is a local algebra check, not an
audit certificate for the whole particle filter. Analytical derivation and the
independent numerical derivative checks supply the remaining local evidence.

## GPU execution and conditional comparisons

`nonlinear-iapf-gpu-01/run-manifest.json` records the exact launch, environment,
source, driver hash, study specifications, observations, seeds and output paths.
The source is clean and frozen. The final filtering and score kernels run on the
RTX 4080 SUPER in TensorFlow FP32/TF32/XLA with verified memory growth and one
trace each. Offline density fitting uses explicitly declared FP64. Physical data
and mesh/domain-refined references use CPU FP64 and are excluded from GPU
performance claims. Concurrent unrelated workloads prevent speed comparisons.

The six parameters are `[.62,-.8,-.6,.9,.25,-.3]`, horizon two, initial particle
count sixteen. Weak curvature is `(c,b)=(.12,.04)`; curved is `(.35,.12)`.
Each scope uses calibration dataset 900, validation 910 and final 920 with seed
941. These final datasets are now opened and cannot become fresh holdouts.
Calibration selected k=1 in both scopes. Each final replicate refits the same
dataset with the same independent offline stream and then uses a distinct
final filtering stream. Actual repeated fitting work is charged, not amortized.

| Regime | Realized N | iAPF mean squared score error | EKF | UKF | Bootstrap | Local linear |
|---|---:|---:|---:|---:|---:|---:|
| Weak | 32 | 0.314684 | 0.002219 | 0.004152 | 1.719237 | 0.266964 |
| Curved | 16 | 0.548303 | 0.034659 | 0.065885 | 12.386501 | 0.884129 |

The iAPF column averages two final replicates. Stochastic comparators have one
replicate each; Gaussian moment filters are deterministic. These are descriptive
errors against the refined reference, with no sampling interval or ranking.
Comparators use the realized final N, which does not equalize total cost. The
machine-readable verdict in `conditional-heuristics.json` is
`promotion_veto_descriptive` in both regimes. Kalman-style moment approximations
are effective cheap adversaries in these mild, short-horizon fixtures.

The weak final count history is `[16,16,32]`: each final replicate consumes 128
offline particle-time points, 64 final points and 71 optimizer steps. The curved
history is `[16,16,16]`, with 96 offline points, 32 final points and 397 optimizer
steps. Three of four per-time fits in each final procedure touch a declared
bound. They satisfy the numerical stopping rule, but boundary activity and poor
relative shape fit are explanatory evidence that the inherited family and
stopping controls require calibration. They are not evidence of an accurate
lookahead. The deliberately permissive tau=100 was a mechanics setting.

All twenty saved result digests, ninety-one dependency hashes and the exact
driver match the frozen run. `nonlinear-iapf-verification.json` records the
check; `nonlinear-iapf-gpu-01/driver-used.py` preserves the executed driver.

## Decision and uncertainty

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Complete scalar nonlinear implementation | Conditional law, analytical derivative and real consumer checks pass | No numerical/data/stream/source veto in frozen GPU run | Broader dimensions and more difficult observations | Retain shared implementation; widen only after scope calibration | General nonlinear or DSGE applicability |
| Do not promote current mechanics controls | Model-score improvement unestablished | Observed heuristic promotion veto in both regimes | One dataset, two iAPF replicates, active fit bounds | Calibrate the full control family on new partitions | iAPF is intrinsically inferior |
| Continue master | Research direction remains valid | No continuation veto | Joint tuning and equal-cost evidence incomplete | Audit and execute the next calibration phase | Whole master complete |

| Inference status | Finding |
|---|---|
| Hard veto screen | All frozen GPU rows pass engineering/numerical checks; one main test artifact correctly invalidated by concurrent source change and passed on retry |
| Statistically supported ranking | None |
| Descriptive-only differences | All score errors, fit bounds, optimizer counts and timings |
| Default-readiness | Not established; no default changes |
| Next evidence needed | Target-specific calibration, new validation/final datasets, independent replicates, equal-total-cost comparisons and paired uncertainty |

The strongest alternative explanation for poor score accuracy is the inherited
bounded fit and its permissive stopping rule, compounded by small clouds and
fixed discrete-selection derivatives. A calibrated fit and a different score
estimator could alter the result. The weakest evidence is the single final
dataset per regime. The implementation would be invalidated by a conditional
density or derivative mismatch; neither occurred. The current candidate failed
the promotion screen, not the mathematical construction or research direction.

## Repair, budget and next phase

No frozen numerical repair is required. The only retry repaired an invalid
main-checkout test artifact; the manuscript builder used available `pdflatex`
after discovering that `latexmk` was absent. No packages were installed.

The GPU allocation closes at one of three launches, 48 of 120 numerical
row/recursive-fit attempts (20 rows plus 28 fits), and 52.04446 of 1800 GPU wall
seconds. Timed CPU process use through frozen tests/commit and main integration
checks is 248.67 of 5400 seconds, plus separately recorded document build time.
The two unused launches are not needed to close this implementation slice.
The budget reconciliation preserves untimed historical work as unknown rather
than assigning it a fabricated zero.

Next, audit comprehensive numerical-control calibration for the already
implemented nonlinear score consumers. Treat both LEDH control-safety failures
and iAPF bound/stopping failures as repair triggers. Use disjoint fresh
calibration/validation/final partitions, freeze the selected procedure before
final data, retain all failed candidates, and account actual offline plus online
cost. Heuristic screens remain vetoes and may not become tuning objectives.
The full control family must be enumerated before a new comparison is called
calibrated. A further narrow cap-only or k-only sweep cannot close this debt.
