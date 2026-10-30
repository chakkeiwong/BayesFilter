# FP64 covariance-restoration guard repair, 2026-10-13

This is the range-force localization repair in the existing surrogate-HMC master.
The exact replay at q=(-.15,-.15), N4032/T20, mixed arithmetic fails at observation9.
Its covariance residual9.370636310897042e-14 exceeds100*eps*d=8.881784197001252e-14.
Actual force consumer and exposed graph agree exactly. Other recorded margins are
positive and all displayed quantities finite. The complete predicate conjunction
has not yet been exposed, so do not claim this is its sole failure until replay.

Question: does a particle-count-aware roundoff guard remove this immaterial
false rejection while retaining all genuine numerical vetoes and unchanged
healthy computations? Baseline is the preserved exact-input force-health.json
under surrogate-hmc-exact-force-range-20261013-01, plus protected source copies.
No density, derivative, particle map, reset, factorization, ridge, correction cap,
precision mode, start, data or proposal weight changes.

For a covariance accumulation Z'Z/N, the usual floating-point model
fl(a op b)=(a op b)(1+delta), abs(delta)<=eps, gives the conservative coefficient
gamma_m=m*eps/(1-m*eps) after m accumulated operations. The absolute-product
Gram bound satisfies || |Z'| |Z| ||_F <= trace(Z'Z) <= sqrt(d)||Z'Z||_F.
Two independently evaluated covariance sums therefore contribute an allowance
2*sqrt(d)*gamma_(N+2). Add this term to the existing100*eps*d FP64 allowance.
The two additional operations cover weight multiplication/scaling. This is a
conservative accumulation allowance, not a complete error theorem for the
factorization/centering pipeline. Existing rank, factorization, finite, mean,
correction and final likelihood/score checks remain mandatory. Keep the FP32
owner policy at1%; cap the new FP64 covariance allowance at1% for extreme sizes.
No tuning to likelihood, score or acceptance is permitted. The additive term is
derived from accumulation size; the factor100 is the preserved matrix-operation
margin, a baseline assumption whose failure would trigger separate localization.

Implement one TensorFlow helper used by scalar value/reference and all-P batch
consumers. Expose per-step validity predicates as boolean telemetry and emit
the actual covariance tolerance in diagnostic results. Avoid new arithmetic in
the scientific outputs; no pfor, eager fallback or copied algorithm.

Primary engineering checks:
- Reject nonfinite covariance errors and deliberately material residuals.
- Preserve FP32 policy exactly; test that N changes FP64 allowance.
- Shared consumer wiring and existing all-P/scalar parity pass CPU/XLA fixtures.
- Repeat actual range force with identical failed-pilot inputs; require four
  healthy full-horizon forces, consumer/direct agreement within1%, and unchanged
  healthy-row forces and pre-failure traces to numerical roundoff.
- A further invalid predicate is a repair trigger, not authority to relax it.
Fresh-scope calibration and validation must precede any new mechanics. Use new
partitions offset40 (calibration26101251/52, validation26101253/54). Same
three-member control family remains an exploratory hypothesis, with uniform
FP64 as arithmetic comparator. Final value and every score component must meet
the owner's1% criterion. FP64 is not a model oracle.

Budget at entry:6346.616 GPU-worker seconds remaining of36000,56 attempts completed.
Planned cap: one repair replay450s; range fresh tuning900s and mechanics400s;
SIR fresh tuning up to3800s and mechanics up to1200s, always clipped to the
remaining cumulative budget. Phases consume actual time; they cannot jointly
exceed the owner cap. Reserve enough for a smallest useful mechanics check before
starting the long SIR tuning. Raise the internal attempt cap from60 to62 only
to fit these five bounded phases plus one localized retry. This changes no
owner-authorized compute, method, target, hardware or promotion criterion.
Fresh versioned roots; one GPU worker; no external review or messaging.

Default/assumption audit: N4032, TF32off and FP32 default are owner choices.
Mixed FP64 recursion is an optional previously tested precision candidate.
Exact pilot starts/beta are frozen diagnostic inputs; old selection is not
current-source authority. Guard epsilon is dtype spacing; gamma uses a
conservative sequential sum rather than assuming a GPU reduction tree. The1%
cap preserves the existing material covariance ceiling. No method ranking or
posterior claim follows from passing a validity guard.

Skeptical audit: the prior nearby-input comparison was insufficient and has
been replaced by exact replay. Raw finite traces do not establish every
predicate; expose them. Enlarging a tolerance could hide an implementation
bug: retain the observed residual, preserve all other vetoes, require unchanged
healthy values/forces and fresh FP64 precision checks. Full posterior inference
remains outside what these short mechanics can establish. No heuristic
superiority comparison is attempted; this is same-algorithm validity repair.
Exact Kalman/integration/bootstrap comparisons remain required for model
accuracy and downstream inference claims.

Result artifacts: precision-guard-result.md, focused CPU logs, before/after
source snapshots and exact-force-range-20261013-02/force-health.json.
Actual commands/environment/source/memory/seed/wall data remain in master
manifests and the ten-hour ledger. This repair was audited before implementation.


Execution refinement after successful exact replay: use the existing master's
advance command for automatic tuning-to-mechanics progression and next-phase
plan refresh. Add an explicit --mechanics-timeout option (default1200s unchanged)
so the audited range400s and SIR1200s phase caps are enforced by the master.
This harness-only change is tested for propagation and positive finite values.
It neither authorizes extra compute nor changes scientific inputs. Historical
precision evidence only nominates the scopes; every candidate receives fresh
offset40 calibration/validation and same-input uniform-FP64 comparison.

Execution environment: /home/chakwong/anaconda3/envs/tftwogpu/bin/python,
GPU selector0 (CUDA enumeration; recorded TensorFlow device is RTX4080SUPER),
growth verified before initialization, XLA enabled, TF32 disabled, N4032/K2016.
The execution plan is precision-guard-execution.json in this directory.
Range: advance --model range_bearing --budget-seconds1300
--mechanics-timeout400. SIR: advance --model sir_d18, bounded by actual
remaining cumulative budget, --mechanics-timeout1200 and tuning cap3800.
Both use --execution-plan, --precision-results at
surrogate-hmc-shared-gaussian-precision-20261012-01/precision.json, and the
existing ten-hour --campaign-ledger. Output roots:
surrogate-hmc-guard-advance-range-20261013-01 and
surrogate-hmc-guard-advance-sir-20261013-01. Exact commands are in run manifests.
Audit: fresh source binding, explicit arithmetic exception, separate fresh
partitions, downstream health vetoes and preserved phase outputs address the
prior scope/partial-run failures. If tuning fails, no mechanics is launched;
the result is a candidate repair trigger, not an inference verdict.
