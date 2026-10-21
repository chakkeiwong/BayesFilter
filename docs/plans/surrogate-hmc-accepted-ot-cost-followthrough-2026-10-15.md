# Continued surrogate-HMC cost repair, 2026-10-15

This is the next cost-repair step of the authorized five-model master, within
the existing 86,400-second campaign ledger. The accepted epsilon=1/five-step
scope advance finished for all five models: fresh tuning, actual HMC mechanics
and consumer parity/cost passed. These are not posterior results.

The immediate question is which unchanged analytical stage now dominates the
LGSSM and SIR full-horizon force, after reducing transport iterations. The
measured four-chain L1 cost models are 16.64 and 33.17 seconds/transition,
respectively; those few fixed-position measurements are descriptive, not
convergence forecasts. The remaining campaign budget is 48,644.887 seconds
before this followthrough. No additional compute is authorized or assumed.

Use the existing full-sequence component diagnostic with the newly selected
controls and beta from accepted-ot-scope-advance-01. LGSSM uses uniform FP32;
SIR uses its existing FP64-recursion/FP32-pair exception. N4032, K2016, T50,
GPU/XLA, TF32 off and seed offset100 remain fixed. Materialized component
timings are diagnostic instrumentation; the comparator is the unchanged fused
XLA filter on identical parameter rows and designs. There is no numerical
algorithm or posterior-target change.

## Evidence contract and audit

Primary criterion: complete finite outputs and existing full-sequence parity
against the fused baseline. Omitted derivatives, changed inputs or controls,
nonfinite outputs and failed parity veto interpretation of stage timing.
Stage fractions, staged/fused ratio and timing repeats are explanatory only.
No speed ranking, statistical superiority, posterior convergence or canonical
LEDH admission is inferred. Source/data corruption, unavailable trusted GPU,
or campaign exhaustion veto continuation; a failed diagnostic triggers repair.

The baseline and cheap adversary are the same fused implementation: this is
cost localization, not method selection. Conditions are reported separately
for LGSSM and SIR and for each observation time. Existing model-reference
comparisons are not repeated. The inherited two-row diagnostic, .01 offset
and three timing repeats give local descriptive evidence only; they cannot
establish posterior-wide performance. Reduced reset iterations can shift cost
away from transport, so older component fractions cannot answer this question.

Skeptical audit: passed for localization. Prior source snapshots are not used
as a mathematical oracle, selection is checksum-checked, and unchanged fused
outputs must agree before interpreting materialized timings. The principal
risk is instrumentation changing fusion; report that ratio explicitly.

## Execution and next phase

Executable remains scripts/run_surrogate_hmc_performance_repair.py run,
--component-cost --component-sequence --controls-warm-start-root pointing at
accepted-ot-scope-advance-01, --tuning-seed-offset 100, --gpu 0. Set
--runtime-precision fp64_recursion_fp32_pairs only for SIR. Each command gets
--budget-seconds 1200 --per-model-timeout 1200 and the existing campaign ledger.
Use unique outputs accepted-ot-components-lgssm-01 and
accepted-ot-components-sir_d18-01 under that campaign; preserve logs, source
snapshots and per-time measurements. Total phase cap 2400 seconds, two attempts.

Refresh checkpoint and campaign next-phase state after each result. Inspect
the dominant exact call chain and evaluate only a bounded, mathematically
equivalent cost repair justified by these measurements. Do not repeat failed
caches unchanged or add arbitrary convergence gates.

The full posterior phase remains part of the original master. Its benchmark
normal prior was explicitly mechanics-only in the previous plan. The owner now
requires model-specific scientific priors and will provide their definitions;
posterior sampling must wait for those definitions. Independent cost repairs
remain authorized. The sampling wrapper now restores and validates the selected
precision/seed scope: 12 focused tests and all five real selection checks pass.
No duplicate validation chain
is requested: adaptive warmup followed by cumulative retained R-hat <=1.01,
up to 20,000 retained draws per chain and the actual remaining wall budget.

## Component result and bounded redundant-work repair

Both full-sequence diagnostics passed. LGSSM stage means are 2.4905 seconds
proposal/observation, 1.1404 density/weights, and 4.8763 transport/reset; staging
costs 1.0154 times the fused filter. SIR means are 10.8588, 5.1840, and 6.4875
seconds, respectively; staging costs 1.2931 times the fused filter. These are
two-row local descriptive measurements. In particular, SIR stage fractions
cannot be treated as exact fractions of the fused consumer's runtime.

The inspected batch transport recomputes its final column log-normalizer after
the Sinkhorn loop, although the final iteration has just computed the identical
quantity. For C_ij=-||x_i-x_j||^2/epsilon and the final source potential alpha,
L_j=log(sum_i exp(C_ij+alpha_i)); gamma_j=-log(N)-L_j. Carry L and its existing
analytical total tangent as loop outputs. Do not reconstruct L from gamma,
which would introduce a subtraction and different rounding. Preserve the same
finite five iterations, primal/tangent formula, row diagnostics, and normalized
transport/reset. Positive iteration counts are already required by Controls.
This removes one redundant tiled reduction; it changes no mathematical target,
numerical control, prior, or derivative definition. No speed gain is assumed.

Skeptical audit: the exact claimed consumer calls the shared batch transport;
preserve that module before editing and compare both full-horizon graphs on
identical inputs and controls with the existing covariance comparison harness.
Require healthy complete outputs, every value and score within the owner's 1%
materiality screen, and no newly unhealthy initial-region row. Cloud/tangent
differences and timing are explanatory, not oracle or posterior evidence.
The baseline includes the same accepted epsilon=1/five iterations. Prior
cache/arithmetic candidates are not reused. Cheap baselines here are unchanged
finite-program execution and deterministic replay, not competing statistical
methods. Source or input mismatch invalidates comparison.

Budget: at most five 1,200-second GPU attempts (6,000 seconds total) from the
same campaign, first LGSSM and SIR, then remaining models if parity passes.
Use the existing --covariance-baseline diagnostic with a protected complete
batch module and --controls-warm-start-root accepted-ot-scope-advance-01,
seed offset100; SIR retains its mixed precision exception. Output directories
are accepted-ot-normalizer-reuse-MODEL-01. A parity or health regression rejects
the repair and restores the protected source. A descriptive runtime regression
over 10% triggers inspection rather than a speed claim. Retain source snapshots,
commands, timing, memory and full output comparisons. Existing selected-scope
artifacts remain historical to their source; do not relabel them after a code
change. Fresh scientific-prior tuning is required before posterior sampling.

## Terminal cost-repair decision

The paired LGSSM and SIR runs completed with exact equality of likelihood,
score, particle cloud and total tangent at nominal/offset rows. Both additional
initial-region checks had healthy rows and exact likelihood/score equality.
The candidate/baseline warmed runtime ratios were 1.054185 and 0.995510,
respectively. Three repetitions at fixed positions do not support a statistical
speed ranking. The candidate has not demonstrated a useful cost benefit.

Plan amendment before further execution: do not spend the other three planned
attempts on this candidate. Archive its source and restore the protected batch
implementation byte-for-byte. This is a decision not to adopt an unhelpful cost
change, not evidence against the surrogate-HMC method or a numerical failure.
The completed five-model tuning/mechanics/consumer checks remain the current
computational evidence. The master scope-handoff repair and prior guard remain.
Scientific posterior execution waits for owner-specified model priors; no
benchmark q-normal prior is substituted. Continue within the existing master
when those target definitions are available.
