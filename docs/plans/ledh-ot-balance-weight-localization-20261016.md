# LEDH transport balancing and first-step weights — 16 October 2026

Execution complete: see [result](artifacts/ledh-ot-balance-weight-20261016/result.md)
and [checkpoint](artifacts/ledh-ot-balance-weight-20261016/checkpoint.md).
Iteration-only repair fails mass validity. Zero denominator passes tested
mass and derivative checks but fails drop-in state non-harm. Defaults and
guards are unchanged; continue to flow-aware proposal design.

## Active question and scope

Continue the owner's instruction to refresh the master and execute on
`ledh-refactor-with-policy-fix`. Why does the shared scalar Gaussian LEDH
reference fail its transport mass check, and why is its first weighted cloud
already too concentrated before reset? These are distinct questions. Resolving
transport validity alone cannot establish model-score accuracy.

This is an UNTUNED FP64/XLA diagnostic of the full shared UKF, guided-proposal,
LEDH flow, Contract-E and dual-cap correction consumer. It is not production
TF32 evidence, a tuning artifact, a method ranking, or a new default. The zero
guide and physical Gaussian transition remain fixed. All scores use the
existing analytical recursion. No learned guide is fitted in this stage.

## Evidence contract and research intent

Baseline: `REFERENCE_CONTROLS`, with flow48, epsilon1, Sinkhorn5 and
terminal-balance32, N128/T6. Reconstruct the previous rejected persistent
dataset3/stream1 case from saved observations, parameters and CPU noise seed.
Archive actual noise tensors now; this is not recovery of the unarchived GPU
tensors from the earlier failure. The exact Kalman conditional is the
first-step reference; the existing guided adapter is the call-chain parity
authority for the diagnostic wrapper.

1. Capture the full shared consumer's reset inputs and first-step densities.
   Require wrapper value/score parity with the existing consumer on a healthy
   fixture and identical failure classification on the reconstructed rejection.
2. On frozen inputs, evaluate terminal counts 32,64,128,256 through the shared
   reset, preserving epsilon1/Sinkhorn5, ridge, design and analytical tangents.
   The implementation uses `sinkhorn_steps + balance_steps` identical scaling
   iterations. Therefore the actual totals are 37,69,133,261; changing the
   terminal count changes the finite program. It is not a distinct exact
   projection after five updates. Report this explicitly.
3. Check the unchanged row-mass <=1e-6 and column-TV <=1e-4 conditions,
   finite values/tangents, covariance positivity and restored moments. Retain
   per-cloud residuals and output changes from32. A possible safety repair must
   pass the actual mass check and preserve previously valid outputs to scaled
   1e-6 for states/covariances and 2e-5 for tangents. These diagnostic tolerances
   are not a production acceptance rule. If no count passes, reject the
   iteration-only repair and investigate the additive denominator floor;
   do not weaken the mass check or silently change that floor.
4. Freeze any candidate before four fresh validation fixtures (independent,
   persistent, and one explicit large-innovation persistent stress case).
   Preserve failing validation as holdout evidence. Check common-noise
   analytical finite differences at steps1e-4 and3e-5, scaled error <=2e-5,
   before any scope-specific optional recommendation. If there is no viable
   count, these promotion checks are unnecessary; continue attribution.
5. Independently inspect the first-step proposal law, physical-prior density,
   Jacobian and likelihood factors against their scalar Gaussian formulas.
   Distinguish the exact model conditional from the sampled ancestor mixture
   conditional. Compare flow48 with the already specified flow192 diagnostic
   on identical noise, using concentration and variance only for explanation.
   This does not select a flow count or re-run the old score leaderboard.

Promotion criterion for a numerical repair is validity and non-harm, never
score-MSE improvement. Mass/finite/covariance failure vetoes that candidate;
it is a repair trigger, not rejection of LEDH or backward guidance. Failed
consumer parity, derivative identity, input provenance or budget is a
continuation veto for dependent experiments. Missing exact old GPU noise
prohibits claims of exact replay but does not prevent new fixed-input tests.

Heuristic comparison is not the question in this localization. The constructed
prior-SIS, bootstrap and fully adapted Gaussian comparators remain required
for the later score comparison. This stage cannot overturn their previous
promotion veto, establish superiority, or certify nonlinear/HMC readiness.

## Assumptions and skeptical audit

| Choice and provenance | Role and justification | Risk and early check |
| --- | --- | --- |
| epsilon1/Sinkhorn5, owner instruction | Untuned development baseline, held fixed | Hidden extra iterations; record total count and exact update |
| balance32, inherited adapter | Baseline, not validated default | Incomplete convergence or regularization bias; frozen-input count curve |
| counts64/128/256 | Geometric diagnostic ladder, not selected for score | More iterations may not remove the defect; stop expansion at256 |
| additive1e-7 denominator term, shared kernel | Inherited numerical protection, justification unproved | Perturbed mass equations; derive the fixed-point relation if residual stalls |
| ridge1e-6 and existing flow/correction controls | Unchanged baseline hypotheses | Other safeguards may affect score accuracy; retain validity/moment diagnostics |
| FP64 CPU/XLA then bounded GPU/XLA | Explicit debugging/reference exceptions | No TF32 transfer; save tensors for cross-device replay |
| N128/T6, prior failing scope | Direct localization, not generalization | Few cases; fresh holdouts only for a viable repair |
| non-harm/FD tolerances | FP64 engineering screens, not statistical evidence | Passing is not equality or model-score accuracy; report actual discrepancies |

Skeptical audit passes for localization: the actual guarded consumer and
unchanged threshold answer the numerical question; fixed inputs separate reset
effects from earlier weighting; the additive term is an explicit competing
explanation; no proxy is promoted to model-score accuracy. The inherited
protection is a hypothesis needing evaluation, not justified by historical use.
Pre-mortem: additional iterations could reduce residual while changing the
finite likelihood or concealing a density mismatch. Archive both values and
derivatives and inspect the pre-reset cloud separately.

## Execution, budget and preservation

Use the LEDH worktree and tftwogpu Python. CPU diagnostics deliberately set
CUDA_VISIBLE_DEVICES=-1 and two TensorFlow threads. Stable signatures and XLA
are required; non-XLA is only a recorded localization exception. Trusted GPU
confirmation, if justified by CPU results, uses the explicit RTX5080 UUID,
memory growth configured before initialization, and TF32 off.

This repair stage has at most300 CPU wall seconds including focused checks and
document builds, and400 GPU wall seconds including probes/compilation. These
fit within the previous stage's remaining allowances (less than371.297 CPU
and430.943 GPU seconds); do not reuse spent time or old replication budgets.
Localized retries share these totals. The earlier two-launch rule applied to
the stopped grid, not all subsequent repair work; this refreshed plan replaces
that obsolete launch-count restriction without restarting the old comparison.

Artifact root: `docs/plans/artifacts/ledh-ot-balance-weight-20261016/`.
Every launch uses a fresh directory. Save command, Git/source hashes, environment,
data/seed/input identities, device/memory policy, timing, diagnostics and decision.
The checkpoint records remaining time and exact next action. Stop new launches
at the budget, preserve partial evidence, and never overwrite old results.

Initial command (150-second internal cap plus short shutdown allowance):

```sh
CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/plans/artifacts/ledh-ot-balance-weight-20261016/diagnose.py --output docs/plans/artifacts/ledh-ot-balance-weight-20261016/cpu01 --budget-seconds 150
```

Record the terminal decision/inference table and strongest alternative
explanation in `result.md`. Update the original master and active checkpoint
after each substantive result. No commit, push or branch change is requested.

## Adaptive diagnostic: denominator perturbation

CPU01/02 reproduce the rejected reset and show that counts32 through256 do
not clear its guard. The iteration-only candidate is rejected. CPU01 also
exposed a float32-rounded constant in the independent density checker;
CPU02 repairs only that checker, and all density identities agree to1.5e-14.
The first-step persistent cloud remains much narrower than both the exact
model and sampled-ancestor-mixture conditionals. This is separate from reset.

Use at most40 further CPU seconds to compare an independent log-domain
matrix-scaling reference on the saved inputs at the same37 total updates,
with additive terms1e-7,1e-8,1e-9,0. Match the current transport at1e-7 before
attributing an effect to that term. This is a primitive diagnostic reference,
not a new LEDH lane or an eligible score implementation. Record actual
transport changes and mass errors for every saved cloud. Keep any numerical
runtime repair pending this causal check and its own non-harm/derivative tests.

Reason: at a fixed point of u=a/(Kv+delta), v=b/(K^T u+delta),
the unnormalized coupling has row masses a-delta*u and column masses
b-delta*v. Row normalization alone cannot generally restore the column
constraint. This exact algebra identifies a possible bias, not evidence that
it dominates the observed finite-iteration error. The frozen-input comparison
distinguishes the explanations. The overall300-second CPU budget is unchanged.

## Bounded repair evaluation after localization

The independent reference matches the shared reset at delta1e-7 to1e-15.
At the same37 updates, delta0 reduces the rejected cloud's column-TV from
2.705e-4 to4.45e-11. This identifies the additive denominator as a material
cause on these inputs. It does not establish downstream non-harm.

Expose an optional denominator regularization argument in the existing shared
reset and its canonical single-cloud consumer; retain1e-7 as the unchanged
default. Evaluate delta0 because it is the unperturbed matrix-scaling equation,
not because it optimizes score error. The additive term is fixed with respect
to theta, so the existing analytical quotient derivatives apply. Accumulate
finite positive denominator checks into validity; invalid denominators must
reject the evaluation, never silently select a different scientific target.
This is a Class C numerical candidate. Zero is justified by the exact marginal
equations; its underflow/overflow risk requires explicit fail-closed checks.

First evaluate all18 saved clouds, preserving epsilon1/Sinkhorn5/balance32,
and compare particles, covariances and all available analytical tangents.
Use the predeclared non-harm limits1e-6 for values and2e-5 for tangents on
previously valid clouds. Check zero's transport against the independent
log-domain reference and its analytical derivatives against finite differences.
If non-harm fails, reject this candidate under this contract and preserve
the result; do not silently relax the threshold or promote a different value.
Still check four fresh full-consumer cases as a bounded diagnostic of whether
the mass repair transfers, explicitly separating that evidence from promotion.
No tuned setting or production claim follows from these checks.

Skeptical audit: passing mass checks alone could conceal a material change in
the accepted finite program. The frozen-cloud non-harm criterion, total tangent
checks, exact default parity and fresh consumer evaluations address that risk.
The remaining first-step proposal concentration is a separate research issue;
no denominator change can resolve it before the first reset. Use at most the
remaining250.724 CPU seconds and400 GPU seconds; no budget expansion.
