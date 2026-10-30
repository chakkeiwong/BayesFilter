# Configurable NeuTra training precision: engineering result and continuation

This file preserves the engineering and interim execution record. The complete
training, paired comparison, HMC outcome and current decisions are in the
[terminal evaluation](bayesfilter-neutra-training-evaluation-results-2026-09-24.md).
The pending-stage decisions below describe their recorded earlier stage.

FP32 transport weights, gradients and Adam state can now train against the
existing FP64 value/score target. TensorFlow's TF32 option governs eligible
FP32 matrix multiplication; it is not a storage dtype. FP64 compatibility
defaults and old saved maps retain their meaning. Finalization makes an
explicit FP64 copy and performs the standard 1,000-point diagnostic on that
exact evaluation map before export.

The shared numerical authority supports both the source-based affine IAF and
conditional NAF/DSF, using standard reverse-KL or the layerwise path estimator.
Fixed outer affine initialization is serialized. Inverse tolerances depend on
dtype, and checkpoint restore rejects incompatible precision configurations.
Existing same-dtype checkpoints retain exact optimizer continuation.

## Measured engineering evidence

Three trusted GPU/XLA processes on an RTX 4080 SUPER evaluated matched
represented parameters and inputs against an FP64 CPU analytic reference.
Each covered IAF/NAF and standard/path gradients, eight updates, tail inversion
and FP64 export. Memory growth was configured and verified before GPU use.

| Arithmetic mode | Maximum scaled gradient L2 discrepancy | Declared screen | Result |
|---|---:|---:|---|
| FP64 | 6.79e-16 | 1e-9 | Passed |
| FP32, TF32 disabled | 4.10e-7 | 1e-4 | Passed |
| FP32, TF32 enabled | 4.01e-4 | 0.01 | Passed |

These are engineering screening hypotheses, specified before the checks. All
inverse screens passed. Mode-dependent differences support distinct arithmetic
behavior; saved HLO does not directly establish Tensor Core instruction use.
The three processes consumed 323.519 worker-seconds. Their small analytic
runtimes do not establish q20 training speed or quality.

The focused precision suite has 14 passing tests. Other completed checks cover
legacy facades, configured IAF/NAF, weighted training, independent derivatives,
1,000-point finalization, and public fixed-transport HMC tuning and replay.
Per-attempt XML preserves repeated tests and failures; counts must not be added
as if they were distinct tests. Integration disposition and final accounting
are in the plan and engineering manifest.

The q20 refresh controller's unlabeled training price and obsolete classical
preparation requests were repaired. A migration fixture now expects the actual
budget pause at its protected reserve. The intended Gaussian posterior success
test initially remained a strict expected failure because continuous-tail ESS
was applied to a binary sign indicator. The subsequent repair explicitly types
binary quantities, applies event-indicator ESS and rank split R-hat to them,
and preserves raw undefined binary quantile/folded diagnostics. Physical
coordinates keep their continuous rank/folded and tail checks; event MCSE and
information floors are unchanged. An exact-half fixture exposed the folded
binary degeneracy and is included in the regression tests. Constant events and
stuck binary chains still fail. The final six focused checks, including the
complete supervised Gaussian master success/resume test, pass. The existing
posterior regression checks also passed. One intermediate supervisor test was
invalidated by an edit during its source snapshot; it was rerun from a fresh
directory after source stabilization. All attempts are charged.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Unsupported conclusion |
|---|---|---|---|---|---|
| Admit explicit FP32/TF32 for q20 calibration | Precision equations, inverse and resume checks passed | No precision veto observed | Target-specific gradient noise and training stability | Price and calibrate the actual q20 target | Universal TF32 adequacy |
| Continue training investigation | Mechanism implementation is testable | Training quality has not yet been evaluated | Capacity, optimization and nonlinear fit | Fund and execute replicated sustained training | Improved q20 learning |
| Defer q20 posterior promotion | Full q20 posterior checks unavailable | Binary diagnostic defect repaired; no q20 posterior has been run | Actual q20 mixing and coverage | Qualify the exact final map within budget | Convergence or production readiness |

| Inference status | Evidence |
|---|---|
| Hard veto screen | No invalid gradients, inverse failures or broken same-dtype resume in precision checks |
| Statistically supported ranking | None; this is an equation/engineering check |
| Descriptive differences | Arithmetic discrepancies and short process timings |
| Default readiness | Configurable feature verified; no universal scientific dtype/default claim |
| Next evidence | Actual q20 variability calibration, sustained seeded training, independent loss/geometry and downstream assessment |

The strongest alternative explanation for any later apparent training gain is
more optimization or a better affine warm start, rather than the nonlinear
architecture. Preserve the original map and a continuation control on common
evaluation banks. A short favorable loss or score residual cannot replace the
complete replicated training protocol. The weakest current evidence is the
absence of actual q20 results for these configured maps.

Evidence root: `artifacts/neutra-precision-training-2026-09-24/`. Large HLO files
remain local and are indexed with hashes. The active execution plan is
`bayesfilter-neutra-precision-training-plan-2026-09-24.md`. Work is isolated in
`/tmp/BayesFilter-neutra-precision-20260924` to preserve concurrent dirty work.

## Sustained training and numerical repair checkpoint

All three initial IAF fits reached 4,096 updates, passed their 1,000-point
checks and exported-map inverse checks. Their descriptive residual medians
are 0.3352, 0.4364 and 0.6732; means are 0.4208, 0.5763 and 0.8092. The
fractions above one are 4.8%, 13.6% and 22.9%. These independent probe banks
are not the common final comparison bank; no replicated-improvement verdict
has yet been issued. Clipping occurred on 38, 59 and 41 of 4,096 updates.

NAF seeds 0 and 1 stopped at updates 2,328 and 462. Exact replay showed an
implementation error: the log-domain mixture rejected finite log weights
when their unnecessary exponentiation underflowed to zero in FP32. The
minimum log weights were -118.800 and -107.520, at exactly the rows and
coordinates that acquired NaNs. The same represented maps and minibatches
were valid in FP64 under both path and standard gradient estimators.

The corrected validity check uses finite log weights and retains all other
domain checks. It changes neither the mixture formula nor the target and
adds no clipping, floor or discarded component. The 46-test equation and
precision suite passes, including a tiny-weight component that dominates a
tail and must survive the log-domain calculation. Both formerly failing GPU
updates now pass with finite gradients and advance Adam to the expected next
iteration. Evidence is in `naf-weight-underflow-01/result.json` and
`naf-invalid-update-repair-verification-01/result.json` under the evidence root.
The pre-repair replay and one failed diagnostic serialization attempt remain
preserved and charged.

NAF seed 2 reached 4,096 accepted updates but failed in post-training AutoGraph
source lookup after numerical source was edited during its live process.
The saved optimizer checkpoint is intact. The refreshed controller resumes
all three seeds in fresh processes to 8,192 total updates, preserving moments
and noise streams. This replaces the unjustified capacity repair for the
now-localized implementation failure. The final heldout banks remain untouched.

The downstream assessment has executable CPU reference checks for both IAF
and NAF: frozen-map loading, public tuning, numerical member export/reload,
and the shared sequential posterior controller. Eleven checks passed; fifteen
including the repaired campaign controller passed. These fixtures establish
call-chain mechanics, not q20 convergence. The current balance before resumed
training is 84,453.730 worker seconds. Exact revised reservations are in
`accounting-before-repaired-continuation.json`.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Unsupported conclusion |
|---|---|---|---|---|---|
| Resume NAF16 after log-weight repair | Both exact failed updates now pass | Old FP32 guard defect fixed; final quality still pending | Sustained fit and generalization | Three preserved seeds to 8,192, then untouched bank | NAF successfully trained |
| Preserve all three IAF candidates | Required probe and inverse checks pass | No numerical veto observed | Paired final-bank differences and posterior coverage | Complete common-bank comparison | Posterior correctness from residuals |

| Inference status | Current evidence |
|---|---|
| Hard veto screen | Original NAF invalid updates retained; exact repaired replays valid; all initial IAF maps numerically valid |
| Statistically supported ranking | None; final paired analysis pending |
| Descriptive-only differences | IAF residual distributions and clipping rates |
| Default-readiness | Not established by these training results |
| Next evidence needed | Completed NAF replication/control, paired final banks and bounded downstream assessment |

The strongest alternative explanation for the IAF diagnostics remains added
optimization and initialization. The continuation control and fresh baseline
evaluation address that explanation. The NAF repair is supported by exact
failure localization and independent equations, but still needs sustained
training. No result rejects the NeuTra research direction.
