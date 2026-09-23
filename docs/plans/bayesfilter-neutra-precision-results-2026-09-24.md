# Configurable NeuTra training precision: engineering result and continuation

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
test remains a strict expected failure because continuous-tail ESS is applied
to a binary sign indicator. That diagnostic issue must be repaired before a
posterior promotion claim. It is not evidence against the learned map.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Unsupported conclusion |
|---|---|---|---|---|---|
| Admit explicit FP32/TF32 for q20 calibration | Precision equations, inverse and resume checks passed | No precision veto observed | Target-specific gradient noise and training stability | Price and calibrate the actual q20 target | Universal TF32 adequacy |
| Continue training investigation | Mechanism implementation is testable | Training quality has not yet been evaluated | Capacity, optimization and nonlinear fit | Fund and execute replicated sustained training | Improved q20 learning |
| Defer posterior promotion | Full posterior checks unavailable | Binary-tail information diagnostic defect | Correct discrete diagnostic semantics | Repair before downstream admission | Convergence or production readiness |

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
