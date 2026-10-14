# Current replication root causes and remaining gaps

8 October 2026. Both replications remain **INCOMPLETE**. This is a completed
investigation, not numerical acceptance of either port. The target remains the
author's behavior with no worse numerical performance; intrinsic author
failures are admissible reference outcomes. Original iAPF and changes to the
authors' algorithms remain deferred.

The controlled-SMC discrepancy is localized to policy fitting and its linear
algebra. KDM's checked IWSG and learning equations agree in double precision,
but the cause of the complete-training quality gap is still unresolved. A new
optimizer isolation test identifies one small finite-precision difference;
it does not establish the cause of that quality gap.

## Evidence and execution scope

Checkout: `/home/chakwong/BayesFilter`, branch `surrogate-hmc`, base commit
`89065bc6354801cb368d5e163ba49fd9c4372d10`. Relevant implementation and harness
files include uncommitted work; the new diagnostic manifests preserve hashes.
Unrelated changes were preserved. Governing plan:
[root-cause investigation](controlled-smc-kdm-root-cause-investigation-20261008.md).
Acceptance remains the
[owner-approved numerical criteria](controlled-smc-kdm-acceptance-criteria-20261006.md).

The cSMC execution reference is the author's MATLAB code running in the recorded
Octave/system-LAPACK environment. It is not a reconstruction of the original
MATLAB runtime. KDM uses the actual Younis/Sudderth source at commit
`b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18`, including its installed PyTorch Adam.
Local technical paper copies inspected for these equations are
`.localresources/papers/heng2020-controlled-smc-published.txt` (Section 4.2,
equations 20–21) and
`.localresources/papers/younis-sudderth-2023-long-range-tracking.txt`
(equations 14–15 and 18, and the bearings training description).

No new full training, qualification or source-method repair was launched. The
only new numerical work was a bounded Adam isolation diagnostic, first on the
default RTX 4080 SUPER and then repeated on the original training RTX 5080 UUID.
The first run omitted the intended GPU pin; it is preserved as mechanics
evidence. The second corrects that execution identity, and both produce the
same comparison rows. CPU PyTorch references deliberately hid GPUs. GPU TF
runs used verified memory growth, TF32 off and XLA; the CPU TF graph arm is an
explicit debugging exception. Commands, environments, source/input hashes,
wall times, device policy, trace counts and complete logs are in:

- `artifacts/controlled-smc-kdm-replication-20260928/root-cause-20261008-01/`;
- `artifacts/controlled-smc-kdm-replication-20260928/root-cause-20261008-02/`.

The two runs consumed 7.697 seconds of reference CPU time and 28.442 seconds
of candidate-process wall time, conservatively charged entirely to GPU. The
second includes a CPU graph control in that same candidate process. A separate
120-second conservative CPU allowance covers routine inspection/validation;
this allowance is not a measured numerical-run duration. No acceptance margin
or runtime numerical policy was changed.

## Controlled SMC: what actually differs

The current call chain is the Lorenz refinement wrapper in
`bayesfilter/replication/lorenz_refinement_tf.py` into
`MultivariateGaussianControlledSMC`, its backward policy fit, and `_regress`
in `bayesfilter/replication/multivariate_controlled_smc_tf.py:313`.
The autonomous wrapper selects `centered_qr`. The fitting comparison harness
also exercises `source_normal_equations`. The source-branch compatibility
diagnostic is a separate harness; its existence does not make the autonomous
implementation source-compatible.

### 1. The current linear solver does not implement the reference solver

The author computes

\[
 \widehat\beta=(D^\mathsf{T}D)\backslash(D^\mathsf{T}y)
\]

at `StateSpaceModels/Lorenz/cSMC_Resample.m:193`. The backslash operation in the
recorded Octave runtime includes matrix classification, condition estimates,
Cholesky/LU decisions and an SVD least-squares fallback. Relevant source anchors
are `.localresources/code/octave-6.4.0-dMatrix.cc:1405`, `:1435`, `:1524`,
`:1622` and `:2205`. The singularity test includes whether `1 + rcond == 1`
in floating-point arithmetic.

The TF branch named `source_normal_equations` instead calls an unconditional
`tf.linalg.solve` at `multivariate_controlled_smc_tf.py:327`. **It is wrong to
describe that branch as implementing the reference's complete solver behavior.**
It forms the same normal equations, but not the same conditional algorithm.
`centered_qr` solves the same least-squares objective in exact arithmetic using
a centered/scaled design; that does not imply equal computed policies in this
ill-conditioned problem.

This distinction is consequential. If the singular values of a full-rank design
are `s_i`, those of its Gram matrix are `s_i^2`, hence
`condition_2(D.T D) = condition_2(D)^2`. The saved 8-dimensional quadratic fits
have 45 coefficients; some Gram systems are near the double-precision rank
boundary. Changing the solver changes the retained directions and fitted
polynomial, not just a harmless last printed digit.

The earlier checked evidence makes this localization strong:

- All 22 valid fixed-policy forward stages passed; ancestry agreed exactly,
  with maximum particle error about `1.42e-14` and log-likelihood error
  `1.53e-9`. This checks the consumer with the author's policy supplied.
- Fitting then forwarding failed the strict comparison for all 16 cases of
  each of the two TF solvers. The source-policy controls passed.
- A reference using the matching system LAPACK reproduced all 128 saved
  systems exactly. Merely using the same branch logic with another LAPACK
  did not. Forced-branch TF CPU and GPU controls each passed only 72/229
  comparisons.
- At saved time 20, source DGELSD retained rank 45 while a different SVD route
  retained 44. Even a time-28 case where both retained 45 had a materially
  different fitted prediction. A small residual or the same rank does not
  certify the policy.

Detailed preserved evidence:
[forward](controlled-smc-p0-forward-result-20261005.md),
[fitting](controlled-smc-p0-fitting-forward-result-20261005.md),
[solver semantics](controlled-smc-p0-solver-semantics-result-20261005.md),
[DGELSD arithmetic](controlled-smc-dgelsd-arithmetic-result-20261006.md).
The sensitivity experiment also retained a roughly 1.1% likelihood offset on
its particular saved case, about 49 times the median response to the source
`1e-8` perturbation. That is diagnostic evidence, not a population estimate or
a replacement acceptance tolerance.

**Remaining gap:** no TF source-compatible conditional solver is wired into
the autonomous fitting endpoint, and no alternative TF fit has passed the
accepted downstream equivalence/noninferiority tests. Bitwise coefficient
identity is not itself an acceptance requirement. A numerically different
fit may qualify if it preserves the declared method and passes those tests.

### 2. The source covariance failure is intrinsic; failure comparison is unfinished

The full-quadratic author fit sets an unconstrained quadratic coefficient `A`,
then forms the controlled covariance `(Q_inverse + 2*A)^(-1)` and its log
determinant (`cSMC_Resample.m:197–224`). The source's projection applies only
to the one-dimensional or `purequadratic` branch. There is no general theorem
that unconstrained least squares makes this precision positive definite.

The saved author failure has a negative covariance eigenvalue near
`-0.00458167` and finite complex policy coefficients. This is not just a
missing TensorFlow exception handler. Under the user's replication target,
that source failure is an acceptable baseline outcome.

TF's positive-definiteness guard rejects the diagnosed bad precision, but
end-to-end comparison of the complete failed/complex policy trajectory is
still unconfirmed. The distinction matters: rejecting this known bad case
does not establish no added failures on author-valid cases. The next failure
test must preserve the invalid policy explicitly; it must not discard imaginary
components and present the resulting real array as the author's input.

The incomplete 3537/4000 Lorenz source matrix and dependent APF run are also
not evidence that the port must repair the author. They limit the original
full-grid paper analysis; matched replication can account for source failures
without completing an impossible all-successful source grid.

## KDM/IWSG: what is checked and what remains unexplained

The current training endpoint is
`docs/benchmarks/diagnose_kdm_fresh_training_cost_tf.py:42`, which calls
`diagnose_kdm_source_epoch_tf.py:9`. That driver uses
`SourceWeightPretraining` and `SourceFullTraining` in
`bayesfilter/replication/kdm_source_stages_tf.py`, then
`BearingsNeuralReplay._sequence` and `_apply_update` in
`kdm_neural_replay_tf.py:190` and `:259`. The sequence calls the native
resampler and `kdm_normal_von_mises_tf.py` for KDE/IWSG and posterior NLL.
These are the actual full-training consumers, not unused helper capabilities.

The source anchors are `src/trainers/specific_trainers/full_sequence_trainer.py:174`,
`src/trainers/trainer.py:328`, `:509`, `:538`, and
`src/models/kde_particle_filter/kde_particle_filter.py:708`, under the pinned
author checkout. The source KDE implementation is
`packages/kernel-density-estimator-bandwdidth-prediction/packages/kernel_density_estimation/kernel_density_estimator.py:238`.

### 3. There is no demonstrated missing IWSG formula or training stage

For a sample drawn from the frozen current mixture, the paper's IWSG factor is
`m(z; phi) / stop_gradient(m(z; phi))`. Its value is one and its derivative
is the mixture score. The checked TF implementation uses
`exp(log_q - stop_gradient(log_q))`, with the author's small additive epsilon.
The sample itself is stopped. MDPF shares posterior/resampling parameters;
A-MDPF separates them. The checked code preserves that distinction.

The inspected call chain also preserves the author's four-step truncated
backpropagation, sparse training labels, particle initialization, weight-model
pretraining, stage transfers, complete training/validation rows, partial
batches, scheduler and early-stopping mechanics. Complete source and TF
development training have terminated and passed independent protocol audits.
Those audits establish that the specified program ran; they do not establish
equal learned quality.

At trained checkpoints, the exact-input double-precision replay passed all
928 checks, with maximum observed discrepancy about `2.27e-13`. The float32
screen passed 898/928 and failed 30, principally raw/clipped gradients in MDPF.
The source's own float32-versus-float64 gradients also differed on these inputs.
Thus the inspected finite program is algebraically compatible in these cases;
the evidence does not make author float32 gradients an exact real-arithmetic
oracle. It also does not clear the outstanding float32 screen.

### 4. Two finite-precision mechanisms are identified, but neither explains the final quality gap yet

The proposal represents angles by a sine/cosine pair and converts them back
using `atan2`. Its derivatives divide by the squared vector length, so a short
vector amplifies perturbations. The earlier
[angle-conditioning result](controlled-smc-kdm-neural-angle-conditioning-result-20261007.md)
explains the direct-angle discrepancies with checked conditional bounds. This
is a conditioning mechanism, not evidence that all downstream gradient or
training errors are acceptable. The source uses the same construction.

The new diagnostic isolates Adam by supplying identical saved gradients,
parameters, rates and moments to the actual TF `_apply_update` and the
installed reference `torch.optim.Adam`. It uses both zero moments and the
actual saved TF moments at step 24,332, for 16 prescribed updates. It deliberately
holds fixture learning rates fixed and is not a replay of the resumed training
controller. Gradients are already clipped and no weight decay is applied;
this tests full-stage Adam, not pretraining decay or evolving neural gradients.

The installed source Adam computes bias corrections `1-beta**step` using
Python double-precision scalars (`torch/optim/adam.py:529`). The TF code computes
these scalars in the model dtype (`kdm_neural_replay_tf.py:272`). Its moment
updates and tensor arithmetic can also round differently. A diagnostic-only
intervention changes just the precision of those two bias corrections.

| Sixteen prescribed updates | Fresh moments: max parameter error | Saved mature moments: max parameter error |
|---|---:|---:|
| Current TF float32 GPU/XLA | 9.54e-7 | 5.96e-8 |
| Current TF float32 CPU graph | 9.54e-7 | 2.98e-8 |
| TF float64 GPU/XLA control | 2.22e-16 | 2.78e-17 |
| Float32 GPU with diagnostic double-precision bias corrections | 5.96e-8 | 5.96e-8 |

All updates were finite and committed; each diagnostic graph traced once.
Both GPUs produced identical comparison rows. This isolates a real arithmetic
difference and reduces the fresh-moment parameter discrepancy sixteenfold in
this test. It does **not** demonstrate a training repair, improved NLL, or
that Adam is the dominant cause. Runtime code was not changed. Other Adam
arithmetic still differs; the mature-state result did not improve.

### 5. The trained-quality gap is real in the observed checkpoints; its population cause is unresolved

The TF-trained checkpoints were converted and evaluated by the **same actual
author evaluator** as the source checkpoints. That removes the TF evaluation
kernel as an explanation of this particular comparison. Across the three
completed source training units and one fixed TF unit, averaging the two
filter seeds within each training unit, TF NLL was higher by **0.300–0.600
nat per joint state** and RMSE by **2.79–9.73%**. See
[source variability](controlled-smc-kdm-source-variability-result-20261008.md).
These descriptive differences exceed the prospective margins of 0.01 nat
and 1% RMSE. There is only one TF training unit, so a population ranking or
formal noninferiority rejection is not established.

The complete runs use independent PyTorch and TF random streams, including
different batch permutations and sampler/proposal draws. Current tests do
not follow both evolving models through many matched updates. The first-step
and fixed-input replay results therefore cannot distinguish:

- accumulated arithmetic/gradient drift, including angle conditioning and Adam;
- ordinary training-seed variability and different optimization trajectories;
- sampler/initialization distribution differences not exposed by fixed-draw replay;
- a training-state or controller discrepancy that appears only after many updates.

Training at horizon 17 and evaluation at 150 follow the source setup. That
longer evaluation may expose a learned error, but changing the horizon would
change the replication experiment rather than explain its cause.

This is the main investigation-design gap: previous full runs changed random
inputs and arithmetic together. Additional independent full training runs
alone are an expensive way to locate a code error. The three-source pilot's
unpaired precision estimates are also not a trustworthy final sample-size
calculation: TF training variance and matched-pair covariance are unknown.

## XLA, Python loops and NumPy

The current numerical routes do not support a blanket diagnosis of an XLA bug:
cSMC's forced-branch discrepancy remains on CPU, and the new float32 Adam
fresh-state discrepancy is the same on CPU graph and GPU/XLA. A historical
XLA failure in the first conditional transition was handled by peeling that
transition out of the loop; the remaining time evolution still has one traced
TensorFlow loop body.

There are Python loops over epochs/batches in the host training driver, and
fixed loops over network layers and parameter tensors while constructing the
graph. Those are distinct from Python loops executing once per particle or
time step. Particle operations are batched; filter time, RK4 substeps and
circular rejection use `tf.while_loop`. Stable batch views cover full and
partial batches. Host `.numpy()` calls occur at reporting, checkpoint and
assertion boundaries. The inspected runtime path does not perform NumPy
numerical computation or use pfor. Independent author/reference harnesses
are a separate allowed role.

No retracing explosion was found in the inspected logs and new diagnostic.
This is a bounded call-chain finding, not certification that the entire
repository is free of eager code, NumPy or unrelated XLA issues. TensorArray
diagnostics and host materialization have resource costs, but are not a
demonstrated explanation of the observed numerical quality gaps.

## What will close the gaps

| Gap | Next discriminating work | Completion evidence |
|---|---|---|
| cSMC solver behavior | Use saved matrix/RHS pairs to test a bounded TF solver candidate against the recorded source branch/rank behavior; explicitly estimate feasibility before attempting a full DGELSD port. Retain centered QR as a numerically different candidate, not a claimed exact source solver. | Valid autonomous fit/forward behavior, followed by the agreed downstream confidence bounds. Coefficient residuals alone cannot qualify it. |
| cSMC intrinsic failures | Exercise the actual consumer on saved valid and failed policies, preserving complex/invalid inputs and reporting rejection explicitly. | Matched source failure accounted for, plus no unexplained additional failures on source-valid cases. |
| KDM accumulated drift | Build a bounded coupled-update diagnostic from real checkpoints and optimizer/controller states. Hold row order and underlying random variates fixed; transform those variates using each evolving model. First isolate the earliest differing update, then propagate to short validation. Include pretraining before MDPF/A-MDPF. | Attribution to a particular operation or state transition, with source controls and unchanged method. Fixed realized draws alone test conditional gradients, not full coupled training. |
| KDM Adam candidate | Compare the isolated scalar-precision change under that coupled diagnostic and healthy-input no-harm checks. | Reduced source discrepancy without a new failure; no runtime promotion based solely on the 16-update result. |
| Numerical qualification | After implementation checks, freeze candidate/settings and estimate matched-pair variance on development units before an untouched qualification cohort. | cSMC: 90% log-mean-likelihood equivalence interval inside ±log(1.01), upper 95% log-likelihood variance ratio ≤1.05. KDM: upper 95% NLL difference ≤0.01 nat and RMSE ratio ≤1.01, with the declared failure bounds. |
| Full selected source scope | Recover or explicitly disposition Lorenz Table 1 exact data, multimodal driver/initialization details, KDM Appendix B specifics and the original multi-trial checkpoint/evaluation scope. | Executable, source-anchored specifications and corresponding evidence; no silent substitution of an easier model. |

The broad source-scope gaps are not all numerical defects. Some exact original
inputs or experiment details remain unavailable locally. The published bearings
configuration generation and new development datasets/training are implemented;
that does not supply the original multi-trial checkpoints or reproduce every
paper experiment. All 22 inspected bearings evaluation tasks load NLL-trained
checkpoints, including tasks named MSE; comparison must follow the executed
source wiring and disclose this distinction.

The first next action is the bounded KDM coupled-update diagnostic, not another
uncontrolled full training campaign. Its plan must freeze a short update budget,
observed quantities and stop conditions before execution. cSMC can continue
separately with the saved fitting/failure fixtures. None of these steps requires
repairing the author's intrinsic failure or broadening into E0 enhancements.

## Decision and uncertainty

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| cSMC replication open | Autonomous no-worse qualification not passed | Solver/policy comparison failures remain; matched failure path incomplete | Whether a feasible TF numerical route satisfies downstream margins | Saved-system solver and failure-consumer work | Original MATLAB reproduction or author-method defect beyond the observed source failure |
| KDM replication open | Checkpoint development quality outside descriptive margins; untouched CI test absent | Float32 diagnostic failures remain; complete-run protocol audits passed | Cause of the quality gap and TF training variance | Coupled multi-update localization | IWSG mathematics wrong, final noninferiority, or source superiority in the population |
| Optimizer mechanism localized | Exact prescribed-input control passed | No nonfinite or invalid update; float64 control passed | Effect under changing gradients and full training | Evaluate isolated candidate within coupled test | Sixteen small errors explain the observed NLL gap |

| Inference status | Verdict |
|---|---|
| Hard veto screen | Known cSMC solver mismatch and KDM float32 failed screens remain visible; author intrinsic covariance failure is accepted reference behavior. New optimizer test had no execution/finite-value veto. |
| Statistically supported ranking | None from three source training units and one TF unit. Filter seeds are nested evaluations, not independent training replications. |
| Descriptive-only differences | The 0.300–0.600 nat NLL and 2.79–9.73% RMSE gaps, training epochs/times and source variability. |
| Default readiness | Not established by these replication diagnostics. |
| Next evidence needed | Matched evolving-training localization, independent paired training units and untouched downstream qualification with the frozen criteria. |

Post-run red-team: the strongest alternative explanation of the KDM gap remains
an unfavorable single TF training realization rather than an implementation
defect. Conversely, passing local float64 and optimizer checks cannot exclude a
later controller, distribution or float32 learning discrepancy. A controlled
sequence showing no extra drift, followed by adequate paired qualification,
would overturn the concern about systematic TF degradation. The weakest part
of present evidence is the one-unit TF training sample and missing coupled
trajectory. For cSMC, exact source-system replay supports localization, but
equivalence to the original MATLAB runtime and downstream acceptance remain
separate unproved claims. Nothing inspected establishes that either paper's
central algorithm is fundamentally wrong.
