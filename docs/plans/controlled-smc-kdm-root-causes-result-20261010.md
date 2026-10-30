# Replication root causes and repair results — 10 October 2026

**Planning-error correction, 11 October 2026:** the five-cell cSMC scout was
selected to resolve the 1%/5% rule, which was wrong planning for controlled SMC.
The scout recommendation is withdrawn; the active next step is the
[author-relative acceptance reset](controlled-smc-author-relative-acceptance-20261011.md).
KDM repair recommendations remain active.

The investigation found and repaired one definite translation defect: KDM's
TensorFlow Adam computed scalar bias corrections and learning-rate arithmetic
in float32, while the author's PyTorch optimizer computes them in binary64.
On the same four-update development fixture, failing comparisons fell from
49 to 8 out of 3,616. All remaining failures concern four float32 proposal-network
gradients and their clipped copies. Neither replication is qualified by this
diagnostic result.

For controlled SMC, the source regression is extremely ill-conditioned and its
conditional linear solver does not have the same finite-precision behavior as
either TensorFlow's general dense solve or the centered-QR candidate. That
explains coefficient sensitivity; it does not establish worse downstream
performance. The completed Q2 comparison found zero additional TensorFlow
failures and no statistically supported material degradation, but most variance
intervals cannot establish the accepted 5% noninferiority margin.

The [reviewed execution plan](controlled-smc-kdm-root-causes-20261010.md) is
complete. The [terminal review](artifacts/controlled-smc-kdm-replication-20260928/root-causes-20261010/terminal-review.json)
accepts the bounded engineering repair, with replication still incomplete.
This was Codex self-review, not independent review. The main
[master program](controlled-smc-kdm-replication-master-program-20260928.md)
retains the author-relative criteria and separates the remaining experiments.

## What was traced and established

### Controlled SMC: normal equations amplify arithmetic and solver differences

The pinned source is Heng commit
`2696f6d611aae172d10102cb7bfbfd314bb6b88f`. In
`.localresources/code/heng-controlled-smc/StateSpaceModels/Lorenz/cSMC_Resample.m:187–193`,
the backward target is the negative incremental log weight minus the future
log normalizer. The fitted coefficients solve

\[
G\beta=b,\qquad G=D^\mathsf{T}D,\quad b=D^\mathsf{T}y.
\]

The actual source operation is Octave/MATLAB backslash on the rounded Gram
matrix. The inspected source execution selected its positive-definite branch
at 87 times and its singular branch at 14 times. In the port,
`MultivariateGaussianControlledSMC.fit -> _fit -> _regress`
(`bayesfilter/replication/multivariate_controlled_smc_tf.py:312`, backward loop
at 560) reaches either a general dense Gram solve or centered QR. Those are
different numerical procedures, although they address the same ideal
least-squares problem when full rank.

The saved fixture has 101 backward times, forcing 4.8801, seed 20260930.
Replaying each **saved rounded Gram matrix and RHS** through the same Octave
installation reproduces all 101 source coefficients and predictions exactly.
The fixture is therefore a valid source control. Independently forming the
Gram matrix already changes entries by up to 3.49e-10 at the first backward
time, time 100. There the design condition number is about 4.43e7, so the
ideal Gram condition number is about 1.97e15; the source reciprocal-condition
estimate is 1.40e-16. Squaring the condition number follows from the singular
values of a full-rank design: the eigenvalues of its Gram matrix are their
squares. Ordinary rounding can consequently produce large coefficient changes.

On identical source designs and targets, both actual TensorFlow regression
endpoints remain finite. The strict prediction screen fails at 99 of 101
times for the general Gram solve and 97 of 101 for centered QR; maximum
prediction errors are approximately 0.6135 and 0.4516. Both first fail at
backward time 99. Time 100 passes this absolute/scaled screen because its
target is tiny; that is not evidence of a well-conditioned first fit.

Independent CPU solver comparisons further separate mechanisms. On the 14
source singular-branch rows, a Gram SVD solve using LAPACK's default cutoff
reduces the maximum prediction difference to about 0.006915; dense, Cholesky
and direct-design least squares remain around 0.4516. The residual difference
still depends on driver, cutoff and arithmetic. A lower residual or a closer
selected fixture does not justify changing the candidate. No SVD rule, ridge,
damping, author repair or new fitting default was introduced.

Evidence: [source replay](artifacts/controlled-smc-kdm-replication-20260928/root-causes-20261010/root-causes-03/cpu/octave-replay.json),
[CPU solvers](artifacts/controlled-smc-kdm-replication-20260928/root-causes-20261010/root-causes-03/cpu/csmc-solvers.json),
and [actual TensorFlow regression endpoints](artifacts/controlled-smc-kdm-replication-20260928/root-causes-20261010/root-causes-03/gpu/csmc-exact-input.json).
The confirmed cause is sensitivity to finite-precision solver semantics.
It does not prove that the paper is wrong or that the centered-QR port is
materially worse under the owner's replication criterion.

### KDM: Adam scalar precision was a translation defect

The pinned Younis source is
`b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18`. Its installed reference optimizer,
`/home/chakwong/anaconda3/envs/tf-gpu2/lib/python3.11/site-packages/torch/optim/adam.py:529–547`,
uses Python scalar arithmetic in the noncapturable branch:

\[
s_t=\frac{\eta}{1-\beta_1^t},\qquad
d_t=\frac{\sqrt{v_t}}{\sqrt{1-\beta_2^t}}+\epsilon,
\qquad p_{t+1}=p_t-s_t m_t/d_t.
\]

The earlier TensorFlow implementation calculated the powers and scalar divisions
in the parameter dtype. With float32 parameters this changes updates even when
the gradients, parameters and moments are identical. The repaired
`bayesfilter/replication/kdm_neural_replay_tf.py:261` computes these scalar
operations in binary64 and casts at the source tensor-operation boundaries.
Learning-rate variables now preserve binary64 values. The moments, loss,
clipping, weight decay, optimizer hyperparameters and angle transform retain
the source procedure.

The isolation test injects identical source gradients and prior state through
the actual `_apply_update` endpoint. Maximum float32 parameter error drops
from 1.19209e-7 to 7.45058e-9, a factor of 16 on this fixture. Float64 remains
within approximately 1.11e-16. This is an engineering comparison against the
author implementation, not a statistical performance claim.

The executable wiring audit checks direct calls to `_apply_update` from replay,
native replay, full training and both weight-pretraining consumers. It is an
AST wiring check; it does not imply that complete training was rerun. Four
variant/dtype checkpoint tests verify exact learning-rate round trips, shared
batch-state rates and rejection before mutation of incompatible old states.
Training checkpoints now use `kdm_neural_training_state_v2` with
`adam_scalar_policy=pytorch_noncapturable_binary64_v1`. Old v1 optimizer
checkpoints are deliberately rejected instead of being silently resumed under
different arithmetic. Preserved historical results remain intact.

| Same four-update comparison | Before repair | After repair |
|---|---:|---:|
| Failed checks / total | 49 / 3,616 | 8 / 3,616 |
| Failed raw-gradient checks | 18 | 4 |
| Failed clipped-gradient checks | 18 | 4 |
| Failed first-moment checks | 12 | 0 |
| Failed gradient-norm checks | 1 | 0 |
| Failed parameter, second-moment, loss or particle checks | 0 | 0 |
| Float64 checks passing | 1,808 / 1,808 | 1,808 / 1,808 |

After repair, all 552 parameter checks pass; their largest absolute difference
is 2.68e-7. All 552 first-moment, 552 second-moment, 16 loss, 16 gradient-norm
and 576 particle checks pass. The eight failures are raw/clipped copies of the
last proposal-layer weight gradient: MDPF updates 0 and 3, A-MDPF updates 1 and
2. Their maximum absolute errors range from 2.34e-5 to 4.85e-5. The diagnostic
screen remains unchanged at `128 * eps32 * (1 + abs(reference))`.

Evidence: [before repair](artifacts/controlled-smc-kdm-replication-20260928/root-causes-20261010/root-causes-01/coupled/comparison.json),
[after repair](artifacts/controlled-smc-kdm-replication-20260928/root-causes-20261010/root-causes-03/coupled/comparison.json),
and [isolated optimizer and reset-state tests](artifacts/controlled-smc-kdm-replication-20260928/root-causes-20261010/root-causes-03/gpu/kdm-isolation.json).

### Remaining KDM gradients: sensitivity is supported; the exact operation is not localized

Resetting the port to the exact source pre-update parameters and moments before
each step leaves three failing raw-gradient comparisons. The evolving run
before the optimizer repair had 18. Thus accumulated training-state differences
explain much, but not all, of the discrepancy. Adam cannot explain an initial
gradient discrepancy computed before any update.

The source bearings transformer at
`.localresources/code/younis-mdpf-neurips-2023/src/models/particle_transformer.py:111–160`
uses internal coordinates `u=sin(theta), v=cos(theta)`, normalizes their radius
and returns `atan2(u/r, v/r)`. The port reaches the same construction through
the proposal MLP in `kdm_neural_replay_tf.py:156–173`. For nonzero radius,

\[
\theta=\operatorname{atan2}(u,v),\qquad
\nabla\theta=\frac{(v,-u)}{u^2+v^2},\qquad
\|\nabla\theta\|=1/r.
\]

The derivative therefore amplifies input changes near zero radius; its own
sensitivity scales as `1/r^2`. On 16 update/dtype/variant cases, selected from
the first 32 and the 32 smallest-radius vectors in each case, the actual source
autograd, TensorFlow primitive and analytic derivative all pass. The smallest
radius is about 0.001046, implying amplification near 956. A requested 1e-8
perturbation changes the wrapped angle by as much as 8.11e-6. Artifacts record
the realized perturbation because float32 cannot represent every requested
change. This is an angle diagnostic, not a bound on final training error.

The experiment supports shared conditioning and float32 arithmetic sensitivity.
It does **not** establish an incorrect `atan2` derivative, identify the precise
network reduction responsible for the remaining errors, or justify a relaxed
gradient threshold. That operation-level cause remains not checked beyond the
reported localization. No angular floor or clipping was added.

Evidence: [source primitive](artifacts/controlled-smc-kdm-replication-20260928/root-causes-20261010/root-causes-03/cpu/angles.json)
and [TensorFlow primitive](artifacts/controlled-smc-kdm-replication-20260928/root-causes-20261010/root-causes-03/gpu/angles.json).

### Compilation warnings: redundant wrappers, not repeated shape tracing

The previous cSMC diagnostic created six graph instances for three forcings
and two solvers. Each traced once, but TensorFlow's warning grouped calls from
the shared Python definition. `docs/benchmarks/compare_lorenz_fitting_forward_tf.py:103–168`
now reuses two fixed-signature engines and assigns one forcing variable before
each case. All 16 source forward controls pass; fit, forward and policy
evaluation trace counts are one. Source controls and numerical fit results
match the preserved diagnostic. The repaired logs contain no retracing warning.

The inspected repeated scientific kernels use TensorFlow graphs and native
loops; the static MLP-layer loops and host experiment loops are not Python loops
over particles. NumPy/SciPy occurs in explicitly independent reference analysis,
not in the repaired optimizer. The confirmation uses GPU XLA, with memory growth
verified and TF32 off. No XLA-off or eager fallback was introduced. These checks
do not establish that every historical XLA problem has been repaired.

## What the remaining qualification gaps mean

Q2 is preserved as a completed, fixed-sample experiment: 41 cells, 320 paired
trials each, 14 intrinsic source failures, five matched TensorFlow rejections
and zero additional TensorFlow failures on source-valid trials. Only five
cells satisfy both the interval-adequacy rules and the numerical margins.
The 1% likelihood intervals are within the margin at 39 of 40 author grid
points and at the generating parameter, but interval adequacy still matters.
There is no supported out-of-margin mean discrepancy or variance degradation.
An inconclusive interval proves neither degradation nor noninferiority.

The variance criterion is one-sided: its upper bound must be below `log(1.05)`.
The [precision diagnostic](artifacts/controlled-smc-kdm-replication-20260928/root-causes-20261010/root-causes-03/cpu/precision-feasibility.json)
reports two scenarios. Assuming zero effect and square-root sample scaling
suggests about 10.38 million variance pairs summed over cells. Retaining observed
effects instead gives about 20,469 pairs over 37 cells with finite forecasts;
four observed variance ratios already exceed the margin and have no finite
forecast if those point estimates persist. Neither forecast is a power
calculation or establishes tail adequacy. In particular, the zero-effect
scenario is not uniformly optimistic because many observed effects favor the
port. These calculations reject a blind large rerun, not the replication target.

For KDM, three complete author development runs and one pre-repair TensorFlow
run exist. Their independent training randomness does not provide a paired
estimate of training variability. The old port's descriptive differences were
0.300–0.600 nat in NLL and 2.79–9.73% in RMSE. They cannot establish a population
ranking or describe complete training under the repaired optimizer. Previously
estimated unpaired sample needs of 1,645–3,415 units per arm are unaffordable
within the current allocation and themselves uncertain from only three source
units. Paired calibration is the next useful experiment; another unpaired
single run would leave the same inference problem.

## Next phases, with review conditions

1. **Localize the remaining proposal gradient.** Use the four preserved failed
   update cases, exact source states and common incoming adjoints. Compare the
   proposal's forward activations and vector-Jacobian products backwards through
   angle normalization, sigmoid/residual construction and each dense layer.
   Use float64 as an independent arithmetic reference, preserving the FP32
   source baseline. A stage can be called causal only if replacing its arithmetic
   in a diagnostic replay removes the corresponding downstream discrepancy.
   Cap the diagnostic at 300 CPU and 900 GPU seconds; no threshold or source
   algorithm change follows merely from a strict mismatch.
2. **Target cSMC development where the uncertainty matters.** Examine fresh
   paired runs at F=2.5 and author-grid cells 12, 20, 21 and 30
   (F approximately 4.34615, 5.57692, 5.73077 and 7.11538). These nominate the
   likelihood concern and the four variance point estimates above 5%. Use 32
   pairs per cell as a cost-bounded engineering scout, not a statistically
   justified sample size or new default, with the frozen centered-QR candidate;
   capture influential pairs and the first source/port regression divergence.
   Reserve at most 1,200 CPU and 2,400 GPU seconds, verify the per-case cost
   before completing the set, and stop on broken source controls or extra
   nonfinite outcomes to repair the harness/port. Selection from Q2 makes these
   diagnostic cells unsuitable for a full-grid qualification claim.
3. **Measure paired complete-training variability after the repair.** First
   implement and validate a 64-update common-random-input smoke. Pair data,
   initial parameters, batch order and random inputs to sampling; each model
   must still sample from its own current distribution. Reusing realized author
   particles after the models diverge would change the port's training law and
   is not an eligible comparison. Preserve full author stage caps (400/800/800),
   validation, stopping and learning-rate rules. Start with one complete paired
   development unit; use its cost and paired evaluation variation to determine
   whether further units are feasible. One pair is a cost probe, not a
   population-level comparison. Historical cost is about 6.63 CPU hours
   and 3.30 GPU hours per unit, not a guaranteed upper bound. Three such units
   are a calibration target only, not an automatic qualification sample size.
4. **Freeze fresh qualification only after calibration.** Predeclare seeds,
   sample count, comparison unit, interval construction, adequacy, failure
   accounting and budget. Keep the cSMC 1% likelihood / 5% variance and KDM
   0.01-nat NLL / 1% RMSE criteria unchanged. Preserve Q2 and all development
   evidence; do not append until a desired interval appears. Report an
   under-budgeted qualification honestly if calibration cannot justify the
   required precision. Missing selected source datasets/checkpoints and paper
   experiment scope remain explicitly separate, as listed in the master.

These are the next research phases, not claims that they have run. Each long
training/qualification launch needs its concrete sampling design and bounded
supervisor under the existing authorization. The review rejects unchanged
unpaired reruns, artificial exact-coefficient requirements, reuse of failed
qualification data for candidate selection, and a promise that more compute
alone will close the gaps. No new user approval is required for localized work
within the authorized scientific scope and remaining allocation.

## Execution and terminal assessment

The literal approved supervisor prefix was reused; no allow-list expansion was
needed. The successful confirmation command was:

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python /home/chakwong/BayesFilter/docs/benchmarks/run_replication_gap_localization.py --phase root-causes --output /home/chakwong/BayesFilter/docs/plans/artifacts/controlled-smc-kdm-replication-20260928/root-causes-20261010/root-causes-03
```

| Attempt | Outcome | CPU wall seconds | GPU wall seconds |
|---|---|---:|---:|
| root-causes-01 | Baseline diagnostics complete; scalar mismatch isolated | 5.3773 | 165.2156 |
| root-causes-02 | Octave lacks `jsonencode`; serialization failed before GPU work | 1.7179 | 0 |
| root-causes-03 | TSV/standard-library JSON repair and numerical confirmation complete | 4.3254 | 175.6813 |
| Total | Within 600 CPU / 1,800 GPU second cap | 11.4206 | 340.8969 |

Attempt 02 is retained. Its localized exporter repair changed no numerical
operation and was reviewed before the one allowed recovery attempt. The
[successful manifest](artifacts/controlled-smc-kdm-replication-20260928/root-causes-20261010/root-causes-03/manifest.json)
records Git `89065bc6354801cb368d5e163ba49fd9c4372d10`, source snapshots and
hashes, input versions/seeds, commands, environments, GPU identity, memory policy,
logs and charges. The terminal review verified 29 source/input hashes.
The active remaining allocation is 220,860.17 CPU seconds (61.35 hours) and
117,789.92 GPU seconds (32.72 hours); the larger inherited ledger balance is
not an additional grant. No run remains active from this phase.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept Adam translation repair | Matched-gradient optimizer error reduced; consumer regression passes except preserved FP32 gradient screen | No new nonfinite/commit failure; four checkpoint cases pass | Complete-training effect | Paired training after common-random-input validation | Full KDM replication |
| Accept graph-harness repair | Source controls and numerical outputs preserved; one trace per function | No source-control or placement failure | Other uninspected routes | Retain stable wrapper reuse | Repository-wide compilation health |
| Retain cSMC candidate; qualification inconclusive | Full-grid noninferiority not established | Zero extra TF failures; no supported material degradation | Heavy tails and interval adequacy | Targeted development, then prospective design | Port worse than author, or replication complete |
| Keep remaining gradient discrepancy open | Four raw/clipped FP32 tensors still fail diagnostic screen | No downstream parameter/loss/moment failure on this fixture | Exact network operation and accumulated training effect | Layer/adjoint localization and downstream paired calibration | Incorrect angle derivative or justified new tolerance |

| Inference status | Assessment |
|---|---|
| Hard veto screen | Source controls, hashes, finite/commit and GPU/XLA checks pass. Strict FP32 gradient mismatches remain explanatory/repair triggers, not new acceptance prerequisites. |
| Statistically supported ranking | None. Q2 has no supported material degradation; it does not establish full-grid noninferiority. |
| Descriptive-only differences | Four-update error reductions, selected angle sensitivity, solver residuals and old unpaired training metrics. |
| Default-readiness | Not assessed; this is author replication and a bounded translation repair. |
| Next evidence needed | Source-preserving paired complete-training calibration and fresh adequately powered downstream qualification. |

Post-run skeptical review: the strongest alternative explanation is that a
smaller short-run optimizer difference need not improve final training
agreement. A fresh paired run with additional failures or persistent material
downstream loss would overturn any proposed qualification, even if component
checks pass. The weakest evidence is extrapolation from selected short fixtures
and heavy-tailed precision estimates. The work invalidates neither source method;
it repairs one port defect and identifies which uncertainty remains.
