# What the original implementations establish about the failed forward fits

The available evidence does not establish that the original programs fail to
fit our current two- and three-component development targets. A matched full
training comparison is missing. The demonstrated failure belongs to our shared
IAF training procedure and its automatic forward-KL-to-RKL handover. Local
numerical checks support the implemented density and gradients but do not prove
that the complete procedure reproduces an author's successful experiment.

This is a read-only source/evidence audit prompted by the user's question. No
new training, package installation, GPU execution or upstream experiment was
performed. It examines the recorded FAB equivalence artifacts, Gabrié fixtures,
NeuTra tests, AFT/CRAFT source and tests, and current campaign records. The
skeptical review explicitly separates primitive parity, complete training
behavior, raw-flow accuracy, and corrected-sampler accuracy. Earlier statements
that FAB's JAX code had never executed are historical: later artifacts prove
that it did execute for bounded equivalence checks.

## Original-code execution inventory

| Method | Evidence actually present in inspected records | What remains untested |
|---|---|---|
| NeuTra IAF | Source-based architecture tests, TFP mask references, local derivative/inverse tests and GPU/CPU comparisons | Original author's full training program on the current mixtures, with a matched TF port |
| FAB | Actual pinned fab-jax operations, seven upstream test functions after documented compatibility changes, and common-draw complete iterations | Converged original-versus-port training on the current separated two-/three-mode targets; current recipe and full dependency equivalence |
| Gabrié / flonaco | Selected unchanged original global-MH/MALA functions on controlled draws, plus original Croissants density/score | Original RealNVP training controller on the current targets, or complete matched adaptive-controller parity |
| AFT / CRAFT | Source inspected; local TensorFlow free-energy gradients, population separation and update-order tests | Execution/parity of the complete pinned JAX controllers and their trained results on the current targets |
| AIS / SMC | Local weight, mutation, resampling, normalizer and ancestry controls; original AIS operations also covered inside the FAB fixture | Matched standalone original-versus-local experiments on the current target families. These samplers do not themselves require a terminal IAF forward fit |

No matched full-training evidence was found in the inspected current scientific,
six-method and warm-start records. The corrected r2/r3 scientific campaign
failed the exact-teacher student prerequisite before its native method cells
ran. A failed common student cannot be counted as six failed original methods.

## FAB: real, substantial, but bounded equivalence evidence

The September 26 result at Git commit
`079eb5626eab0267ced0ca07f16ab7672c3fcdb1` supersedes the September 25 initial
audit's statement that JAX had not run. Its manifest is
`artifacts/neutra-fab-equivalence-2026-09-25/equivalence-manifest.json`.
The actual pinned original repository is `lollcat/fab-jax` at
`c9f991366ca94b2678a7ed620bc9e12655cfef1d`.

The inspected raw results show:

- `upstream-tests-compat-05.json`: seven functions completed after visible
  compatibility repairs. Three SMC functions are visual diagnostics, so this
  is not seven quantitative correctness proofs. Changes included import aliases,
  correcting test configuration, disabling the incompatible act-norm test branch,
  a nonzero FP32 inverse tolerance and a valid buffer sample request.
- `tensorflow-fp64-stress-03.json`: 1,804 checks passed, including original
  sampling/replay/loss/optimizer calls and four iterations in each of four lanes
  (HMC/Metropolis, with/without replay).
- `tensorflow-fp64-4d16-01.json` and
  `tensorflow-gpu-fp32-no-tf32-03.json`: 1,763 checks passed each.
- `tensorflow-gpu-tf32-compiled-callback-04.json`: seven declared gradient
  checks failed. Its largest reported parameter-gradient discrepancy was
  approximately 4.01e-4. This is a scoped precision-screen failure, not evidence
  explaining the FP64 r3 forward-fit failure where FAB was not running.

These tests found and repaired genuine port differences: fresh-loss scaling,
Optax-equivalent Adam equations/state, invalid-row handling and replay details.
Their nonlinear reference mixture has weights (0.4,0.6) and nearby centers
starting (0.6,-0.4) and (-0.7,0.8), not the present well-separated targets.
The reference driver's IAF callback is a test implementation checked against
the canonical map; the FAB operations themselves are imported from upstream.
Four matched iterations do not test whether training eventually fits a target.

The archived FAB module was recovered from the exact recorded commit and its
SHA-256 checked against the manifest. Comparing that module with the active
one found only the adapter change from `transport.dtype` to the actual first
trainable parameter's dtype. This does not imply that all current dependencies
or new targets inherit the historical equivalence result. It also does not
justify calling this substantial earlier test effort nonexistent.

## Gabrié: the paper reports related map defects and a successful sampler

The local paper is
`.localresources/fab-coverage-followup-20260928/gabrie-adaptive-flows.pdf`.
The inspected method is Section III.B, equations (8)-(9), Algorithm 1 and
Sections IV.A-C; the relevant experiments are Appendices F and G.1-G.2 and
Figure 5. Appendix G.1 explicitly reports:

> The residual connection between modes in (d3) originates from the
> transformation of the uni-modal base measure into the bi-modal target.
> Additional iterations would make it thinner.

It also reports failure to recover mixture weights when only nonmixing local
chains are used, and failure to discover the second mode when every chain starts
in the first. Its successful example initializes both modes and alternates local
sampling with flow-based global proposals. That example uses two unit-covariance
Gaussians separated by ten standard deviations, with weights 1:2. This is
related to our fixed mixture, but is not our current randomized development
experiment and does not establish the magnitude of our tail/moment errors.

The published mixture training protocol is materially different:

| Choice | Gabrié Appendix F / G.1 | Current width-64 r3 fit |
|---|---|---|
| Map | RealNVP, six pairs of coupling layers | IAF, three autoregressive stages |
| Conditioners | Depth-three MLPs, width 100 | Two hidden layers, width 64 |
| Gradient batch | 400 samples, 40 walkers advanced ten times | 64 samples resampled from a fixed exact pool of 4,096 |
| Optimization | Adam, 1,500 iterations, learning rate 0.005 | Adam, 8,192 forward updates at 0.001 |
| Sampling/training | Concurrent adaptation and sampling | Fixed teacher, then separate map fitting |
| Subsequent objective | Forward adaptation; the method discusses combining objectives after sufficient fit | Unconditional 256-update RKL finish at 0.0003 |

These are source-described choices, not proposed new defaults. The archived
CLI defaults differ from the paper's experiment settings too; launching an
upstream script with arbitrary defaults would not reproduce Figure 5.
Original source confirms `RealNVP_MLP` in
`.localresources/flonaco-author-20260929/upstream/experiments/gaussian/run_training_mog.py:109`
and the fresh/adaptive sample generation, loss-jump retries and training loop
in `flonaco/training.py:60`, `:212` and `:222`.

Our actual original-code fixture is
`tests/reference_neutra_warm_start_author.py`: it extracts and executes unchanged
`run_MALA`, `run_metropolis` and `run_metromalangevin` functions, with three starts
and two controlled transition rounds. It does not execute `train` or fit
RealNVP. Its output `author-reference-r2.json` is checked at 2e-14 tolerance in
`tests/test_neutra_warm_start_pipeline.py:57`. The current scientific arm is
explicitly a fixed-map global-MH/MALA control at
`bayesfilter/testing/neutra_scientific_campaign.py:153`. It is not the original
concurrently adapting learner.

## Different mathematical objects are being assessed

Forward KL is not intrinsically wrong for mixture fitting. At the population
level, D_KL(p||q)>=0, with equality iff p=q almost everywhere. A represented
global optimum therefore recovers p. Finite model capacity, finite samples and
nonconvex optimization can prevent reaching that optimum. The previous report's
example shows that small average forward KL need not exclude excess tail mass;
it does not prove that any particular published algorithm must produce it.

Gabrié's sampler can remain useful while q differs from p. For a frozen positive
proposal q, its independence-MH acceptance is

\[
\alpha(x,y)=\min\{1,p(y)q(x)/(p(x)q(y))\}.
\]

Then p(x)q(y)alpha(x,y)=min(p(x)q(y),p(y)q(x)), symmetric in x,y. This proves
invariance of p for the fixed MH kernel even if q is imperfect. Invariance alone
does not establish finite-time convergence. The adaptive controller needs its
own analysis; the paper's theory and its ULA-based numerical examples must not
be silently replaced by an unconditional finite-time correctness claim. The
paper explicitly notes its use of ULA in experiments and describes corrected
MALA as an option. Reported sampling success is consequently distinct from
exact raw-flow fitting and from Gaussian whitening.

Other methods solve still different subproblems. AIS/SMC return corrected
weighted populations. AFT/CRAFT train stage transports while retaining weight
and Markov corrections. FAB targets its alpha-two objective using AIS and
optional replay. Their success does not imply that a separate IAF, fitted to
their samples and then subjected to our RKL finish, is accurate. The October 4
exact-teacher control localizes a downstream student failure, not a failure in
any of those unexecuted native teachers.

## Attribution and the missing discriminating comparison

| Decision | Primary evidence | Veto / limit | Main uncertainty | Next justified action | Not established |
|---|---|---|---|---|---|
| Attribute the current failure to our complete fitting procedure | Exact-teacher maps have large shape errors and RKL coverage regression | Existing maps rejected | Optimization, finite family, initialization and remaining integration defects | Matched learner calibration | A published algorithm is impossible |
| Retain scoped FAB source parity | Original operations and common draws actually checked | TF32 screen failed; short finite domain | Long-run training and current dependency scope | Reproduce selected upstream training before generalizing failures | Full training equivalence |
| Treat Gabrié/AFT/CRAFT reproduction as incomplete | Primitive/local ordering tests and inspected source | Complete original-controller comparison absent | Differences in architecture, sampling and optimizer policy | Compare original controller, port, then IAF substitution separately | Same failure in original code |
| Keep map accuracy separate from sampler accuracy | MH/importance-correction identities | Finite sampling still needs validation | Efficiency with imperfect map | Report native sampler and common IAF results separately | Accurate samples imply perfect whitening |

The discriminating sequence is: reproduce the author's documented example with
its original architecture/controller; compare a faithful port under matched
inputs and settings; change only the map to the canonical IAF; then test the
current random targets with forward and post-RKL endpoints assessed separately.
Original-code success plus port failure would expose a reproduction gap. A
matched port passing until IAF substitution would implicate architecture or its
training calibration. Matched implementations both failing would establish a
failure of those settings on that target, not a theorem against the algorithm.

No new stochastic comparison or ranking was performed in this audit. No default
or readiness status changes. The strongest alternative explanation to an
algorithmic limitation remains an inadequately reproduced or calibrated local
training procedure. Existing derivative tests do not eliminate that explanation.
