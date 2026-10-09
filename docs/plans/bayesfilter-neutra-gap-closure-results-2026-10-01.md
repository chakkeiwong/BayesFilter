# NeuTra remaining-gap repair: execution record

Status: executed and reviewed. All three previously unresolved cases passed
fresh sequential HMC and final-reference checks. No worker remains active.
Plan: `bayesfilter-neutra-gap-closure-plan-2026-10-01.md`.
Controller: `scripts/run_neutra_gap_closure.py`.
Evidence root: `artifacts/neutra-warm-start-master-2026-09-29/campaign-r1/`.
The terminal audit is `gap-closure-20261001-r1/terminal-review.json`; current
phase and resource charges are in `gap-closure-20261001-r1/state.json`.

The repairs address two demonstrated mechanisms: forward training near a
Gaussian stationary configuration, and qualification that stopped at the first
verified HMC member. Representative training rows, zero clipping, and checked
directional derivatives separate the first problem from the earlier suspected
sampling or gradient defects. Optional initializer/capacity settings produced
viable maps, and testing the next verified trajectory closed the posterior
precision failures. These results do not make either setting a universal default.

| Case | Selected map | HMC ε, L | Warmup/chain | Retained/chain | Maximum R-hat | Maximum MCSE/SD | Final reference |
|---|---|---|---:|---:|---:|---:|---|
| Mixture, seed 11 | Width16, variance .2; 65,536 forward + 2,048 RKL updates | .25, 9 | 2,000 | 2,000 | 1.002537 | .026601 | Passed |
| Mixture, seed 37 | Width8, variance .2; 65,536 forward updates; warm map retained | .25, 9 | 2,000 | 2,000 | 1.001998 | .027327 | Passed |
| Wiggle, seed 23 | Preserved previously trained map | .1767766953, 9 | 2,000 | 6,000 | 1.003115 | .028274 | Passed |

All three selected members passed the declared numerical, movement, R-hat,
bulk/tail ESS and precision requirements. Minimum bulk ESS values were
1965.98, 1846.73 and 1305.97; minimum continuous-tail ESS values were 1485.61,
922.76 and 1301.95. Every case selected its **second** verified member. Each
first member exhausted 10,000 retained transitions per chain without passing.
Final-reference values were withheld until selection and evaluated once.

## What the code and mathematics establish

The implemented Gabrié update samples with the current IAF's independence-MH
proposal followed by MALA, then differentiates minus the mean log flow density
at the detached sampled states. It does not train on the Laplace proposal or
the placeholder pool in the caller. Both fixed-map kernels preserve the target;
their composition preserves the target but need not be reversible. Finite
adaptive training still needs an empirical check of its realized measure.
Gabrié et al., section IV.B equation (11) and IV.C, explicitly allow learning
from a finite walker distribution initialized outside equilibrium. Missing
burn-in is therefore not, by itself, a defect in this implementation.

For the unwarped mixture, the coordinates are independent. At the
moment-matched diagonal Gaussian, their standardized values satisfy
E[U_j]=0 and E[U_j²]=1. A conditional shift has density derivative
U_j h(U_<j); a conditional log scale has derivative (U_j²−1)h(U_<j).
Both expectations vanish by independence. Consequently the forward-KL
objective has a stationary configuration in the diagonal-affine subfamily of
the stacked autoregressive map. This derives a possible optimization trap,
not a local-minimum theorem or a limitation on the architecture's expressivity.
The full derivation is in `bayesfilter-neutra-failure-causal-findings-2026-09-30.md`.

The wiggle failure was different. Its first verified HMC member passed
numerical checks, R-hat and ESS, but missed MCSE/SD ≤ .03. The old consumer
never assessed the other nine verified members. Qualification now tries at
most three members, choosing the earliest distinct trajectory lengths first.
It uses health, convergence and precision for selection, then opens the final
reference once. A failed final-reference check cannot trigger another member
selection on the same holdout.

## Frozen mixture sampler and derivative check

Both saved 8,192-update maps used the calibrated MALA step .1. Each diagnostic
ran 64 independent fixed-map chains for 256 discarded and 1,024 retained steps
from saved walkers, mode 0 only, and mode 1 only. All six groups passed the
existing development-reference screen, using between-chain means for
uncertainty. These are bounded operational checks, not a mixing-time theorem.

| Saved map seed | Start | Right-mode frequency | Valley frequency | Global acceptance |
|---|---|---:|---:|---:|
| 11 | Saved walkers | .659271 | .001099 | .301300 |
| 11 | Mode 0 | .669495 | .001450 | .303391 |
| 11 | Mode 1 | .672104 | .001099 | .305878 |
| 37 | Saved walkers | .672821 | .001236 | .300461 |
| 37 | Mode 0 | .669159 | .001480 | .304657 |
| 37 | Mode 1 | .668747 | .001251 | .302490 |

The target's right-half-space mass is 2/3 up to its Gaussian tail correction;
its valley probability is Φ(7)−Φ(3)=.00134990. All proposals were numerically
valid. One parameter directional derivative per saved map, evaluated on actual
retained rows, differed from the central finite difference by 1.44e−11 and
8.56e−12, against the predeclared 1e−6 tolerance. This checks those directions,
not every parameter direction or the population optimum.

The evidence does not support changing initial mode weights or adding burn-in
as the first repair. It permits the planned optimizer/initialization diagnosis.
The six frequencies and acceptance values above are descriptive, not a ranking.

Artifacts: `attempts/gap-20261001-measure-s11-v0-r1/` and the corresponding
seed-37 directory. Each contains initial states, full warmup/retained tensors,
per-chain observable means, derivative checks, source identities and a run
manifest. Total worker cost: 44.134 GPU process-seconds and 85.648 CPU
core-seconds. GPU1 memory growth was configured and verified before device
initialization; the numerical blocks ran with XLA in the existing tfgpu env.

## Restored controls and initialization experiment

Both restored training controls subsequently reached 32,768 lifetime updates
with Adam state preserved and no clipped updates. Seed 11's heldout forward KL
was .947331, and its flow assigned .297729 to the valley whose target mass is
.001350. Across its 6,291,456 actual training rows, right-mode frequency was
.666276 and valley frequency .001320. The adaptive training measure therefore
does not share the flow's large valley error. This directly separates the
sampler's empirical fit to the target from the optimizer's failed flow fit.
These are descriptive training totals; correlation forbids treating the row
count as the number of independent observations.

Fresh runs with the inherited .02 initializer also exhausted their declared
65,536-update cap without passing the fit screen. The .2 initializer arms use
the same two seeds, rung schedule, architecture, objective and sampler. Their
initial weight standard deviations are multiplied by √10. All eight fresh
arms recorded zero clipped updates. Final development results were:

| Width | Initializer variance | Seed | Forward KL | Flow valley mass | Warm-fit screen |
|---:|---:|---:|---:|---:|---|
| 8 | .02 | 11 | .946927 | .295288 | Failed: Gaussian plateau |
| 8 | .02 | 37 | .947108 | .298462 | Failed: Gaussian plateau |
| 8 | .2 | 11 | .040998 | .021362 | Failed: still improving at cap |
| 8 | .2 | 37 | .016886 | .013550 | Passed |
| 8 | 1 | 11 | .039949 | .021362 | Failed: still improving at cap |
| 8 | 1 | 37 | .012171 | .007446 | Passed; earlier viable arm retained |
| 16 | .2 | 11 | .017718 | .008789 | Passed |
| 16 | .2 | 37 | .023083 | .016602 | Failed; earlier viable arm retained |

The table describes fixed runs. It does not establish a general width or
initializer ranking. In particular, width16 did not pass for both seeds.
The conditional 131,072/262,144 continuation mechanism was implemented and
reviewed because some arms were improving at their cap. Its execution check
recorded it as unnecessary after both seeds qualified; no extra training or
new budget was spent on it.

Seed 11's RKL checkpoint reduced development forward KL to .011923 and valley
mass to .005737 and was selected under the existing rule. Seed 37's RKL
checkpoints did not replace its forward-trained map. The selected maps' final
1,000-point probes are below; these probes use their recorded seeds and do not
support a ranking of extreme tails.

| Selected map | Residual median | q95 | q99 | Maximum | Log-density-ratio range |
|---|---:|---:|---:|---:|---:|
| Mixture seed11 | .149 | 3.713 | 37.942 | 1707.834 | 7.054 |
| Mixture seed37 | .161 | 3.399 | 340.734 | 878.754 | 20.298 |
| Wiggle seed23, preserved probe | 4.413 | 20.736 | 34.310 | 49.298 | 12.155 |

These maps are **not uniformly whitened Gaussians**. The probe tails and the
finite large-energy-error alerts during HMC remain substantive explanatory
findings. Passing the posterior screens establishes only their declared scope.

## A posterior-qualified mixture repair, and why the first kernel was insufficient

Seed 37, width 8 and initializer variance .2, passed the warm-fit screen at
65,536 updates (heldout forward KL .0168864, flow valley frequency .0135498).
The frozen Gabrié sampler passed from both modes. All three inspected RKL
checkpoints passed the coarse shape screen but had higher development forward
KL, so the existing selector retained the warm map. This is not evidence that
RKL cannot improve HMC geometry; those other maps were not compared downstream.

Fresh qualification then supplied direct evidence for the consumer repair:

| Verified member | Warmup/chain | Retained/chain | Maximum R-hat | Largest MCSE/SD | Outcome |
|---|---:|---:|---:|---:|---|
| ε=1, L=3 | 2,000 | 10,000 | 1.002555 | .034240 | Precision failed for x1²; final reference unopened |
| ε=.25, L=9 | 2,000 | 2,000 | 1.001998 | .027327 | Numerical, R-hat, ESS, precision and fresh final-reference screens passed |

The first kernel's coordinate means had bulk ESS about 64,367 and 25,208,
whereas x1² had bulk ESS about 974. Its failure is consistent with a known
Gaussian resonance, which supplies an exact reason that good acceptance and
coordinate-mean behavior cannot establish a useful HMC kernel. For the
one-dimensional standard Gaussian and identity mass, one leapfrog step is

\[
\begin{pmatrix}q'\\p'\end{pmatrix}
=M_\epsilon\begin{pmatrix}q\\p\end{pmatrix},\qquad
M_\epsilon=\begin{pmatrix}
1-\epsilon^2/2&\epsilon\\
-\epsilon(1-\epsilon^2/4)&1-\epsilon^2/2
\end{pmatrix}.
\]

At ε=1, M has diagonal entries 1/2 and off-diagonal entries 1 and −3/4.
Multiplication gives M² with diagonal entries −1/2 and the same off-diagonal
entries; hence M³=−I. A three-step proposal therefore sends q to −q regardless
of the refreshed momentum, preserves energy exactly, and leaves q² unchanged.
Its acceptance is one, yet it cannot estimate the variance from new independent
positions. The actual map is not exactly Gaussian, so this derivation does not
prove resonance is its sole failure mechanism. The saved quadratic-quantity
precision failure is the observed reason for rejection.

The selected map is still imperfectly whitened. Its warm 1,000-point probe has
median residual .181, q95 6.028, q99 214.605 and maximum 823.307. The successful
HMC run also has finite large-energy-error alerts, retained as explanatory
under the unchanged policy. Passing the declared posterior screens is scoped
operational evidence, not a claim of uniformly Gaussian latent geometry.

## Verification and interpretation limits

The 51-test focused suite passed in 87.90 seconds, followed by controller
holdout/retry and continuation regressions. A terminal configuration-preservation
repair passed its regression together with the affected master/queue suite
(21 tests, 3.15 seconds). Across these runs, 54 distinct focused tests passed.
Tests use the actual fit,
training and qualification consumers. Tuner/member outcomes are injected for
the selection regression, so that test establishes wiring rather than HMC
validity. The separate GPU campaign supplies the numerical evidence.

The full Python source copy is preserved in `gap-closure-20261001-r1/source/`.
Ordinary SHA-256 identities and Git provenance record exactly what ran despite
the unrelated changes in the shared worktree. Existing results remain intact.

The terminal audit checked 628 saved source files, 155 non-config inputs,
22 exact launch configurations and 12 new post-training probes without a
mismatch. The launcher previously used a shared config path refreshed between
phases. For this cycle, its exact input bytes were reconstructed from preserved
effective settings and matched to each pre-run SHA-256 before saving
`launch-config-recovered.json`. Original manifests remain unchanged. Future
launches now read their own saved `launch-config.json`; a regression changes the
shared file after launch and verifies the worker input remains intact. This is
a host-side reproducibility repair, not a change to the numerical experiment.

There were 22 completed workers: 19 GPU phases and three CPU reference phases,
with no infrastructure retry. Total worker charges were **3826.078 GPU process
seconds (1.062799 hours)** and **4558.271 CPU core seconds (1.266186 hours)**,
within the unchanged 6/12-hour sub-cap. The shared conservative ledger retains
82690.797 GPU process seconds (22.969666 hours) and 172323.215 CPU core seconds
(47.867560 hours). Routine engineering test time is reported separately above.

The prior review preserved 12 earlier passes and found 15 repaired passes with
three unresolved cases. This cycle closes those three. The coverage audit now
finds a passing saved execution for all 30 target/arm/seed groups, while retaining
every failed attempt. Older passes retain their original policies; they were
not rerun here. This selected collection is not an estimate of a 100% success
probability, nor evidence for untested q20 or high-dimensional targets.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Reject Gaussian-plateau fits | No nonlinear-learning or shape pass despite 65,536 updates | Shape veto; no clipping or observed sampling-measure failure | Stationarity derivation is not a local-minimum theorem | Preserve as failed initializer controls | Failure of IAF expressivity or Gabrié invariance |
| Retain the two mixture repairs | Fresh sequential HMC and final reference passed | No declared hard vetoes; score tails remain | Few seeds, selected attempts, imperfect geometry | Preserve exact selected maps and kernels | Uniform whitening or universal initializer/capacity default |
| Retain the wiggle repair | Second verified trajectory passed precision and reference | First member remains rejected | Source-dependent random streams differ from historical run | Use the current source-bound member evidence | A changed source seed alone proves a causal algorithm improvement |
| Close this bounded cycle | All three unresolved cases have passing fresh executions | Source/input/config/probe audit passed; budget respected | General reliability and q20 transfer remain untested | Stop workers and write local reset/status notes | Production or q20 readiness |

| Inference status | Finding |
|---|---|
| Hard veto screen | Selected members have none; failed fit and precision screens remain recorded |
| Statistically supported ranking | None |
| Descriptive differences | Loss, valley frequencies, residual tails, acceptance, ESS, precision and runtime |
| Default readiness | Not established; initialization changes remain optional hypotheses |
| Next evidence | A separate target-specific protocol and replicated evidence before generalization or q20 |

Post-run red-team review: the strongest alternative explanation is sensitivity
to stochastic optimization and selected successful attempts, rather than a
universal benefit from the larger initializer. Replicated failures of the
chosen settings would overturn a reliability claim, which is not made here.
The weakest numerical evidence is control of rare geometry/energy tails:
medians are favorable but maxima remain large, and neither the 1,000-point
probe nor the operational posterior screens establish exhaustive behavior.
