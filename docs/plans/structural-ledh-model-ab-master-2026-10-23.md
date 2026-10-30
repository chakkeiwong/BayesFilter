# Structural Model A/B LEDH and surrogate-HMC campaign

## Active question and authority

Can the shared LEDH machinery handle a rank-one structural transition without
changing its deterministic equation, and then support parameter inference for
Model A and nonlinear Model B? This tests LEDH values/scores and their use in
HMC; fitting a Gaussian sigma-point likelihood is not an answer to this question.
The owner requested implementation, thorough review and execution, and granted
**48 GPU hours** on 2026-10-23. Use the isolated branch
`structural-ledh-model-ab-20261023`; do not touch `surrogate-hmc`.

## Targets and mathematical repair

### Controller call-chain audit amendment, 2026-10-24

Before training/sampling, inspection found two implementation defects: the
sequential controller checked only affine-unwhitened NeuTra coordinates, and
the continuation path omitted explicit GPU placement. Neither changes the
likelihood or score target, but both matter to executing the stated campaign.
Repair the repository-issued binding to carry an identified, frozen parameter
diagnostic map. Check R-hat jointly on sampler coordinates, unconstrained model
parameters and physical model parameters, at the existing warm-up and cumulative
retained checkpoints. Preserve the original sample archives and MH target.
Fresh runs, continuation and archive verification must use the same mapping.
Test a correlated-chain counterexample whose coordinate marginals pass while a
mapped parameter fails. Test the resume device call explicitly.

Skeptical audit: this is a missing implementation of the existing convergence
criterion, not a new validation posterior or an ESS/energy rejection rule.
Acceptance remains tuning information; no accuracy or superiority follows from
passing R-hat. The final untouched probes completed before source changes;
47.4137 GPU hours remain. Conservatively rerun numerical receipts after source
migration, charging all previous work. No campaign budget increase is needed.

Repair validation: 39 controller/master tests passed in 17.24 seconds, 43
resume/binding regressions passed in 39.29 seconds, and the six focused mapped
diagnostic tests passed again after the compatibility edit. The correlated-chain
fixture passes marginal coordinate R-hat but fails the mapped sum, demonstrating
why the missing map matters. It is checked through fresh, resume and final-check
consumer calls. The feature is opt-in; existing unrelated bindings keep their
payload identity. Logs are in `/tmp/ledh-structural-test-design-20261023/`.
The restarted A GPU integration also passed, reporting four convergence
quantities (two q parameters and two physical parameters).

Both models have state dimension 2, observation dimension 1 and innovation
dimension 1. Predict before observing; the initial state is random.

* A: `m'=.35*m-.10*k+sigma*e`, `k'=m`, `y=m'+tau*eta`;
  initial covariance I; truth `(sigma,tau)=(.25,.15)`.
* B: `m'=rho*m+sigma*e`, `k'=.55*k+beta*tanh(m')`,
  `y=m'+k'+.30*eta`; initial covariance diag(.25,.20), **variances**;
  truth `(rho,sigma,beta)=(.70,.25,.80)`.

The inspected full-state proposal builder factors Q. Here Q is singular for A,
and B does not even have an additive full-state Gaussian transition. Jittering
Q would change the model. Introduce a general completion protocol `C(x,e)` and
its analytical first derivatives, retaining the same proposal-map, transport,
moment restoration and correction authorities. Construct the three proposal
maps in innovation space, using the pulled-back observation `h(C(x,e))`.

For a uniformly selected parent j, use the conditional innovation mixture
`q(e|j)=sum_b beta_b q_b(e|j)`, including the standard-normal defensive component.
The augmented target is `(1/N)*phi(e)*g(y|C(x_j,e))`; therefore
`w=phi(e)*g(y|C(x_j,e))/q(e|j)`. The parent probability cancels. There is no
ambient two-dimensional transition density and no missing completion Jacobian:
both densities are with respect to counting measure on j and Lebesgue measure
on e. Do not sum densities over other parents' distinct curves. This is an
explicit structural adaptation, **not finite-N parity with the old marginal
Gaussian-mixture proposal**. After weighting, the shared reset approximates the
marginal filtering distribution. Its moment restoration does not prove equality
of distributions or unbiasedness of the subsequent likelihood estimator.

All parameter and parent-state sensitivities propagate together analytically.
For B, `dm=rho*dm_old+m_old*drho+sigma*de+e*dsigma` and
`dk=.55*dk_old+tanh(m)*dbeta+beta*sech(m)^2*dm`.
Autodiff/finite differences check this recurrence; they are not its runtime score.

## Evidence contract and research intent

| Field | Declared role |
|---|---|
| Baselines | A exact Kalman; B actual short-path integration; long-B independent replicated structure-preserving bootstrap PF. Gaussian/EKF projections are cheap explanatory comparators only. |
| Mechanism | Innovation-space proposal and exact deterministic completion, shared analytical batch recurrence/reset; canonical IAF whitening for B. |
| Primary engineering pass | Exact completion; analytic-score vs fixed-program FD; batch vs separate-row parity; correct endpoint wiring; no full-state jitter or scalar-row training. |
| Primary scientific pass | A posterior vs Kalman posterior; B short-target posterior vs refined integration when affordable; long-B reference likelihood-shape/score with measured reference uncertainty. Retained HMC R-hat <=1.01 for every parameter. |
| Materiality | Constant log-likelihood offsets cancel in a posterior. Compare centered likelihoods; posterior means within max(.1 reference SD, 2 combined MCSE plus reference numerical error). Quantile differences are reported in posterior SD units. No component percentage at a zero score. |
| Promotion veto | Invalid finite-program derivative, failed deterministic constraint, stale/mismatched tuning identity, unresolved reference refinement, scalar training, wrong target/prior/Jacobian, or nonfinite accepted samples. A failed approximation blocks that scope's promotion, not unrelated repair phases. |
| Continuation veto | Corrupted data/artifact, demonstrated invalid target/math, exhausted 48-hour budget, unavailable required hardware, or irreparable missing evidence. Localized harness/compile failures trigger repair within budget. |
| Explanatory only | Acceptance (target .70, useful .65–.75), ESS, energy error, map covariance/condition, losses, runtime, moment residuals, truth coverage on one dataset. Large energy error alone does not stop HMC. |
| Nonclaims | No universal default, exact long-B posterior, superiority from one seed, convergence from short smokes, or canonical admission just because shared code is called. Direct LEDH HMC targets the explicitly frozen approximate posterior. |
| Artifacts | Unique `docs/plans/artifacts/structural-ledh-ab-20261023-*` root: manifest, checkpoint, phase attempts, references, tuning, frozen map, all warmup/retained chunks and terminal interpretation. |

Heuristic adversaries are constructed from the problem: bootstrap filtering with
the exact completion (no flow), EKF/UKF Gaussian projection (cheap local nonlinear
approximation), and A's Kalman filter / B's refined short-path integration. Report
separately by A/B, horizon and the evaluated parameter region. Current probes
cover truth, positive/negative local shifts and the prior center. Boundary/tail
behavior has not been evaluated and must not be inferred from these probes.
Do not pool these situations. A heuristic loss is a promotion veto for claims of
computational advantage, not evidence that an implementation is mathematically
wrong. No optimization against the final reference/heuristic holdout.

## Phases, limits and refresh mechanism

The executable master owns a JSON state machine. Each phase records its command,
commit/source digest, environment, seeds, wall time, GPU time, evidence and next
phase. It refreshes the next-phase plan from actual results. Completed phases
are not rerun on resume; retries use new attempt directories. An interrupted
attempt remains preserved. Defaults below are hypotheses until checked.

1. **References and mechanics (CPU diagnostic).** A T=2,10,50 Kalman; B T=2,3
   path quadrature (integrate initial k analytically); refinement and exact
   completion/score/chart tests. Test B beta=0 and A singular-Q controls.
2. **Generic structural repair and GPU pilot, cap 4 GPU hours.** Add the protocol
   to the shared batch recurrence; batch 1/2/4 parity, all-parameter derivatives,
   same-flow/reset call-chain tests; FP32/XLA N=4032 pilot and memory/runtime.
   Tiny N is mechanics only and obeys K=N. No production claim from that smoke.
3. **Offline per-scope calibration, cap 6 hours.** A T=2,10,50; B T=2,3,10,50.
   Disjoint generated calibration and validation datasets, then freeze controls
   per exact scope before untouched inference data. Start flow=16 and
   Sinkhorn=(epsilon=1, iterations=5), compare a bounded 3-candidate ladder
   (flow 8/16, reset 5/10). Preserve moment, finite and cost diagnostics. Use
   at most 3 candidates per scope and 2 repair rounds. Do not interpret
   inherited settings as tuned. B independent PF references use repeated seeds
   and a particle ladder until measured uncertainty is adequate or cap reached.
4. **B T=50 NeuTra, cap 16 hours.** Native batch target B>1, matched finite LEDH
   log posterior and analytical score. Shared canonical author IAF, ELU masks,
   reverse permutations, bias outside c=2 cap. Target-specific width 18/36,
   learning-rate and training-budget ladder, at least two training seeds;
   disjoint heldout loss/gradient checks and downstream pilot. Record poor
   whitening as a tuning repair, never infer quality from architecture alone.
5. **Adaptive HMC, cap 20 hours.** A T=50 Kalman MH endpoint with frozen LEDH
   force (surrogate-force HMC); B T=50 frozen LEDH posterior and NeuTra map
   (approximate posterior, explicitly labeled). Shared sequential controller;
   four chains, warmup min 2000/latest1000 R-hat<=1.05/max10000; retained draws
   grow in chunks to all-parameter modern R-hat<=1.01 or max20000/chain or cap.
   All warmup archived and excluded. No independent validation chain. Keep
   ESS and energy diagnostics; neither is a standalone stopping rule here.
6. **Reserve and terminal audit, 2 hours.** Unspent phase budget may transfer
   within 48 total hours, recorded before use. Summarize evidence and unresolved
   scopes honestly. Budget/iteration exhaustion is incomplete, not success.

Attempt limits and elapsed time count failures. Supervisor termination preserves
partial evidence and a resumable checkpoint. No fixed target-independent speed
estimate before the N=4032 pilot. CPU references have a separate 2-hour cap.

## Defaults and assumption audit

| Choice / provenance | Justification and failure mode | Early check / status |
|---|---|---|
| FP32, N=4032 / owner | Requested speed/memory; roundoff or particle bias possible | FP64 small parity and 4032 pilot; reviewed execution choice, not accuracy proof |
| TF32 off / owner FP32 preference | Unambiguous FP32 comparison | Manifest and device probe |
| GPU/XLA / repo | Default numerical execution | Growth verified before initialization; compile/allocator telemetry |
| K=2016 / repository chunk selector | Exact divisibility at 4032 | Enforced by shared transport |
| Priors / synthetic experiment choice | No original paper defines these synthetic models' priors | Prior predictive/score checks; declared hypothesis, not literature attribution |
| A log(sigma)~N(log(.4),.7), log(tau)~N(log(.3),.7) | Positive broad scales, not truth-centered | Chart/Jacobian equivalence test |
| B atanh(rho)~N(.5,.8), log(sigma)~N(log(.4),.7), log(beta)~N(log(.7),.6) | Stationarity, positive noise and positive nonlinear loading; this defines an identifiable restricted target | Prior predictive and boundary probes; no claim for negative beta |
| Initial laws / model constructors | Preserve specified variances and timing | Simulation/Kalman tie-out |
| Defensive mix (.2,.4,.4) | Positive prior component bounds phi/q by 5; other arms are warm starts | Record importance weights, nominal only until scope calibration |
| Flow16, epsilon1, Sinkhorn5 / previous owner choices | Starting hypotheses, not cross-model evidence; OT regularization can alter later filtering | Scope ladder and centered reference errors |
| Correction enabled / canonical direction | Shared GenUT dual-cap machinery; no invented structural fork | No-fire finite regression, correction diagnostics, scope calibration |
| IAF 3 stages, widths18/36, c2 / canonical architecture | Author block masks require widths divisible by B's parameter dimension 3; B quality unknown | Target-specific seed/capacity/LR ladder, heldout and HMC |
| HMC limits / owner and shared policy | Adaptive readiness rather than arbitrary1000 draws | All draws/decisions retained; cumulative R-hat only |

For priors expressed as normals on chart coordinates, evaluate that chart-space
density directly. The equivalent physical prior is density divided by the chart
Jacobian; adding another Jacobian to the chart-space normal is wrong. This is
tested and the chosen convention is written into the target identity.

## Skeptical pre-execution audit (2026-10-23)

**Revised before execution.** The first possible baseline, the existing Model B
SVD sigma-point adapter, computes a different approximate likelihood and cannot
establish LEDH correctness. The full-state Gaussian proposal cannot evaluate a
density on B's curve. Both were removed as claim-bearing routes. The proposed
innovation importance ratio is derived above, including its augmented parent
measure. Exact deterministic completion is separated from reset approximation.

No loss, ESS, acceptance or point-score percentage is promoted to posterior
convergence. Reference refinement and Monte Carlo uncertainty are explicit;
long-B PF is not an exact oracle or a random ordinary-HMC endpoint. All scopes
tune separately and final inference data stays untouched. Training configuration
is target-specific and canonical IAF rather than a historical map. GPU/CPU,
XLA, batch and memory policies are explicit. The plan has finite attempt/time
caps and distinguishes failed candidates from invalid experiments. Execution may
start with references/mechanics; scientific promotion remains conditional on
the actual reference and posterior evidence.

## Execution command

From this worktree, use the dedicated master entry point (implemented in this
campaign) with `run --gpu-hours 48`; use `status` and `resume` against the same
output root. Every command requiring a GPU runs with trusted/escalated tool
permissions. CPU-only reference workers hide CUDA before TensorFlow import.

Executable: scripts/run_structural_ledh_ab.py, using the tf-gpu environment.
Initial command: /home/chakwong/anaconda3/envs/tf-gpu/bin/python
scripts/run_structural_ledh_ab.py run --gpu-hours 48 --until pilot
--output docs/plans/artifacts/structural-ledh-ab-20261023-01.
The resume command uses that same output root and remaining budget. GPU0 UUID
is fixed in the master; GPU1 is occupied by unrelated work.

## Implementation review and evidence

Nine CPU mechanics/reference checks pass: completion and derivatives, conditional
innovation importance density, full analytical recurrence finite differences,
batch parity, A singular-Q Kalman, B quadrature linear limit, and independent
Fisher path-score agreement within measured Monte Carlo error. The Sylvester
custom op was built locally; no packages were installed.

The master uses disjoint calibration/validation/final seeds, fixed designs,
scope-bound tuning receipts, native batch4 targets, batch8/shard4 GPU training,
and the existing frozen-position-force sampler with sequential assessment.
Calibration loss and heldout ELBO nominate candidates only; statistical ranking
or exact nonlinear posterior claims do not follow from these losses.

Training starts at the declared prior center and half-prior scales (hypothesis,
not learned whitening): widths18/36, learning rates .001/.0003, seeds17/29,
256-update screen then1024-update extension. Early checks are finite target and
optimizer updates, independent heldout loss, inverse accuracy, and downstream
adaptive HMC. Adam(.9,.999), epsilon1e-7 and clipping1 are explicit numerical
hypotheses. Retain gradient norms and rejected updates. A poor map triggers
target-specific training repair; it does not reject structural filtering.

HMC targets .70 acceptance, with nomination bands overall .60–.80 and per-chain
.40–.90; retained .65–.75 is a tuning diagnostic. Energy and ESS do not terminate
posterior sampling. Shared health checks still reject corrupted or nonfinite
transitions. All postwarmup samples count for inference, including fields named
verification by the shared historical API.

Pre-launch audit: CPU evidence supports adapter mechanics, not N4032 feasibility
or approximation quality. The bounded pilot measures those costs before training.
Posterior/reference comparisons and the constructed heuristic table remain
downstream obligations; sampling completion alone is not scientific success.
The first launch is therefore explicitly capped at the pilot phase while those
remaining reporting and recovery obligations are reviewed and implemented.

## Pilot repair: compact Cayley action (2026-10-23)

The N4032 FP32/XLA pilot consumed 440.08 GPU seconds. Model A T2 batch4
was finite, with 51.70 seconds steady evaluation and 13.31 GB peak allocator
memory. Model B requested a 15.12 GB allocation and failed. These are engineering
failures, not evidence against the structural model or its posterior.

The shared moment correction builds dense N-by-N Cayley matrices and their
parameter/basis sensitivities, then solves N-by-N systems. Its declared basis
already has only 2r columns U (r=4). Write G=U A U', S=U'U, where A contains
the same paired skew coefficients. Keep S explicitly: floating-point cosine
columns are not exactly orthogonal. The original normalization is exactly
sqrt(1+trace(A' S A S)); its derivative with respect to v_j is
trace(A' S E_j S)/scale. Let D=kappa*A/scale. Then

    O' z = z - U w,
    w = (I + D S/2)^(-1) D U' z,
    dw/dv_j = (I+D S/2)^(-1) (dD/dv_j) (U'z-Sw/2).

This follows by substituting the solution into (I+K/2)O'z=(I-K/2)z,
K=U D U'. It retains the complete finite objective, every mixed derivative,
the declared radius, and the shared correction path. It changes no model,
particle count, precision, target or tuning control. Analytical Jets propagate
theta derivatives through these small solves; no autodiff score is introduced.

Skeptical audit: assuming exact orthogonality or dropping scale derivatives
would change the finite program, so both are retained. Check dense-vs-compact
actions, all mixed derivatives, nonorthogonal U, and end-to-end filter parity
on small arrays before repeating both N4032 pilots. FP64 checks use 2e-10
relative/absolute tolerance; FP32 uses 1e-4. A speed/memory improvement cannot
override a parity failure. Preserve the original attempts and budget charges;
record a source migration before retry. This repair uses the existing pilot
and reserve allocation; no additional GPU budget is requested.

## Reviewed execution repair and terminal assessment

The repaired N4032 T2 pilots pass on GPU0. Model A's repeated evaluation fell
from 51.70 s to 0.0824 s and peak allocation from13.31 GB to0.40 GB; its reported
FP32 values and scores are identical. Model B now passes at0.0867 s and0.40 GB.
These timings describe one paired run, not a statistically supported ranking.
The original failed attempt remains preserved and charged. At this checkpoint
47.866642483 GPU hours remain from the owner's48 hours.

Skeptical review found that finite probes alone did not exercise HMC or the
canonical map. The master now runs an actual XLA endpoint-corrected transition
for each model, checks force/endpoint health and the frozen map's inverse and
finite-difference log determinant. These are mechanics checks, not convergence
evidence. The same map checks run after training. Model A's MH endpoint is
independent exact Kalman, with its prior recomputed at the endpoint dtype;
Model B's MH endpoint is the explicitly rounded FP32 finite-filter target.
Its analytical FP32 force is a deterministic proposal field; no claim of an
exact derivative of floating-point rounding or of the exact B posterior follows.

Final assessment now includes a129/257-point-per-axis Model A posterior grid,
prior-tail boundary mass, refinement error, and the predeclared MCSE-aware mean
agreement test. Model B short-path likelihood/score checks remain separate from
long-target posterior accuracy; no short B posterior is sampled in this bounded
campaign. The heuristic table reports EKF, spherical-radial cubature, and the
available bootstrap/reference estimates separately by horizon and probe region.
Without an equal-runtime comparison no computational-advantage claim is made.
No boundary-regime claim is made from the local/prior-center probes.

The audit also rejects inference on a numerically invalid final scope and
invalidates numerical/reference receipts on source migration. Prepared data
are preserved only when their generator is unchanged. Budget is never reset.
These repairs close harness gaps; they do not relax a scientific criterion.
Remaining risks are finite-design B approximation error, PF bias/path-score
variance, and insufficient target-specific training under the available budget.
Successful sampling alone cannot resolve those risks.

Integration repair note: A's XLA transition passed after moving a host health
diagnostic into the compiled function; eager GPU IsFinite on this installed
CUDA build failed with INVALID_HANDLE. Artifact restore and dtype-conversion
assertions are therefore host operations; numerical HMC and training remain
GPU/XLA. The B smoke constructor now supplies the canonical required cap and
seed. Inspection found a non-None nomination energy limit would request a step
repair, contrary to the owner's diagnostic-only instruction; nomination now
sets it to None. The sequential controller's legacy energy veto remains off.

# Integration device repair
The call-chain audit also found that candidate nomination received one initial
position despite requiring an explicit four-chain bank. The master now passes
the declared dispersed starts. Optional numerical-device selection is serialized
in the shared tuner and used by tuning, sequential chunks and archive consumers;
host orchestration remains on CPU. Its unset behavior remains unchanged. The
integration phase executes the nomination implementation used by the typed
adaptive campaign, with eight mechanics-only transitions before serious sampling.
An independent finite-
difference determinant diagnostic runs on CPU because LogMatrixDeterminant is
unsupported by GPU XLA. None of these smoke outputs establish convergence.
Model A compiled transitions passed. Model B failed before inference in an eager
FP64 Sub kernel during map initialization (CUDA_ERROR_INVALID_HANDLE on the RTX
5080 TensorFlow build). Frozen smoke-map setup, restore checks, initial coordinates
and posterior reporting now run on CPU. Compiled force, endpoint, transition and
training updates remain GPU workloads. This changes setup/reporting placement,
not the target or algorithm. Repeat GPU integration before advancing; preserve
and charge all failed attempts.

The first nomination smoke mistakenly supplied an ordinary TensorFlow config to
the public dispatcher, which selects main's different candidate-set controller
and rejects evidence rungs below 64 decisions. The actual campaign uses the typed
VerifiedTensorFlowHMCConfig route and its shared nomination implementation. The
smoke now calls that exact implementation; the serious run still uses the public
typed adaptive entrypoint. This is a smoke wiring repair, not relaxation of an
evidence threshold. Failure occurred before sampling and is preserved/charged.

## Integration checkpoint, 2026-10-24 Hong Kong

Both compiled integration attempts passed: `model_a-integration-06` and
`model_b-integration-04` in the campaign attempt directory. These include the
actual shared nomination stage used by the typed adaptive campaign. Model B's
maximum inverse error was 4.44e-16; its finite-difference log-determinant error
was 1.06e-11. This checks a canonical smoke map, not a trained posterior map.
The numerical-device setting is GPU:0 with XLA; host setup and reporting are
CPU. All 33 focused master/controller tests and 54 existing shared-proposal,
precision and compact-Cayley regressions pass. No convergence, learned-map
quality or posterior agreement follows from these mechanics checks.

The master advanced automatically to independent references and then scope
tuning. At this checkpoint 47.63 GPU hours remain, including charges for failed
attempts. The next decision is whether frozen per-scope controls produce valid
untouched probes; learned whitening and adaptive HMC remain subsequent phases
of this same master program.

## Training setup repair, 2026-10-24 Hong Kong

All seven scopes completed tuning and untouched probes, but all four Model B
training arms failed before an optimizer update. The first failing operation
was eager GPU multiplication during prior-scale setup (CUDA invalid handle).
This is an execution-placement failure, not evidence against NeuTra or the
structural likelihood. The failed attempts remain charged to the 48-hour cap;
169578 GPU seconds remained at the failure checkpoint.

Repair: perform initialization arithmetic and external latent draws on CPU,
place trainable and optimizer resource variables on GPU, and explicitly invoke
the complete native-batch training step and heldout numerical graph on GPU/XLA.
The model, prior, objective, architecture ladder, seeds and stopping rules stay
fixed. Add an actual two-update Model B T=2 integration check using the same
builder as serious training. It must show finite accepted updates, changing
parameters, GPU resource and result placement, and an XLA-compiled full step.
Passing this check establishes execution wiring only; it cannot promote a map.

Skeptical audit: moving all variables to CPU would silently change the training
backend and is rejected. Moving only the first multiplication would miss network,
optimizer and random-draw setup. Check the complete builder and update call chain.
CPU/GPU rounding parity is an engineering check, not a posterior criterion.
No numerical protection or threshold changes are authorized by this repair.
Resume with source migration, preserved data, fresh numerical receipts and the
unchanged total budget. A setup failure triggers another localized repair.

Repair validation: `model_b-training-setup-01/result.json` records two accepted
updates, optimizer iterations 1 and 2, changed network parameters, GPU placement
for every model/optimizer resource and loss tensor, and the compiled full-step
XLA attribute. Compile plus first update took 16.51 seconds; the second update
took 0.153 seconds at T=2. These are descriptive smoke timings, not T=50 cost
estimates or posterior-quality evidence. Twelve focused master/controller tests
also pass. The resumed master proceeds through integration, tuning, training
and sampling without another authorization boundary.
The 21 native-sharded-update regressions also pass (113.93 seconds, intentional
CPU test execution). They cover complete Adam update and checkpoint/resume parity
against unsharded updates, FP32/FP64, padding, invalid-shard rollback and mixed
target/map precision. CPU parity supports engineering correctness; the separate
two-update GPU artifact establishes actual campaign device placement.


## 24 October continuation: Model B reset precision

Model A has completed inference and exact-Kalman posterior assessment. Model B
recovery follows structural-ledh-ab-endpoint-recovery-2026-10-24.md. The master
now supports a scoped, budget-preserving migration, opt-in conditional FP64 reset
with FP32 recurrence, production arithmetic checks, parameter-only NeuTra warm
start with fresh optimizer, and fresh adaptive HMC. Original B draws are not
combined with the changed numerical target. Historical reference receipts remain
labelled as original precision; the completed Model A result is preserved.
