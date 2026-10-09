# NeuTra mixture repair: controls, training, and posterior evidence

Status: reviewed research design, not an implemented or executed campaign.
Prepared in response to the owner's request for a thorough remedy grounded in
the literature and current findings. The completed rare-region campaign stays
closed. No numerical run, architecture/default change, or budget extension is
made by this document.

## Research intent and scope

Main question: can the shared canonical IAF produce useful fixed transports
for posterior estimation on the unwarped and warped two-dimensional mixtures,
and which of teacher error, optimization, capacity, geometry, and HMC limits
prevents that result?

The objective is usable posterior estimation. Perfect Gaussianization is not
required for HMC correctness, and a low diagnostic residual does not establish
either convergence or coverage. The methods that already passed deliberate
probability-estimation screens remain useful; there is no need to rerun all
four methods before testing the transport repair.

| Research role | Definition |
|---|---|
| Candidate mechanism | Persistent weighted forward training during RKL refinement, plus measured continuation and targeted geometry diagnosis |
| Exact reference | Normalized analytic mixture and its known inverse warp |
| Comparators | Analytic transport; canonical IAF with exact versus estimated teachers; matched FKL, RKL, and combined-objective continuations |
| Expected failure | Lost mode weights, inaccurate bridge geometry, optimizer plateau, inadequate teacher, or insufficient HMC information |
| Promotion criterion | Frozen-map shared HMC procedure passes declared physical-space convergence, precision, and independent-reference checks on both targets |
| Promotion veto | Invalid target/derivatives, nonfinite map, noninvertibility, lost required coverage, failed HMC health/reference checks, or insufficient information |
| Continuation veto | Shared mathematical/implementation invalidity, corrupted evidence, unavailable required hardware, or exhausted allocated budget; blocks affected dependent work |
| Repair trigger | Candidate-specific plateau, missing mode, inadequate teacher precision, ill-conditioned geometry, or an inconclusive bounded kernel search |
| Explanatory diagnostics | KL, score residuals, Jacobians, clipping, gradient noise, acceptance and runtime; large finite residual alone does not reject the research direction |
| Not concluded | Exhaustive mode discovery, impossibility of IAF, reliability on arbitrary targets, or readiness for q20 from a simple-target pass |

## Existing evidence that the next plan must retain

The October 2 campaign trained new rare-region banks for only 1,024 FKL plus
1,024 RKL updates, without establishing convergence. However, earlier October 1
work already continued other teacher/initialization combinations to 65,536 and
131,072 updates. Some improved and some plateaued. Thus the diagnosis is not
that every historical failure is explained by short training. Those earlier
passes also used earlier posterior profiles; they are not retrospective v4
confirmations.

The earlier controller already supports restoration of map, Adam, lifetime
update counts, and sampler state. The new `neutra_rare_region_phases.fit_bank`
uses a separate fixed-count selection/continuation loop and saves frozen maps
without a resumable training state. Sharing the IAF numerical core did not
prevent orchestration drift. The repair must preserve its corrected stratified
minibatches while reusing the tested continuation/state machinery.

References: `bayesfilter-neutra-causal-repair-results-2026-10-01.md`,
`bayesfilter-neutra-geometry-selection-results-2026-10-01.md`, and
`bayesfilter-neutra-rare-region-diagnosis-2026-10-02.md`.

## Literature basis and local adaptations

| Source inspected | Supported mechanism | What is proposed locally |
|---|---|---|
| Noé et al. 2019, Methods Eq. (9), energy/example losses Eqs. (1)–(2), supplementary Fig. S2; archived `train_flexible` and double-well notebook | Retain maximum-likelihood and energy/RKL terms together; energy-only training can select a single metastable state | Combine corrected weighted FKL and exact-target RKL in the canonical IAF; a source-based objective adaptation, not reproduction of the author's architecture/experiment |
| Gabrié et al. 2022, III.B, IV.A–E, VI.A, Appendix G.1; original independent MH and global/local code | Forward fitting with global/local exploration, initialization across known modes, and mixtures of separately trained maps | Retain mode-conditioned sampling banks; freeze adaptation before retained teacher assessment; use corrected global/local sampling as a teacher mechanism |
| Thiede et al. 2016, EMUS II–V and VII | Overlapping biased windows can reconstruct target expectations and estimate uncertainty under stated assumptions | Validate the actual weighted bank and overlap, including between-replication uncertainty, before it supplies training |
| Hoffman et al. 2019, 4.1.1 and 5; existing author architecture audit | Canonical IAF can help HMC; insufficient geometry can slow tail mixing | Retain the author IAF and identity latent mass; require downstream evidence rather than loss-based promotion |

Exact source anchors are at the end. Gabrié's fixed-kernel global MH acceptance
is `min(1, pi(y) q(x)/(pi(x) q(y)))`. Combining two pi-invariant kernels
preserves pi; their deterministic composition need not itself be reversible.
This invariance statement does not prove finite-time convergence or justify
using adapting-chain draws as stationary teacher samples. Freeze and assess.

The archived Noé code includes an energy `linlogcut`. Blindly copying it would
change the mathematical objective in the affected region. This plan uses the
exact target energy and `explore=1` semantics. It borrows the inspected combined
objective, with this implementation difference stated explicitly.

## Phase A: establish known-correct downstream controls

For marginal CDF F_1 and standard-normal CDF Phi, use the reference map

    T_1(z_1) = F_1^{-1}(Phi(z_1)),
    T_2(z)   = z_2                         (unwarped),
    T_2(z)   = z_2 + 0.1*(T_1(z_1)^2-26) (warped).

Its pullback is exactly standard normal in real arithmetic. Validate the
numerical CDF inversion, log Jacobian, transformed value, and total score in
FP64, including the valley and tails. Use implicit inverse derivatives rather
than differentiating bisection decisions. The pointwise cancellation becomes
ill-conditioned in some regions; tolerances must follow floating-point error
and resolution checks, not a copied universal absolute threshold.

Keep three control layers distinct:

1. Exact iid target draws through the existing posterior diagnostics. Reuse
   the recorded v4 iid calibration where scope matches; extend only missing
   estimands or new precision rules.
2. Gaussian latent HMC through the public tuner and shared sequential
   controller, with the analytic map used to decode physical observables.
   This checks search, retained diagnostics and budget without learned geometry.
3. Numerical evaluation of `log pi(T(z))+logdet J_T(z)` and its score, tied to
   the analytic Gaussian result. This checks the actual transform computation.

The current artifact loader does not advertise arbitrary quantile maps. Use a
repository-owned analytic-reference binding/fixture if needed; do not stamp a
quantile transform as a trained canonical IAF or inject a hand-verified kernel.
A direct Gaussian calculation alone must not be described as a test of the
learned-transform implementation.

If an exact control fails, repair the precise inversion, tuning, assessment,
or budget problem before interpreting dependent learned-map rejection. One
stochastic control failure is evidence to investigate, not automatic proof of
a deterministic bug. Preserve the failure and assess the expected failure rate.

## Phase B: qualify the teacher actually used

Create one predeclared pooled bank from independent complete estimator
replications. For normalized within-replication weights w_ri, equal-replication
pooling has weights w_ri/R. Do not concatenate differently normalized weights
or let a run with more saved rows acquire more probability mass. Preserve
replication labels to assess uncertainty and sensitivity to each replication.
Pooling self-normalized estimators is not a finite-sample unbiasedness theorem.

Assess this bank's mode masses, broad-valley mass, narrow-event mass, moments,
and within-mode shape, alongside between-replication uncertainty and method-
specific overlap or ancestry diagnostics. An aggregate method screen is not a
substitute. Improve the failed conditional/window/mass estimate identified by
these checks; do not add particles indiscriminately.

Use three disjoint data roles: training, development/calibration, and final
confirmation. A fixed stratified allocation retains rare rows with the original
probability correction. Never resample the entire bank by posterior weight
before training and thereby erase the deliberately collected rare rows.

Add a separately labeled oracle-teacher experiment on these known targets:
exact conditional reference samples with analytic stratum weights, versus the
estimated teacher, at matched architecture, initialization, objective and
optimization protocol. The completed master prohibited oracle training in its
method arms; this new diagnostic control is an explicit scope difference.
Oracle-trained success would isolate teacher/optimizer questions and would not
establish that our practical teacher works.

For these 2D targets, compute development mode/event masses and losses using
target-specific integration or stratified references with numerical resolution
and tail-truncation studies. Fixed finite grids cannot certify every off-grid
defect. Final draws, seeds and data remain untouched by this selection.

## Phase C: retain forward coverage during refinement

For the normalized proposal q_theta and unnormalized target gamma, define

    L_F(theta) = -E_pi[log q_theta(X)],
    L_R(theta) = E_phi[-log gamma(T_theta(Z)) - log|det J_T_theta(Z)|],
    L_lambda(theta) = L_R(theta) + lambda L_F(theta).

Additive constants do not affect gradients. At lambda=1 and with an exact
teacher, this is the sum of the two KL directions up to a constant. Both terms
are means in nats, with probability weights normalized correctly. This is the
equal-weight source baseline from Noé's combined ML/KL experiment, not a
universal calibrated coefficient. Approximate teachers replace the forward
expectation by their empirical measure, which must be stated in results.

The mathematical benefit is limited but precise. For any partition B_k with
p_k=pi(B_k) and v_k=q(B_k),

    KL(pi||q) = KL(p||v) + sum_k p_k KL(pi(.|B_k)||q(.|B_k)).

Consequently, losing a mode with p_k>0 incurs an unbounded forward cost as
v_k goes to zero. RKL alone lacks that protection. With a full-support flow,
exactly zero mass is a limit, but severe underweighting remains penalized.
This does not guarantee successful nonconvex optimization, rare-region relative
accuracy, or derivative accuracy. In particular the p_k coefficient can make
an extreme rare region weakly represented in the objective.

From a common forward checkpoint, compare FKL-only, RKL-only, and FKL+RKL
with lambda=1. Keep the pure-RKL branch as a diagnostic comparator. Reset Adam
identically in these branches when isolating objective effects, and retain a
separate FKL continuation with its original Adam state. Otherwise optimizer
reset and objective change are confounded. Record the additional inverse/target
cost of the combined objective; equal update counts are not equal compute.

Implement the combined loss as an explicitly combined gradient update through
the existing shared numerical authority. Alternating separate Adam updates is
another schedule, not exactly optimization of the simultaneous loss. Keep
gradient-estimator choice fixed initially. Validate the combined parameter
gradient against the sum of independently checked term gradients and a bounded
directional-difference reference.

Preserve forward checkpoints and evaluate both KL directions, mode/valley
coverage, and geometry after refinement. A supported coverage loss rejects
that refinement and retains the parent. Failure to establish a difference is
not evidence of equivalence. The latest checkpoint receives no automatic
priority. A forward-only map may proceed if it satisfies downstream requirements.

## Phase D: measure optimization and capacity instead of declaring convergence

First reuse the tested continuation mechanisms in `neutra_warm_start_campaign`
and `neutra_warm_start_closure`; keep corrected stratification in the common
trainer. New frozen artifacts alone are insufficient for continuation: save
map parameters, Adam moments/iteration, RNG state or next unique stateless
stream, objective, teacher identity, and lifetime work. A frozen historical
map with no Adam state can start an explicitly named reset intervention; it
cannot be described as an exact resumed optimizer.

At distinct checkpoints record training and heldout loss for each objective
term, per-stratum contributions, gradient noise, per-layer parameter updates,
clipping, mode masses, and the diagnostic vector. Use paired heldout changes
for ordinary independent reference rows and replication/block uncertainty for
correlated or weighted banks. Do not treat resampled copies as independent.

Continuing supported progress triggers continuation under the measured budget.
A noisy loss difference is inconclusive. Call a plateau only after the
uncertainty interval excludes an improvement larger than a predeclared
scientifically relevant delta, with adequate evaluation precision. The delta
must be linked to the target's fit/coverage question or an explicit engineering
stopping tolerance. It cannot be inferred merely from failure to reject zero.
This is an operational repeated-look stopping rule, not global-optimum proof.
At the resource ceiling, an improving run is labeled budget-limited.

For a demonstrated plateau, localize one cause at a time:

- FKL train/development discrepancy or inaccurate individual teacher: repair
  data and weighting before expanding the network.
- Noisy/conflicting updates: calibrate batch size and LR using actual gradients
  and Adam updates. Include a no-change control.
- Nonlinear fit failure with a qualified/oracle teacher: compare canonical
  widths 16 and 32, then a single width-doubling extension if needed. Width64
  would be a new capacity hypothesis, not an author default. Keep three stages,
  masks, ELU, reversal and the free-bias conditional scale convention fixed.
- A stage-depth change, DSF/NAF substitution or new base distribution is a
  separate architecture proposal, not a hidden repair to this comparison.

The .02 author initializer and .2 recent-repair initializer are separate
hypotheses. Use the recorded parent for continuation. If initialization is the
remaining suspect, compare these explicitly rather than silently transferring
.2 to all future fits. RKL gradient estimators are also separate hypotheses;
an estimator with lower variance changes optimization, not the KL objective.

Validation cadence is priced: if one update costs C, validation costs V, and
the chosen validation time share is rho, use blocks of at least
ceil(V*(1-rho)/(rho*C)) updates. The share rho is an explicit engineering
preference, not a scientific threshold. Full directed diagnostics belong at
candidate/checkpoint boundaries rather than every update.

## Phase E: test the geometry that losses miss

Retain the standard 1,000 Gaussian-base probes, supplemented by independent
posterior-reference points mapped through T^{-1} and directed physical valley,
tail, and mode-boundary points. Their distributions and roles must remain
separate. Replace the single valley line by several conditional slices or
stratified two-dimensional probes; include the unwarped second coordinate in
the warped target. A one-dimensional line can miss off-axis defects.

Decompose the residual

    r_theta(z) = J_T(z)^T score_pi(T(z))
                 + grad_z log|det J_T(z)| + z.

Check the cancellation, inverse stability, physical coverage and fixed-step
Hamiltonian energy behavior. Large J alone is not an error: the exact quantile
map already has derivative about 244,565 at the midpoint. Do not regularize
away that legitimate transport feature by imposing an unsupported global
Jacobian cap. Directed energy probes diagnose integrator resolution; they do
not qualify a posterior kernel or replace fresh public tuning.

If qualified teachers and converged density fits still give poor geometry,
consider an optional, explicitly new score-residual objective. For fixed
physical training distributions nu_k and declared allocation coefficients a_k,
one candidate is

    G(theta) = sum_k a_k E_{X~nu_k}[||r_theta(T_theta^{-1}(X))||^2].

This is a local derivation requiring its own experiment, not a claim about the
original NeuTra/Boltzmann-generator recipe. With fixed X, the target score
score_pi(X) can be evaluated once; the total theta derivative still passes
through the inverse map and map derivatives. Stopping that inverse derivative
would optimize a different quantity. Mixed/higher map derivatives can be costly
and require their own parity, compilation and memory checks. A Gaussian-only
version would retain the missed-mode blind spot. No coefficient or default is
chosen before the cheaper controls identify an actual need.

Noé's optional reaction-coordinate entropy loss is another distinct mechanism:
it deliberately reallocates proposal probability toward transition regions.
That can help exploration with exact correction, but is not exact posterior
Gaussianization. Do not treat it as an interchangeable whitening penalty.

## Phase F: qualify posterior estimation at an attainable precision

Use `tune_fixed_transport_hmc_kernel`, fixed identity mass in latent space,
fresh verification for each map and kernel, and the shared sequential
controller. Consult `HMC_TUNING_INTERFACE_CAPABILITIES`; a chain runner or
hand-authored receipt is not an artifact-authority tuner. No mass adaptation
is introduced. Preserve eligible earlier checkpoints without using the final
reference to select among them.

The existing v4 gates remain the baseline. They require physical moments,
mode/CDF behavior, and observation of both broad-valley outcomes in each
retained chain, not the 3.09e-7 narrow event. A standardized MCSE criterion
does not imply a small relative error for an event. For event probability p,

    relative_MCSE approximately sqrt((1-p)/(p*N_eff,event)).

At p=0.001349898, 20% relative MCSE requires about 18,500 independent-equivalent
draws; 10% requires about 74,000. The existing four-chain, 10,000-per-chain cap
has only 40,000 raw draws. A 10% goal is therefore not a routine feasible
assumption under that cap, even before autocorrelation. These numbers are
derived feasibility estimates, not finite-sample guarantees or hard bounds
when negative correlation is possible.

Proposed additional broad-event precision target: 20% relative MCSE as a
limited feasibility objective, matching the earlier deliberate-estimator
screen's stated purpose. This is an explicit stronger evidence contract than
v4, not an assertion that v4 already provides it. Assess small-count and MCSE
calibration against the exact controls before launch. Preserve ESS, event counts,
between-run variability, reference error and the uncertainty limitations.
Leave the much narrower event to deliberate estimators and weighted geometry
checks; ordinary HMC is not budgeted to estimate it precisely.

Assess model-drawn starts and an independent mode-spanning start bank as
separate robustness checks. Known physical representatives locate modes, not
equilibrium volume. Generate starts with within-mode variation; inverse-map
them, check roundtrips, and retune for that start scope. Agreement of all
chains initialized inside a collapsed proposal does not establish coverage.

Selection and final confirmation remain separate. Final confirmation uses
fresh streams and is never reopened repeatedly to select another candidate.
An observed failure becomes preserved holdout evidence; a later repair needs
a new prospective confirmation. Record run-level seed variability rather than
claiming a method ranking from a few favorable trials.

## Conditional alternative if a single global map remains inadequate

Keep Gabrié-style mixtures of local canonical IAFs as a source of corrected
training samples and an independent sampling check. Their ability to jump
between modes can be useful even when one smooth global map is hard to train.
The mixture density is evaluated as a log-sum-exp; a random mixture component
is not a globally invertible single NeuTra map.

If the controlled canonical-map study still cannot support plain NeuTra HMC,
the already requested tempered NeuTra ensemble is a separate next inference
route. It must preserve exact target/swap corrections and receive its own
calibration and evidence; local-map success does not certify that ensemble.
Do not turn failure of one global map into a requirement to replace the
canonical IAF or abandon the research direction.

## Numerical assumptions and resource plan

| Choice | Provenance/status | How to determine adequacy |
|---|---|---|
| Same two targets and events | Existing analytical benchmarks | Exact probabilities and inverse-warp identity |
| Three-stage canonical IAF | Owner/source architecture | Core conformance and oracle-teacher fit; no capacity guarantee |
| Width16/32, optional64 | Existing candidates; optional geometric extension | Matched teacher/optimization and downstream checks |
| Lambda1 combined loss | Noé equal-weight ML/KL source baseline | Separate term gradients, coverage and downstream ablation |
| Batch256 and LR .0003/.001 | Previous campaign hypotheses | Measured noise, update size and continuation; not universal defaults |
| Initializer .02 versus .2 | Author versus prior local intervention | Isolate only when needed; never silently conflate |
| Update doubling | Work-allocation convenience | Progress intervals and measured cost; counts are not convergence |
| 1,000 random probes | Owner standard | Add disjoint reference/directed checks; no global guarantee |
| 20% relative broad-event MCSE | Proposed feasibility criterion | Required event ESS and known-correct control calibration |
| Existing R-hat/ESS/count caps | Shared operational policy | Exact-control pass behavior and downstream requirements; no silent relaxation |
| Seeds | Two existing development seeds can localize mechanisms | Fresh independent confirmation; number chosen for declared uncertainty goal and priced budget, not a reliability claim from two fits |
| Precision | FP64 analytic-control reference; FP32/TF32 canonical training direction | Same-computation forward/inverse/score and trained-map checks; FP64 stays labeled reference if TF32 cannot preserve valley cancellation |

Price the actual mixed update, inversion, checkpoint diagnostics, and HMC work
before committing a new total ceiling. Reconcile the current shared CPU/GPU
ledger; the completed campaign's unspent balance is not a runtime forecast.
Charge summed worker time including failures. An implementation-ready campaign
must freeze its total and per-phase attempt/compute ceilings after pricing;
this design does not invent an hour estimate for work not yet measured.

Use the available GPUs for independent arm/seed workers, with explicit device
assignments and memory growth, subject to current occupancy. Avoid concurrent
updates to one optimizer. External teacher/sample generation uses bounded CPU
workers. The canonical implementation stays TensorFlow/TFP, batch-native and
XLA with stable signatures. CPU-only and FP64 diagnostic exceptions are
labeled. No broad environment mutation or new external review service is needed.

## Implementation handoff and outcome-driven master

| Component | Required bounded change |
|---|---|
| Shared transport/training authority | Optional combined FKL/RKL update with exact objective and full restorable state |
| `neutra_rare_region_phases` | Preserve stratification; replace duplicate fixed-count lifecycle with shared continuation; qualify actual pooled teachers |
| `neutra_warm_start_qualification` and diagnostic helpers | Analytic-reference controls, physical start robustness, declared event precision; retain public tuning authority |
| Master/controller | Stop reasons distinguish invalidity, teacher failure, progress, plateau, coverage loss, geometry failure, tuning inconclusiveness and precision cap |
| Focused tests | Real saved-state next-update equality, stratum measure preservation, combined-gradient parity, collapsed-map false-pass regression, and the actual downstream consumer |

After each phase record the result, failure classification, remaining budget
and next justified phase. Infrastructure repairs get fresh output directories
and focused regressions. A candidate failure proceeds to its planned repair;
a shared invalidity stops dependent work. Reuse successful evidence only where
target, data, numerical route and criterion still match. Do not make a new
controller with another inconsistent definition of training completion.

Suggested new artifact root:
`docs/plans/artifacts/neutra-controlled-repair-2026-10-02/`, with unique numbered
attempt directories. No root or job has been created by this design. Preserve
the completed campaigns as historical evidence of their actual scopes.

## Skeptical plan review

This design passes as a proposal for the following reasons: an exact control
can falsify the downstream explanation; an oracle teacher separates data from
fitting; objective arms share a parent and account for Adam reset; earlier
long-run plateaus prevent blaming everything on a short budget; rare-event
precision follows event probability; teacher uncertainty remains explicit;
and no loss or finite residual is promoted into posterior evidence.

Specific draft flaws resolved before finalizing:

- More data/updates alone cannot guarantee geometry: added objective and
  capacity controls with a conditional geometric branch.
- Blind author-code copying can clip target energy: retained the exact local
  target and named the source difference.
- An exact Gaussian run alone bypasses transform numerics: split control
  layers and require an explicit analytic-reference binding where needed.
- A fresh Adam state could explain an apparent objective benefit: added the
  reset-matched arms and preserved-state continuation.
- A zero improvement significance test is not a plateau test: require an
  uncertainty-aware practical-improvement criterion, or label inconclusive.
- A mixture proposal is not a single invertible map: preserve that distinction.
- Strict probability accuracy could be impossible under the HMC cap: derive
  event sample requirements before selecting the precision contract.

Remaining work before execution is routine implementation, targeted regression,
actual cost pricing, a reconciled finite allocation, and freezing the final
numeric protocol. This is skeptical self-review, not independent experimental
validation or proof that the proposed remedy succeeds. No requested execution
has been left running or blocked by an approval ceremony.

## Source anchors inspected for this design

- Noé et al., 2019, https://arxiv.org/abs/1812.01729. Local
  `.localresources/neutra-mode-initialization-literature-20260929/`
  `noe-boltzmann-generators.txt`: main text 302–395 and 638–694; Methods
  1326–1360, 1407–1499 and 1510–1599; supplementary Fig. S2, 2320–2335.
  Author archive `noe-archive-invertible.py`: 462–474, 581–607, 640–652;
  `noe-doublewell-training-cells.txt`: notebook cells35 and39 retain
  `weight_ML=1`, `weight_KL=1` after ML pretraining. These are architecture-
  specific experiments, not calibrated parameter values for our target.
- Gabrié et al., 2022, https://arxiv.org/abs/2105.12603. Local
  `.localresources/fab-coverage-followup-20260928/gabrie-adaptive-flows.layout.txt`:
  216–244, 275–395, 405–431, 810–834, 1701–1728 and 2032–2045. Author code:
  `.localresources/flonaco-author-20260929/upstream/flonaco/sampling.py`:
  151–167 and207–249. Fixed-kernel invariance is the local composition claim;
  finite adaptive convergence is not inferred from those code functions.
- Thiede et al., 2016, https://arxiv.org/abs/1603.04505. Inspected paper,
  equations, CLT assumptions and source functions are recorded in
  `.localresources/neutra-rare-events-20261002/reading-notes.md`. Its older
  pre-campaign status prose is historical; the technical source anchors apply.
- Hoffman et al., 2019, https://arxiv.org/abs/1903.03704. Local paper and author
  code correspondence are recorded in `../reference/neutra-implementation.md`.
  Section4.1.1 supplies the source architecture/training experiment; Section5
  states the concern about inadequate maps slowing tail mixing.
