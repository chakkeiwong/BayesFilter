# Remedy proposal: distinguish reproduction, fitting and sampling failures

Status: revised on October 5 after selective incorporation of Claude's review;
execution subsequently authorized and underway. The active execution record is
`bayesfilter-neutra-source-fit-execution-2026-10-05.md`, with findings in
`bayesfilter-neutra-source-fit-results-2026-10-05.md`. The owner added 24 GPU and
24 CPU hours for continued study. Numerical balances below describe the proposal
at revision time; the active ledger preserves these costs and the new allocation.
The owner requested the reasonable changes, not wholesale adoption of the review.
The revision does not select a new flow default. The later resource grant is
recorded in the active execution note.

The [review assessment](../reviews/bayesfilter-neutra-fit-remedy-review-assessment-2026-10-04.md)
records the accepted advice and corrections to the review's mathematics and
decision rules. The [pre-revision proposal](artifacts/neutra-remedy-review-incorporation-2026-10-05/r1/baseline/bayesfilter-neutra-source-and-fit-remedy-plan-2026-10-04.md)
preserves the version Claude inspected. Full execution still requires measured
pricing and prospectively declared scientific accuracy bands.

## Research intent and existing evidence

The question is whether a reproducible, target-specific training procedure can
produce a useful canonical NeuTra IAF on simple multimodal targets and retain
that ability on untouched randomized two- and three-center targets. The desired
endpoint is reliable posterior computation with a frozen map. Good density fit,
Gaussian score diagnostics and corrected-sampler performance are separately
assessed quantities, not interchangeable definitions of success.

The October 4 corrected r2/r3 campaigns failed at an exact-teacher student
prerequisite. Native FAB/Gabrié/AIS/SMC/AFT/CRAFT trials did not then execute.
The r3 forward maps already have excess off-component mass. The unconditional
RKL finish reduced one three-component responsibility mass from 0.21572 to
0.02240 against target 0.249591, while reducing RKL. Local inverse, density and
derivative checks passed; clipping and sampled strong scale saturation do not
explain these particular fits. Capacity, optimization, initialization and
integration remain unresolved. Width and update count were confounded.

The October 2 campaign already tried forward, reverse, joint and joint
continuation on eight target/teacher/seed cases, without fresh posterior
confirmation. All forty-eight fit jobs reached their allocations without an
established plateau. Do not repeat those configurations while calling the
combined objective new or guaranteed to work. Do not transfer the October 4
invalid-r1 uncertainty screen into this proposal.

Primary evidence: `bayesfilter-neutra-scientific-root-causes-2026-10-04.md`,
`bayesfilter-neutra-upstream-training-evidence-audit-2026-10-04.md`,
`bayesfilter-neutra-literature-remedies-2026-10-04.md`, and
`bayesfilter-neutra-controlled-repair-results-2026-10-02.md`.

The comparator ladder must remain visible. Identity and training-data-fitted
affine Gaussian maps are the naive density controls; the exact target/transport
is an oracle diagnostic, never a generic learned competitor. The documented
author learner is the source baseline. The calibrated canonical IAF with forward
training is the plain local method, and its matched joint/representation branches
are the enhanced candidates. For actual sampling, include a calibrated classical
AIS/SMC control under its own native evidence contract. Parameter choices for
the affine control use calibration/training data, not heldout or final samples.
The ladder answers different questions at the density and sampling levels; a
stronger oracle does not replace the tuned practical comparator.

| Intent item | Definition |
|---|---|
| Mechanisms | Faithful complete training, calibrated canonical IAF, refreshed training data, preserved forward pressure, and explicitly optional capacity changes |
| Expected failure | Incomplete reproduction; under-optimized or insufficiently expressive map; coverage regression during objective switch; corrected-sampler failure despite acceptable density loss |
| Primary promotion criterion | Fresh fixed-map posterior confirmation against independent exact references on every target in the declared scope, with calibrated uncertainty and required precision |
| Promotion vetoes | Invalid density/derivatives, nonfinite computation, lost coverage, failed physical-space posterior checks, missing independent confirmation or missing scope-specific retuning |
| Continuation vetoes | Invalid common reference/harness, corrupted or mismatched artifacts, prohibited execution path, unsupported required consumer, exhausted allocation, or an unapproved material direction change |
| Repair triggers | Candidate fit failure, stalled optimization, objective-switch regression, missing sampled regions, inconclusive bounded HMC search; these do not reject all later phases |
| Explanatory diagnostics | Loss trajectories, clipping/update/gradient noise, conditional scales, Jacobian stretching, 1,000-point residuals, runtime and acceptance |
| Nonconclusions | No default promotion, exhaustive mode discovery, algorithm ranking, or q20/state-space transfer from these exploratory mixtures |

## Proposed sequence and decisions

Phase identifiers are retained for correspondence with the review; they do not
imply that all of P1 precedes P2/P3. The immediate order is P0 historical audit
and pricing, P2 controller repair, then one small matched P3 fitting comparison.
A bounded P1 original-source baseline is a separate track when affordable;
complete adaptive-port reproduction is not a prerequisite for the local fitting
study. Findings select the next P3 continuation or P4 representation/geometry
test, followed by P5 teacher transfer and P6 untouched confirmation.

### P0. Audit the earlier failures and make the comparisons executable

Preserve the terminal campaign and all failed checkpoints. Snapshot only the
source dependencies actually required by the new experiment. Verify the learner
sees density, score, dimension and target signature; exact mixture components,
CDFs and responsibilities belong to the reference evaluator and declared oracle
controls, not to a purported generic algorithm. Audit the actual target generator
and record its induced distribution: do not assume independently uniform pairwise
distances for three centers unless the construction really provides that law.

Audit all eight October 2 target/teacher/seed cases before recommending another
joint continuation. Link every fitted endpoint to its ancestor, objective,
Adam state, lifetime work, heldout trajectory, distribution checks, kernel
qualification and retained-sampling outcome. Use the raw results to distinguish:

- poor density or mode-weight agreement;
- invalid numerical computation or failed chain-health checks;
- incomplete kernel search or warm-up at its cap;
- insufficient retained-event information or precision; and
- resource exhaustion with the substantive question unresolved.

These categories can coexist for one map. The saved summary reports 8 numerical/
health outcomes, 76 resource-limited outcomes, 36 retained-cap outcomes and 106
warm-up-cap outcomes across repeated checkpoints/kernels; these are not counts
of independent fitted models. Reconcile the counts with the actual stop reasons
instead of treating "no posterior confirmation" as proof of undertraining.
The previously inspected mixture/oracle/seed-11 joint sequence reduced both
reported objectives but still failed its coverage screen. It motivates this
full post-mortem; it does not establish that joint training succeeds or cannot
help. Record the missing evidence where a saved artifact cannot resolve a cause.

Use the existing exact Gaussian/analytic-transport controls, repeating only checks
affected by a change. Verify independent reference versus candidate uncertainty,
weighted banks, resuming optimizer/RNG state, loss normalization and checkpoint
selection. A sample count is not a count of independent training replications.
Keep the ordinary 1,000-point post-training diagnostic, plus target-space and
directed geometric checks. Directed maxima are explanatory unless a numerical
validity failure occurs; no arbitrary residual threshold becomes an HMC theorem.

Price complete comparisons from saved worker and assessment timings, including
setup, all declared fit seeds and confirmation where required. Separate measured
costs from extrapolations to new architectures, batches and sampling schemes.

Output: a per-case failure matrix, a complete comparison cost forecast and a
runnable phase manifest. The matrix must identify the next discriminating repair
and what evidence would overturn its diagnosis; it must not infer a cause from
the terminal status alone.
If a common check is invalid, repair it and run a focused regression before
using dependent numerical results. A failed map with valid controls advances
to its discriminating repair, rather than stopping the entire research program.

### P1. Retain a bounded original-author baseline

The user asked whether the original methods exhibit these failures, so this
track remains necessary for attribution even without a suspected primitive bug.
Its first priced unit is one complete original training recipe on the author's
example; a truncated mechanics run cannot answer the same question. A single
run provides a descriptive reference, not reproduced success probabilities or
a stochastic ranking. If unaffordable, record it as deferred and leave the
original-versus-port question unanswered while independent P3 work proceeds.

Start with Gabrié's Appendix G.1 Gaussian example because it directly addresses
separated modes. Its original learner is PyTorch, not JAX. Preserve the author's
RealNVP, initialization, sampler order,
sampling cadence, optimizer and documented experiment settings. Distinguish the
paper settings from CLI defaults and distinguish the paper's ULA example from a
separate corrected-MALA adaptation. Record every compatibility patch. Compare
raw-flow shape, corrected sampling, mode weights and learning histories; reported
80--85% proposal acceptance is a descriptive author result, not a reproduction
acceptance threshold or convergence test.

Run the original full learner before interpreting a substituted IAF as a failed
reproduction. A full TensorFlow adaptive-controller port is a subsequent priced
attribution step when needed to resolve the observed difference; it is not an
unconditional first task. Any original-versus-port claim must compare the same
architecture and target. Use common deterministic arrays for transition/loss/optimizer parity;
compare full-run behavior with replication, since backend floating-point
differences can change stochastic trajectories. Check both implementations on
the same exact-iid training control to separate sampling from fitting. Explicitly
distinguish this new control from reproducing the author's adaptive experiment.

Noé's double-well ML-then-joint example is the second source baseline if the
objective switch remains at issue. Its energy transformation, data distribution,
architecture and optimizer state must be preserved for reproduction. An
unmodified analytic energy or altered weights are separately labeled adaptations.
The original NeuTra code supplies a further architecture/gradient authority;
do not demand that its original RKL recipe solve a multimodal benchmark that
the author did not claim to solve.

Foreign frameworks are confined to isolated original-code reference processes.
No broad environment mutation or replacement of the TF/TFP runtime is implicit.
Price feasibility first; if a source environment cannot be reproduced within
scope, record that gap and continue independent local fitting controls without
claiming full author reproduction.

Decision: original failure first triggers environment/setting/source audit;
original success with port failure triggers a port repair; matched success with
later IAF failure points to the substitution or its calibration. Failure in both
implementations on a new target is a failure of that tested recipe, not a theorem.

### P2. Repair the training controller and preserve causal comparisons

The canonical numerical source remains `neutra_transport_core.py`, with the
`hoffman_author_iaf` profile and its defining masks, scale, initialization and
reversal. Historical constructors must not silently select legacy masks.
Remove the unconditional forward-to-RKL handover from the new procedure.
Keep forward checkpoints as first-class candidates. Make forward continuation,
RKL-only diagnostic continuation and joint continuation explicit branches from
the same frozen ancestor. Match Adam reset/preservation across comparable arms;
one summed-gradient Adam step is not two sequential optimizer steps.

Qualification, repair choice and stopping must use separate heldout information.
Keep each attempted checkpoint and account for repeated selection. A loss still
improving at its cap is classified as improving at the cap, not converged;
further affordable optimization is a testable hypothesis, not a promised repair.
An imprecise difference is not a plateau.
A practical plateau requires an interval contained in a declared negligible-
improvement band, not merely a nonsignificant test. Freeze that band from the
intended density resolution and feasible validation precision before looking
at candidate continuation outcomes.

Output: focused regressions for controller branching, real checkpoint comparison,
matched optimizer state, finite weighted gradients and deterministic resumption.
Exercise the actual master-to-worker-to-shared-core call chain and inspect the
emitted checkpoints; tests of an unused helper do not establish controller repair.
These changes do not by themselves establish a better trained map.

### P3. Start with a small matched exact-teacher fitting study

Begin with the existing two development specifications, without consuming final
targets. The first funded unit is one paired mechanism comparison on both
development targets across the declared fit seeds, including endpoint assessment.
Do not launch a Cartesian product of every setting below. Keep architecture,
update budget and initialization matched while changing one factor at a time;
record sample/target-evaluation counts and process time as well as updates.
The sequential choices are:

1. Fixed-bank versus fresh-iid forward training, using the same map and optimizer
   settings. Keep heldout samples independent in both arms. Refreshing data is
   not assumed to help an underfit model.
2. At fixed width and update allocation, assess learning rate and batch size
   using gradient noise, actual parameter changes and heldout progress. Preserve
   the author initializer as the baseline; test any changed initializer explicitly.
3. At matched training effort, compare widths through the canonical constructor.
   Record both update counts and actual target evaluations/process time; different
   inverse costs preclude treating equal updates as equal cost.
4. Branch from the same forward endpoint into further forward fitting and joint
   fitting. Retain RKL-only as a diagnostic control. Start the joint comparison
   with the source-based lambda=1, then investigate relative loss/gradient scale
   if it fails. No universal lambda or successful coverage guarantee is assumed.

The P0 post-mortem determines whether an already answered contrast can be skipped
and whether a different listed contrast is the next discriminator; state that
choice before execution. Persistent failure may move directly to the bounded
P4 representation comparison without exhausting every LR/batch/width combination.
Joint continuation need not wait for a perfect forward fit, but it must preserve
the ancestor and face the same distribution checks as forward continuation.
For J=RKL+lambda*FKL, positive lambda penalizes complete loss of a target region
in the population objective. It does not guarantee empirical region coverage or
an adequate fit after finite training.

Use three independent fit seeds as an exploratory replication minimum; retain
all outcomes and do not infer superiority from three runs. This count is a
convenience choice to expose seed sensitivity, not a power calculation. Freeze
the seed list before execution. Use paired initializations/noise when meaningful,
without pretending differently sized architectures share identical parameters.

The final procedure must contain an objective/scaling audit, architecture and
optimizer calibration, a priced continuation ladder, explicit seed policy,
heldout selection and downstream validation. A convenient one-off configuration
does not become a target-specific training protocol by passing a smoke test.

### P4. Use distinct triggers for representation and geometry tests

Persistent density or tail failure after controlled optimizer checks is itself
a reason to test representation; good density fit is not an entry requirement
for that comparison. Compare the author's RealNVP family and the already implemented,
separately named conditional DSF/NAF research option. Test density, inverse,
Jacobian, gradients, full support and consumer compatibility before training.
Compare an extended-depth IAF only as an explicitly labeled departure from the
canonical three-stage profile. A spline implementation is a later option, not
an immediate prerequisite. Explain why the existing alternatives are insufficient
before adding another implementation.

The rationale for these alternatives is demonstrated deformation difficulty,
not a false disconnected-support impossibility claim for positive Gaussian
mixtures. Each alternative remains a comparator; replacing the canonical default
requires owner direction. A mixture of maps is a proposal/sampler branch until a
separate HMC coordinate treatment is derived and supported.

If density fit is adequate but directed geometry remains harmful in actual HMC,
evaluate the existing derived geometric objective as a local extension. Check its
total inverse-map parameter derivative, activation/knot regularity, derivative
cost and numerical equivalence. Do not omit terms to avoid higher derivatives.
The existing objective is

\[
G(\theta)=\mathbb E_{X\sim\nu}
\left\|J_{T_\theta}(T_\theta^{-1}X)^\mathsf T
 [s_p(X)-s_{q_\theta}(X)]\right\|^2,
\]

with physical X and reference measure nu fixed independently of theta. The
target score can therefore be precomputed; a target Hessian is not inherently
required. Mixed/higher map derivatives and the inverse-map dependence remain
necessary. Replacing this with generated moving points changes the objective
and derivative requirements; it is not an implementation shortcut. The checked
derivation is in `docs/chapters/ch26f_neutra_controlled_training.tex`.

Keep the coefficients and sampling measure explicit. Stratified, correctly
weighted training reduces estimator variance but cannot increase a rare region's
expected weight in the same KL objective. A deliberately reweighted geometric
objective is a new objective and needs a separate comparison.

### P5. Test native methods and the common learner separately

Retain every method row. For each, record native source fidelity, native sampling
accuracy and downstream IAF quality separately; do not count an unexecuted cell
as a scientific rejection. This phase can collect bounded native controls in
parallel with learner repair when affordable, but cannot attribute a known
exact-teacher student failure to those native methods.

| Method | Required discriminating work |
|---|---|
| FAB | Resolve alpha-two integrability for the actual target/proposal and AIS accuracy first; preserve original finite parity evidence and TF32 discrepancy. Compare a full author training run, not only four-step parity. |
| Gabrié | Reproduce the adaptive controller and architecture; keep the current fixed-map MH/MALA arm explicitly separate. Test known-mode starts and a discovery challenge as different questions. |
| AIS | Validate annealing overlap, mutation and replicated weighted reference agreement without resampling. |
| SMC | Assess the same bridge with resampling plus ancestry and mutation diagnostics; replicated samples must not be treated as independent particles. |
| AFT | Reproduce training/validation/test populations and stage learning; assess corrected output separately from each stage map. |
| CRAFT | Reproduce complete repeated passes, fresh populations and update ordering; label the finite-particle objective correctly. |

Then feed separately assessed teacher banks to the frozen calibrated IAF
procedure. A bad teacher triggers sampler repair; good teachers with bad students
trigger checks of weighting, finite-bank information and training as well as
sampler uncertainty. A successful exact-teacher fit does not prove every later
student failure is a teacher defect. Good raw maps with failed corrected sampling trigger geometry,
tuning or finite-information diagnosis. Keep the appropriate control at each step.

### P6. Freeze, generalize and confirm posterior computation

Freeze the complete selection/training procedure before final targets. The
corrected campaign reserved two-center seeds 1103/1104 and three-center seeds
2103/2104; verify they remain untouched before using them. Preserve the fixed
warped and unwarped targets as known development/reference cases. If a final
target informs repair, it becomes development evidence and a new prospectively
recorded final target is needed. The evaluator may know exact components; the
generic learner may not.

For supported frozen maps use the public `tune_fixed_transport_hmc_kernel`
authority in `HMC_TUNING_INTERFACE_CAPABILITIES`, with the exact transformed
target and fixed identity mass in latent coordinates. Do not restore mass
adaptation or introduce a private tuner. Any changed map creates a new tuning
scope. Inverse-map the same physical start bank, verify roundtrips and run fresh
kernel verification before the shared sequential retained sampler.

Assess physical-space mode mass, means, covariance and predeclared event
probabilities against exact references, together with the existing sequential
convergence and precision requirements. Retain all warmup but exclude it from
posterior estimates. An impossible ordinary-HMC rare-event count requirement
must not masquerade as a training failure: derive necessary event information
from MCSE approximately sqrt[p(1-p)/ESS_event], and use a separately justified
rare-event estimator when needed. Changing a scientific accuracy requirement
after seeing failures is not an allowed repair.

Only a procedure passing this scope may motivate a new q20-specific plan.
The 20-dimensional/state-space transfer needs its own objective, cost, capacity,
tuning and validation study; the simple targets cannot certify it.

## Numerical choices, statistical decisions and compute

The current remaining allocation, read from campaign-r3 `state.json`, is
3,895.501185 GPU-process seconds and 8,660.105809 CPU-core seconds. These are
measured ledger values, not a new allocation. Three concurrent GPU workers charge
their summed time. Prior failures and reference diagnostics remain charged.

Saved r3 `student-*/forward-learning-history.json` files provide these block
timings on the two development targets:

| Target | Width / forward updates | Training blocks | Intermediate assessments |
|---|---|---:|---:|
| Two centers | 32 / 4,096 | 12.513 s | 9.253 s |
| Three centers | 32 / 4,096 | 15.721 s | 9.125 s |
| Two centers | 64 / 8,192 | 29.091 s | 11.473 s |
| Three centers | 64 / 8,192 | 28.016 s | 11.405 s |

These exclude full process setup, endpoint diagnostics and later HMC. They are
not prices for new batch sizes, architectures, fresh-data schemes or replicated
comparisons. Use them as measured inputs, then reconcile against the saved
worker totals. The review's three-, four- and ten-hour estimates are unmeasured;
three GPU-process hours would already exceed the approximately 1.08 hours left.
No particular extension request is justified until complete comparisons are priced.

This remainder has not been shown sufficient for the full proposal. Before any
campaign launch, price complete primary comparisons, all required seeds, setup,
validation and retained confirmation. Use measured compile/setup time plus
per-update/per-draw cost to compute C=sum_jobs(C_setup+n*C_unit+C_validation).
Include failed-attempt/retry reserves explicitly from the observed failure rate
or label an engineering contingency as a hypothesis. If the required complete
comparison exceeds the remainder, present one consolidated allocation request
with measured estimates; do not quietly shrink it into a canary or reset the
ledger. No additional allocation is assumed in this document.

| Choice | Starting value or determination | Provenance, risk and smallest useful check |
|---|---|---|
| Original Gabrié reproduction | Six coupling pairs, width 100, depth 3, batch 400, 1,500 Adam updates, LR .005 | Appendix F/G.1; use paper experiment, not CLI defaults. Verify actual effective settings before launch. |
| Canonical IAF | Source constructor; 3 stages; author masks, ELU, reversal, free scale bias | Owner/source baseline, not a proven adequate family. Check exact profile after reload. |
| Initialization | Author variance scale .02; earlier .2 only a labeled comparator | Source versus local hypothesis; inspect initial density/scale/gradient/update distributions. |
| Width | 32/64 at matched allocations initially | Existing r3 hypotheses; do not confound width with steps. |
| LR and batch | Initial LR 3e-4/1e-3, batch 64/256; select a small discriminating subset after pricing | Inherited October 2/4 choices, not calibrated defaults. Check noise and sustained heldout progress. |
| Joint coefficient | Lambda 1 first; further values determined from loss and gradient scales | Noé example baseline; expected-objective distinction and coverage checks remain necessary. |
| Scale cap | Existing c=2, free bias outside cap | Local source-option hypothesis; revisit only if diagnostics implicate it. |
| Continuation rungs | Geometric checkpoints based on measured cost, using 1,024/2,048/4,096/8,192 as inherited initial locations | Reporting locations, not convergence criteria. Maximum work must fit the complete comparison allocation. |
| Fit replications | Three exploratory seeds initially | Convenience minimum, not statistical superiority evidence; expand only through a priced inference plan. |
| Gaussian probes | 1,000 per trained endpoint, with separate heldout physical/directed probes | Explicit user requirement; never a uniform guarantee. |
| Distribution screens | Preserve historical thresholds for comparability; new scientific accuracy bands must be declared from desired observable error and calibrated using exact controls | Existing 0.15 responsibility screen is only a coarse veto, not adequate proof of stable weights. Do not invent a tighter universal number. |
| Precision | FP64 references; separately assessed FP32/TF32 runtime | FP64 success cannot establish TF32 equivalence. Revisit recorded FAB gradient mismatch in its own scope. |
| Stop conditions | Invalid common evidence, genuine scope/permission boundary, exhausted priced budget; otherwise follow the planned candidate repair | Failed fit is a promotion veto and repair trigger, not automatically a continuation veto. |

Use independent fit repetitions for training uncertainty and replicated populations
for teacher uncertainty. Use paired integration draws only for comparing fixed
maps. For selection use heldout data; for confirmation use fresh streams.
Nomination diagnostics do not establish a method ranking. If superiority is
claimed later, predeclare paired contrasts, meaningful effect sizes, uncertainty
intervals and repeated-look/multiplicity handling, and price the needed sample
size. Until then retain all viable arms without a descriptive winner.

## Implementation and output organization

Extend the existing scientific master with explicit source-reference, fitting,
representation, native-method and confirmation phases. Reuse the canonical
transport/training code and public HMC interfaces rather than create another
learner or tuner. The proposed phase names are a design, not a claim that new
CLI verbs currently exist. Existing `check`, `preflight`, `price`, `run`,
`resume`, `status` commands remain the integration surface after implementation.

Every attempt gets a fresh versioned directory under
`docs/plans/artifacts/neutra-source-fit-remedy-2026-10-04/`; do not create a
new scientific attempt by resuming terminal r3. Record phase, target/data/source
identity, exact effective settings, environment, seeds, devices, growth policy,
XLA/TF32, optimizer/RNG checkpoint, wall/core/process time, outputs, decision and
remaining budget. Refresh the next phase from actual findings after each result.
Bounded local harness/resource repairs may retry within the unchanged allocation;
preserve failed evidence and charge it. No one-use approval tokens or new broad
allow-list rules are required by this plan.

GPU training remains batch-native TF/TFP/XLA with verified memory growth and
trusted execution. Sample-bank generation uses multicore CPU work unless an
explicit scientific exception is justified. No NumPy runtime fallback, scalar
sample loop, implicit pfor or unreviewed precision/backend change is introduced.

## Review disposition and remaining execution details

The initial tempting plan, "use joint loss, then run all six teachers," was
rejected: joint loss already failed local confirmation, and the current failure
precedes the teachers. This revision first audits those outcomes and isolates
learning while preserving a separate source-reproduction comparison.
It also corrects width/step confounding, distinguishes raw versus corrected
sampling, retains original FAB execution evidence, protects final targets and
refuses to infer a new compute allocation from prior budgets.

Accepted review changes are the detailed historical post-mortem, measured complete
pricing, a smaller sequential first comparison and explicit objective branches.
The representation condition is corrected to allow persistent fitting failure
to trigger a capacity test. Original-author reproduction remains a bounded
attribution question rather than a prerequisite for all local progress.

The review's regional-KL stationary solution and oscillatory-density KL proof
are wrong as derived in the linked assessment; neither is used here. Its new
KL/tail cutoffs, factor-two deterioration/cost allowances, .001 point-estimate
plateau rule and 10,000-point probe requirement have no calibrated basis and
are not adopted. The existing required 1,000-point diagnostic remains descriptive.

The skeptical re-audit supports this order because it addresses the checked
exact-teacher failure, preserves competing explanations and stops invalid common
evidence without converting a failed candidate into rejection of the direction.
Before a research launch, finish P0's per-case diagnosis, select and price the
complete first comparison, freeze its settings/streams/accuracy bands, and verify
the P2 call chain. The remaining uncertainty is whether affordable optimization,
representation or downstream geometry is limiting. This is a revised proposal,
not a completed repair or evidence that NeuTra is now correctly trained.
