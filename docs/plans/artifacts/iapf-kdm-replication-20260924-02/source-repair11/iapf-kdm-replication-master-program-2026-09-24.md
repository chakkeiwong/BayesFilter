# iAPF and KDM/IWSG replication master program

Updated: 2026-09-25. Status: **48 additional GPU hours authorized; execution continuing; both replications incomplete**.

Active execution/budget amendment:
[48-hour continuation](iapf-kdm-replication-48h-continuation-2026-09-25.md).
This supersedes the exhausted-attempt stop below. The same ledger now allows
173700 cumulative GPU seconds and 60 cumulative attempts; old charges remain.
The CPU cap is unchanged. TG/IRG probes run first; longer phases require their
specific evidence contract, not another approval within the authorized scope.
Owner direction: establish the two original methods before further local
extensions or integration. This document is the single active authority for
that order of work, subordinate to the current owner instructions and AGENTS.md.

Canonical file: /home/chakwong/BayesFilter/docs/plans/iapf-kdm-replication-master-program-2026-09-24.md.
The similarly named file in the execution worktree is a pointer, not another
editable master. Keep current status in the [checkpoint](iapf-kdm-replication-checkpoint.md)
and supporting anchors in the [source audit and plan review](artifacts/iapf-kdm-replication-master-20260924-01/source-audit-and-plan-review.md).

## Purpose and completion rule

The question is whether we can implement each original filter faithfully,
reproduce a defined set of its experiments, and then verify the corresponding
BayesFilter TensorFlow implementation. These are three separate obligations.
A successful kernel comparison, a completed process, a fitted Gaussian, or an
XLA timing result cannot substitute for any of them.

| Required method | Meaning here | First required experimental scope |
| --- | --- | --- |
| iAPF | Guarniero, Johansen and Lee's iterated auxiliary particle filter: repeated twisted particle filtering, backward function fitting, and adaptive particle/stopping rules | Section 5.2 linear-Gaussian study, Algorithms 3–5 and Equations (15)–(16), through the published dimensions and replication counts |
| KDM/IWSG | Younis and Sudderth's **mixture density particle filter (MDPF)**, with continuous kernel-mixture resampling and importance-weighted sample gradients; KDM names its mixture representation, not a different complete filter | Appendix B synthetic learning experiment, followed by the Gaussian/von-Mises MDPF bearings-only benchmark in Section 6 and Appendix D |

This is a scoped first replication program. The iAPF stochastic-volatility and
other paper examples, MDPF DeepMind/House3D experiments, A-MDPF, and
Epanechnikov fine-tuning remain outside this first scope. A completion statement
must name the completed experiments; it must never say that both entire papers
have been reproduced.

Later LEDH combinations, model-score extensions, NeuTra/HMC, and default or
leaderboard promotion wait for the required replication evidence. This
reorders the former score master; it does not change the canonical LEDH,
reset, gradient, tuning, or backend policies.

## What is already established

| Evidence | Current status | What remains |
| --- | --- | --- |
| September 18 public-reference comparison | 66 KDM component checks passed after explicit target/epsilon alignment; the iAPF public R implementation was found to differ from the paper | Complete filters, learning, paper experiments, and matching TensorFlow consumers |
| Independent R iAPF paper attempt | Exact-twist controls passed; learned-fit replication remained unresolved | Equation (15), positive floor, numerical solver behavior, complete adaptive runs and paper-scale uncertainty |
| Later adaptive score/QR ladders | Useful evidence about their declared local objectives | They do not close the original Equation (15) replication gap |
| MDPF source training/evaluation | Full-sized short source training and T=150 diagnostic evaluation pass; fixed-sample and truncated-sequence gradient parity pass | Repeated-draw/source-target analysis, full training, untouched scientific evaluation and uncertainty |
| September 23 precision confirmation | Interrupted local-variant engineering study | Deferred; its remaining budget is not transferred into this program |

Existing evidence is preserved, but may be reused only for the exact operation,
target, source revision, and execution scope it actually tested. Historical
LEDH evidence remains subject to the repository's explicit invalidation rules.

## Work sequence and exit evidence

Source closure comes first. Independent iAPF and MDPF tasks may then progress
without waiting on an unrelated repair, within their allocated resource limits.
No extension becomes the substitute for a difficult original-method stage.

| Stage | Work | Evidence needed to close the stage |
| --- | --- | --- |
| 0 — Source and execution closure | Resolve paper versions, mathematical targets, author/reference identity, complete configuration dependencies, runnable endpoints, environments, tolerance design, seeds and cost estimates | A source-to-consumer map; resolved configuration; executable wiring checks; short reference diagnostics; exact commands and budget for the next stage |
| I1 — iAPF mechanics | Independent R implementation of the paper model, twisted initial/transition laws, likelihood normalization, backward targets, fit and controller | Exact Gaussian/Kalman identities and adversarial controller fixtures; all discrepancies classified |
| I2 — iAPF fit reconstruction | Diagnose the original Equation (15) objective and unspecified numerical choices on saved and fresh calibration fixtures | A frozen, justified fit/floor/solver policy with failure diagnostics; a fresh full-horizon pilot; unresolved source choices disclosed |
| I3 — iAPF paper experiment | Run the frozen policy and paper comparators at the published Section 5.2 settings | Every planned replication accounted for, failure denominators retained, Kalman-relative results and uncertainty; explicit reproduction verdict |
| K1 — MDPF/IWSG mechanics | Follow the actual MDPF configuration through sampling, density evaluation, gradient injection, next-step normalization and loss | Executable endpoint wiring and matched-target value/gradient checks, including circular coordinates and epsilon behavior |
| K2 — Synthetic learning replication | Reproduce Appendix B's sequence learning with IWSG, IRG and truncated gradients | Full training trajectories, mixture-Kalman reference, independent repeats, and uncertainty for the declared learning/gradient claims |
| K3 — Bearings-only MDPF replication | Reproduce base MDPF pretraining, training and long-horizon evaluation; run the prespecified source comparators | Complete data/split/configuration lineage, all training runs and failures, untouched evaluation, conditional NLL/error and uncertainty |
| P — TensorFlow port conformance | Verify the actual BayesFilter consumers against the established independent references | Matched mathematical targets, full-sequence parity, compatible training objective/gradients, stable GPU/XLA graphs and recorded numerical limits |
| C — Joint completion assessment | Assess both methods across source fidelity, experimental reproduction and port conformance | One two-method decision table, unresolved limitations, and a justified next action |

This document governs the replication program. Its executable entry point is
[scripts/run_replication_master.py](/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/scripts/run_replication_master.py)
in the numerical checkout. It currently executes **Stage 0 diagnostics only**,
with plan/run/status commands, cumulative budgets and recorded repair attempts.
Later stages are unimplemented and cannot be launched by this dispatcher.
A completed process does not mark a replication complete.

The first diagnostic tranche finished twelve launches and exposed a failed iAPF
local-fit repair. The separately authorized continuation has used all twelve
attempts, with eight successful diagnostic job types: fixed-sample and
truncated-sequence IWSG parity, iAPF amplitude escape, short full-size source
training, actual T=150 evaluation, one-step expectation checks on 32,768 clouds,
the four-arm source protocol inventory, and DIS-PF's MSE-to-bandwidth training.
One observer failure and three loader failures remain charged and preserved.
The loader repair passed a real CPU import check; TG-MDPF and IRG-MDPF have
not yet been timed successfully. Both replications remain incomplete. See the [continuation result](/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/docs/plans/artifacts/iapf-kdm-replication-20260924-02/continuation-review.md)
and [verified evidence](/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/docs/plans/artifacts/iapf-kdm-replication-20260924-02/terminal-evidence-review03.json).
The original [gap-resolution review](/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/docs/plans/artifacts/iapf-kdm-replication-20260924-01/gap-resolution-execution-review.md)
remains historical evidence of the failed local-fit candidate.


## Gap-resolution phases and execution order

Revision: 2026-09-24, following the owner's request to update, review and execute.
Completing a prerequisite does not complete either replication. Preserve the
first ledger and all twelve attempts. The owner's later continuation instruction
authorized a separate bounded tranche; all twelve launches are now used.
See the [comparator review](/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/docs/plans/artifacts/iapf-kdm-replication-20260924-02/comparator-cost-review.md) for its final
source inventory, loader repair and DIS-PF result.

| Phase | Concrete work and evidence | Exit condition and next action | Current state |
| --- | --- | --- | --- |
| 0R — Repair resumability | Bind reusable checks to numerical source dependencies, interpreter identity and inputs. Retain historical results but invalidate changed/legacy checks. Add explicit job selection and regression checks. | Changed dependencies cannot inherit PASS; ordinary candidate failures do not stop independent checks. | DONE; 16 focused tests pass; unchanged resume checked at its recorded revision |
| 0M — Execute author training | Call actual author setup_training, trainer constructor, generated bearings data, persistent loaders, scheduler, optimizer, validation and checkpoint writing. Exercise short weight pretraining and full-sequence training on GPU, with all outputs isolated. | Finite losses/gradients, observed updates, endpoint traces and reloadable checkpoints. This closes short execution wiring only; failures trigger localized repairs. | PASS on short reference, attempt 12 |
| I2a — Discriminate fit repair | Apply a damped Gauss–Newton local solver to the unchanged joint Equation (15) density residual on 32 saved failure records (ten distinct fit inputs). Check analytic Jacobians, exact-Gaussian controls and complete objective/shape/scale trajectories. | Stationarity without collapse nominates a candidate only. Lower scalar residual alone never passes. Failure identifies the next source/numerical question. | DIAGNOSTIC FINISHED; general repair failed |
| I2a.1 — Resolve the failed local fit | Use the ten distinct inputs and their trajectories to distinguish poor local initialization/steps from amplitude escape. Audit unrecovered author optimizer and Eq16 floor choices. Define a source-grounded local-solution policy; declare any bounds/damping and test non-harm. | Freeze a justified fit policy using calibration and fresh shape/normalization checks, then pass it to I1/I2b for the adaptive pilot. No objective substitution or success claim from tiny raw loss. | ESCAPE DIAGNOSIS PASS; local-fit/floor policy remains unresolved |
| I2a.2 — Recover or qualify the fit policy | Inspect original-author solver/floor material and published-version differences. If unrecovered, specify a qualified local-solution reconstruction with declared constraints/stopping/floor provenance and calibrated non-harm checks. | Exact Gaussian and fresh shape/normalizer checks must pass before a pilot. Objective substitution or materially different algorithm requires an explicit direction decision. | NEXT iAPF prerequisite; official record yielded only accepted PDF, not solver code |
| I1/I2b — Freeze and validate iAPF | Resolve floor/solver provenance and controller/normalizer fixtures. Freeze choices using calibration data, then run a fresh T=100 adaptive pilot with an independent final draw and Kalman reference. | A complete valid pilot, disclosed source ambiguity and measured cost; otherwise return to fit reconstruction. Do not tune to the paper table. | OPEN; fit/floor policy first |
| K1 — Verify IWSG mathematics | Compare actual resampling-to-loss endpoints against independent full-mixture value/gradient calculations at fixed samples, including small weights, normalization and angle wrap. | Matched targets and declared derivative tolerances pass. Finite gradients alone do not close K1. | FIXTURE PARITY PASS: eight endpoint and six truncated-sequence cases; estimator expectation/protocol gaps remain |
| K1b — Close estimator/source protocol differences | Analyze repeated-draw behavior of the epsilon-modified sampler/density/NLL and source BPTT; locate the Appendix B consumer and specify expectation targets. Resolve evaluation seed semantics before freezing trials. | An explicit source/paper discrepancy disposition and executable estimator checks with stated uncertainty; no blanket unbiasedness claim from fixture parity. | PARTIAL: one-step normalized-statistic expectation derived and checked; source-fidelity evaluation freezes effective seed zero. Appendix B consumer and whole-NLL expectation remain unresolved |
| B — Freeze protocols and budget | Freeze seeds/splits, tolerances, uncertainty units, margins and conditional heuristic checks. Measure pretraining, training, evaluation and iAPF cost; include every repeat, comparator and retry. | Concrete cumulative budget; materially expanded compute needs the owner's decision. Unknown components stay unbudgeted. | OPEN; reliable full budget absent |
| B1 — Measure representative cost | Time the real source route without pervasive call profiling; separately measure data generation, full-size training/validation, pretraining, checkpoint/plot costs and T=150 evaluation. Record early-stopping assumptions and account for all comparators/repeats/retries. | A defensible cumulative CPU/GPU/attempt budget with uncertainty and stop conditions before requesting expanded compute. | MDPF and DIS-PF short costs measured; TG/IRG loader failures repaired but timings absent; full budget open |
| I3 — Execute iAPF paper study | Run Section 5.2 settings and specified repetitions/comparators, preserving failures and Kalman-relative estimates. | Complete accounting, uncertainty and explicit reproduction verdict. | NOT RUN / NOT BUDGETED |
| K2 — Execute synthetic learning | Implement Appendix B's source objective, mixture-Kalman reference and TG/IRG/IWSG comparisons with independent training repeats. | Full trajectories and uncertainty; the bearings smoke cannot substitute for this experiment. | NOT RUN / NOT BUDGETED |
| K3 — Execute bearings study | Generate isolated source datasets; complete weight pretraining, 11 full-training trials, source comparators and untouched T=150 evaluation. | Freeze metric/branch-to-checkpoint mapping first; preserve data/checkpoint lineage, all trials, conditional NLL/error and uncertainty. | NOT RUN / NOT BUDGETED |
| P — Validate TensorFlow consumers | Implement established targets in TensorFlow/TFP; check actual consumers against references using stable signatures, batching, GPU memory growth and XLA. | Full sequence/value/gradient parity, tracing, compilation, memory and downstream checks. A matmul smoke cannot close P. | NOT CHECKED at method level |
| C — Assess both replications | Assemble source fidelity, mechanics, experimental results, uncertainty, heuristic vetoes and TensorFlow conformance separately. | Explicit PASS/FAIL/QUALIFIED/INCOMPLETE verdicts and justified integration decision. | INCOMPLETE |

Completed work: 0R, short 0M wiring and compatibility repairs, failed I2a
repair, I2a.1 escape diagnosis, K1 pointwise/truncated-sequence fixtures, K1b
one-step expectation, four-arm protocol inventory, and B1 short MDPF/DIS-PF
training and MDPF evaluation costs. The next scientific work is I2a.2 and K1b,
plus TG/IRG timing and complete long-run accounting under B. A source policy
must be justified before another iAPF adaptive pilot. Scientific splits,
uncertainty units, tolerances and conditional heuristic comparisons must be
frozen before I3/K2/K3; P then checks actual TensorFlow consumers. Full
experimental phases remain specifications, not implemented handlers, and are
NOT_BUDGETED. Remaining diagnostic seconds cannot fund the multi-day campaigns.

The [gap-resolution review](artifacts/iapf-kdm-replication-master-20260924-01/gap-resolution-review-20260924.md)
records the pre-run audit, diagnostic contract, defaults and execution.

## Source checkpoint identity prerequisite

The checked source configuration inventory covers MDPF, TG-MDPF, IRG-MDPF and
DIS-PF at author commit b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18. In
configs/our_experiments.yaml, the MSE-named evaluation still resolves to
full_dpf_fixed_bands/models/full_dpf_model_best.pt, the NLL-stage checkpoint.
DIS-PF's active MSE evaluation loads its pre-bandwidth checkpoint; its
bandwidth-calibrated NLL evaluation is a separate disabled entry. Names alone
therefore do not identify the evaluated scientific object. Before K3, freeze an
explicit metric, trained stage, configuration and actual checkpoint/hash map
against Section 6/Appendix D.4.1 and the actual evaluation consumer. A deliberate
configuration correction must be disclosed; neither these labels nor this
inventory prove paper reproduction. See the comparator review and attempt08's
resolved configurations for exact source anchors.

## iAPF mathematical and experimental requirements

The initial state is the paper's X1 distribution, not an extra local X0
transition. Match the initial twist normalizer, each twisted transition,
potential and terminal factor before comparing a likelihood estimate. Use
Algorithm 5's ESS-adaptive resampling with kappa = 0.5, including accumulated
weights and the correct normalizing-constant factors when resampling is
skipped. Every-step resampling is a separately named comparator.

Algorithm 3 fits backward targets using the next twist. Equation (15) fits
Gaussian density values against a scaled positive target; Equation (16) then
adds a strictly positive constant. A log-density QR regression, score fit,
relative-shape objective, or the independent public R repository's objective
is a different fit. None can silently replace Equation (15).

The unrestricted Equation (15) objective admits an amplitude-escape direction:
as Gaussian variance grows and the positive scale approaches zero, both sides
of the residual can vanish. A small objective or an optimizer success flag is
therefore insufficient. I2 must record density/target scales, variance and
scale trajectories, gradients, boundary hits, and final usable shape. Diagnose
the existing failure before another ladder. Compare candidate local solver
choices on calibration fixtures, freeze the selected reconstruction, and use
fresh validation; do not choose a solver or floor by matching the published
performance table. Any bound, ridge, floor, damping, or other alteration needs
its mathematical role, source status, and numerical effect stated explicitly.

Algorithm 4 uses N0 = 1000, k = 5, tau = 0.5 in this study. Match the strict
iteration condition, the k+1-estimate window, particle doubling rule, and an
independent final filter draw after adaptation. Early-window indexing that
the paper leaves implicit must be documented and tested. Computational caps
produce an incomplete replicate; they are not replacements for the paper's
stopping rule.

I3 uses T = 100, d = 5, 10, 20, 40, 80, identity initial/process/observation
covariances, observation matrix I, initial mean zero, and
A[i,j] = 0.42^(abs(i-j)+1). Use 1,000 independent complete iAPF replications
per dimension, BPF N = 10,000 and fully adapted APF N = 5,000, with exact Kalman
likelihoods. Preserve the paper's distinction between reported comparator
cells and unreported high-dimensional cells; additional cells are supplemental.

The paper's original data and numerical optimizer settings have not been
recovered. Freeze new data and adaptation/final-filter random streams before
running, share data across methods, and report inference conditional on those
data. Additional data sets are needed for population-level claims. Report
likelihood ratios in a numerically stable representation, log-likelihood
error, standard deviation, iteration/particle counts, failures and wall time.
Do not compare mean log likelihood to likelihood unbiasedness.

The paper table has no recovered sampling uncertainty for its original data.
Its entries are reference points, not known population constants. Do not call
the old “within twice the paper SD” diagnostic a formal replication test.
Stage 0 must prespecify numerical tolerances and the scientific comparison
criteria before untouched runs; numerical table equivalence remains
unestablished if a defensible equivalence margin cannot be justified.

## MDPF/IWSG mathematical and experimental requirements

MDPF learns a posterior state density from supervised trajectories through
learned dynamics/proposal and measurement-weight networks. Its training loss
is state-posterior NLL. This does not automatically estimate the likelihood
score of a separate generative state-space model.

Paper Equations (14)–(15) differentiate m(z;phi)/q(z) with the sampled location
and proposal q held fixed; at the reference parameter q = m and the weight is
one. The numerator is the full mixture density. Differentiating through the
proposal denominator or a sampled component label computes another estimator.
Finite-difference checks must freeze the same samples and denominator. Testing
unbiasedness additionally requires repeated draws from the actual matching
proposal; pointwise gradient agreement alone cannot establish it.

Use the pinned original-author repository. The correct base bearings
configuration is:

.localresources/code/younis-mdpf-neurips-2023/src/experiments/bearings_only_experiments_methodical_25_particles_multiple_runs/experiment0002_importance/config.yaml

It selects KDEParticleFilter with differentiable resampling enabled and
ImportanceSampling selected. The directory importance_sampling_pf_learned_band
selects a discrete comparator and cannot serve as the MDPF replication.

K1 must retain Gaussian position kernels, the von-Mises angular kernel,
bandwidth transformations, detached sampled locations, full-mixture score
injection and subsequent weight normalization. Audit the author's density
epsilon and added resampling-weight epsilon separately. Where sampling and
density smoothing differ, distinguish exact source execution from the ideal
matched-proposal derivation. No blanket unbiasedness claim covers that
difference.

The [expectation diagnostic](/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/docs/plans/artifacts/iapf-kdm-replication-20260924-02/expectation-review.md)
derives and checks an additional distinction. For a linear statistic and constant
measurement weights, the finite-N normalized consumer has expected gradient
(N-1)/(N c) times the covariance of the statistic with the source density score,
where c=1+1e-8. In the ideal matched-density limit this is about 0.96 of the
mixture-mean derivative at N=25. Actual source sampling/normalization agrees
with the derivation on 32,768 clouds. This supports the stated one-step result,
not unbiasedness of the training NLL or an untruncated sequence. Preserve
source behavior for replication; any bias correction is a different estimator.

K2 follows Appendix B: scalar dynamics with bimodal observations, the stated
trainable parameters, fixed noise assumptions, distinct posterior/resampling
bandwidths, 1,000 sequences of length five, batch size 64, SGD, and
final-time state NLL. Resolve initialization, learning rate, repeat count and
source availability before launch. An independent reconstruction must be
labelled as such if the actual synthetic author runner cannot be recovered.
The finite-horizon mixture-Kalman calculation is the reference for the
specified generative model, not an oracle for arbitrary learned proposals.
The located author toy_problem_dataset.py implements nonlinear dynamics and
a quadratic observation, so it is not the Appendix B consumer. Its name cannot
justify substituting that easier-to-find experiment for the requested one.

K3 first reproduces the Gaussian/von-Mises bearings model at N = 25:
5,000 training trajectories and 1,000 validation trajectories of length 17;
5,000 evaluation trajectories of length 150. Match the observation model,
true-state-plus-noise initialization, sparse training labels, dense
evaluation, four-step truncated BPTT, and architecture from Appendix D and the
resolved author files. For the NLL branch the paper gives neural-network
learning rate 0.0005, bandwidth rate 0.00005, and gradient clipping at 100.
The inspected source adds Adam, batch 64, up to 800 epochs, validation
early-stopping/scheduling, and an 11-trial configuration generator. These are
source settings to reconcile, not evidence that 11 complete runs have occurred.

Do not run the checked-in run.bash blindly. Its generated YAML is absent in
the inspected checkout, pretraining checkpoints are prerequisites, and the
training experiment switches are false. Resolve all includes and save a
readable final configuration before execution. Trace the actual loss and
checkpoint selected by evaluation; a directory suffix is insufficient.

Run MDPF with source TG-MDPF, IRG-MDPF and DIS-PF comparators under matching
data, architecture, initialization, training effort and evaluation rules,
recording any paper-mandated difference in their objectives. NLL training and
optional MSE fine-tuning are different branches. Freeze which published result
each branch targets; no unreported fine-tuning or test-based selection.
For the source-fidelity evaluation arm, record and preserve effective RNG seed
zero: the current source ignores its configured random_seed during RNG setup.
A corrected seed setter or independent resampling streams must be separately
identified. Training seeds, paired comparator streams and inference units
still need to be frozen before the full experiment.
A-MDPF requires trained MDPF initialization and remains a later extension here.

## Evidence contract, inference and failure handling

| Evidence role | Definition for this program |
| --- | --- |
| Primary correctness criteria | Exact claimed target and reachable implementation; mechanism identities; source-matched complete sequence behavior; declared numerical tolerance checks |
| Primary replication criteria | Complete prespecified experiment and all planned repetitions; reproducible estimates for the stated paper claims with appropriate uncertainty |
| Promotion vetoes | Wrong objective/conditioning, missing derivative terms, invalid numerics, failed source parity, unreported failures, data leakage, failed heuristic adequacy, or inadequate statistical support for the proposed claim |
| Continuation vetoes | Invalid target/data/reference, corrupt evidence, unresolved implementation identity for a dependent run, budget exhaustion, or an external/irreversible boundary requiring authorization |
| Repair triggers | Local fit failure, source/configuration mismatch, harness/serialization/resource failure, or numerical precision defect with a bounded discriminating repair |
| Explanatory diagnostics | Compilation/steady-state timings, trace counts, ESS, fit loss, memory, training curves, and kernel parity outside the target claim |

A failed candidate is not a rejection of either research direction. First
decide whether the failure invalidated the experiment or exposed the candidate.
Continue a planned repair within the unchanged contract and remaining budget.
Never drop failed seeds or replace them with successful ones. A faithful
negative experiment can complete a replication attempt; it does not establish
successful reproduction or suitability for downstream promotion.

Use paired comparisons where observations and evaluation samples are shared.
For learned MDPF, training runs are the independent units for claims about
training stability; thousands of correlated time steps are not thousands of
independent training replications. Retain per-run and per-trajectory records.
Prespecify intervals, resampling units, multiplicity treatment and effect
margins in Stage 0. Report 95% uncertainty intervals; bootstrap whole relevant
units rather than individual dependent time steps. Too few independent units,
unstable bootstrap behavior or unsupported tail comparisons yield
inconclusive results, not rankings.

Construct these cheap adversaries before interpreting the complex methods:

| Experiment and salient situations | Adversaries and rationale | Reporting rule |
| --- | --- | --- |
| iAPF, each dimension separately and low/high ESS segments | Published BPF allocation: ordinary SMC; BPF at the initial iAPF particle allocation: cheap resource baseline; published fully adapted APF: one-step lookahead | Kalman is the separate exact oracle. Keep published and equal-resource comparisons distinct; account for adaptation cost. |
| Synthetic MDPF, separated/overlapping modes and early/final times | Frozen initial model: benefit of learning; persistence density fitted on training residuals: simple dynamics; generative BPF with parameters fitted from the same training data: classical filter | Keep true-parameter mixture Kalman separate as the oracle; do not grant its information to only one competing method. |
| Bearings MDPF, short/long horizons and high/low ambiguity/outlier regimes | Constant-motion predictive density fitted on training data; fitted generative EKF; fitted generative BPF with the same observation family | All methods must supply a normalized state density for NLL and use the same training information and evaluation observations. |

Define regime boundaries on calibration/training data, never on test outcomes.
Use circularly correct angular errors/densities. Document where a Gaussian
EKF approximation cannot represent multiple modes. Poor performance against
a simple eligible baseline is a headline and a promotion veto; it does not
license changing the paper algorithm to improve its replication score.
Heuristics are falsification checks, not new tuning targets.

## Material assumptions to resolve and freeze

| Choice | Provenance and justification | Failure mode and earliest check | Current classification |
| --- | --- | --- | --- |
| First two experiment scopes | Owner's replication-first direction; tractable paper experiments that exercise the methods | Mistaking partial scope for entire-paper reproduction; explicit scope labels | Planning choice |
| iAPF fit/floor/solver and controller indexing | Paper specifies objective/positive floor/controller but not every numerical detail | Vanishing objective, unintended method, biased seed exclusion; saved-fit and controller fixtures | Unresolved reconstruction |
| iAPF fresh observations and random streams | Original data/seeds not recovered | Apparent table discrepancy caused by data; Kalman tie-out and split manifest | Required disclosed limitation |
| MDPF exact architecture, preprocessing and pretraining | Paper Appendix D plus pinned source includes | Wrong effective experiment despite a successful process; resolved config and endpoint test | Source closure pending |
| MDPF epsilon and circular bandwidth conventions | Actual author numerical program | Density/sampler or derivative mismatch; matched fixtures at small weights and angle wrap | Source behavior, not a universal mathematical default |
| Training optimizer, schedule, initialization and early stopping | Target-specific paper/source protocol | Undertraining or incomparable runs; short loss/gradient and checkpoint checks | Source baseline, not transferred tuning |
| TF dtype, TF32 and XLA | Repository production direction, subject to numerical evidence | Wrong compiled result or precision effect mistaken for method failure; reference parity first | Engineering hypothesis until verified |
| Numerical tolerances, statistical margins and capacity estimates | Independent calibration/error analysis and bounded pilots | Post-hoc acceptance or under-budgeted full runs | Must be frozen before dependent experiment |

No safety protection is accepted or rejected merely because it was inherited.
Pure observability and reject-only numerical checks should be retained.
Numerics-changing protection needs a dedicated non-harm evaluation and a
declared effect on paper/source comparability.

## Execution resources, approvals and artifacts

The first authorized tranche used all twelve launches, 18.325 CPU-worker
seconds and 130.263 GPU-worker seconds. Its ledger remains unchanged. The
owner's subsequent “if not, continue” instruction authorized the separate
[bounded continuation](iapf-kdm-replication-continuation-2026-09-24.md), capped at
1,800 CPU-worker seconds, 900 GPU-worker seconds, twelve launches and 300 seconds
per child. Eight diagnostic job types passed in twelve attempts, including one
observer and three loader failures. Consumption is 26.154373 CPU seconds and
283.127934 GPU seconds; zero launches, 1,773.845627 CPU seconds and 616.872066
GPU seconds remain. The attempt cap is exhausted. No child remains running.
Routine source reading,
documentation and routine development checks are recorded separately.

The unprofiled source cost check used 5,000/1,000 trajectories, source T=17/N=25
and batch sizes 64/256. Full training plus validation averaged 33.132 seconds
per epoch; inclusive Trainer.train averaged 34.952 seconds. The actual T=150
evaluation of 5,000 separate diagnostic trajectories took 36.403 seconds for
the whole process, including generation, metrics and output persistence.
It used a short-trained checkpoint and supplies no replication accuracy claim.

At these short-run rates, eleven MDPF trials at 800 epochs imply approximately
81–85 GPU-hours for full training alone; one 400-epoch pretraining adds about
0.35–0.50 hours. These are descriptive arithmetic scenarios, not confidence
bounds or an executable full budget. They supersede the old profiled
177-hour illustration for this scope. The DIS-PF source requires full-network MSE training followed by frozen-network,
bandwidth-only KDE_NLL fitting. Its corresponding eleven-by-800-epoch scenario
is 105.37–110.38 GPU hours for MSE and 59.21–62.70 hours for bandwidth fitting,
164.58–173.08 hours combined, excluding pretraining and evaluation. TG/IRG,
optional MSE refinements, Appendix B, iAPF, long-run plotting/early stopping and
repairs remain uncosted. See
[cost arithmetic](/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/docs/plans/artifacts/iapf-kdm-replication-20260924-02/cost-projection.json).
See also [comparator cost arithmetic](/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/docs/plans/artifacts/iapf-kdm-replication-20260924-02/cost-projection02.json).
These accounting ranges are descriptive one-seed extrapolations, not uncertainty
intervals, commitments or method rankings. Full-stage budgets remain NOT_BUDGETED.

I3 and K3 are **not yet budgeted for launch**. Their full reference/source
protocols must not be silently shrunk to fit a convenient timeout. Stage 0
must measure representative cost, inventory environment/data dependencies,
then fill a single cumulative CPU/GPU/attempt budget here for I1–C, including
pretraining, all comparators, all planned repeats and retries. If a reliable
full-run budget cannot be established, report UNDER_BUDGETED with the
measured estimate. A request to execute this master authorizes its stated
tranche; a material compute expansion needs the owner's decision after the
concrete costed plan is ready. Do not borrow the old Phase21 balance.

Use the existing numerical checkout
/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922,
branch ledh-refactor-with-policy-fix, base commit
89065bc6354801cb368d5e163ba49fd9c4372d10, after verifying its current dirty
changes. Main checkout is the planning authority. Never overwrite historical
result directories.

New execution artifacts belong under the numerical checkout's
docs/plans/artifacts/iapf-kdm-replication-YYYYMMDD-NN/<stage>/attemptNN/.
Each stage keeps the actual command, resolved settings, source snapshots or
ordinary hashes, Git commit and dirty diff, environment, device settings,
seeds/data partitions, timeout, wall time, exit/failure class, remaining
budget and result. A single concise checkpoint links these records.

Use direct commands with an explicit working directory and stable wrapper
prefixes. Before the first GPU command in a stage, verify escalated
nvidia-smi and the actual framework device probe. GPU work always uses
escalated permissions. TensorFlow must establish and record memory growth
before device initialization. Deliberate CPU reference work hides GPUs before
framework import. No package installation or broad environment change is
included in this plan.

Prepare the exact runnable command and match it against the approved rules
before asking for any missing sandbox permission. Request a narrow reusable
wrapper prefix once when necessary; do not ask anew for unchanged authorized
repairs. Do not expand a bare Python/bash/Claude approval to avoid command
matching. Configuration, command validation, and local reversible repairs
need no additional self-imposed scientific approval ceremony.

## Runner inventory and TensorFlow requirements

| Existing entry point | Permitted use here |
| --- | --- |
| [R mechanics](../../tests/reference_iapf_paper.R), [R paper runner](../benchmarks/replicate_iapf_paper_linear.R), [Python R harness](../benchmarks/run_iapf_r_replication.py) | Starting points for I1–I3; audit the currently selected fit/controller before reuse |
| [Equation (15) solvers](../../tests/reference_iapf_eq15_solvers.R), [old resolution harness](../benchmarks/run_iapf_eq15_resolution.py) | Reuse focused diagnostics only. The old Python supervisor has an expired fixed deadline and historical output root; do not launch it. Historical results do not certify a new policy |
| Original MDPF src/main.py with experiment0002_importance/config.yaml | Import/factory, source sequence/loss/backward and short setup_training/trainer/optimizer/data/checkpoint paths pass. Full pretraining/training, comparators and untouched evaluation remain open; production main is not yet launched |
| [Replication master](/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/scripts/run_replication_master.py) | Executable prerequisite dispatcher with --jobs, dependency-aware resume and cumulative budgets; selected continuation diagnostics finished, five launches retained |
| [Author-training diagnostic](/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/docs/benchmarks/diagnose_replication_training.py) | Implemented 0M short independent-reference check; three isolated source compatibility adaptations |
| [Eq15 local-solver diagnostic](/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/docs/benchmarks/diagnose_iapf_eq15_local_solver.R) | Implemented I2a hypothesis/controls; unsuitable for promotion after saved-input collapse |
| [IWSG endpoint diagnostic](/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/docs/benchmarks/diagnose_replication_iwsg.py) and [sequence diagnostic](/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/docs/benchmarks/diagnose_replication_sequence.py) | Independent source value/gradient fixtures; cannot establish unbiasedness or learning quality |
| [IWSG expectation diagnostic](/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/docs/benchmarks/diagnose_replication_expectation.py) | Actual source one-step linear-statistic expectation, independent quadrature and Monte Carlo uncertainty; finite-N normalization and epsilon differences quantified |
| [Eq15 escape diagnostic](/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/docs/benchmarks/diagnose_iapf_eq15_escape.R) | Diagnoses the unrestricted fit objective; does not select a solver/floor |
| [Source evaluation cost](/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/docs/benchmarks/diagnose_replication_evaluation.py) | Executes actual T=150 source evaluation on separate diagnostic data and short-trained checkpoint; no accuracy claim |
| [Old score master runner](../../scripts/run_younis_score_master.py) | Later score/integration study only; not the driver that reproduces the two original filters |

Original R and author PyTorch code serve as independent references. A local
BayesFilter implementation uses TensorFlow/TFP. NumPy remains diagnostic-only;
do not import reference numerical code into an eligible runtime. Match the
actual target before converting code or benchmarking it.

Repeated local kernels require stable explicit input signatures and bounded
tracing, native tensor batching, and TensorFlow loops for time/adaptation
where necessary. Host scheduling over independent experiments is not itself
retracing or an XLA defect. No implicit or explicit pfor without the required
written exception. Keep file I/O, strings, manifests and Python callbacks
outside numerical kernels.

GPU/XLA is the local default target. For P, record XLA compatibility,
compilation cost, peak host/device memory, numerical equivalence and
steady-state performance separately. A CPU/non-JIT diagnostic localizes an
error but cannot close the GPU/XLA obligation. Trace counts and speed are
engineering evidence; they do not prove a replicated algorithm.

## Completion record and next action

For each method, report source fidelity, mechanics, paper-experiment
execution, scientific reproduction and TensorFlow conformance separately as
PASS, FAIL, INCOMPLETE or QUALIFIED, with evidence links. QUALIFIED must name
the unresolved reconstruction or missing source detail; it cannot silently
become PASS for an exact-source claim.

The joint decision table must include the primary criterion, veto status,
main uncertainty, next justified action and unsupported conclusions.
A separate inference table records hard failures, statistically supported
rankings, descriptive differences, default readiness and the next needed
evidence. One substantive source/plan review and one terminal result review
are sufficient; independent review is advisory, and a procedural reviewer
delay is not a new veto.

Both scoped replications and the relevant port checks must support the
intended claim before integration resumes. An unresolved source issue or
scientific failure triggers the next bounded repair or an explicit direction
decision, not a quiet switch to an easier local variant.

**Next action:** resolve iAPF local-fit/floor provenance under I2a.2, close
the Appendix B consumer and remaining NLL/protocol questions under K1b,
then inventory remaining
comparators/costs for B. The selected [continuation](iapf-kdm-replication-continuation-2026-09-24.md)
diagnostics have finished. Full experiments remain NOT_BUDGETED, neither
replication is complete, and the old precision supervisor remains deferred.

The [continuation review](/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/docs/plans/artifacts/iapf-kdm-replication-20260924-02/continuation-review.md)
records decision/inference tables, the repaired observer failure, scientific
vetoes, exact evidence and limits. The [terminal evidence review](/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/docs/plans/artifacts/iapf-kdm-replication-20260924-02/terminal-evidence-review02.json)
verifies snapshots and outcomes. Plan/run/status expose only implemented
diagnostics; they cannot launch unimplemented full experiments. A source
fingerprint labeled stale after a dispatcher extension remains historical
evidence for its recorded numerical program, not an automatically reusable
current-manifest PASS.
