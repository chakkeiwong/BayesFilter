# Simulated moment training of LEDH guides

Owner request: create, review and execute an implementation/test plan. Worktree
`ledh-iapf-execution-20260922`, branch `ledh-refactor-with-policy-fix`.
This completes the missing moment-loss optimization, rather than repeating the
previous score-loss selection pilot. No commit, push or default change is part
of this request.

20 October follow-up: the [numerical repair plan](ledh-moment-restoration-repair-20261020.md)
has completed diagnosis and bounded validation. The current-covariance
Cholesky failure is fixed, but materialized moment fidelity still fails on
the exceptional cloud. See the [repair result](artifacts/ledh-moment-restoration-repair-20261020/result.md).
This supersedes the earlier next-localization action, not the preserved
training/test results. No efficacy ranking or resumed holdout is authorized
by finite value/score computation alone.

Further follow-up: [collapse repair](ledh-moment-collapse-repair-20261020.md)
and its [result](artifacts/ledh-moment-collapse-repair-20261020/result.md)
repair endpoint admission and identify an upstream flow/proposal mismatch.
An exact fixed-cloud affine-preimage identity passes, but no integrated
flow-aware guide or new efficacy comparison is claimed. The kernel consumer
uses mixture-product moments, so its capacity is not excluded by the negative
information eigenvalue found for the quadratic adapter. This finding refines
the next implementation, not the preserved training/test evidence.

## Question and evidence contract

Can training the existing quadratic cSMC and positive-RBF backward-guide
coefficients against simulated terminal moment labels reduce the terminal
moment error of the full shared LEDH executor on fresh SIR datasets?

The scope is continuous SIR d18, theta=(0,0,0), T=5, N=64, GPU/XLA FP64 as an
explicit diagnostic exception. Use the existing shared UKF/flow/Contract-E/
dual-cap GenUT analytical executor. OT remains epsilon=1, five Sinkhorn
iterations and 32 terminal balancing updates; chunk K=N. No new LEDH lane.
The RBF guide is the existing KDM-inspired adapter, not a newly trained original
neural KDM. The experiment establishes at most a result for these guide families
and this scope, not canonical admission, default readiness or general KDM merit.

Primary target: terminal filtering first and raw second moments conditional on
all observations Y[1:T]. Earlier filtering moments are deliberately excluded:
the guides see later observations, so an earlier-time supervised loss would
target smoothing unless the information set were restricted separately.
The parent/future conditional proposal objective in the manuscript is a distinct
possible experiment, not silently identified with this terminal filtering loss.

For calibration-fixed center m and coordinate scales d, set z=(x-m)/d and
Phi(x)=(z,vech(zz^T)). Fix feature scales from calibration-label standard
deviations with a declared positive floor. Loss is one half the average scaled
first-feature squared error plus one half the average scaled second-feature
squared error. This is a fixed positive semidefinite quadratic loss in the
original first/raw-second moment errors, including centering cross terms.
Calibration controls depend on neither validation nor test labels.

For fixed settings and independent particle randomness, conditional expansion
gives E||A-Phi(X)||_W^2=E||A-E[Phi(X)|Y]||_W^2+C. C is independent of the guide.
Thus paired differences in simulated-label losses estimate differences in
population moment MSE without a posterior oracle. They do not reveal exact
posterior errors for individual sequences or directly measure covariance MSE.

Promotion requires all validity checks and paired dataset-bootstrap
98.75% intervals (four comparisons, nominal Bonferroni family-wise 95%) strictly below
zero for each selected trained family versus its untrained counterpart and
unguided LEDH. Use 10,000 bootstrap draws; the independent unit is a dataset,
averaging its particle replicates before resampling. No ranking from means alone.
In addition, neither learned family may have a worse mean than any heuristic
in either predeclared observation regime. This is a promotion veto only.

Constructed heuristic adversaries, all run afresh on identical datasets/streams:

* Unguided LEDH: tests whether adding guidance helps at all.
* Half-strength cSMC: cheap shrinkage against an overconfident fitted message.
* Original fitted cSMC: tests whether supervised coefficient changes add value.
* Original fitted RBF product guide: tests whether training improves kernel use.

Regimes are low/high terminal mean observed infectious count, split by the
calibration median. Freeze that threshold before training. Report per-regime
counts and losses as well as pooled results. These sanity checks never select
training parameters. Test score-label loss, raw first/second moment losses,
validity margins, timing and optimizer progress are explanatory only. They
cannot substitute for moment improvement or certify score accuracy.

## Implementation and genuine optimization

Fit each original guide using observations and independent guide-fit noise
only. Labels never enter those fitting routines or the filter inputs. Across
datasets, learn shared, dimensionless guide adaptations eta. For time t<T,
eta contains log precision strength a and translations of susceptible and
infectious coordinates in process-noise SD units. For the quadratic guide,
P_eta=exp(a)P and b_eta=exp(a)(b+P delta). For the RBF guide, translate every
center by delta and multiply every covariance by exp(-a). Preserve logits.
The t=T guide remains inactive. eta=0 exactly recovers each untrained guide.
These are actual continuous coefficient changes, not bank selection.

Compare a time-pooled three-parameter adaptation with a time-specific
3(T-1)-parameter adaptation. This bounded capacity comparison tests whether
time dependence is needed without attributing a small adapter's limitations to
the original algorithms. Translations are tied across the nine regions because
the model uses common physical coefficients. Region-specific adaptation is not
tested; report this as a capacity limitation.

Optimize the actual full-LEDH moment loss with common-random-number symmetric
directional differences and backtracking descent. This avoids introducing an
autodifferentiated model-score lane. All numerical operations use TensorFlow;
the repeated executor and adapter/loss kernels have stable signatures and XLA.
Use minibatches of four independent training datasets; keep their particle
streams identical across the two probes and trial updates. Fresh streams are
used for the next update. A rejected or invalid trial retains the prior eta.
Record probes, trial losses, accepted updates, eta, calls and elapsed time.

Initial protocol: two independent optimizer seeds per family/capacity, 48
updates, perturbation radius 0.1, initial descent length 0.25, at most three
backtracking trials (1,1/2,1/4). Normalize directions to unit Euclidean norm;
normalize the estimated descent to avoid arbitrary label-loss units setting
the step size. Bound log strength to [-log(4),log(4)] and translations to
[-2,2] process SD. These define a modest adaptation hypothesis, not an optimal
or numerically protective default. Emit bound hits and accepted step lengths.
Evaluate validation loss at initialization and updates 16,32,48; freeze each
family's candidate before opening test labels. Validation chooses capacity,
optimizer seed and checkpoint only, using moment loss. Never score loss.

## Partitions, budget and repairs

Fresh simulation seeds: calibration 262020001 (256 batched draws), training
262021001..262021064, validation 262022001..262022016, test
262023001..262023064. All are disjoint from the previous pilot. Use separately
namespaced fitting, training-particle, validation-particle, test-particle and
optimizer seeds. Two particle replicates on validation/test, independent of
training streams. Store exact seeds and labels, but load test only after writing
the frozen selection artifact. All methods share evaluation particle streams.

Remaining inherited campaign allowance is 33303.439085210 GPU-process seconds.
This stage has a hard 21600-second total GPU-process cap, including failed
attempts, probes and retries; reserve the remainder. Target completion is much
shorter; first measure compilation and call time. Use new versioned roots under
`docs/plans/artifacts/ledh-moment-training-20261020/`; never overwrite evidence.
At most three campaign launches, with bounded infrastructure repairs. Record
actual device/memory-growth policy, dtype, XLA/TF32, environment, source hashes,
Git revision/dirty state, commands, all partitions and process wall time.

Repair triggers: missing endpoint wiring, invalid simulation labels, nonfinite
baseline, failing density/derivative parity, no parameter movement despite a
known decreasing synthetic objective, or a failed infrastructure attempt.
These require repair before interpretation. Individual invalid optimizer
trials are rejected and recorded; repeated baseline invalidity is a continuation
veto. A valid candidate losing on heldout moments is a promotion failure, not
an implementation failure or evidence against all moment training.

If fewer than two updates are accepted in any run, or validation never improves
over initialization, first inspect perturbation signal and bound-hit diagnostics.
Run a predeclared repair on training/validation only: perturbation 0.025,
descent length 0.0625, 96 updates, same two capacities and seed partitions.
Do not inspect test until this decision and final selection are frozen. A second
failure is an optimization/capacity limitation for this bounded study; execute
the heldout evaluation of the frozen candidates and report it honestly.
Exhausted compute, baseline invalidity or corrupted/missing required evidence
stops dependent execution. Do not call an under-budgeted training run successful.

## Default/assumption audit and pre-mortem

| Choice | Provenance / reason | Failure / earliest check | Status |
|---|---|---|---|
| T5,N64,FP64 | Previous scoped consumer mechanics; fresh runs | Small-particle/short-horizon result may not transfer; scope explicit | Diagnostic baseline |
| Shared OT/flow/correction controls | Existing post-rebuild consumer; owner OT freeze | Inherited controls may limit attainable accuracy; require no-fire validity/parity and save margins | Comparator, not optimal |
| Terminal centered features, equal block weights | Exact risk identity, removes coordinate-unit imbalance | Centering can alter priorities; save raw feature losses too | Declared objective |
| 256 calibration draws; SD floor 1e-6 | Cheap simulation scale estimation | Degenerate scales; fail on nonfinite and record floor activations | Hypothesis |
| Pooled/time-specific S/I adaptation | SIR shared coefficients and distinct time roles | Misses region-specific shape changes; capacity comparison | Hypothesis |
| Finite differences/backtracking | Full executor remains unchanged; no mixed derivatives required | Weak/noisy descent; synthetic optimum and perturbation diagnostics | Tested optimizer |
| Bounds and update sizes | Factors 1/4..4, translations within two process SD | Optimum on boundary or unresolved fine steps; bound hits and planned small-step repair | Hypothesis |
| 64 train/16 validation/64 test | Dataset-level replication within budget | Validation selection or weak test intervals; two optimizer seeds and simultaneous intervals | Bounded study |
| Message fit 512 particles/32 kernels | Existing observation-only adapter | Poor approximation; paired original guide and training-loss progress | Warm start only |

A successful optimizer test does not prove scientific improvement. A lower
training loss can overfit latent labels, so only independent datasets matter.
The guide is applied before LEDH; moment optimization can adapt that composition
within its family but does not prove that family can represent a useful guide.
Unknown exact moments do not invalidate simulation risk, but noisy labels can
make the study inconclusive. RBF adaptation is not original neural KDM training.

## Skeptical plan review before execution

Reviewed 20 October 2026 against the user request and actual pilot call chain.
REVISED the initial temptation to supervise all time steps: future-informed
guides make that a different conditioning target. Terminal-only loss fixes it.
REVISED finite-bank selection into continuous coefficient optimization, with
an executable update test and logged parameter changes. Score labels are
excluded from training/selection. No old result is reused as a baseline.
Fair comparisons share source executor, controls, datasets and particle streams.
Untrained and shrinkage adversaries prevent a favorable complex-only comparison.
Separate heldout data, explicit compute caps and numerical stop conditions are
present. Source fidelity to neural KDM, covariance MSE, HMC and broad score
improvement are not claimed. PASS for this bounded moment-training study;
independent external review is not claimed.

## Execution order and deliverables

1. Implement coefficient adapters, calibrated moment loss and real optimizer;
   tests must check identity, known optimum, label isolation and call-chain parity.
2. Implement one GPU runner using the existing shared executor; run bounded
   smoke and numerical checks before the campaign. Adjust only infrastructure
   after failures, preserving this objective and all evidence.
3. Run training, validation, any declared repair and untouched evaluation.
   Freeze coefficients/selection before test and preserve all optimizer history.
4. Analyze dataset-paired uncertainty, heuristic regimes, validity, raw moments
   and explanatory analytical-score labels; update the manuscript implementation
   status, master checkpoint and result note with decision/inference tables.
5. Audit that the executed optimizer and selector consumed moment loss, that
   every LEDH call resolves to the shared executor, and that reported claims
   match the heldout evidence. Build the updated PDF if the manuscript changes.

## Bounded failure localization

Campaign-01 stopped at test dataset262023017, particle274000034: the unguided
baseline failed validity. Training and selection are complete, but the partial
test must not support a ranking. The invalid row omitted its computed checks;
replay that exact baseline with unchanged sources, controls and random seeds
to retain those checks and locate the first failing time. This is an
observability diagnostic, not another training or selection attempt. Cap the
replay at 600 process seconds within the existing stage allowance, use a new
failure-replay-01 directory, and compare regenerated inputs with saved inputs.
Skeptical review: it answers the failure question without changing the target,
relaxing a threshold or discarding a failed test. Stop comparison if the failure
is genuine. Only a demonstrable harness fault permits the unchanged campaign
to resume; a numerical-method change requires a fresh scoped experiment.

## Terminal execution status

Implementation, pre-run skeptical review, training, prescribed retries,
selection, exact failure replay and terminal audit are complete. Thirty-five
focused tests pass. Heldout efficacy evaluation is INCOMPLETE: the exact
replay confirms a genuine unguided baseline nonfinite cloud at time4 of the
seventeenth test dataset. No partial-data ranking or guide promotion is issued.
The predeclared continuation veto applies. See
[result and inference tables](artifacts/ledh-moment-training-20261020/result.md).

The result does not invalidate the simulated-label identity or reject the
learned families. It identifies a common numerical failure that must be
localized before this scientific comparison can be valid. The next justified
action is a bounded examination of affine-restoration inputs/factors/tangents,
then a separate non-harm test of any numerical change. A repaired-method
comparison must refresh its scope and use fresh heldout data. Stage charge is
6041.481965510 seconds; inherited allowance remaining is 27261.957119700.
