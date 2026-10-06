# Rare-region sampling and NeuTra training master

Status: COMPLETE. All four methods, declared repairs, 16 fits and 12 eligible
HMC qualifications finished. The terminal integrity audit passed; no HMC fit
qualified. See `bayesfilter-neutra-rare-region-results-2026-10-02.md` for the
separate estimator/training/posterior decisions and remaining gaps. Owner requested
the survey, four-method executable experiment, review, execution, and persistent
narrow launch permission. This is a new bounded continuation of the existing
academic campaign, not q20 or a change of canonical architecture.

## Question and evidence contract

Can deliberate rare-region exploration estimate known small probabilities and
supply a more informative weighted training bank for the canonical IAF?
Test the existing exact two-dimensional mixture and its unit-Jacobian warped
variant. Report the broad valley |x0|<2, the much narrower |x0|<0.1, right-mode
probability, first/second moments, and pullback geometry separately.

The four arms are (1) overlapping Gaussian umbrellas with EMUS recombination,
(2) self-normalized importance sampling with a mode/bridge proposal, (3)
fixed-level splitting with constrained Metropolis mutation and full shell
reconstruction, and (4) a mixture of separately trained canonical IAFs with
global Metropolis correction, local MALA, and warm-up-only mixture-weight
adaptation. Arm 3 tests the multilevel identity, not an unimplemented adaptive
AMS algorithm or its idealized independent-conditional theorem. Arm 4 is the
frozen-after-warmup adaptation of Gabrié IV.E, not a claim to reproduce every
author experiment. Exact source anchors are in the new monograph section and
`.localresources/neutra-rare-events-20261002/reading-notes.md`.

Baseline ladder: exact iid posterior reference (sampling floor, never supplied
to a method), equal-weight Laplace/defensive proposal and corrected global/local
sampler, four proposed arms, then optional capacity/budget repair. Each method
is assessed against analytic event masses and moments, not against whichever
other arm happens to fail. Exact references and the known inverse warp are
evaluation tools; reference draws and true mode weights cannot train proposals.
Known mode locations are declared initial information, as in Gabrié's example.
There is no exhaustive mode-discovery claim.

Probability screen: independently replicated estimates, positive observations
of each required event, aggregate relative standard error <=20%, and absolute
error <= max(4 replicate standard errors, 10% of the exact event mass).
The 20% precision target is a feasibility hypothesis, not production precision;
the 10% allowance is relative to probability, never sqrt(probability). Four
standard errors are a conservative working screen, not exact small-sample,
simultaneous, or sequential coverage. Eight disjoint estimator replications
quantify between-run uncertainty. No viable-method ranking is declared.
Continuous moment screens use four replicate standard errors plus .03 times
the exact observable standard deviation (sqrt(26-25/9),1,sqrt(102),sqrt(2));
this inherited continuous-moment allowance is never applied to event masses.

The numerical validity/finite-weight/invertibility checks veto promotion.
Failed probability precision or geometry is a repair trigger and promotion
veto, not a continuation veto for the other methods. Missing/corrupt artifacts,
an incorrect target or correction formula, noncompliant GPU allocation, a
failed invariant regression, exhausted total budget, or unresolved shared
implementation error stops dependent work. A failed optional method does not
invalidate the target or reject the research direction.

Training and sampler questions stay separate. Weighted FKL followed by RKL is
tested using the existing shared IAF authority; preserve every checkpoint and
standard 1000-point report. Add deterministic physical valley probes to expose
regions random Gaussian probes miss. Geometry and heldout KL are explanatory;
actual HMC uses the public fixed-transport tuner, identity latent mass, shared
sequential controller, and v4 posterior screen. A finite geometry result or
successful probability estimator does not establish HMC readiness. A global
proposal that mixes modes but misses the center has not solved rare estimation.

The inherited v4 HMC profile includes moments, mode occupancy, fixed marginal
CDF cuts, and the broad valley `abs(x0)<2`. It does not require ordinary HMC
draws to resolve the 3.09e-7 central event `abs(x0)<.1`. That much rarer event
belongs to the deliberate-estimator screen and weighted-bank diagnostics.
Neither a v4 HMC pass nor its finite 1000-point probe would establish precise
ordinary-HMC estimation of that central event.

## Numerical choices and assumptions

| Choice | Provenance and purpose | Failure check / repair |
|---|---|---|
| Two 2D targets | Existing author-derived benchmark plus declared warp | Exact analytic masses; independent reference data |
| Events 2 and .1 | Existing valley and explicitly illustrative central interval from the survey | Report both; no post-hoc event selection |
| Eight replication seeds | Disjoint fixed identifiers, limited-cost uncertainty evidence | No superiority claim; twofold work repair if imprecise |
| Umbrella sigma .2 | For unit-variance means +/-5, log marginal curvature <=24; bias precision25 makes each unwarped window strongly log-concave | Warped conditional geometry still checked; refine spacing/burn-in |
| Centers -8 to8, spacing .4; extra center sigma .05 | Neighbor spacing2 sigma, covers modes plus3 local standard deviations; positive biases preserve support beyond range; narrow window resolves .1 event | Overlap connectivity, positive normalizers, linear residual and independent replications; spacing .2 repair |
| Four walkers/window; burn512/2048, retained1024/4096 | Convenience budget ladder; not convergence by count | Per-window movement/replication check; longer burn and trace in repair |
| Physical MALA dt .002/.01/.05; umbrella .0005/.002/.005 | Physical-coordinate curvature hypotheses; the narrow bias has precision400 | Independent pilot finite movement and >=.5 acceptance; no retuning on final replications |
| IS local/bridge/defensive mixture | Equal mode weights, Laplace Hessians; bridge centers follow known valley, never exact masses | Pilot estimate of event-specific variance over bridge weights .2/.5; final fresh seeds |
| Split levels4,3,2,1,.5,.2,.1 | Explicit nested geometric approach to the two fixed events | Survival, ancestor counts and finite mutations; increase particles/mutation on failure |
| Particles4096/8192; mutations32/128 | Bounded feasibility and repair hypotheses | Independent replication, shell mass conservation; no ideal-conditional unbiasedness claim |
| Canonical IAF widths16/32, learning rates .0003/.001 | Author architecture, existing width range plus optional capacity extension; rates are calibration hypotheses | Target-specific crossed pilot, disjoint validation, clipping/update checks |
| Batch256, FKL256/1024, RKL256/1024 | Existing batch-native protocol, affordable diagnostic ladder | Preserve all distinct checkpoints; continuing improvement at cap means undertrained |
| FP64 analytic benchmark and transport reference mode | Inherited checked reference benchmark to isolate rare geometry, not a new TF32 default | Record dtype/TF32; no production throughput claim |
| Standard1000 probes plus fixed valley line | User's standard probe and directed explanatory diagnostic | No finite cutoff promoted to readiness criterion |
| CPU sample workers2, threads2 each | Owner CPU sample-generation policy, resource sharing | Actual worker CPU charge, no GPU claims from CPU artifacts |
| GPU1, growth, TF/XLA | Existing campaign assignment, verified at launch | Fail closed if unavailable/busy or memory policy invalid |
| HMC initial epsilon.5, L3/9/18, bounded public search | Inherited starting hypotheses; tuner measures/repairs epsilon per L | Fresh tuning per map; no hand-issued tuning artifacts |

Implementation-specific hypotheses, frozen before launch: the extra narrow
umbrella has sigma.05 (precision400), so its MALA pilot is .0005/.002/.005,
around the inverse precision; .01/.05 are excluded for that window. Gaussian
IS bridge scales1/.08 are broad-valley/narrow-event hypotheses, with .05
Student-t5 defensive mass and bridge mass .2/.5; nomination minimizes the
pilot's estimated maximum relative SE, without analytic probabilities. Local
training uses32 walkers,512 burn and256 retained transitions, with a final
temporal block for local validation; these are approximate local banks, not
exact conditional samples. Split initialization uses32/64 global/local steps;
the reflected proposal scales are .5 min(bound,1) and .75, symmetric in the
truncated region. Source/invariance tests precede use. Local mixture logits
use gradient step.01 during512/2048 warm-up transitions only. The IAF initializer
variance scale.2 is a prior same-target repair hypothesis; canonical masks,
stages, ELU, permutation and conditional-cap semantics remain unchanged.
Clipping is five times the maximum of eight actual stratified-gradient pilot
norms; observed clipping is recorded rather than assumed harmless. Directed
geometry uses1001 evenly spaced physical valley points, a finite diagnostic
resolution with no uniform-error claim. Numerical roundtrip/EMUS residual
screens1e-8 and shell conservation1e-10 are FP64 reference tolerances, checked
against independent algebra and corruption fixtures. Two local seeds and eight
estimator seeds are fixed stream identifiers, not tuned numerical parameters.

## Executable phases and refresh

`bash /home/ubuntu/python/BayesFilter/scripts/run_neutra_rare_region_campaign.sh run`
is the fixed launch command. The same wrapper accepts `status`, `check`, and
`resume`; it sets the interpreter and environment itself. It cannot execute
arbitrary supplied shell commands. A narrow prefix permission covers these
subcommands. The managed permission layer remains authoritative.

1. Freeze configuration/source and run focused correction/invariant checks.
2. CPU preparation: disjoint references, local training banks, Laplace baseline,
   pilot calibration, deterministic analytic checks.
3. GPU local IAF fits for the mixture-proposal arm; retain hyperparameter pilots
   and select by heldout FKL solely for proposal nomination.
4. CPU four-method replications; preserve samples, weights, overlap/mutation
   diagnostics, elapsed time, exact commands and seeds. Methods run independently.
5. Refresh the phase plan. Execute one predeclared numerical/budget repair of
   a failed arm. Preserve failed attempts. Do not repeat the identical candidate
   and call it a repair. Infrastructure failures receive a fresh output path and
   a focused correction before retry; unknown errors stop dependent work.
6. Train canonical global IAFs from each method's weighted bank, two training
   seeds, crossed width/LR pilot, FKL/RKL checkpoints and standard/directed probes.
   A failed teacher remains diagnostic-only. Weighted bank sampling must not
   erase the rare strata: use stratified allocation with exact importance
   correction inside the minibatch rather than resampling by posterior weights.
7. Fresh bounded public HMC qualification for numerically valid maps from
   teachers that pass the probability screen; otherwise record the unmet
   prerequisite. Final reference is not used to select checkpoints.
8. Terminal result review, source/artifact audit, manuscript build, budget
   reconciliation and reset memo. State separately which methods, training maps,
   and posterior checks passed and what remains uncertain.

Every finished phase writes `next-phase.json` before launching its successor.
Controller checkpoints include completed jobs, failures, repairs, next action,
and remaining budget. Resume consumes preserved successful outputs, detects
live workers, and never overwrites evidence. Numerical workers use the frozen
source; an infrastructure repair gets a new source revision and attempt.

Output root: `docs/plans/artifacts/neutra-rare-region-2026-10-02/campaign-r1`.
Sub-budget: 7200 GPU process-seconds and14400 CPU core-seconds, taken from the
last reconciled shared remainder81480.811/170544.791. These are ceilings,
not runtime forecasts. Recheck the shared ledger before launch. Each worker
has a900-second wall ceiling, HMC240 seconds; concurrency charges summed
worker time. Two independent CPU workers maximum and one GPU worker maximum.
At most two scientific settings per method/target and two infrastructure
attempts per job. No extension or package/environment mutation is authorized.

## Skeptical review before implementation

The first draft would have pooled oversampled points without preserving their
measure and would have treated a mixture-proposal sampler as a rare-probability
estimator. Both are wrong. This plan requires explicit weighted measures,
shell conservation, and separate mode/rare-event outcomes. It also avoids the
previous sqrt(p) tolerance, does not transfer exact-sampling confidence claims
to MCMC, does not assume training coverage controls derivatives, and does not
use a low KL checkpoint as a posterior promotion criterion. Known coordinates,
analytic targets, limited seeds, inherited IAF settings, and approximate split
conditionals are explicit limits. The bounded first comparison can reject a
candidate; it cannot rank viable methods or validate a production default.

The unit suite includes a two-update, batch8, width4 CPU-hidden training
mechanics exception solely to check stratified weights and graph wiring; it
cannot support training-quality or GPU conclusions.

Implementation review must exercise wrong-weight, disconnected-window,
missing-event, biased-proposal-correction, budget/retry, and resume cases.
Review passes for implementation under these conditions. Final execution
review must record actual tests and source-to-equation checks before launch.

## Execution review, 2026-10-02

Fifteen focused CPU-hidden reference/mechanics tests passed, including an
independent discrete and Gaussian-quadrature EMUS check, disconnected overlap,
SNIS normalizer cancellation, asymmetric-proposal detailed balance, reflected
kernel symmetry, shell conservation, tiny stratum mass preservation, actual
two-update weighted graph execution, corrupted frozen-map rejection, missing
rare events, and controller budget/resume bookkeeping. The numerical kernels
tested with XLA enabled. A trusted readiness probe found GPU1 idle and verified
TensorFlow memory growth; GPU2 was busy and is excluded.

The code review repaired three draft gaps before launch: IS pilot selection
now uses estimated probabilities, not oracle masses; map loading uses the
shared hash-validating artifact loader; HMC now consumes the full latest-first
checkpoint shortlist and opens final reference once, rather than discarding
earlier maps. The budget dispatcher waits for in-flight reservations to clear
instead of falsely declaring exhausted budget. Nonfinite loss means candidate
rejection, never a successful repaired map. The method screen also checks the
four continuous moments; local sampler bank adequacy remains an empirical risk.

MathDevMCP's document audit indexed the argument but marked12 semantic
obligations not checkable; that is not proof of correctness. Its SymPy route
checked shell telescoping, quantile-score cancellation, and event-optimal
allocation algebra. Manual review checked density normalizers, proposal
orientation, weights and the assumptions behind the quoted theorems against
the inspected sources. The tests provide independent finite checks, not
universal equivalence. The master is approved by this skeptical self-review
for the bounded experiment, with the stated limitations and repair branches.

## Terminal review scope

The cheap adversaries are the implemented exact CDF calculation (an oracle
available only for these targets), exact iid posterior sampling at 4096 draws,
and equal-weight local Laplace importance sampling with a 5% Student defensive
component. Their rationale is respectively to expose deterministic probability
error, the passive-sampling floor, and whether learned or elaborate exploration
is needed beyond known-mode Gaussian approximations. Report each on the broad
valley, narrow valley, and mode probability for each target; do not pool away
the narrow-event failure. These are evaluation controls, never training data.
The exact reference is not a general-purpose competitor for an unknown
posterior. A finite experiment cannot certify dominance of the viable methods;
the terminal machine-readable verdict must therefore retain that limitation
and forbid default promotion.

The review will verify frozen source hashes, recorded input hashes, required
outputs, 1000-point probe identities, device allocation, and agreement between
the local and shared resource ledgers. It will report missing HMC prerequisites
and failed candidates in their planned denominators. These are read-only
checks of the existing evidence, not additional training or altered screens.
Each teacher bank is the preselected first replication. An eight-replication
probability screen does not certify that individual bank's accuracy; its
own probabilities must be reported alongside the aggregate.
