# NAF approximate-forward-KL to reverse-KL master

Status: completed, including the October 6 owner-authorized warm-start criterion
repair and all 12 reserved random cases. All randomized teacher/warm-start/final
screens pass; the master is idle. See the results note and terminal r2 audit
for evidence and limits. NAF is the default
transport for this work. The active criterion is
`bayesfilter_neutra_forward_warm_start_then_final_v2`; the exact revision and
skeptical review are in
`bayesfilter-neutra-forward-warm-start-criterion-repair-2026-10-06.md`.
This plan supersedes the old scientific master's IAF repair/resume sequence.

## Research intent and evidence contract

Question: can an approximate posterior teacher initialize a NAF with adequate
coverage, and can pure reverse-KL training preserve that coverage on fixed
unwarped/warped mixtures and then unseen randomized two/three-mode geometries?
The intervention is posterior-teacher weighted forward cross entropy followed
by exact-target reverse KL. The frozen forward endpoint is the primary
comparator. Raw proposal importance sampling is the naive teacher comparator;
annealed AIS and annealed SMC are the first complete teacher candidates. The
existing six-method inventory remains visible: FAB is blocked by unresolved
auxiliary-tail eligibility; Gabrié's current frozen-map control is not its
adaptive training algorithm; AFT/CRAFT controller equivalence remains unchecked.
These three implementation gaps must not be recorded as algorithm failures.

The primary pass criterion is a complete, finite forward and reverse training
pair whose independent teacher, forward warm-start coverage/validity screen,
and final RKL full distribution screen pass. The intermediate feature-z screen
is explanatory; the old requirement that both maps pass full accuracy is retired
by the owner's explicit criterion change. Saved old outcomes remain preserved.
Pure RKL is an experiment, not an assumption that reverse loss must improve
coverage. Save and retain both maps even if the later one fails. Numerical
invalidity, missing provenance/probes, wrong objective, or leakage of exact
target samples into native training is a continuation veto. A failed teacher
or fitted map is a candidate rejection/repair trigger: continue the declared
calibration ladder, not the rejected candidate's dependent promotion.
The existing 1,000-point latent score probe is explanatory unless nonfinite
or incomplete. KL estimates, clipping, acceptance, ancestry and timing explain
failure; none establish posterior convergence. No q20, HMC, all-mode discovery,
TF32 readiness or universal method ranking follows.

## Default and source boundary

Use the existing `NeuTraTransportConfig.huang_dsf` with `author_cmade` through
the single `neutra_transport_core` authority. Three autoregressive stages,
two width-64 ELU hidden layers, four sigmoid components, existing source
initialization/positivity/inversion are the initial study configuration. The
owner explicitly promotes this architecture for this work; its numerical
recipe remains target-specific. Preserve explicit canonical IAF baselines and
old snapshots. No generic deserialization/API default is silently changed.

The completed attribution study and source anchors are recorded in
`bayesfilter-neutra-nonlinearity-attribution-results-2026-10-06.md` and its plan.
Native teachers reuse `discover_modes`, `LaplaceMixtureProposal`, and
`AnnealedSMC`. Mode discovery sees only density/score/Hessian/dimension, never
true centers, component count or reference samples. The local Laplace mixture
has normalized full-support defensive tails; importance/annealing corrections
use its actual density. This is the documented local combination, not a claim
of reproducing a full author's controller. A discovered mode list is not an
exhaustiveness proof.

## Executable phase order

1. Focused CPU mechanics: NAF default wiring, weighted training objective,
   reverse objective, independent banks, forward-map restore, complete probes,
   failed-candidate continuation, terminal resume and budget refusal. A tiny
   CPU/non-XLA Gaussian smoke is only implementation validation.
2. Price native teacher generation on CPU and forward/reverse fitting on GPU
   separately before reserving full stages. Native sampling uses two concurrent
   CPU workers; TensorFlow GPU devices are intentionally hidden there. GPU NAF
   workers use XLA, verified memory growth and the existing shared budget lock.
3. On the two development geometries (9101/9201), try the bounded native teacher
   ladder. Record raw proposal IS, AIS and SMC results separately. Four whole
   populations supply training data; four independent populations assess
   teacher reproducibility and supply teacher validation. Exact reference draws
   belong only to evaluator files. Never use repeated/resampled particles as
   independent populations. Reject a failed teacher before student fitting.
4. Train NAF using normalized pooled importance weights, 8,192 forward updates
   at .001 followed by 8,192 at .0003, batch 64. Save the frozen forward map,
   optimizer and full 1,000-point diagnostic. These inherited attribution
   settings are warm-start hypotheses for approximate data; training and
   independent-teacher losses expose finite-bank overfitting.
5. From the forward map, reset Adam for pure RKL. Calibrate rates .0001/.0003
   and retained rungs 256/1,024/4,096 on development data only. Preserve each
   endpoint, all diagnostics and rejection reasons. Select the first rate/rung
   that passes the warm-start and final criteria with all declared development fitting seeds; never choose by
   reverse loss alone. Select the smallest successful teacher ladder, then
   freeze its complete recipe. Failure of all rungs is under-calibration, not
   permission to change screens or claim the direction failed.
6. With the recipe frozen, test fixed unwarped and warped targets, then the
   reserved target seeds 1103/1104/2103/2104 with fitting seeds 51/52/53. Freeze
   the entire specification catalog before launching any test. All six fixed
   cases must pass the active warm-start/final pair rule before exposing randomized holdouts.
   A failed fixed stage returns to development-protocol repair and keeps the
   randomized tests untouched. It rejects this frozen recipe, not NAF or the
   research direction. A final failure is retained as holdout evidence; do not
   retune on that target. Report both
   before/after RKL outcomes, including candidate failures. These are bounded
   transfer screens, not a powered ranking.

## Numerical choices, provenance and early diagnostics

| Choice | Value / origin | Failure mode and early check |
|---|---|---|
| Native teacher candidates | 4,096/8,192 particles per population; 32/64 bridges; 8/16 MALA moves; dt .05/.025 | Explicit calibration hypotheses, not established defaults; particle weights, mutation validity, ancestry and independent population feature checks |
| Mode preparation | 64/128 broad starts, 200/400 L-BFGS iterations; score tolerance 1e-8, squared Mahalanobis merge threshold 1e-6, curvature floor 1e-7 | Inherited diagnostic values; missing/nonstationary/singular modes reject preparation; log every start and curvature |
| Proposal | Existing local Laplace components plus 10% broad Student-t | Inherited defensive construction ensures support, not adequate mass; normalized density and raw importance baseline screen |
| Broad starts and defensive component | Multivariate Student-t with df=5 and scale 4, centered at zero | Inherited full-support hypothesis from `BroadStudentProposal`; missing remote modes remains possible and independent teacher/evaluator checks must reject inadequate coverage |
| Annealing / resampling | Uniform beta grid from 0 to 1; SMC resamples at ESS < .5N; AIS does not resample; waste-free retention off | Explicit simple baseline; recorded cESS .8 is inactive with this fixed schedule. Weight collapse or mutation failure rejects the candidate; no adaptive-temperature claim |
| Teacher independent repeats | Four train plus four validation populations | Bounded exploratory uncertainty; whole-population variance and between-bank agreement; four replicas cannot certify equilibrium |
| Forward objective | -sum normalized teacher weights times log q | Finite empirical target is approximate; validate on independent teacher populations and keep exact evaluator samples out of training |
| Forward schedule | 8,192 at .001 + 8,192 at .0003, fresh Adam at switch; batch 64 | Measured exact-teacher recipe; overfitting/plateau on approximate banks remains a tested hypothesis |
| RKL | Rates .0001/.0003, rungs 256/1,024/4,096, independent latent noise | Target-specific calibration; coverage collapse vetoes that endpoint even when RKL falls |
| Adam / clipping | .9/.999/1e-8; emergency norm cap 1,000 | Attribution used this cap rarely; retain counts/norms, stop numerical-invalid training; majority clipping triggers repair |
| Precision | FP64 diagnostic exception, GPU/XLA | Attribution's numerical reference setting; does not establish TF32 readiness |
| Map assessment | 32,768 reference rows, 2,048 map draws, cross entropy minus identity-Gaussian cross entropy<=.25 nats, feature z<=5, mass error<=.15, complete finite 1,000-point probe | Inherited exploratory screens from `freeze_and_assess`; preserve all rejections and uncertainty, no convergence claim |
| Forward warm-start eligibility (v2) | Reloaded finite map, complete finite probe, identity density guard, mass error<=.15 and each component at least .5 times reference mass; feature z is explanatory | Owner-authorized role change; .5 is an explicit factor-two undercoverage heuristic, not an accuracy/convergence proof; final full screen stays unchanged |
| Teacher assessment | ESS fraction>=.20, feature z<=5, mass error<=.15 | Weight ESS alone misses undiscovered modes; exact features used only by evaluator |
| Target geometry | Uniform pair distances [6,10], variances [.5,2], random rotation/translation, component masses >=.1 | Existing frozen benchmark family; no arbitrary high-dimensional generalization |

The .0001 RKL rate is an explicit lower-step calibration hypothesis relative
to .0003; the longer rungs test whether the old 256-step canary was adequate.
Training-bank particles are approximate data, never relabeled exact draws.
Training reuse is declared and monitored rather than confused with fresh
population gradients. All choices remain in manifests.

## Resources, commands, review and stop conditions

The last settled ledger has 38,185.09 GPU-process seconds and 33,267.18
CPU-core seconds remaining. This is a ceiling, not a forecast. Use the existing
shared campaign accountant and fresh `forward-reverse-*-rN` output directories.
Do not promise the entire multi-method matrix fits. Measured full-stage
reservations include teacher preparation, fitting, RKL branches and assessment;
stop as under-budgeted before a partial final stage if its reservation cannot
fit. Per-worker timeouts are resource controls, not scientific failures.
At most two infrastructure attempts per job/source; preserve and charge both.

The first full calibration pair supplies pricing. Initial engineering ceilings
are 600 wall seconds / 1,200 CPU-core seconds per teacher and 2,400 wall seconds /
3,000 CPU-core seconds per fit. These are bounded timing hypotheses, informed
by the attribution fits for the GPU side, not numerical eligibility criteria.
Subsequent reservations use the maximum measured cost times 1.5 (a convenience
margin, not a probabilistic bound). Teacher pricing is specific to its profile:
doubling particles, bridges and mutation moves must not inherit the smaller
profile's timeout. A previously unmeasured profile retains its initial ceiling
until its own complete worker has been measured. If the ceiling fails, preserve
the timeout as resource evidence and revise pricing before scientific inference.

Final call-chain audit identified recovery and dependency risks before calibration:
an interrupted smoke could leave generic `running` state and resume the legacy
sequence, phase-wide teacher timing could underprice the larger ladder rung,
and a failed fixed stage could unnecessarily expose randomized holdouts.
Persist smoke progress independently of individual workers, and price each
teacher profile separately. Require the fixed stage to pass before randomized
testing. Focused recovery, reservation and holdout-preservation tests must pass.

The fixed wrapper gains `forward-reverse-plan` (readiness/queue only) and
`forward-reverse` (execute/resume). Normal `resume` follows an active
forward/reverse state instead of re-entering historical IAF repair. Refresh
next-phase status after every completed worker; completed phases do not rerun.

Skeptical pre-implementation audit: the old master has an IAF constructor,
4,096-point repeated exact-data bank, 256-update RKL, no native CPU generation
lane and a historical resume path. Reusing that command would not answer the
new question. The refreshed plan separates native teacher error, forward fit
error and RKL coverage loss; freezes independent target geometries; retains
all failed candidates; and distinguishes architecture policy from evidence of
learned-map quality. The implementation must pass call-chain and recovery
tests before reporting executable readiness. Price/calibration remain pending
scientific work, not reasons to block routine implementation.

## Readiness validation, October 6

Implementation validation is complete. The initial focused suite passed 42
checks; the native CPU tracing repair passed seven checks; the final queue and
recovery suite passed 22 checks. These runs overlap and must not be summed as
distinct tests. Final tests cover fixed-before-random ordering, complete-stage
budget refusal, larger-profile timing isolation, failed-candidate continuation,
terminal idle resume, smoke recovery and pending-charge settlement.

The first bounded smoke failed during concurrent TensorFlow derivative graph
construction before GPU fitting. The repaired route traces one immutable
sampling kernel before two CPU threads execute independent seeded populations.
The native CPU retry completed in 12.83 wall / 27.61 CPU-core seconds. The GPU
fit smoke completed in 154.37 wall / 192.76 CPU-core seconds, including XLA
compilation and assessment. All four tiny training blocks were batched GPU/XLA;
all three saved endpoints reloaded and had complete finite 1,000-point probes.
Verified memory growth was enabled before GPU initialization. These are
mechanics checks, not calibration or training-quality evidence.

The monograph compiled to 654 pages; the rendered proof and result table were
inspected. New section: `docs/chapters/ch26f_neutra_nonlinearity_results.tex`.
Build and PDF: `docs/plans/artifacts/neutra-naf-forward-reverse-2026-10-06/monograph/`.

The settled remaining allocation is now 38,030.72 GPU-process and 32,834.95
CPU-core seconds. Full calibration has not launched. The program is ready to
start measured calibration, subject to normal GPU tool permissions and the
declared budget; the entire subsequent campaign is not yet priced. Saved
readiness: `bayesfilter-neutra-naf-forward-reverse-readiness-2026-10-06.md`.

## Execution authorization and skeptical review, October 6

The owner requested another 24 GPU hours and 24 CPU hours and instructed a
thorough program review followed by execution. Add 86,400 seconds to each
existing cap without resetting prior charges. New local cumulative caps are
178,800 GPU-process seconds and 184,800 CPU-core seconds. Before subsequent
checks the remaining allocation is 124,430.72 GPU-process seconds (34.56 hours)
and 119,234.95 CPU-core seconds (33.12 hours). The corresponding shared cap is
incremented once under the existing master lock; the allocation record preserves
the owner's wording, previous limits and arithmetic.

Pre-execution review traced proposal construction and native weight updates,
independent complete-population uncertainty, CPU sampling/GPU fitting boundaries,
the categorical estimator of the finite-bank forward gradient, the exact-target
reverse objective, Adam reset and parent-map restore, frozen-map reload and
all screens. The .25-nat identity-Gaussian cross-entropy-difference screen was present in
code but missing from the numerical table. The first review note mislabeled
it as forward KL; tracing `evaluate_student` corrected that description. It
is a weak baseline veto, not an absolute density-accuracy guarantee. Actual
forward KL is recorded separately as an explanatory estimate. No threshold changed.
The reviewer also checked shared/local cumulative accounting, worker limits,
failed-candidate continuation, larger-profile pricing, idle terminal recovery
and the fixed-stage prerequisite for exposing randomized geometries.

The review passes for the stated exploratory training question. Weakest
assumptions remain the finite teacher ladder and four-population uncertainty,
inherited forward schedule on approximate data, and the possibility of RKL
coverage collapse. The explicit screens and saved forward comparator expose
these failures; passing them does not establish HMC or posterior validity.
First-pair timing is still required before stage reservations. Failed candidate
screens continue the declared ladder. Local infrastructure failures may be
repaired and retried within the unchanged scientific contract and enlarged
budget; preserve each attempt and run a focused regression where applicable.

Execute `bash scripts/run_neutra_scientific_campaign.sh forward-reverse` with
trusted GPU permissions, physical GPU 2 if still available, verified memory
growth, CPU-native teacher populations and the existing FP64 diagnostic exception.
Monitor saved training progress and phase state; do not overwrite evidence or
interrupt unrelated jobs. Record results in
`bayesfilter-neutra-naf-forward-reverse-results-2026-10-06.md`.

## Completed continuation

The revised intermediate rule was implemented and checked without changing the
training kernels or final thresholds. All six fixed pairs were reassessed
retrospectively with their original evidence preserved, then all 12 untouched
random cases ran with the frozen recipe and passed. The final r2 audit verifies
the evidence and zero-worker terminal resume. No further worker is pending in
this plan. Detailed outcomes, remaining large tail score residuals, cumulative
costs and the separate downstream-validation question are recorded in the
results note above.
