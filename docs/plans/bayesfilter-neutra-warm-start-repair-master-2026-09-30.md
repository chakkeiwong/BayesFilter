# Executable repair of NeuTra initialization and validation

Terminal checkpoint: the 30-case matrix finished at 11:26:11 Asia/Shanghai on
September 30. Twelve cases passed the final checks; unwarped mixture and funnel
have no passing cases. No worker is active. The terminal evidence review,
remaining budget and next discriminating work are recorded in
`bayesfilter-neutra-warm-start-repair-results-2026-09-30.md`. Matrix completion
does not mean reliable multimodal training has been achieved.

October 1 audit follow-up: the prior launch reviews do not close the remaining
initialization, effective HMC search, training-continuation, mutation-repair or
temporal-diagnostic findings. The
[audit retrospective](bayesfilter-neutra-audit-retrospective-2026-10-01.md)
records the specific plan/code/test mismatches and executable evidence required
when repairing them. These checks remain outstanding; no numerical policy or
candidate status changes merely because they are documented.

## Authorization and research question

The owner requested a reviewed executable master, automatic repairs and a
refreshed next-phase plan between phases, minimized repeated approvals, and
execution. This continues the existing simpler-target campaign allocation;
the September 30 budget amendment below records the owner's additional 48
hours. It does not authorize q20, another backend/flow architecture,
publication or package changes.

Question: can a separately assessed sample generator provide a useful
multimodal initialization of the canonical IAF, and which preserved stage
remains useful for corrected posterior sampling? Teacher accuracy, flow fit,
RKL refinement and final HMC are separate questions.

The prior study and audit are preserved. The active plan is this document;
historical completion flags do not establish that these phases passed.

## Executable phases and repairs

1. Repair failure handling, source/input/config-sensitive resume and cumulative
   pending-work tracking in the existing master. Test these with deterministic
   fixtures before a worker launch.
2. Prepare fresh development samples and blinded multistart mode discovery.
   Fit a normalized Laplace Gaussian mixture using local negative Hessians and
   local mass approximations, retaining a broad Student component. This is an
   explicitly approximate proposal, not a posterior estimate. Verify density,
   sampling and importance correction. A failed curvature or discovery check
   triggers a larger independent search, never silent oracle substitution.
3. Generate ordinary-SMC teachers from that proposal. Assess independent
   populations against benchmark truth using region, shape, moments and tails.
   Calibrate mutation on representative bridge populations rather than only
   four steps around the final modes. Increase particles or mutation separately
   when the corresponding evidence fails. Archive particles, weights and
   ancestry. Also test Gabrié's concurrent training path, preserving walkers.
4. Calibrate IAF fitting on the actual teacher and objective with a bounded
   width/LR/update ladder. Compare with a simple fitted Gaussian and check
   physical-space shape. Keep each map and optimizer checkpoint. A Gaussian
   plateau triggers a capacity/initialization diagnostic; finite loss alone
   never admits a fit. Failed candidates remain recorded evidence.
5. Validate the warm map. For Gabrié, freeze it and assess fresh sampler
   trajectories from deliberately different mode allocations. Keep flow
   approximation and corrected-sampler evidence separate. A teacher or sampler
   failure triggers its own repair and cannot be repaired by reporting a good
   flow loss.
6. Attempt classic RKL only on an eligible warm fit; check intermediate maps
   and preserve the warm map when RKL damages coverage/shape. Do not require
   RKL success to use a valid warm initializer. Qualify eligible warm and RKL
   candidates through the existing fixed-transport public tuner and shared
   sequential HMC controller, with identity latent mass.
7. Freeze candidate choice using development checks. Generate a new final
   reference bank and fresh sampling streams only afterward. The final holdout
   cannot drive a retry against the same bank. A failed final check returns to
   development and consumes a fresh bank for another bounded attempt.
8. After every phase, write status, evidence, failure class, remaining budget,
   repair decision and the exact next phase/command atomically. Expected
   candidate failures continue to the declared repair or other targets.
   Infrastructure failures receive bounded local retries; unresolved harness
   defects stop dependent work with a concrete repair record. Completion means
   the declared work is resolved, not that every method succeeded.

The first executable matrix is Gaussian, unwarped mixture and warped mixture,
then wiggle and funnel when their preparation/reference checks pass. Ordinary
SMC and Gabrié are the primary sample generators. AFT/CRAFT are quarantined from
new claims until their objective-specific calibration and full state archival
are implemented; they cannot silently inherit the ordinary forward-fit clip.
This quarantine closes accidental reuse, not their open scientific questions.

## Evidence contract

Primary warm-start criterion: agreement of independently generated teacher
clouds and learned-map draws with declared coarse region/shape tolerances,
plus meaningful nonlinear fit beyond a moment-matched Gaussian when the
target is non-Gaussian. Teacher replications assess finite-population
uncertainty; correlated walkers are never counted as iid samples.

Primary final criterion: existing sequential HMC numerical health, modern
R-hat, bulk/tail ESS, Monte Carlo precision and agreement with a fresh target
reference for the declared estimands. A finite map, mode count, acceptance,
weight ESS, forward loss or 1000-point score residual alone cannot establish
posterior correctness or promote a default.

Invalid target/derivatives, broken density/weight identities, nonfinite states,
corrupt artifacts, absent reference, invalid memory policy and exhausted total
budget veto dependent execution. Poor candidate fit, missed mass, or RKL
collapse rejects that candidate and triggers the corresponding repair. Proxy
diagnostics never become an unannounced reason to abandon the research.

## Numerical choices and provenance

| Choice | Provenance and purpose | Failure check / status |
|---|---|---|
| Existing canonical IAF, FP64, TF/TFP, GPU/XLA | Preserve numerical authority and isolate pipeline repairs | No precision/architecture promotion; configured widths remain candidates |
| Local covariance = inverse negative log-target Hessian | Second-order Laplace expansion at each checked mode | Positive curvature and finite density; not a global Gaussianity claim |
| Local weights proportional to target(mode)/sqrt(det H) | Derived Laplace mass approximation, common (2 pi) factor cancels | Importance correction and independent teacher checks; never optimizer hit frequencies |
| 10% broad Student support component | Convenience hypothesis protecting tails and missed basins | Weight concentration, region/shape checks; report its approximation status |
| Search 64 then 256 starts | Initial count inherited; fourfold independent repair is a bounded work hypothesis | Benchmark mode count/score/curvature checks; no unknown-target exhaustiveness claim |
| Teacher populations 1024 → 4096 → 8192, three seeds | Inherited first two scales; final bounded extension and replication | Independent-population variation; cannot infer iid precision from particle count |
| Mutation 4 → 16 steps; pilot grid 0.001/0.01/0.1 | Inherited hypotheses, now tested on bridge populations; factor-four repair isolates movement | No usable step means explicit failure; acceptance never proves mixing |
| Width 8/16 (2D), 20/40 (funnel); LR .001/.003 | Existing reviewed grid, no new architecture | Actual teacher/objective validation and Gaussian comparator |
| Batch 256, training rungs 256/1024/2048/4096/8192 | Existing batch and budget ladder with bounded extensions | Shape/coverage and loss uncertainty; stop or repair a plateau rather than merely spending the cap |
| Coarse probability error 0.05; valley error 0.01 | Warm-start accuracy hypotheses, not posterior precision | Fixed physical bins catch broad-Gaussian false positives; statistical uncertainty reported separately |
| Continuous moment error 0.1 reference SD | Coarse training target; must include uncertainty | Heavy-tail uncertainty prevents a success claim, not automatic target rejection |
| Nonlinear gain over Gaussian: 5% of reference Gaussian FKL, with 3 SE evidence | Small mechanism screen against the observed Gaussian plateau; hypothesis, not optimality | Independent development sample loss differences; Gaussian target has a separate no-regression check |
| Three training seeds 11/23/37; fresh stage-specific streams | Existing replications, distinct role salts | No tuning of seeds or ranking without adequate uncertainty |
| Existing total ceilings CPU 122400, GPU 64800 seconds | Previously authorized allocation; subtract every prior attempt and conservative unknown charge | No new allocation; a new directory does not reset cost |
| Per-job wall cap initially 600 seconds | Conservative engineering ceiling above prior 40–240 s jobs; not a convergence criterion | Price actual phase; bounded extension within remaining allowance for infrastructure timeout |

The first matrix remains a research diagnostic, not a claim of reliable
training on every target. Failed finite candidates must have an explicit
terminal status and a next discriminating experiment.

## Skeptical review before implementation

The old plan confused coarse coverage with learned shape and used final
confirmation for adaptive repair. This plan separates those quantities, uses
simple Gaussian and exact-target controls, preserves warm maps, and generates
final references only after selection. Mode-based proposals remain normalized
and importance-corrected; discovered centers are not posterior samples.
Preserving the author's IAF avoids an unreviewed architecture substitution.

The full upstream Gabrié training controller is not currently replicated.
The active branch is explicitly the tested global-MH/local-MALA core with
canonical IAF and our recorded training controller. It cannot claim full
author-program equivalence. Its frozen sampling validity is assessed directly.

Potential misleading success: a teacher hits both modes but assigns wrong
mass; a map matches masses but fills the valley; RKL improves its own loss but
loses a mode; a final test is repeatedly reused; a stale job is skipped after
repair. The first phases and regression tests explicitly target each case.
Potential misleading failure: too few particles, unsuitable bridge mutation,
near-Gaussian optimizer stagnation, or insufficient phase time. Each has a
separate bounded repair. No reviewed threshold is silently relaxed.

Pre-implementation audit passes for this bounded scope. Independent-agent
review is not required; the implementation receives a second skeptical review
and focused tests before campaign execution.

## Permissions and launch boundary

Use one repository-owned shell launcher with fixed interpreter, campaign root,
memory-growth setting and command set. Request a persistent approval matching
its exact absolute path, not arbitrary bash/python. Status and local retries
use the same launcher. Install only that rule after the script is concrete
and inspected. Local rules do not override managed-platform policy. No
credentials, gateway changes, network campaigns or unrelated commands are
part of this approval.

Official Codex rules documentation returned HTTP 403 in both normal and trusted
fetches. Installed local rule examples and the installed policy checker will
be used to verify the narrow rule; no claim is made that local allow overrides
the managed reviewer.

## Execution record

Implementation now connects the existing budgeted worker to seven explicit
repair phases. The master writes `repair-next-phase.json` and Markdown after
every phase/decision. The launcher fixes the repository, interpreter and shared
ledger and accepts only start/status/check/configure. AFT/CRAFT remain explicitly
quarantined from this new campaign; their full repair is not falsely reported
as completed.

The focused suite passed 54 tests in 33.24 seconds (CPU-only reference/smoke
exception, GPUs hidden). It includes the three reported controller defects,
the normalized defensive proposal density, rejection of a moment-matched
unimodal approximation, actual trainable refinement gradients, fresh-final
phase ordering, and rejection of dependent training after teacher failure.
The final post-review changes separate Gabrié from SMC failure and preserve a
1000-point probe for the selected refined map; affected tests will be rerun.

Implementation review found and repaired two additional issues before launch:
frozen maps needed an explicitly trainable copy for refinement, and a failed
SMC teacher must not block the independent Gabrié branch. A frozen sampler's
constant observable may have undefined raw split R-hat; it is ignored only as
a constant while the reference-distribution screen still applies. The frozen
Gabrié coarse screen uses raw split R-hat below 1.1 as a declared sensibility
hypothesis; final HMC continues to use the existing modern diagnostics. The
512 then 1024 discarded/retained steps per walker are bounded sampler
diagnostics, not posterior-promotion evidence. Fixed-map sampling begins
separately from each discovered mode to expose initialization dependence.

Teacher MCMC and in-graph Gabrié sample generation remain on the GPU as a
reviewed exception to external CPU sample generation, keeping their batched
target evaluations resident. Exact/reference banks and mode searches run on
CPU with GPUs hidden. Shape/fit reporting is an explicit host-side diagnostic
boundary; optimizer updates and repeated sampler kernels use stable-signature
TensorFlow/XLA graphs.

Trusted GPU readiness passed with growth enabled on host GPU 1; GPUs 0 and 2
had other compute jobs. The installed `codex execpolicy check` returned allow
for the exact staged launcher rule. This is local rule verification, not an
override of the managed reviewer. Scientific execution follows focused checks
and rule installation. No new budget has been allocated.

Launch review completed with 55 focused tests passing in 33.32 seconds. The
additional regression checks that a failed final test receives a different
reference stream for the bounded retry. One such retry widens the existing
kernel search and, if applicable, returns to the preserved warm map; it never
tunes against the consumed final reference. Final stream identifiers bind the
candidate, generator, seed and retry number.

The narrow rule was installed as
`~/.codex/rules/bayesfilter-neutra-warm-start-repair.rules`, and the exact launcher
prefix was also saved by the platform approval flow. The service launched at
02:26:26 Asia/Shanghai, September 30. First real execution: Gaussian discovery
passed, three independent 1024-particle teachers passed the coarse screen,
the selected width-8/.003 warm map passed after 1024 updates, and the fresh
final HMC check passed for the SMC seed-11 candidate. The master then advanced
to Gabrié without an approval pause. These are first-case results, not proof
of reliable multimodal training. The live state and exact subsequent decisions
are in `campaign-r1/repair-master-state.json` and `repair-next-phase.json`.

The first live repair also completed: Gaussian/Gabrié seed 11 passed its frozen
sampler assessment, but its first bounded HMC search found no verified pair.
The master automatically wrote the repair decision, widened the supported
search and generated a distinct final reference (stream seed 39821776 became
259818701). The retry passed with 2000 warm-up and 1000 retained transitions
per chain. Gaussian/SMC seed 11 also passed at those lengths. These two
finite-case results demonstrate execution, phase refresh and the bounded repair
mechanism; they do not resolve the multimodal training question. Remaining
seeds/targets continue autonomously under the unchanged shared budget.

## September 30 additional 48-hour allocation

The owner has authorized "48 hours more to conduct the work." Use the same
accounting convention as the September 29 extension: 32 CPU core-hours and
16 GPU process-hours, a total of 48 additional compute-hours. The split is
inherited from the measured approximately 2:1 CPU/GPU cost ratio documented in
`bayesfilter-neutra-warm-start-continuation-2026-09-29.md`; it is an accounting
allocation, not a runtime prediction or a scientific tuning choice. Cumulative
ceilings become 237,600 CPU core-seconds and 122,400 GPU process-seconds.
Previously charged attempts and the conservative uncertain CPU charge remain.

The active controller caches its configuration. A one-time budget handover
therefore waits for its existing master lock rather than interrupting a worker
or changing source used by the current numerical jobs. At that boundary it
updates only the resource ceilings and authorization text. If the controller
stopped for exhausted budget, the handover resumes the same repair controller
with its saved attempts and pending work. If the matrix completed, or a
different continuation veto fired, it records the available reserve without
starting more work. While waiting, the current controller continues under the
previous, tighter ceilings. The additional allocation is already authorized;
the delayed application prevents an in-memory configuration overwrite.

Skeptical audit: increasing ceilings must not reset costs, rerun completed
science, duplicate a live controller, relax a failed validity check, or multiply
the 48-hour allocation across both resources. The existing master lock and
absolute before/after ceilings prevent accidental simultaneous controllers and
double allocation; resource ceilings and authorization are already excluded
from scientific job identity. Check idempotent application, preserved costs
and inputs, and budget-only resumption with CPU-only controller fixtures before
launch. No numerical source, target, acceptance criterion, attempt count,
per-job duration or final-reference policy changes. The extension is not a
requirement to spend all available time.

The allocation record, configuration snapshot and handover status are under
`campaign-r1/budget-extension-20260930-r1/`. The exact handover command is
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/apply_neutra_warm_start_budget_extension.py`.
It runs in a durable user service; only a budget-exhaustion exit can trigger
automatic resumption. Scientific findings remain in the existing per-phase
evidence and repair result.

The handover passed nine CPU-only tests, including preservation of historical
charges, repeated application, stale terminal state and budget-only resumption.
A direct check confirmed that changing only the two ceilings and authorization
leaves the existing scientific invocation identity unchanged. The durable
`neutra-warm-start-budget-extension-20260930.service` is waiting on the master
lock; the original campaign controller remains PID 12049. All six Gaussian
cases passed the final checks. The unwarped-mixture seed-11 SMC fit exhausted
its nonlinear-shape repair; the remaining planned candidates continue. This
records a candidate failure, not rejection of the research direction.

## Funnel calibration recovery, September 30

The controller stopped at 04:43:53 Asia/Shanghai after 24 of 30 cases were
resolved, with 12 passing the final posterior screen. The additional allocation
was subsequently applied. The failing funnel teacher preserved its pilot table:
all four time steps have beta-zero acceptance 0.078125, while later bridges
accept much more often. `select_mala_candidate` correctly refused them, but an
uncaught `ValueError` mislabeled this candidate rejection as a harness failure
and stopped the remaining program.

The implementation uses children conditional on v with variance exp(2v), and
log density at zero children is constant - v^2/2 - 9v. Thus the density mode is
v=-9 and the local child variance is exp(-18)=1.523e-8, matching the saved
Laplace covariance. MALA proposes x + dt score(x) + sqrt(2 dt) noise. In a
Gaussian direction with variance s^2, its linear mean multiplier is
1-dt/s^2; the local Euler stability range is 0 < dt/s^2 < 2. This is a scale
diagnostic, not an acceptance or mixing guarantee. The old minimum dt=1e-4
is approximately 6,566 times this local variance. Merely increasing particles
does not repair that pilot range.

Repair: retain the original grid and all its pilot streams. Only if it fails,
add dt equal to 1, 0.1 and 0.01 times the smallest eigenvalue of the proposal's
local component covariances, retaining values below the original minimum.
These three dimensionless ratios are bounded geometric search hypotheses;
the covariance supplies target-specific scale. At the recorded funnel mode
they are about 1.523e-8, 1.523e-9 and 1.523e-10. All candidates must still pass
the original finite/movement/acceptance screen at every pilot bridge. Preserve
every trial and its provenance. A typed calibration failure is returned as a
failed phase by both teacher and Gabrié-fit callers, so the controller can
exhaust the declared repair attempts and continue independent candidates.

Evidence contract: the engineering pass criterion is that a calibration
rejection cannot bypass the screen or abort unrelated planned candidates.
A stiff-Gaussian reference fixture must demonstrate that the old range fails
and a curvature-scaled candidate passes. The research question remains teacher
accuracy and downstream frozen-map HMC; tiny accepted steps do not establish
useful exploration. The Laplace proposal's concentration at a density mode
may still miss the funnel's typical mass. Existing teacher, fit and final
posterior checks remain decisive. Calibration is a repair trigger, not a new
posterior criterion, and no acceptance threshold is relaxed.

Skeptical review passes for this limited repair: the preserved table identifies
the failing beta-zero proposal, the scale is derived from inspected code and
saved covariance, and all target equations and scientific criteria remain.
The main alternative explanation is inadequate proposal overlap even after
calibration; the existing independent teacher tests can reject that case.
Run the focused regression suite, then resume only `--targets funnel` using
the shared budget and new versioned attempts. Earlier target outcomes remain
evidence of their recorded source version and are not silently rerun or
relabeled as observations under changed source. The extra allocation leaves
26.73 GPU process-hours and 53.47 CPU core-hours before this repair.

The focused suite passed 67 tests in 36.61 seconds with GPUs deliberately
hidden. This includes the stiff-Gaussian reference, both rejected-calibration
callers, prior scientific invariants and budget handover. The prior controller
state and results were copied to `campaign-r1/recovery-funnel-20260930-r1/`.
The same durable service resumed with `run --targets funnel` on available
GPU 1; the worker enforces and records memory growth. The preserved results
are Gaussian 6/6 final passes, ordinary mixture 0/6, warped mixture 4/6 and
wiggle 2/6; six funnel cases remained pending at resumption. These descriptive
counts establish neither a method ranking nor reliable multimodal training.

The first resumed GPU attempt exercised the repaired path: time step
1.523e-9 passed the unchanged pilot with minimum bridge acceptance 0.9766 and
no invalid proposals. Its 1,024-particle teacher then failed the separate
accuracy/precision screen. The controller preserved that result and advanced
automatically to the 4,096-particle repair. Thus calibration no longer crashes
the program, while teacher correctness remains unresolved. Evidence:
`attempts/closure-teacher-funnel-s11-v0-r2/` under the campaign root.
