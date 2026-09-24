# q20 training hyperparameter audit and calibration design

Latest owner direction: first apply the
[low-cost sanity canaries](bayesfilter-q20-training-sanity-canaries-2026-09-22.md).
Those checks screen the current recipe for obvious problems; they are explicitly
not hyperparameter optimization or completion of this larger calibration design.
The [executed canary report](bayesfilter-q20-training-sanity-results-2026-09-22.md)
now records clipping-role failure, scale/noise warnings and limited healthy
epsilon/LR/initialization checks. Use those findings to focus subsequent work;
do not launch the whole calibration design merely because it is documented.

The owner requires proper calibration before treating the training settings as
production choices. The problem is broader than clipping. Several numerical
choices remain uncalibrated, and the current execution and selection logic
cannot carry out the calibration described in earlier plans. This document
records the complete active training/validation inventory and the necessary
repair order. It does not report an executed hyperparameter search.

The [training-gap investigation](bayesfilter-q20-training-gap-results-2026-09-22.md)
contains the measured training histories, scale saturation, local derivative
checks and saved-data gradient decomposition. The September 15
[mathematical audit](bayesfilter-ssl-lstm-q20-parameter-mathematical-audit-2026-09-15.md)
already marked most choices as hypotheses and proposed relevant checks.
Documentation of those checks was not their execution. Its old comparison,
mass-preparation and large validation-bank proposals are not reinstated here.

The [machine-readable inventory](artifacts/q20-training-gap-investigation-2026-09-22/training-calibration-audit.json)
checks that every active training and validation configuration field is covered
and records the live values, budget and source checksums. The inspected current
training/controller files match the archived execution source; the low-level
batched sigma-point module differs. The new calibration's numerical source
must therefore be fixed explicitly before transferring old target evidence.

## Research intent and evidence contract

Question: which target-specific training settings produce a numerically valid,
reproducible transport that helps obtain the owner's q20 posterior estimate
within the remaining allocation? The target remains the four-parameter q20/T30
float64 UKF approximate posterior. Plain NeuTra and the tempered NeuTra
ensemble, with identity latent mass, remain the allowed inference methods.

Use the exact existing protocol as an explicitly uncalibrated baseline. A
local continuation comparison clones the same weights, Adam state, iteration
and RNG position, records every changed setting, and uses deliberately paired
base draws and evaluation points. A new architecture that changes the initial
map needs a fresh-training comparison, or a demonstrated function-preserving
extension. The uneven historical training roots cannot serve as an equal-effort
comparison without addressing that difference.

The calibration nomination criterion is resolved learning progress on common
development points at declared matched work, acceptable numerical/map health,
and evidence that the implicated defect has been addressed. Selection must
consider uncertainty across training roots as well as evaluation rows. Fresh
confirmation after freezing the selected protocol and downstream sampling are
required before production/default-readiness. Low loss, infrequent clipping,
small score residuals or short-chain acceptance alone are insufficient.

Numerical invalidity, wrong target/map/source identity, a scalar training
fallback, broken checkpoint state, contaminated confirmation data, incorrect
GPU memory policy or exhausted budget veto the affected experiment. A poor
candidate, continuing learning or a failed bounded HMC trial triggers the
relevant funded repair; it does not reject NeuTra. Gradient norms, saturation,
noise, optimizer moments and proposal-region occupancy are explanatory
diagnostics and repair triggers. They are not posterior pass criteria.

## Confirmed problems in the calibration mechanism

1. `q20_master_program.py:561` calls a mode named `calibration_only` and returns
   `TRAINING_CALIBRATION_COMPLETE`. `q20_production_training.py:136,194`
   restricts that mode to the first root and first rung: 128 updates, four
   width/LR combinations for plain NeuTra. It does not calibrate batch size,
   clipping, Adam moments, epsilon, architecture depth, scale range or the
   temperature schedule. Treat this output as a short learning pilot only.
2. A candidate can become `hmc_trial_nominee` at the 512-update scope floor
   while its genuine incremental loss is still improving. The scheduler then
   skips its later training rungs. All twelve direct maps exhibited ongoing
   improvement. Larger configured rungs are not a working continuation policy.
3. `choose_maps` returns the first eligible candidate in width/LR/root order.
   This is an operational ordering, not calibrated hyperparameter selection.
   After failure of plain tuning, the master does not try the remaining maps
   or resume the failed map's training before proceeding toward the ensemble.
4. Candidate IDs enter training and validation seed derivation. A shared root
   number does not produce common random numbers across widths/LRs. The old
   audit's common-root pairing claim does not describe the executed design.
5. Root-0 direct maps have 1,024 lifetime updates; other roots have 512. A
   migration reset scope counters while preserving actual weights/moments.
   Later identical-map reassessments produced zero increments. Neither unequal
   training comparisons nor identical-map increments establish convergence.
6. The validator forces two-stage tanh maps and optimizer carry across beta.
   The default pricing loop uses only batch 32 unless explicitly given other
   batches. Listing batch alternatives or capacity repairs in a document does
   not make them executable search arms.
7. The current supervisor executes one worker at a time. Three-GPU calibration
   needs explicit independent jobs, separate state/cache/output directories,
   assigned devices and aggregate resource accounting. Device count is not
   evidence of achieved parallel speedup.

These are implementation/design repairs, not thresholds to optimize until
the existing controller produces a favorable result.

## Complete training and validation choice inventory

Every row below gives the executed value, provenance, plausible failure and
the discriminating calibration or check. Related controls are grouped because
their effects interact. None of the proposed contrasts is a selected default.

| Controls and current values | Provenance and present evidence | Failure mechanism | How to determine adequacy |
| --- | --- | --- | --- |
| Global gradient cap 10 | Inherited; 99.32%–100% of historical updates clipped; 31/31 fixed-map diagnostic batches exceed it | Routine batch-dependent normalization changes Adam's moment weighting; raw cap is parameterization-dependent | Preserve raw and supplied gradients, actual updates and map movement. Propose bounded caps from measured distributions and compare matched continuations, including a supported unclipped control if implemented. A lower clipping fraction alone is not success. |
| Batch 32; proposed pricing sizes 8/32/128 | Inherited candidates; measured RMS batch variation 63.68 versus norm of bank mean 24.42 at the selected map; bank-mean RMS uncertainty estimate 11.44 | Noisy updates; more samples can help but may cost too much or change optimization dynamics | Use nested common base points to estimate gradient variance, then measure complete GPU graphs, memory and total learning cost at feasible sizes. Do not choose by updates/second. Recheck clipping/LR after changing batch. |
| Learning rates .0005/.001; constant schedule | Limited historical width/LR runs, with no uncertainty-supported selection; documented half-rate continuation is not active | Overshoot, noise floor, or unnecessarily slow movement; interaction with clipping and output scaling | Inspect actual parameter and physical-map changes, loss change and curvature along proposed steps. Bracket rates around measured stable/unstable behavior, then compare funded paired continuations. Decay is a hypothesis to test after demonstrated diminishing progress. |
| Adam beta1=.9, beta2=.999, epsilon=1e-7 | Inherited; earlier shadow updates verified implementation, not suitability | Stale moments or epsilon-dominated coordinates; sparse/masked parameters can distort naive summaries | Record active-coordinate moment/gradient alignment and epsilon relative to sqrt(v), plus per-layer update/weight and map-displacement measures. Conditional moment/epsilon contrasts require measured evidence; epsilon has not been shown to be harmful. |
| Two IAF stages; widths (16,16)/(32,32) | Inherited, hard-restricted family; no depth search or demonstrated sufficiency | Insufficient conditional dependence or constrained contraction; additional width can add optimizer sensitivity | Separate insufficient optimization from capacity by continuing the baseline, then test one capacity change at comparable declared cost. Recalibrate affected optimizer controls and replicate the selected architecture from fresh initialization. |
| Tanh; full reversal between stages; fixed autoregressive degree masks | Inherited ordering/topology, not target-specific selection | Saturated activations or conditioning order poorly matched to target dependence | Inspect activation derivatives, conditional scale/shift behavior and map Jacobians. Evaluate an ordering/activation contrast only when implicated; do not infer universal hidden saturation from the current limited observations. |
| Per-stage log-scale bound s_max=2; extra scale-linear paths disabled | Inherited; selected final fourth scale median -1.966965 with derivative about .03276; similar near-bound contraction across maps | Attenuated scale-learning derivative or restricted representation; larger limits can instead worsen conditioning | Compare a defined scale parameterization or capacity repair, preserving initial-map identity where claimed. Monitor Jacobian conditioning/inverse health and learning. Merely replacing 2 with 3 at unchanged weights changes the map and is not a matched-state continuation. |
| Fixed outer affine scale 4, center [.35,-.08,.65,.05] | Derived exact prior initializer; retaining it as a fixed outer map is an uncalibrated optimization choice | All posterior contraction is delegated to the IAF; physical scale multiplies final-shift gradients | Separate the prior distribution from transport coordinates. Test a learned or reparameterized outer affine only with a preserved initial map and appropriate optimizer-state treatment; changing transport coordinates must not change the prior. |
| Hidden initialization SD .02; final output weights/biases exactly zero | SD inherited without fan-in calibration; zero outputs derive the exact prior map | Small hidden features/derivatives or late root differentiation; identical initial proposal laws mistaken for independent coverage | Measure initial activation, parameter-gradient and map-update scales. Derive candidate scales from fan-in/activation behavior and check actual training. Preserve exact prior law when comparing hidden initializers. |
| IID standard-normal base draws; mean(-log target-logdet); no extra penalties/dropout | Derived RKL reparameterization and deterministic transport definition; checked local scores and optimizer VJP | RKL can fit one region while missing posterior mass; adding penalties or changing coefficients silently changes the optimization problem | Keep the objective exact. Assess proposal-region allocation and use independent posterior/downstream evidence for coverage. Do not divide only the likelihood by T or regard self-generated Gaussian points as posterior samples. |
| Direct beta=1; continuation [0,.5,1]; beta-zero updates=0 | Exact analytic beta-zero initialization justifies zero updates; positive-beta schedule inherited | A large temperature jump can create poor overlap, abrupt curvature and optimizer mismatch | Inspect incremental log-likelihood variation, importance-weight concentration as an explanatory overlap measure, new gradients and learning cost before choosing intermediate beta values. No universal ESS cutoff is inferred. Keep the final target beta=1. |
| Carry Adam across beta=true | Inherited continuity choice; currently forced by validator | Old moments can become inappropriate after objective change | At identical weights compare post-change gradients with stored moments; compare carry/reset only if warranted, with explicit state and matched streams. Do not silently change Adam state during an alleged exact continuation. |
| Three training roots [0,1,2]; root namespace [20260915,150001] | Convenience replication/seed labels, not proof of reliability | Seed-sensitive success, unintended stream differences or selection on a lucky root | Separate initializer, training, selection and confirmation streams. Deliberately pair training/evaluation draws for suitable contrasts, keep all failures, and use between-root uncertainty to determine further replication. Exact integer seed values are reproducibility labels, not tunable scores. |
| Rungs [128,512,2048,8192]; floor512 | Inherited resource/observation schedule; actual controller stops nominees at floor | Undertraining presented as completed training; fixed counts misread as convergence guarantees | Continue real checkpoint increments while progress is supported; diagnose a noise floor or capacity limit before stopping. Price the next increment and downstream work. An improving model at the funded cap is undertrained/under-budgeted, not converged. |
| Checkpoint every128 plus boundaries; resume of weights/slots/iteration/RNG | Cadence inherited; full-state identity is an engineering requirement | Save overhead or excessive lost work; incorrect continuation | Measure checkpoint cost and recovery behavior. Calibrate cadence operationally; exact restore is not a negotiable accuracy parameter. |
| Validation sizes [768,3072,12288] stored in live config; only first768 active | Historical ladder retained as metadata; current evaluator intentionally uses [:1] | Inadequate decision precision, costly unnecessary validation or claiming inactive sizes were evaluated | Size paired development evaluations from measured variance and the decision resolution; reuse legitimate cached values. Keep a fresh final bank after selection. Escalating to an enormous bank to certify a tiny loss plateau is not automatically useful. |
| Improvement resolution .04; half-width .02; two plateau comparisons | Uncalibrated optimization-resolution choices; currently explanatory, not trial gates | Arbitrary plateau labels, especially for unchanged checkpoints | Set a learning-resolution goal tied to a meaningful next allocation and downstream consequences. Require distinct checkpoints and uncertainty-aware progress; these constants are not posterior-error tolerances. |
| Normal interval multiplier1.959963984540054 | Derived pointwise normal-approximation 95% multiplier, not a repeated-look guarantee | Reused-bank or adaptive selection intervals misread as confirmatory evidence | Label development intervals descriptively. Freeze candidates before fresh paired confirmation; account for training-root uncertainty and any claimed multiplicity/sequential coverage. |
| Map reliability32 rows; rtol1e-9, atol1e-10 | Inherited engineering probes/tolerances, with successful checks in their limited scope | Missing tails or conditioning trouble; roundtrip parity mistaken for whitening | Use identity/conditioning/error consequences to justify tolerance and stress coverage. Passing codec parity verifies representation; it does not select training hyperparameters or establish posterior validity. |

Adam's beta2=.999 has an exponential half-life of
`log(.5)/log(.999)=692.80` updates; a contribution is multiplied by about .599
after 512 updates. This derives a memory scale, not a minimum convergence
count. Bias correction removes zero-initialization bias under its assumptions;
it does not prove moments track a rapidly changing objective. Measured moment
telemetry is required before calling this setting defective.

The objective and the target-defining data/prior/UKF controls are not training
hyperparameters to vary until the result looks better. Numerical regularizers,
covariance repair thresholds and posterior-approximation accuracy have a
separate validity audit. Successful training or the twelve local score checks
cannot settle that separate question. GPU/XLA, batching, allocator rules and
privacy/compute boundaries remain execution requirements.

## Calibration order and required master repairs

1. Repair the experiment mechanism first: pilot-versus-calibration status,
   genuine checkpoint increments, deliberate seed pairing, common evaluation
   banks, explicit candidate settings, funded continuation and retention of
   viable candidates. A failed bounded HMC trial must dispatch an explained
   repair or try another supported map. It must not erase a candidate or
   imply all plain NeuTra maps failed.
2. Collect missing optimizer telemetry at the preserved map and at prior
   initialization: active-coordinate moments/denominators, actual updates,
   map displacement, per-layer gradients and batch variability. Reuse saved
   points where they answer the question. Do not repeat the already successful
   derivative checks without a new numerical concern.
3. Resolve implicated output scaling and scale-bound constraints together with
   batch/clipping/LR calibration. Each structural change can alter the relevant
   optimizer scale, so a cap chosen before that change is not transferable by
   default. Freeze small candidate brackets from measured behavior before
   training comparisons; inherited alternatives remain baseline hypotheses.
4. Run bounded, matched continuation contrasts; inspect capacity/initialization
   and temperature/moment changes only through declared conditional branches.
   Avoid an all-combinations grid. Report equal target work and total resource
   cost separately when batch sizes or architectures differ. Extend an
   inconclusive candidate only within its predeclared budget; absence of a
   resolved winner is a valid calibration outcome.
5. Freeze one complete training recipe, evaluate independent training roots
   and untouched confirmation points, and test downstream usefulness. If
   several candidates remain statistically indistinguishable, use an explicit
   operational tie rule among those that meet the required evidence, without
   claiming superiority. Coverage concerns must be preserved.
6. For supported frozen dense-IAF maps, use the registry-authorized
   `tune_fixed_transport_hmc_kernel` with identity latent mass and fresh tuning
   for the actual map. This follows the inspected
   [HMC tuning interface](../reference/hmc-tuning-interface.md) and capability
   registry in `bayesfilter/inference/tuning_contract.py`. Final retained
   estimates still use the shared sequential NeuTra controller. Calibration
   does not introduce mass adaptation or a new method-comparison campaign.

Each numerical phase must preserve exact commands/source/environment, inputs,
candidate settings, initialization and stream identities, GPU placement and
memory-growth evidence, wall and aggregate device time, attempts/failures,
measured criteria, uncertainty and the next justified action. These are ordinary
research records; no new approval tokens or launch-security protocol is added.

## Budget, execution status and skeptical review

At this audit the live campaign is paused, with 155,692.207 recorded campaign
seconds (43.248 hours) and 902.358 diagnostic seconds (15.039 minutes)
remaining. These are existing balances, not new allocations. The current
single-worker accounting must not become 43 hours per GPU when parallel work
is implemented. Account aggregate device time as well as elapsed wall time.

The existing short `--stop-after calibrate` command does not implement this
design. It must not be used as evidence that the settings have been calibrated.
The remaining 15 diagnostic minutes are not a demonstrated budget for a full
hyperparameter study. Price the actual proposed branches, reserve confirmation
and posterior work, and record which work consumes training-campaign versus
diagnostic allocation under the existing accounting rules. Do not relabel work
to evade either limit. Exact new launch commands, candidate brackets, attempt
caps and phase allocations are outputs of the harness repair and pricing;
they have not been supplied by this audit.

Pre-execution skeptical audit identifies unequal histories, false pairing,
first-eligible selection, underfunded larger rungs, misleading calibration
status and inactive settings as material defects. This audit and repair order
are supported by inspected code and artifacts. The current master does not
pass as a numerical implementation of this calibration design. No new training
search was launched and no setting was changed by this audit. Fix and check
those concrete defects before the next numerical calibration phase; this is
an implementation/evidence requirement within the owner's authorized work,
not a request for renewed campaign permission.

## Audit decision and reset note

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Require actual training calibration before production selection | All active training/validation fields inventoried; missing selection experiments identified | Current calibration mechanism cannot answer the intended question | Which recipe works within remaining cost | Repair the specified harness defects and price bounded conditional calibration | Every inherited number is wrong, clipping is the sole failure, or a new recipe is calibrated |

There is no statistically supported hyperparameter ranking from this audit.
The finite/derivative/optimizer checks support their narrow engineering claims;
norms, scale saturation and historical loss differences remain descriptive.
No new default is ready. A calibrated recipe needs controlled training and
independent confirmation plus useful downstream sampling evidence.

The strongest alternative explanation is that sufficient continuation of the
existing recipe would already produce a usable map. Preserve that control:
near-continuous clipping and inherited constants do not prove otherwise. The
most direct established failure is that the current scheduler never completed
the proposed continuation-and-repair process. Calibration should distinguish
that explanation from poor optimizer or architecture choices.

Reset: use this audit with the September 22 training-gap report. Treat the old
`TRAINING_CALIBRATION_COMPLETE` status as pilot completion only. Do not infer
calibration from a plan, a constant's repeated use, a source-level assertion,
or an unused candidate list. Repaired execution must produce the actual
selection evidence before a new production training claim.
