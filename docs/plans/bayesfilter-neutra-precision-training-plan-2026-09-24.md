# NeuTra precision repair and q20 training evaluation

User authorization: implement configurable FP32/TF32 training, commit and merge
with the remote and push, then evaluate the repaired training. The target is
the existing q20/T30 four-parameter UKF posterior approximation. This is not a
method-comparison campaign or a request for latent mass adaptation.

## Engineering question and execution order

1. Separate transport/optimizer precision from target-evaluation precision in
   the shared numerical implementation. Preserve existing FP64 checkpoints and
   facades. Expose FP32 explicitly; use dtype-appropriate inverse tolerances.
   Make promotion of trained FP32 weights to a frozen FP64 evaluation map an
   explicit, recorded conversion, followed by fresh diagnostics.
2. Test independent equations, mixed-precision target gradients, standard/path
   estimators, inverse/tail behavior, checkpoint continuation and frozen replay.
   Run a bounded trusted GPU/XLA FP64 / FP32 / FP32-with-TF32 comparison using
   identical parameters and inputs. TF32-enabled metadata is not proof that
   Tensor Cores were used; inspect compiler evidence where available.
3. Commit only the NeuTra work and its necessary dependencies in an isolated
   checkout, merge the remote, resolve conflicts, rerun affected checks and
   push. Preserve concurrent unrelated dirty HMC and governance work.
4. After synchronization, run target-specific pricing/calibration, sustained
   training, independent post-training verification and affordable downstream
   qualification. Record a fully specified training amendment after actual
   prices are known and before starting the training ladder.

## Evidence contract and intent ledger

| Item | Contract |
|---|---|
| Main question | Can corrected source-based IAF/conditional NAF, adequate optimization and configurable precision produce useful q20 transport geometry? |
| Precision comparator | Identical arrays/inputs evaluated with FP64, FP32 and FP32 with TF32; FP64 is a reference, not presumed necessary for training. |
| Training baseline | Preserved depth-four 2,048-update q20 map, freshly evaluated on common banks; matched continuation controls where affordable. |
| Engineering pass | Finite valid batched gradients, independent derivative agreement at precision-appropriate tolerances, inverse reconstruction, exact same-dtype resume, explicit precision provenance. |
| Training improvement criterion | Predeclared paired heldout loss differences with uncertainty and fresh 1,000-point score geometry, replicated across initialization/noise seeds. Neither alone establishes posterior correctness. |
| Downstream promotion | Exact frozen-map public fixed-transport HMC qualification and existing model-coordinate posterior checks, with identity latent mass. |
| Promotion veto | Invalid target rows, nonfinite map/gradients, failed inverse/derivative checks, missing diagnostic, or failed downstream checks. |
| Continuation veto | Broken target/implementation or corrupted artifacts, deadline, exhausted total budget, or inability to fund the declared complete replication. A poor candidate is not a continuation veto. |
| Repair triggers | Precision mismatch, unstable optimizer, persistently improving map at a rung, scale/conditioning failure, or inadequate capacity; use bounded declared repair branches. |
| Explanatory diagnostics | Gradient/update norms and clipping, validation trajectory, residual quantiles, log-density correction, tail slices, compile/steady timings. No ranking from single-run maxima or timings. |
| Nonconclusions | No universal TF32 adequacy, no exact-paper-experiment reproduction, no convergence from acceptance or loss, no posterior promotion from a short fit. |

## Defaults, resources and skeptical audit

FP64 target arithmetic is inherited from the existing checked q20 evaluator;
it is a compatibility baseline, not evidence that every computation requires
FP64. FP32 flow/Adam with TF32 is the requested candidate. Original source
code uses FP32. Stable log-domain NAF equations remain unchanged. Existing
FP64 inverse tolerances of 1e-11 cannot be reused for FP32: choose the FP32
engineering tolerance relative to machine epsilon and test independent
reconstruction, tails and implicit derivatives. Do not silently relax a
scientific convergence threshold.

The last q20 ledger is
`artifacts/q20-short-fit-canary-2026-09-23/accounting-summary.json`, with
143330.11637 campaign worker-seconds (39.81 hours) remaining. The old canary
allocation is closed and is not reused. This execution draws from the existing
campaign balance, conservatively charging worker wall time including failed
attempts and concurrent workers. The deadline remains 2026-09-25 18:00
Asia/Shanghai. Initially reserve at most 1,800 CPU and 1,200 GPU seconds for
precision engineering; these are convenience caps within the existing balance,
not new compute grants. Reconcile before allocating the serious training study.
Integration amendment: the earlier uncommitted q20 implementation is a necessary
dependency of post-training diagnostics and must be committed with the shared
authority. Its compatibility checks expanded the CPU engineering reservation to
3,000 seconds (a convenience cap within the same remaining allowance); GPU
engineering remains capped at 1,200 seconds. Charge repeated failed attempts.

Artifacts use fresh directories under
`docs/plans/artifacts/neutra-precision-training-2026-09-24/`. Every serious
worker records command, source hashes/commit, environment, memory-growth
verification before GPU initialization, actual dtype/device/XLA/TF32 policy,
seeds, target/data identity, checkpoints, timing and remaining allocation.
GPU occupancy is inspected with trusted permissions; independent training
workers may use free GPUs, and their times are summed.

Skeptical audit before implementation: the material risks are confusing TF32
flags with actual FP32 arithmetic; evaluating a different posterior after a
cast; changing old artifact meanings; false inverse failures from FP64
tolerances; comparing different initial weights; stopping training at a smoke
rung; and including unrelated active work in a commit. The design addresses
these through explicit precision boundaries, parameter-matched checks,
compatibility tests, priced sustained training with independent validation,
and isolated Git integration. No scientific quality conclusion is permitted
from the precision smoke. Audit passes for engineering; the priced training
amendment must pass its own scientific audit before execution.

## Precision check specification

`check_neutra_precision_2026_09_24.py` runs each mode in a separate process on
GPU1, initially observed idle. It uses three stages, two width-16 layers, four
dimensions and 32 rows, matching the order of the q20 training workload. Both
IAF and DSF use standard and path estimators. Eight repeated updates per cell
test compilation/optimizer mechanics only. A coupled precision test and
independent inverse derivatives also run in the CPU-hidden unit suite.

The GPU analytic target is an anisotropic Gaussian plus a quartic term, with
its value and score evaluated in FP64. Identical represented weights and
inputs are used for each CPU reference. Report scaled gradient L2 error and
scaled loss error. Limits are 1e-9 for FP64, 1e-4 for FP32 and 0.01 for TF32:
engineering screening hypotheses allowing operation accumulation beyond the
respective machine precision, not scientific-equivalence tolerances. A failure
requires diagnosis, not automatic widening. Test trained-map inversion on
10-times base-normal inputs; scaled roundtrip limits are 1e-7 for FP64 and
0.002 for FP32/TF32. Exported FP64 maps must also reconstruct within 1e-7.
Preserve compiler HLO, timings and actual variable/target dtypes. Actual q20
calibration must subsequently compare precision drift with gradient variability
and check the exact frozen evaluation map.

## q20 pricing and calibration entry (after Git synchronization)

Use the preserved depth-four checkpoint from
`q20-training-repair-2026-09-23/campaign-01/attempts/00005-continue-depth/worker/data/cohort-00000.json`,
candidate `direct-w16-lr0.0005-r0`, with 2,048 lifetime updates. Check its
target/bridge identities and reconstruction before using it. The historical
scalar correction is explanatory evidence, not the baseline for the new full
conditional NAF.

The first target worker is a pricing/calibration run, capped at 2,400 worker
seconds from the same campaign. Preserve the original baseline and construct
configured IAF and DSF candidates. The control uses the legacy four-stage
width-16 tanh map. New candidate hypotheses are three-stage ELU IAF with free
scale bias at width16, and three-stage conditional DSF with sixteen mixture
components and widths16/32. The mixture count follows Huang appendix E’s
IAF-DSF setting; the conditioner widths are target-specific hypotheses. Width16 comes from the existing workload; width32
is an explicit capacity alternative. Three stages and the IAF initializer come
from the checked source profile; DSF width and component count need local
evidence and are not claimed paper defaults.

Use one fixed diagonal affine initialization derived from 8,192 draws of the
saved baseline for all new maps. This is a warm-start approximation to an
already available map, not a posterior estimate or coverage claim. The draw
count is a cheap power-of-two convenience choice for moment stability. Generate
these internal initialization draws in a batched TensorFlow graph; no external
training dataset or scalar target loop is introduced. Record the derived center
and scales, and reject nonpositive/nonfinite scales.

Price compile and repeated full updates separately at batch32 and batch128,
the two previously executable batch sizes. Inspect standard and path gradients
on common unchanged parameters; compare their sampled gradient variation and
the FP32/FP64 discrepancy. Eight minibatches are a low-cost sanity screen, not
an optimizer/variance ranking. A precision discrepancy greater than one tenth
of the observed minibatch-gradient RMS variability is an investigation trigger;
the fraction is an explicit precision-screen hypothesis, not an HMC tolerance.

The priced amendment must fund complete independent training replications,
preserved checkpoints, disjoint validation and final banks, all 1,000-point
checks, and a downstream qualification reserve. It must not stop because a
candidate becomes trial-eligible. Each continuing-improvement signal at a rung
must reach its funded continuation or receive an explicit budget-limited
disposition. Sparse validation (at checkpoint rungs) replaces per-update
validation. Failed candidates trigger the declared optimizer/capacity repair,
not rejection of the NeuTra research direction.

## Integration audit, 2026-09-24

The checkpoint-migration fixture allocated exactly the protected 7,200-second
repair reserve. After migration/pricing, its correct result is a budget pause;
the test now checks that no unfunded training starts. No budget guard changed.
The refresh controller had drifted from the current estimation objective: it
omitted the method label in training pricing and still requested classical
preparation. It now prices plain NeuTra at beta one with its frozen chart and
identity latent mass. A real supervised-worker test covers exact checkpoint
restore, pricing scope and reuse without duplicate charges.

The integration audit found a downstream defect: the generic posterior information check
applies continuous tail ESS to `positive_theta_2`, a binary indicator. Its upper
tail indicator is constant, so the known Gaussian success fixture is rejected
despite finite R-hat, bulk ESS and event MCSE. It was initially retained as a
strict expected failure and has now been repaired as documented below; the
success test passes without that marker. This did not invalidate the independent
target, transport-gradient, loss or score-residual training evaluation. Repair
this diagnostic with explicit discrete-quantity semantics before using it for
an HMC promotion decision. Do not remove the quantity or loosen ESS thresholds.

## Priced training protocol amendment

Initial q20 measurements are 2.43 seconds per batch-32 update and 6.7 seconds
per batch-128 update across IAF/NAF and standard/path gradients. These are
descriptive prices, not a method ranking. Use a conservative 2.7 seconds per
batch-32 update for admission, then replace forecasts with actual worker time.
The engineering balance is 141,245.38 seconds (39.23 hours), before q20 pricing.
Pricing is capped at 2,400 seconds; the whole campaign still shares the same
deadline and remaining allowance. No individual stage may create new budget.

### Calibration before the sustained fits

For IAF16, NAF16, NAF32 and the saved-map continuation control, inspect eight
fixed-parameter gradient batches at sizes 32 and 128. Record standard/path
gradient variability and matched FP32-with-TF32 / FP64 path-gradient drift.
Drift exceeding 10% of minibatch RMS variability requires investigation.
This is a low-cost plausibility check, not variance optimization.

Use independent disposable 128-update path-gradient pilots at learning rates
0.001, 0.003 and 0.01, initialized identically within a family. The endpoints
come from the inspected author-code/paper range; the middle value is a
convenience log-scale interpolation. The path estimator is the predeclared
training candidate because it removes the score term and has zero variance
at exact fit under its checked assumptions; this is not an empirical claim
that it dominates standard gradients here. The standard estimator remains a
measured calibration comparator and a bounded repair if path training fails.
Use Adam beta1=0.9, beta2=0.999 and epsilon=1e-8 (inherited author/optimizer
baseline hypotheses). The previous q20 epsilon check did not implicate epsilon;
monitor update sizes and instability here rather than assuming transfer proves
adequacy. Clip at ten times the largest initial calibration gradient norm,
an explicit emergency-envelope hypothesis. A majority of clipped updates in
a checkpoint window triggers repair. Preserve the exact threshold and counts.

Use 128 independent common validation rows for the pilots. Select the highest
declared rate that has finite valid updates, no majority clipping, and lower
heldout loss than its unchanged initial map. This nominates a viable starting
rate; it does not establish an optimal learning rate or a quality ranking.
If none passes, extend the same ladder down to 0.0003 under the repair reserve;
do not silently widen validity checks. A matched standard-gradient pilot at
the nominated rate is the repair for a path-specific failure. Pilot states
must never become undisclosed warm starts for the serious seeded fits.
Calibration, including failed attempts, is capped at 9,000 worker seconds.

### Sustained training and continuation

Train IAF16 and NAF16 from the shared diagonal affine initialization, with
three independent initialization/training seeds (indices 0, 1, 2). Each receives
4,096 optimizer updates at batch 32. Preserve full Adam/checkpoint state every
256 updates and assess common heldout banks of 256 rows at 1,024, 2,048 and
4,096 updates. The update floor is a budget-feasible target-specific hypothesis,
not a proof of convergence or a reproduction of the paper's batch-4096 study.
The checkpoints allow the hypothesis to be inspected instead of hardening it
into a default. There is no stop at trial eligibility and no per-update
validation. New final maps are evaluated in FP64 after explicit conversion.

Also continue the preserved legacy four-stage map for 4,096 updates with the
same calibrated estimator/optimizer family and batch size. This control starts
from its saved weights with fresh Adam slots, explicitly recorded; it isolates
additional optimization only imperfectly, since the architecture and warm start
differ. It is a conditional control, not an independently replicated old
training recipe. Do not attribute a difference solely to architecture.

Fund an additional 4,096 updates for each of the three valid NAF16 seeds, making
8,192 the final endpoint and 4,096 an intermediate checkpoint. This decision
was recorded before the first 1,024-update validation: complete the affordable
training allowance rather than make a noisy plateau screen a stopping rule.
Improving heldout loss or poor nonlinear geometry remain repair triggers,
not HMC admission. If NAF16 has numerical or capacity failure and the NAF32 pilot passes,
the same reserved work may instead train three NAF32 seeds for 4,096 updates.
That branch must be recorded before launch, with the capacity hypothesis and
no change to the final evidence threshold. The initial IAF and NAF fits plus
legacy control cost at most 77,415 forecast seconds; the continuation reserve
is 33,178 seconds. Reprice if steady updates exceed 2.7 seconds persistently.

### Independent verification and scientific decision

Every completed training arm gets the standard 1,000-point normal-base
post-training probe: residual norm distribution, coordinate RMS, density-ratio
range, scale/derivative diagnostics, finite/status checks and exact map identity.
Also test the exact exported FP64 inverse on 32 independent normal points
scaled by four. The 1e-7 scaled reconstruction and absolute log-determinant
limits are inherited engineering-screen hypotheses from the precision check;
the tail scale and count are convenience stress choices. Failure rejects the
candidate handoff and triggers repair, while other planned candidates continue.
It is an explanatory geometry diagnostic; no arbitrary residual cutoff is
called convergence. Preserve a single untouched final bank of 2,048 common
normal-base points, disjoint from gradient, training, validation and the
1,000-point banks. Evaluate the saved baseline, control, and all final trained
maps on it. Save individual paired loss and squared residual contributions.

For each fixed checkpoint, estimate mean reverse-KL loss difference relative
to the saved baseline (unknown target normalizer cancels) and mean squared
score-residual difference on the same latent bank. Use a paired bootstrap
with 4,096 resamples and Bonferroni-adjusted intervals across at most ten
predeclared final map-versus-baseline comparisons. The bootstrap count is a
Monte Carlo resolution convenience; report it and the independent bootstrap
seed. A trained family supports replicated improvement over this saved
baseline only when all three independent fits have an upper adjusted loss
interval below zero, no numerical veto, and a reduced squared-residual mean
with adjusted upper interval below zero. Otherwise report which fixed maps
improved and the unresolved seed/geometry uncertainty. This is a stringent
local training criterion; it is not population-level proof from three seeds.
Do not rank IAF versus NAF or tail maxima without a separate valid comparison.
The continuation control determines whether improved optimization alone is
a plausible explanation, without causal attribution or superiority claims.

A 6,000-second envelope covers sparse validation, final banks, required
1,000-point probes and their compile overhead. Reserve 8,000 seconds for
affordable downstream fixed-map qualification, and retain at least 5,000
seconds for localized repairs. Unspent pricing/calibration reservations return
to the shared balance. If the total forecast fails to fit, reduce optional
downstream/repair breadth explicitly before launch; never drop seeds, required
diagnostics, or stop still-improving training without a budget disposition.

Skeptical audit: the preserved baseline and common affine warm start prevent
silent comparator changes, while the conditional control exposes the added-
optimization explanation. Separate banks and checkpoint hashes prevent reused
maps or pilot selection from masquerading as final evidence. The 4,096-update
floor can still be insufficient; explicit funded continuation addresses that
risk. TF32 adequacy is checked on actual q20 gradients. Validation costs are
bounded and sparse. The main unresolved limitation is posterior coverage,
which the downstream qualification must test; a training pass cannot establish
it. The budget arithmetic fits only with measured update prices and at least
two available GPUs before the deadline; defer a busy device and retain the
wall-time veto. Audit passes for calibration and the stated conditional ladder.

### Executable phase controller and reconciled caps

`scripts/continue_neutra_training_campaign.py` waits for the six initial fits,
then runs the saved-map control and the three funded continuation fits. A
rejected NAF16 candidate activates the calibrated NAF32 cohort instead; valid
NAF16 prefixes remain in the report. Infrastructure failures preserve their
checkpoints for localized repair. Completed phases can be reused only with
identical requests; a restart must not duplicate an already completed fit.
The final stage evaluates the untouched common bank, verifies its identity
against each exact finalized checkpoint, and writes paired bootstrap results.
Posterior assessment remains a separate downstream decision.

After pricing, calibration and recorded engineering charges, 134,475.159 worker
seconds remain before sustained training. The six initial jobs have a combined
69,000-second cap. Control plus continuation/repair have a 46,000-second cap;
final banks have 3,000 seconds, downstream work 8,000 and localized repairs
5,000. The 3,475.159 seconds left cover additional engineering/analysis and
forecast error. Each 4,096-update worker has an 11,500-second cap derived from
the measured update price plus compilation, checkpoints and diagnostics.
Every phase checks actual spent time against the existing campaign balance.
These are reservations within the original allowance, not new grants.

Controller audit: every planned seed must have an explicit terminal outcome;
candidate rejection cannot erase surviving peers or imply rejection of NeuTra.
Continuation restores Adam slots and uses noise indices 4,096 onward, while
the legacy control explicitly starts fresh slots from saved weights. Final
evaluation consumes no pilot/validation bank. The largest repair branch has
nine candidate maps plus the baseline, fitting both the final-evaluation cap
and the predeclared multiplicity allowance. The controller cannot grant
posterior status. Focused synthetic checks cover cohort decisions, exact
continuation requests and budget rejection before launch.

## Binary-event posterior diagnostic repair

The narrow repair declares binary quantities explicitly in the common posterior
policy. For an event indicator I, the estimand is E[I], so its information is
the event-indicator ESS and MCSE. With both outcomes observed, pooled rank
normalization maps the two values to a+bI with b>0; the common autocovariance/
between-chain ESS ratios are invariant under this affine transformation.
Therefore rank bulk ESS equals split-chain indicator ESS. Apply both declared
information floors to that ESS for the named binary quantity, while retaining
the existing continuous-tail ESS for each model coordinate. Preserve undefined
binary quantile-tail ESS as undefined in the report. Do not claim to estimate
a continuous binary quantile or remove event monitoring. Missing outcomes,
non-0/1 values, failed R-hat or failed event MCSE still prevent admission.

The first independent fixture also exposed the exact-half case: a binary sample
with median 1/2 folds to a constant 1/2 despite varying events. A Bernoulli
distribution has only its event-probability parameter, so explicitly typed
events use rank-normalized split R-hat; continuous quantities retain the maximum
rank/folded rule. The report has a new quantity-aware summary schema and retains
the complete raw rank/folded diagnostic, including undefined folded entries.
All-zero/all-one or separate stuck-zero/stuck-one chains still fail. This is a
change of diagnostic appropriate to the declared estimand, not evidence from a
relaxed threshold. Direct indicator-ESS equality and those counterexamples are
tested, and the full known-target supervisor test passes after the repair.

Engineering contract: compare the explicit event ESS against independently
constructed raw-indicator split ESS; require the known Gaussian master success
fixture to complete; ensure all-zero/all-one and falsely declared binary data
cannot pass. Existing untyped quantity behavior and continuous-tail formulas
remain intact. These CPU-hidden reference checks establish diagnostic wiring
and its algebraic invariant, not q20 posterior accuracy. Skeptical audit finds
no threshold relaxation: the same floors and event MCSE apply to the correct
estimand, and exact policy identity records the change. No sampler, mass,
target or training arithmetic changes. Tests and their wall time are preserved
under the existing campaign engineering charge.
