# q20 short-first NAF forward/reverse experiment

User authorization: update and review the monograph, commit/push the coherent
NeuTra work, plan/review/execute a q20 test beginning with short whitening checks.
This continues the funded local study. No posterior or architecture-default
promotion is part of this experiment.

## Question and evidence contract

Can approximate weighted posterior particles initialize the full Huang
DSF/author-cMADE NAF so that subsequent reverse-KL updates improve geometry
without losing the q20 posterior's represented sign regions?

The target is the existing `make_q20_tempered_bridge(20)`, beta one: four free
parameters, 20 latent state coordinates, 30 observations, Gaussian prior centered
at (.35,-.08,.65,.05) with SD 4, analytic selected-coordinate derivatives of the
principal-square-root UKF likelihood. Record its actual target/adapter signatures,
data/source hashes and backend. It is an approximate filtering posterior, not
an exact latent-state marginal likelihood. No LEDH artifacts enter this study.

Execution repair: explicitly select `tensorflow_eigh_strict_factor_cached`, the
backend used by `docs/benchmarks/run_q20_configured_training_2026_09_24.py:66`.
The first two attempts inadvertently inherited the bridge constructor's older
`compiled_custom_op` default and stalled before returning their first batch.
That was a missed default in the initial audit. This repair restores the known
q20 execution backend while preserving target identity; finite-difference checks
still run before teacher acceptance. Their scaled tolerance 1e-5 is an engineering
screen for steps 1e-4/5e-5, not a posterior accuracy tolerance.

The candidate uses the shared `NeuTraTransportConfig.huang_dsf` numerical
authority. Synthetic settings are warm-start hypotheses, not calibrated q20
defaults. The exact empirical forward objective is weighted negative log density;
pure reverse KL uses fresh base draws and the actual q20 value/score. Zero
reverse weight must skip target evaluation entirely during forward updates.

The primary pilot decision is whether longer target-specific training is worth
funding. It requires finite target/gradient/inverse calculations, a credible
approximate teacher, decreasing heldout teacher cross entropy, and a finite
whitening diagnostic without represented-region collapse. A complete 1000-point
probe is required at saved forward and reverse endpoints. Absolute whitening,
posterior convergence, full mode discovery, downstream HMC readiness, superiority
and default readiness are not pilot conclusions.

## Stages, budgets and automatic continuation

1. Preserve the manuscript baseline, incorporate the completed synthetic study,
   check equations/numbers/citations, compile the full monograph, inspect affected
   pages. Preserve unrelated changes in the shared checkout.
2. Price a batch-native GPU/XLA q20 value/score and score probe at batch 32.
   Check finite differences on fixed points using two step sizes. Generate
   independent prior banks on CPU threads; evaluate them on GPU. This is a cheap
   full-support importance baseline and teacher feasibility diagnostic. Never
   discard invalid target rows or call an underweighted bank a valid teacher.
3. If prior importance is adequately populated, use independent normalized banks
   as an explicitly labeled approximate-IS teacher for the first mechanism test.
   Otherwise run the existing annealed-SMC implementation from that full-support
   prior. Teacher proposal density and importance corrections must be exact for
   that proposal. SMC adds no guarantee of discovering every mode. This fallback
   is a repair of the teacher, not a change to the training question.
   The completed prior screen failed (ESS 2.31--9.53 of 512). Before generating
   new SMC populations, reuse the eight saved beta-one central populations in
   `ssl-lstm-q20-physical-annealed-smc-repair-2026-08-10/r2` if compatibility
   passes. Freeze populations 0--3 for training and 4--7 for validation.
   Verify terminal tensor hashes, pre-resampling weights, ancestry, source
   target identity and all 800 physical target values with the current backend.
   The stored density is in an affine chart: reconstruct the documented pooled
   proposal covariance and subtract its half log determinant before comparison.
   Require scaled value error <=1e-8 and affine reconstruction error <=1e-9;
   these are FP64 engineering tolerances, not posterior accuracy criteria.
   Preserve current analytic scores for inverse-mapped teacher diagnostics.
   This reuses an approximate teacher covering two known sign regions. Its
   lack of observed inter-region mutation precludes any mode-completeness claim.
   Historical CPU execution does not become GPU performance evidence.
4. Test a short NAF ladder at 0/128/512 forward updates, then 32 reverse updates.
   Start with width 64, three stages, four sigmoid components and batch 64;
   inspect heldout cross entropy, 128 paired Gaussian probes during training,
   and 1000-point saved endpoint probes. Probe seeds remain common for change
   diagnosis; reserve fresh seeds for confirmation. Validate the frozen map by
   reloading it and checking forward/inverse agreement. Preserve every endpoint.
5. If all engineering and teacher checks pass, train a second seed and extend
   the viable settings to 2048 forward updates and 128 reverse updates. If a
   rate is unstable, retry from the saved predecessor at one-third the rate;
   if it is still improving at a rung, continuation is permitted. If the
   teacher fails, repair the teacher first. Poor finite score residuals are a
   candidate/repair signal, not proof against NAF. Do not silently replace a
   failed warm-start with a demand for final-fit accuracy.
6. Decide from the measured costs whether longer 8192/16384 forward ladders,
   rate/capacity alternatives and fresh confirmation fit the remaining budget.
   Write that measured continuation decision before launching them. Stop this
   pilot at a real numerical/teacher/budget blocker and state the smallest next
   discriminating test. Do not start HMC as part of a training-only experiment.

Budget brought forward from the terminal accounting audit: 111947.281704
GPU-process seconds and 104304.108478 CPU-core seconds. All previous charges
remain. The initial pricing/teacher/pilot allocation is at most 7200 GPU-process
seconds and 14400 CPU-core seconds (a convenience cap limiting exposure to a
new target; not an expected runtime). Start with a 900-second price worker.
Timeouts include compilation. Each fresh worker has a bounded timeout, unique
versioned output and a manifest. Charge measured worker wall time to GPU and
measured CPU process/child time to CPU; maintain a local carried-forward ledger
and append compatible charges to the shared accountant. Reserve before launch;
do not reset the account. Longer runs require measured pricing, not another
human approval within the existing scope and allocation.

## Diagnostic roles and mathematical interpretation

With gamma_z=gamma(T(z))*abs(det DT(z)), record e(z)=grad log gamma_z(z)+z
and r(z)=log gamma_z(z)+||z||^2/2. Exact Gaussianization implies e=0 and
constant r. Normalization is unnecessary. Base-draw residuals can miss modes,
so accompany them with inverse-mapped validation-teacher points, importance
ESS, sign-region mass, and weighted means/covariances in physical coordinates.
Quantiles, extremes, loss and finite short-run differences are explanatory;
paired uncertainty is needed before declaring improvement. A low median cannot
override a nonfinite tail or a loss of represented teacher mass.

Constructed inexpensive adversaries: prior Gaussian (no training), diagonal
moment Gaussian (marginal scaling), and full moment Gaussian (linear correlation
removal), all in the same physical coordinates. Fit moment controls using only
training banks. Evaluate coverage and score residuals conditionally on both
signs of observation_weight.0.0, and central/tail base points. The forward
checkpoint is the objective-switch comparator. These are diagnostic controls,
not another HMC comparison campaign. Clear inferiority to a simple control in
a represented region vetoes promotion; the controls are never optimization
targets or an excuse to retune on heldout confirmation points.

Teacher screen (pilot heuristics, not exact posterior certification): at least
four independent populations split into training and validation; within each
bank ESS >=64 and maximum normalized weight <=.10; both observation-weight sign
regions represented; independent population sign masses and standardized means
agree within four estimated between-population SEs plus .05 absolute mass/.25
prior-standardized mean allowances. Report each population, not only a pooled
ESS. First price at 512 particles/population, then increase to 2048 if affordable.
Failing banks cannot provide claim-bearing teacher evidence. Any invalid target
row, mismatched identity, corrupt checkpoint, scalar fallback, batch one,
nonfinite update, or absent GPU growth is a continuation veto until repaired.

Warm-start coverage guard: each represented sign region retains at least half
the independent teacher mass, with absolute discrepancy <=.15. These exploratory
limits are inherited from the synthetic warm-start rule and explicitly do not
assert exact q20 mode masses. Training/teacher uncertainty remains a limit.

## Numerical assumptions and review

Width/stages/components/batch and forward rates .001/.0003 originate in the
synthetic study. Excess curvature or slow fitting exposes failed transfer;
128/512 updates are low-cost diagnostic rungs, not convergence budgets. RKL
starts at .0001, 32 updates, then 128 only if finite and coverage-preserving.
Adam (.9,.999,epsilon 1e-8) and no gradient clipping preserve the tested objective;
record gradients, update sizes and invalidity. No clipping is an empirical
hypothesis supported by the finite unclipped synthetic runs, checked anew here.
Full FP64 is a diagnostic exception for first q20 mechanism tests because the
current cMADE path was studied in FP64. TF32 equivalence remains a separate
required calibration before production or precision claims. CPU dataset generation
uses two threads with explicit device placement; GPU training uses memory growth,
stable signatures and XLA, never row-mapped scalar targets or pfor.

Skeptical pre-execution review: revised an initial temptation to transfer exact
synthetic component screens to q20. True q20 mode masses are unknown. Independent
teacher replication and full-support proposal corrections replace that oracle;
no proxy becomes posterior evidence. Reusing a fixed synthetic training budget
would hide tuning failure, so pricing and short rungs precede continuation.
Forward loss alone cannot assess whitening; paired scores and conditional
coverage are required. A failed intermediate final-fit screen cannot block a
planned RKL repair. The plan passes as a bounded candidate/engineering test,
with teacher adequacy, global coverage and precision explicitly unresolved.

Repair review after pricing: all 4096 prior rows were finite; two-step derivative
errors were 1.41e-7 and 3.53e-8 relative to 1+|score|. A native batch of 32 costs
about 3.00 seconds after its 15.88-second first call. Replaying 800 saved SMC
particles therefore costs about 90 seconds plus setup; reserve 450 seconds.
Reserve 4200 seconds for the first fit, including new-map compilation, controls,
two 1000-point probes and 32 actual-target updates. The three price attempts
already consumed 1955.08 GPU-process seconds. These reservations total 6605.08
seconds, within the initial 7200-second exposure cap. The teacher reuse audit
passes as a bounded warm-start test because it checks the actual target values
and chart change instead of relying on the old weight screen alone. It does
not upgrade the saved populations into exhaustive posterior evidence.

Measured stage-five continuation review: the first short pair completed in
764.70 GPU-process seconds. Forward blocks cost 71.44 seconds for 128 updates
including compilation and 42.86 seconds for the next 384; reverse compilation
plus 32 updates cost 231.75 seconds. All updates, endpoint inverses and 1000-point
probes were finite, and the represented-region coverage guard passed. Large
finite tail residuals are explanatory and a repair trigger, not a continuation
veto. A draft result note that would have stopped here was corrected before
finalizing: it would have silently strengthened the plan's criterion.

Execute `fit-extended --gpu 0 --seed 61008 --teacher <replayed SMC directory>`:
the second seed uses forward rungs 128/512/2048 and reverse rungs 32/128, with
a full 1000-point probe at each terminal objective endpoint. Retain rate .001
for this forward-budget test and .0001 for reverse; compare within that seed's
forward/reverse checkpoints, without a cross-seed or cross-budget ranking claim.
This is a budget/seed calibration, not an 8192/16384 production run. Reserve
4200 seconds, conservatively above the roughly 20--25 minute estimate including
compilation/validation. Prior attempts cost 2896.77 seconds, so the reservation
still fits the initial 7200-second exposure cap. A failed 32-update reverse
coverage guard saves its full endpoint and blocks the 128 rung for that
candidate; the planned lower-rate repair remains available within the total
campaign budget. Decide about long training after this bounded continuation.

Stage-six decision after the second seed: 2048 forward/128 reverse updates
completed in 2323.38 GPU-process seconds. The final 1000-point median/p99/max
residuals were 1.995/74.28/671.22. Coverage, finite updates and reloads passed.
All ten largest residuals had estimated log(p/q)<-3 in the same base bank;
the worst point lies inside ||z||<=3, so these are not exclusively far-base-tail
events. These are explanatory observations, not a new pass/fail threshold.
Additional empirical forward training would not directly penalize excess q
mass where the finite teacher supplies few particles. Reverse KL explicitly
averages log(q/gamma) under q and addresses this excess mass. The smallest
next test therefore extends the saved reverse128 checkpoint to reverse256,
preserving Adam and the RNG counter, at the same rate .0001 in four 32-update
blocks. This tests insufficient reverse refinement without another 8192-step
forward fit. The 256 total is the synthetic study's inherited budget hypothesis,
not a q20 convergence criterion. Retain complete endpoint probes and coverage.

Reserve 1800 seconds: 128 updates at the measured 6.53 seconds/update cost
about 836 seconds; compilation, inverse/teacher validation and the 1000-point
probe justify the remaining margin. Current pilot charges are 5220.15 GPU
seconds, so the reservation totals 7020.15, within the original 7200 limit.
Skeptical review: no new target, data, objective, architecture, precision or
scientific threshold is introduced; comparisons reuse the same probe and
teacher, with no statistical ranking claim. A failed coverage or finite check
rejects the candidate. This bounded refinement closes the initial short-test
allocation; any larger forward/capacity/teacher study needs a separately priced
continuation plan under the remaining campaign budget. Finite residuals alone
must not be relabeled as a numerical or posterior-validity veto.

Artifacts: `docs/plans/artifacts/q20-naf-forward-reverse-2026-10-06/` with unique
attempt directories, source snapshot, manifests, checkpoints, teacher tensors,
paired probe tensors, result and accounting records. Exact commands and phase
decisions will be recorded in the companion results file before/after execution.

## Terminal short-test decision

All six stages above are complete. The last 128 reverse updates took 1052.64
GPU-process seconds including final validation; total pilot GPU charge is
6272.79 seconds. Reverse256 passed finite, reload, inverse and represented-region
checks. Its 1000-point residual median/p99/max is 1.681/57.44/279.99; 76.4% of
points still exceed norm one. These finite errors do not invalidate the harness
or impose a continuation veto. The bounded pilot supports a further calibrated
training experiment but does not promote this map as correctly whitened.

The next discriminating stage is a matched reverse-rate/budget calibration,
not an automatic 8192/16384 empirical-forward ladder. At the measured 6.53
seconds/update, two 512-update reverse arms require about 6687 GPU seconds
before setup and diagnostics; an 8400-second exposure allowance is a
convenience margin supported by the observed compilation/probe costs. This
estimate is recorded for a subsequent continuation plan, not charged or launched.
Preserve the unused campaign allocation. Full HMC remains outside this
training-only pilot.

Terminal skeptical review: the engineering checks answer whether this route
executes the declared objective; they do not certify the approximate UKF target.
The two fitting seeds have different budgets, so they cannot support a seed-only
comparison. The common base bank gives descriptive within-map contrasts, not
independent confirmation. The approximate teacher remains restricted by its
proposal-supported regions. Both the lost-time backend mismatch and the finite
tail errors are preserved rather than excluded from cost or interpretation.
