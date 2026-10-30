# Rare-region methods: execution record

Status: execution and terminal review complete. Three methods resolved the
declared rare-probability task on both small targets; the global IAF/HMC
continuation did not qualify. Plan:
`bayesfilter-neutra-rare-region-master-2026-10-02.md`.
Output: `artifacts/neutra-rare-region-2026-10-02/campaign-r1`.

Read the [standalone survey PDF](artifacts/neutra-rare-region-2026-10-02/monograph-r1/rare-regions-survey.pdf)
or the [updated full monograph](artifacts/neutra-rare-region-2026-10-02/monograph-r1/BayesFilter.pdf).
The executable controller is `scripts/run_neutra_rare_region_master.py`; its
approved entry point is `scripts/run_neutra_rare_region_campaign.sh`.

## Engineering and source checks

Fifteen focused tests pass, including independent EMUS references, MH flux,
SNIS normalizer cancellation, constrained mutation, corrected stratified
training, corrupted-map rejection, rare-event false passes, and controller
bookkeeping. CPU training in the tests is a two-update mechanics exception.
The serious training route uses GPU1, verified memory growth, TF/XLA,
batch-native targets, and the shared canonical IAF implementation.

MathDevMCP's broad document audit returned `completed_with_limits`, with 12
semantic obligations not checkable. This is not a correctness certificate.
Separate SymPy-backed checks verified three elementary identities: splitting
shell conservation, exact quantile-score cancellation, and half-inside/half-
outside allocation for the oracle event SNIS proposal (the last assumes
0<a<1). Source inspection and independent numerical checks supply the remaining
bounded support. The full monograph compiled with resolved citations after
running BibTeX from the output directory; the first latexmk invocation had not
created that bibliography because TeX rejects an absolute output path for
BibTeX. No mathematical content was removed. Render inspection found an
overlong running section heading; short running titles repair it.

The owner-requested allowance is exactly the fixed campaign wrapper in
`neutra-rare-region-campaign.rules`; it was added through the approved installer.
Run and resume use exactly that wrapper. No unrestricted shell/interpreter rule
was introduced by this task. Local allowance does not override platform policy.

## Attempt and repair ledger

The first CPU preparation jobs completed for both targets. Both initial GPU
local-fit jobs then failed at the diagnostic boundary, after completing a
component's training: `PostTrainingProbe` received a TensorFlow beta scalar,
but its graph-construction validation expects a Python number. The error was
`TypeError: must be real number, not SymbolicTensor`. This is a harness/API
failure, not evidence against the IAF or rare-event methods. The two attempts
cost 119.734 GPU process-seconds and were preserved as `local-*-r1`.

Repair: pass Python `1.` at the two probe construction sites. The focused
two-update test now calls the same probe API with 20 rows and checks completion,
finiteness and all rows. All 15 tests pass. A new source snapshot and fresh `r2`
directories were launched with the unchanged target, methods, hardware and
budget. The controller reused completed preparation outputs. No renewed
campaign authorization was needed.

## Sampling results (complete)

All four methods ran on both targets. The declared longer local-map repair
also ran on both; it did not supply any narrow-event observations. Estimates
below are means over eight independent runs; parentheses contain standard
errors of those means across runs. The exact broad/narrow/mode probabilities
are 0.00134989803035, 3.09356535876e-7, and 0.666666571116.

| Target | Method | Broad probability x1000 | Narrow probability x1e7 | Right-mode probability | Screen |
|---|---|---:|---:|---:|---|
| Unwarped | Umbrella | 1.070 (0.162) | 2.469 (0.552) | .638 (.065) | Pass |
| Unwarped | Importance | 1.296 (.033) | 3.012 (.034) | .670 (.004) | Pass |
| Unwarped | Splitting | 1.341 (.049) | 3.057 (.138) | .665 (.002) | Pass |
| Unwarped | Local maps, repaired | 1.373 (.159) | No visits | .666 (.003) | Narrow event fails |
| Warped | Umbrella | 1.027 (.131) | 2.343 (.460) | .589 (.067) | Pass |
| Warped | Importance | 1.298 (.032) | 3.010 (.033) | .669 (.004) | Pass |
| Warped | Splitting | 1.343 (.029) | 2.963 (.130) | .666 (.003) | Pass |
| Warped | Local maps, repaired | 1.694 (.169) | No visits | .666 (.003) | Narrow event fails |

The screen requires observation in every replication, SE/exact probability
<= .2, and error <=max(4 SE,.1 exact probability), plus separate first/second
moment screens. All moment screens passed. These are feasibility screens, not
simultaneous confidence guarantees or statistically supported rankings. The
umbrella estimates have sizeable uncertainty. Fixed-level splitting uses
finite Metropolis mutation; ideal independent-conditional AMS unbiasedness
does not apply.

The local-map failure is specifically a failure to measure the rare event at
this budget. The repaired run retains 8192 states per replication; even 65536
iid posterior draws have only 0.020 expected narrow-event visits. It does not
invalidate the MH correction or the evidence of mode occupancy. The repair
increased warm-up and retained work as predeclared; no threshold was relaxed.

### Simple controls and individual teachers

The constructed analytic CDF/erfc oracle supplies the exact values above.
The other cheap controls are exact iid posterior sampling and equal-weight
local Laplace importance sampling with a 5% Student defensive component, each
at 4096 draws per replication. They are assessed separately on both valleys
and the right mode; neither is pooled into a single favorable average.

| Control | Unwarped narrow estimate x1e7 (SE) | Warped narrow estimate x1e7 (SE) | Narrow screen |
|---|---:|---:|---|
| Exact CDF oracle | 3.093565, deterministic | 3.093565, deterministic | Exact benchmark reference |
| Exact iid posterior | No visits | No visits | Fail |
| Laplace + defensive IS | 3.422 (1.220) | 3.463 (.970) | Fail |

The oracle is deliberately target-specific and supplies no general-purpose
sampler for an unknown posterior. No heuristic-dominance or default-promotion
claim follows from the finite stochastic comparisons. The direct bridge IS
arm is itself a simple useful adversary for more elaborate methods.

The training bank is the preselected first replication, not an oracle bank
or a pooled average. Its right-mode probabilities were 0.759/0.762 for umbrella,
0.649/0.649 for importance, and0.670/0.667 for splitting (unwarped/warped).
The umbrella teacher's narrow estimates were 1.855e-7/1.788e-7. Thus passing
an aggregate screen does not certify an individual teacher. Stratification
preserves these approximate weights; it does not correct their sampling error.

## Training results

Sixteen fits completed: four teachers x two targets x two seeds. Each ran a
crossed width/LR pilot, 1024 total forward-KL updates and reverse-KL checkpoints
at 256/1024 updates. All 48 checkpoints have complete finite 1000-point probes
and finite directed valley checks. Every reported selected forward-continuation
clip fraction was zero. That is evidence about these updates, not a universal
optimizer calibration claim or a record of every reverse-KL update.

The final random-point residual medians range from 0.052 to 0.574. Directed
valley maxima range from 10.252 to 380.096. These are descriptive diagnostics
on different probe measures, not comparable posterior expectations or uniform
bounds. They do not establish a globally whitened posterior. HMC qualification
ran for all 12 fits with passing teachers; the other 4 remain
diagnostic-only. Earlier checkpoints remain in each qualification shortlist.

For these normalized 2D targets, an exactly Gaussian pullback would have
`log(pi_z)+||z||^2/2 = -log(2*pi) = -1.837877`. A map producing only the
right component instead has an ordinary-base-point offset near
`-log(2*pi)+log(2/3) = -2.243342`, and can have tiny local score residuals.
Many recorded offsets lie near the latter value. The chapter derives the
exact translated-mixture example. This is a concrete alternative explanation
for the small residuals, not a posterior coverage test or a proof that each
learned map is identical to that example. Unknown-normalizer targets would
not permit this absolute-offset check without additional information.

## HMC results

**Zero of twelve eligible fits qualified.** The fixed latest-first shortlists
examined 35 of 36 possible checkpoints. One earlier forward checkpoint was
not reached within its shortlist time budget. Ten examined checkpoints had
no verified pair within the bounded search; 24 had no member pass the posterior
screen; one ended with `qualification_budget_exhausted`. These are bounded
failures, not proofs that no workable HMC pair exists.

The public tuner supplied 41 members for sequential posterior checks. Four
reached retained sampling and all four exhausted the 10,000-draw-per-chain
cap without passing. Their ESS and precision requirements failed; three also
failed additional R-hat/event checks. The four were one unwarped importance
member, two unwarped splitting members, and one warped splitting member.
These siblings are not independent training replications. Fourteen other
members stopped on warm-up health: in every case a chain failed the movement
requirement. Their saved states and target/log-acceptance values were finite.
Two members recorded a resource cap. Other warm-up assessments failed R-hat
or binary-event observation requirements. These categories can overlap.

In particular, some raw coordinate R-hat checks passed while the assessment
over modes and other declared quantities failed. This is the intended
distinction between within-mode behavior and global evidence. No final
confirmation holdout was opened, because no selection screen passed.

| Target | Teacher | Eligible fits | Checkpoints examined | Members reaching retained sampling | Qualified fits |
|---|---|---:|---:|---:|---:|
| Unwarped | Umbrella | 2 | 6 | 0 | 0 |
| Unwarped | Importance | 2 | 5 | 1 | 0 |
| Unwarped | Splitting | 2 | 6 | 2 | 0 |
| Warped | Umbrella | 2 | 6 | 0 | 0 |
| Warped | Importance | 2 | 6 | 0 | 0 |
| Warped | Splitting | 2 | 6 | 1 | 0 |

The v4 HMC diagnostics assess moments, modes, marginal CDF cuts and the broad
valley. They do **not** demand observation of the 3.09e-7 central event in
ordinary HMC draws. That event was the deliberate-estimator test. Thus these
HMC failures cannot be dismissed as an impossible narrow-event observation
requirement. They also do not invalidate the separate weighted estimators.

## Terminal audit and resources

The read-only audit is
`artifacts/neutra-rare-region-2026-10-02/terminal-audit-r1.json`, generated by
`scripts/summarize_neutra_rare_region_campaign.py`. It passed: both frozen
source revisions (619 files each), 442 unique recorded input hashes, all
required outputs and probe identities, GPU memory-growth records, CPU hiding,
and one-to-one shared-ledger charges agree. The three new numerical/controller
files still match the executed frozen source. Documentation was expanded afterward.
The audit records a machine-readable `dominance_not_established_no_default_promotion`
verdict and the three simple controls for each target and event.

There were 44 attempts: 42 complete workers and two preserved initial harness
failures. Complete workers comprise 30 GPU jobs and 12 CPU jobs. The ledger
charges both failed GPU attempts, all sampling repairs, and all HMC outcomes.
The largest recorded TensorFlow allocator peak was 150,424,832 bytes; this
is allocator evidence, not a whole-machine capacity guarantee.

| Resource | Allocation | Charged | Unspent |
|---|---:|---:|---:|
| GPU process-seconds | 7,200 | 4,108.190 | 3,091.810 |
| CPU core-seconds | 14,400 | 4,960.260 | 9,439.740 |

The workers ran from 05:16:40 to 06:32:24 UTC on October 2, or 75.73 elapsed
minutes including the repair pause. Summed worker charges are 68.47 GPU
process-minutes and 82.67 CPU core-minutes. These are the campaign workers;
routine tests, document builds and read-only report assembly are separate.
The shared ledger then retained 77,372.621 GPU process-seconds and
166,498.531 CPU core-seconds. Unspent budget is not authorization to silently
change the methods or promote failed maps.

The completed master leaves no worker running. Its `status` command reads the
durable record; `resume` reuses complete jobs. Scientific-screen failure
triggered the declared local-map repair automatically. The one harness defect
was fixed with a regression before resume. Unknown implementation exceptions
stop for diagnosis; the master does not claim to invent arbitrary code repairs.

The full monograph and standalone chapter build successfully. The new chapter
has resolved citations and no overfull-box warnings after equation/layout
repairs. The rendered derivations, table and bibliography were inspected.
This is author/model review; no independent human prose acceptance is claimed.

## Decision table

| Decision | Primary criterion | Vetoes | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Keep umbrella, importance and splitting as viable estimators here | Declared probability/moment screen passes on both targets | No observed correction/finite-weight veto | Eight runs; umbrella uncertainty; approximate split conditionals | Preserve corrected banks and assess target-specific extensions separately | No superiority, general-dimensional or default claim |
| Reject local maps for this narrow-event measurement budget | No central observations before/after repair | Required event-observation screen fails | Far too few ordinary draws for this event | Use deliberate event exploration when this probability matters | MH invariance and ordinary mode sampling are not rejected |
| Do not promote the trained global maps | 0/12 HMC fits qualify | Immobility, convergence/information failures and some resource limits | Limited training and search; individual teacher error | Separate training/geometry repair from sampler-budget calibration | Not an impossibility result for IAF, NeuTra, or the estimators |
| Close this bounded master | All planned phases settled and terminal integrity audit passes | No unresolved harness error | No successful final posterior assessment | Preserve results and define the next discriminating experiment | Execution completion is not scientific success |

| Inference status | Current evidence |
|---|---|
| Hard veto screen | Local-map narrow-event screen fails; 14 HMC members fail warm-up movement; no fit qualifies; initial harness failures preserved |
| Statistically supported ranking | None |
| Descriptive-only differences | Probability estimates/SEs, training losses, random/directed residuals, timing and individual bank weights |
| Default readiness | Not established |
| Next evidence needed | A matched exact-transport control for downstream checks, then a bounded training/geometry repair with mode preservation and fresh posterior assessment |

## Post-run skeptical review

The result rejects the current global-map recipe under the declared training
and HMC budgets. It does not invalidate the target, density corrections or
the four-method experiment. The strongest alternative explanations are an
inadequate training ladder, the noisy single-replication umbrella teachers,
and a restricted HMC search rather than a fundamental representational
obstruction. The small rare-stratum coefficient remains in the correctly
weighted FKL objective; extra rows alone do not amplify that coefficient.
RKL subsequently draws from the learned map and may lose physical coverage.
Those mechanisms are mathematically possible; their separate causal effects
were not identified by this experiment.

The next discriminating control is the exact marginal-transport construction
available for these two benchmarks, assessed by the same downstream procedure.
It can separate inadequate learned coordinates from restrictions in the HMC
qualification. A later training repair should measure preservation of modes
and valley geometry between distinct FKL/RKL checkpoints. The result would
be overturned by a fresh eligible map/member passing the unchanged posterior
criteria, not by a smaller random-point residual or another nominally finished
training run. That further experiment is not part of the completed four-arm
master, and no canonical architecture/default has been changed.
