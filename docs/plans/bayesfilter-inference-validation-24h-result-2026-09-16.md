# Inference validation campaign results

Date: 2026-09-16. Status: the bounded campaign is complete; the larger validation
and tuning-repair program remains open. All launched workers have ended. Design:
[funded campaign](bayesfilter-inference-validation-24h-campaign-2026-09-16.md).
All records are under
[`inference-validation-24h-2026-09-16/`](artifacts/inference-validation-24h-2026-09-16/).

## Findings

The infrastructure now exposes failures that the earlier small integration
tests did not. It does not establish that the complete automatic tuning and
posterior procedure is calibrated.

The separately declared prepared-route confirmation completed **224/224 fits
across 32/32 datasets**, and every selected member passed its declared posterior
checks. The parameter, bounded-radius and data-dependent log-likelihood rank
tests had p-values 0.6385, 0.12 and 0.5075, respectively, against the predeclared
within-family threshold 0.05/3. No discrepancy was detected. Its separately
calibrated test sensitivity addresses a one-posterior-SD location bias, not
small errors or arbitrary defects. This is evidence for the explicit identity
geometry, prepared public entry route and stated output/stopping rule; it is
not calibration evidence for automatic preparation. The complete aggregate is
[`cpu-prepared-sbc-aggregate.json`](artifacts/inference-validation-24h-2026-09-16/prepared-confirmation32-r1/cpu-prepared-sbc-aggregate.json).

The 64-dataset CPU ordinary experiment attempted 448 independent full fits.
Only 15 datasets supplied all seven outputs. Seventy-one fits rejected the
validation design's explicit epsilon 0.6 after preparation produced a smaller
bound; fifteen further fits returned no verified member. The former is a
baseline mismatch in the validation design: this was a supplied initial proposal,
not untouched automatic initialization. No unconditional calibration conclusion
can be drawn from the 15 complete datasets. The corresponding GPU experiment
was cancelled, with partial records preserved and consumed time charged.

A fresh native-initialization CPU check used no search override. It attempted
16 complete fits across eight datasets; seven returned no verified candidate,
leaving four complete datasets. There were no initial-proposal exceptions.
In inspected failures, every L requested a higher epsilon while already at the
preparation ceiling. This identifies a remaining availability problem in the
interaction between preparation's bound and candidate qualification. It does
not show that the posterior itself is intractable or that no valid HMC setting
exists. The campaign preserves the bound and admission criteria under test.

The native GPU check planned eight two-fit datasets. Three of four workers
reached their 1,500-second caps. Five dataset records were saved, including two
complete datasets and four recorded zero-member fits; the remaining dataset
records are missing. The only worker to finish its full shard completed zero
of its two datasets. The group cannot support unconditional calibration or a
precise GPU availability estimate. Partial completed datasets remain visible
in the terminal inventory but do not substitute for finished workers in the
predeclared aggregate.

All eight CPU and all eight GPU Gaussian stopped/fixed replications returned
both arms. All eight CPU
single-mode mixture replications failed posterior readiness: seven stopped at
the warmup cap without retained draws and one reached the retained cap. Fixed
sampling also often missed the known global mode probability. The controller
did not report these mixture runs as posterior-ready. This is evidence about
these starts, target and kernels, not a general guarantee that local diagnostics
detect missed modes. Eight replications do not establish 95% interval coverage
or a ranking of stopping rules.

The completed GPU mixture experiment gives a counterexample to global
readiness inferred from local checks. One of eight replications reported
`POSTERIOR_DECLARED_CHECKS_PASSED` after 6,500 discarded transitions and 2,000
retained transitions per chain, but every retained draw was in the right mode.
Its estimated left-mode probability was zero, against the exact
`0.3000001146606287`. Its estimated x mean was `4.9560705856`, against the exact
2.0, while the reported mean MCSE was `0.0379627688`. The other seven GPU
mixture replications did not pass readiness: four hit the warmup cap and three
hit the retained cap. These counts have only eight independent replications;
they establish the observed counterexample, not a precise false-readiness rate
or a CPU/GPU comparison. The separately seeded fixed arms also have mode errors.
The maximum modern R-hat was `1.0068966028`; the exact binomial interval for
the observed one-of-eight false-favorable count is 0.00316--0.52651. The full
record is
[`gpu-mixture-stopping-0/attempt-001-result.json`](artifacts/inference-validation-24h-2026-09-16/repair-gpu-r1/gpu-mixture-stopping-0/attempt-001-result.json),
replication 0. The saved mode probability and exact mean comparisons support
the missed-mode verdict independently of descriptive acceptance or runtime.

This does not show an arithmetic error in R-hat or lugsail MCSE. Those local
statistics were calculated on chains that missed a mode. The finite-variance
CLT and adequate-exploration assumptions needed to interpret MCSE as uncertainty
about the full posterior were not established by the local screen. A known
mode-indicator diagnostic was available in the independent assessment but was
not one of the controller's declared parameter mean/median checks. Future
target-specific posterior policies should include such scientific quantities
and dispersed starts, with unavailable mode precision preventing a readiness
claim for that quantity. This is a posterior-validation repair; it must not
make R-hat or mode occupancy into a kernel tuning requirement.

The simplex reporting repair was exercised by the CPU pilot on all 23 verified
members: two active coordinates correctly produced three named probabilities.
The all-member GPU pilot exceeded its 600-second allocation and remains
incomplete. Its failed attempt is retained; CPU results do not fill GPU cells.
The later predeclared GPU subset experiment completed both replications,
retaining 21 and 23 verified candidates. Its two selected members each returned
1,000 retained draws per chain, passed the declared posterior checks, and
produced all three probabilities. The other 42 members remain explicitly
unassessed for posterior accuracy; their tuning qualification is preserved.

## Sensitivity and numerical checks

| Experiment | Observation | Interpretation |
| --- | --- | --- |
| 64-dataset rank primitive, one-SD posterior shift | 256/256 detected; 95% interval 0.986--1 | Sensitive to the declared large analytic defect |
| 64-dataset primitive, half-SD shift | 141/256; interval 0.488--0.613 | Insufficient sensitivity to rule out this smaller error |
| 128-dataset primitive, half-SD shift | 232/256; interval 0.864--0.939 | Higher sensitivity, but not affordable for every full procedure |
| 32-dataset primitive, one-SD shift | 232/256; interval 0.864--0.939 | Meets the separately declared 0.8 lower-bound planning criterion |
| Actual Gaussian HMC, reversed MH ratio, epsilon 0.3 | 3/32 detected | Weak detection at this kernel setting |
| Same actual defect, epsilon 0.6 | 1/32 detected | Weak detection; severe-control success cannot hide this |
| Same actual defect, epsilon 1.0 | 32/32; interval 0.891--1 | Strong observed detection for this setting |
| GPU actual defect, epsilon 0.3 / 0.6 / 1.0 | 1/32, 1/32, 32/32 | Same subtle-defect sensitivity limitation; device results are separate |
| Actual-kernel baseline and no-op controls | 0--2 rejections per 32 experiments | Reported with wide binomial intervals, no exact-size certification |
| CPU and GPU density/score mechanisms | All thirteen target laws passed on each device | Numerical checks on declared probes, not full-pipeline correctness |

The thirteen laws are Gaussian, rotated Gaussian, banana, funnel, Student t,
Cauchy, Gaussian mixture, Gamma, Beta, Dirichlet, normal-normal conjugate,
beta-binomial and a one-parameter LGSSM location posterior. The latter's
independent likelihood uses scalar Kalman recursions. It does not validate
inference over arbitrary unknown state-space parameters.

Small full-fit CPU LGSSM SBC completed two datasets with two fits each. The
beta-binomial case completed only one of two datasets. Cauchy and Gamma posterior
cases returned assessed outputs. The frozen affine rotated-Gaussian cases
reached the warmup cap without retained output. These are distinct findings;
the mechanics tests do not replace the missing posterior evidence.
Both GPU affine rotated-Gaussian replications likewise reached the 10,000
warmup cap with no retained draws.

GPU beta-binomial SBC completed both two-fit datasets. GPU LGSSM-location SBC
completed neither two-fit dataset because each had one zero-member fit; it
remains incomplete. The ordinary GPU
funnel run failed in operational mass preparation with `hard_veto`, before
candidate search. Its saved exception lacks the underlying numerical veto
detail, so the precise trigger remains unclassified; it is not evidence of
an epsilon-proposal mismatch or a posterior failure. None of these tiny cases
provides a powered target-specific calibration claim.

The frozen dense-IAF GPU check completed two replications, retaining four and
two verified members respectively. All six members passed their declared
posterior checks and returned assessed model-coordinate output within the
predeclared descriptive reference tolerance. This tests the synthetic frozen
payload, transformed density, candidate qualification and retained replay. It
does not test learning the transport, broad L coverage or statistical calibration
of trained NeuTra procedures.

## Resources, provenance and terminal audit

| Accounting | CPU reference lane | GPU lane |
| --- | ---: | ---: |
| Authorized worker-seconds | 86,400 | 86,400 |
| Measured numerical worker-seconds | 77,475.83 | 40,224.74 |
| Conservative test/report/build charge | 1,000.00 | 0 |
| Total charged hours | **21.80** | **11.17** |
| Unused seconds | 7,924.17 | 46,175.26 |

The accounting unit, declared before execution, is summed worker wall time in
each lane, not calendar duration or a CPU/GPU speed comparison. It includes the
cancelled GPU experiment (22,602.56 seconds), the timed-out native GPU and
simplex jobs, the failed funnel job and every other numerical attempt. CPU
reference workers used two intra-op threads and one inter-op thread with GPUs
intentionally hidden. GPU workers used device 1, an RTX 4080 SUPER, TensorFlow
2.20.0, TFP 0.25.0, XLA enabled, TF32 enabled, and verified memory growth before
initialization. The interpreter was the existing `tfgpu` Python 3.13.13.
Per-worker commands, resolved designs and seeds are in the run indexes and
attempt manifests; the recorded base Git commit is
`d31e1b7618f613b9f7a7ec7ace7f11556870fb4e` plus each saved source snapshot.

The final audits found no inconsistency in 101 completed result hashes, 1,658
tensor checksums, 110 worker runtime manifests, seven frozen source snapshots,
or 794 tuning records. Those tuning records contain 46,183 candidate records
and 9,051 verified-member instances across separate fits. These are inventory
counts, not independent statistical replications. Finished pipeline records
preserved every verified member; one capped GPU pilot retained its tuning
observation without a finished pipeline. In 4,629 verified-member instances,
a qualifying verification observation had a nonpassing reported R-hat, confirming
that those reported values did not prevent tuning retention. The tests separately
cover that policy; this count alone is not a proof about all inputs.

The [terminal ledger](artifacts/inference-validation-24h-2026-09-16/terminal_audit.json),
[dataset inventory](artifacts/inference-validation-24h-2026-09-16/terminal-inventory-audit-r2.json),
[source/runtime/candidate audit](artifacts/inference-validation-24h-2026-09-16/terminal-execution-audit.json)
and [result summary](artifacts/inference-validation-24h-2026-09-16/terminal-result-summary.json)
are the reproducible terminal records. Their scripts launch no numerical workers.
Unstarted cancelled jobs keep their statistical denominator and zero worker
cost. For partial GPU shards, the native aggregate's `unstarted_datasets` field
means records unavailable to aggregation; the terminal dataset inventory
distinguishes genuinely unstarted datasets from started but unfinished ones.

The repair, prepared and dense-IAF snapshots share source identity
`19f15741104c93927b3843445bf0b9e030dd09c05d6bb3ce76acdcbdb9534c41`.
Independent concurrent Q20/training edits changed the current whole-package
hash after the freeze. The [source comparison](artifacts/inference-validation-24h-2026-09-16/terminal-source-delta.json)
lists those paths. The generated coverage table therefore labels campaign
executions historical relative to the current checkout. The tested snapshots
remain intact; no untested source version inherits their findings.

## Implementation and verification

The shared posterior controller now permits a fixed reported dimension
different from the HMC state dimension, while preserving chain/draw axes,
active continuation state, named quantities and empty shapes. The retained
bridge records actual elapsed time and native execution metadata. First-call
timing combines compilation and execution; it is not reported as pure sampling.

Validation supports predeclared member subsets, independently seeded fixed
comparators, actual-stop reference functionals, complete missing-arm denominators,
bounded concurrent workers, frozen source snapshots, predeclared SBC aggregation,
and live accounting of failed/cancelled attempts. All verified siblings remain
saved. `native_search` explicitly distinguishes automatic initialization from
supplied proposals. Coverage rendering recomputes source status rather than
trusting an old report's claim to match the current checkout.

The expanded regression command passed **187 tests**. Separate accounting and
coverage-source checks passed, including a final **30-test** reporting/documentation
command after repairing the unstarted-job denominator and overhead accounting.
These commands overlap and are not 217 distinct tests. The tuning chapter was
rebuilt as a standalone 29-page document after refreshing coverage from 23 saved
reports; its changed validation prose and table were visually inspected.
The [compiled chapter](artifacts/inference-validation-24h-2026-09-16/guide-build-r2/tuning-chapter.pdf)
and build manifest are preserved with the campaign.
The standalone build retains an unresolved reference to the diagnostics chapter,
which is outside that extracted document, plus pre-existing long-code overflows.
The reference guide and suite documentation now describe subset selection,
dimension-changing transforms, fixed comparators, native initialization and
incomplete calibration consistently.

## Decisions

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Preserve supplied-proposal ordinary experiment as failed/incomplete | 15/64 complete datasets | Missing fits veto calibration | Availability combines design mismatch and bounded-search failure | Use native-initialization findings separately | Automatic default failure rate or calibration |
| Retain prepared-route rank evidence | 32/32 datasets and 224/224 fits; no discrepancy in three declared tests | No missing outputs | Smaller biases and arbitrary defects unresolved | Preserve as prepared-route baseline; test repaired ordinary route on fresh data | Automatic preparation calibrated or all candidates validated |
| Retain native ordinary availability finding | 4/8 CPU datasets complete | Seven zero-member fits | Small sample; ceiling behavior target-dependent | Investigate safe requalification of exhausted proposal bounds | No valid kernel exists |
| Keep actual-stop and fixed-arm comparison | Complete Gaussian arms; mixture failures preserved | Mixture readiness and global reference failures | Only eight replications per regime | Broader starts, mode functionals and more replications | Sequential coverage or superiority |
| Reject global readiness for the GPU mixture counterexample | Local checks passed, exact mode probability and mean disagreed | Missed mode vetoes posterior interpretation | One observed case; rate uncertain | Add target-specific global quantities to posterior assessment and test dispersed starts | Local checks establish global mixing |
| Keep validation infrastructure changes | Focused and integration regressions pass | Terminal inventory and checksum audits found no inconsistency | Limited route/target coverage | Use preserved failures to drive the next focused repair | Universal sampler validation |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Missing ordinary fits block unconditional calibration; mixture cap and global reference failures block posterior use even when local checks pass |
| Statistically supported ranking | None sought or established |
| Descriptive differences | Runtime, posterior errors and small-sample stopping coverage |
| Default readiness | Not established; automatic search availability remains unresolved |
| Next evidence needed | A reviewed bound-requalification repair followed by fresh complete ordinary SBC; target-specific global posterior quantities and dispersed starts; stronger subtle-defect power and matched consumer references |

## Scope and next work

The owner allowance funds a substantial bounded campaign, not every target,
defect severity or external reference. Actual eight-schools, regression and
MacroFinance reference bundles remain absent. The generic reader is available,
but no external reference or consumer fit is fabricated. Frozen transport
mechanics do not establish transport training quality.

In the incomplete ordinary experiments, success-conditioned data can explain
apparent rank agreement: only easy fits may return outputs. The missing-fit
denominator prevents that agreement from becoming a calibration claim. The
prepared32 experiment has no missing-fit selection, but its limited test
sensitivity still leaves small or differently structured errors unresolved. The
strongest alternative explanation for ordinary no-member outcomes is an overly
restrictive inherited proposal ceiling, rather than a defect in invariant HMC
transitions. The next discriminating evidence is measured same-target search
under a separately justified, requalified domain, followed by fresh full-procedure
data. The current experiment does not silently change that domain.

The source-level chain is specific. `hmc_warmup.py::find_reasonable_epsilon`
returns a finite step whose mean acceptance over the declared momentum probes
is between 0.25 and 0.75 for the preparation trajectory. Operational warmup uses
four momentum probes and sets its active upper bound to that selected step,
including after metric changes. `hmc_kernel_tuning.py::_fixed_mass_step_upper_bound`
checks final-metric identity and passes that bound to ordinary candidate search.
This is an inherited engineering restriction; the probe does not prove that the
same ceiling includes an acceptance-qualified pair for every later L. The
seven native CPU zero-member fits each start at the ceiling and record
`repair_step_higher` for all six primary L values.

The next repair should separate a starting epsilon from the justified search
domain. First preserve the native failures as deterministic regression fixtures
and reconstruct their exact frozen geometry, starts and seeds. Then evaluate a
bounded, explicitly recorded domain-requalification procedure in those same
coordinates, with numerical health checks and fresh per-pair measurement and
verification. A changed domain needs a new immutable search identity; it cannot
retroactively alter an existing candidate or receipt. Acceptance thresholds and
all-member retention stay fixed, and R-hat stays reporting-only in tuning.
Only fresh complete ordinary SBC after that repair can address its calibration.
The prepared identity experiment is not evidence that the changed ordinary
procedure works.

The remaining work is ordered by what this campaign actually found:

1. Repair the ordinary preparation/search ceiling interaction, preserving the
   final metric, health requirements, immutable identities and all qualified
   members. Diagnose the existing failures before selecting a bounded expansion
   policy; validate the revised procedure on fresh complete ordinary SBC.
2. Preserve detailed preparation veto reasons before raising exceptions. The
   funnel failure currently exposes `hard_veto` without enough saved detail to
   distinguish the numerical cause. Also name aggregate missing-record fields
   accurately for capped shards.
3. Exercise target-specific global posterior quantities and dispersed starts
   against the saved mixture counterexample. Undefined/degenerate mode MCSE must
   remain unavailable. Adding a mode statistic or requiring more local samples
   is not a general solution to undiscovered modes; keep independent references
   and sensitivity experiments.
4. Calibrate actual acceptance-screen operating characteristics near boundaries
   and strengthen subtle-defect power at the declared scientific tolerance.
   The fixed-look invariance implementation still lacks the optional sequential
   wrapper. More repetitions of an insensitive test do not by themselves make
   the tested defect detectable.
5. Add matched eight-schools, regression and MacroFinance references, then
   validate the relevant complete consumer procedure. Frozen synthetic transport
   replay cannot fill training or real-consumer evidence gaps.

The campaign stopped because its predeclared questions and final route check
had terminal outcomes, not because an expected candidate failure invalidated
HMC. Unused budget remains unspent. None of the numerical findings justifies
relaxing qualification thresholds or declaring a new scientific default.
