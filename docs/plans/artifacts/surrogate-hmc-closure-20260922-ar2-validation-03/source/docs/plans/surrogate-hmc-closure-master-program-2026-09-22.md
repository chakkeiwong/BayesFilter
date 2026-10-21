# Surrogate-HMC closure master program

Active stage, 2026-09-23: the AR2 structural implementation, independent model
reference, first scoped control selection and shared small-matrix cost repair
have passed their declared checks. Refreshed controls and complete-transition
timing also passed in ar2-validation-01/02. The primary B forecast now fits the
remaining budget after diagnostic qualification. The master remains incomplete.
Continue scoped uncertainty calibration under
`surrogate-hmc-ar2-validation-continuation-2026-09-23.md` (3000 CPU / 5000 GPU
seconds reserved from the existing global balance) and
`surrogate-hmc-ar2-uncertainty-2026-09-23.md`. Then allocate B posterior testing;
allocate A once actual usage leaves its forecast affordable. The chronological
updates below preserve earlier accounting;
the active checkpoint and latest receipts carry current balances.

Continuation update: the initial pass is complete but the master is not. The
next existing-roadmap stage is now the general additive structural canonical
integration, specified in
`docs/plans/surrogate-hmc-structural-integration-2026-09-22.md`. Its separate
2,400 CPU / 3,600 GPU second reservation comes from the existing campaign
balance. Select `--closure-stages structural-integration`, using a fresh
`surrogate-hmc-closure-20260922-structural-NN` directory. The owner requests that
every next action identify whether it is in the master, and that in-scope work
continue without stopping at arbitrary subpass boundaries. No new direction
or extra campaign budget is proposed here.

Execution update: numerical confirmation 04 passed its unchanged scoped
diagnostic criteria. Before further launches, the final integration review
repairs shared budget discovery: both masters must read top-level receipts
from both artifact prefixes, reject unfinished closure accounting, and charge
interleaved campaign work without debiting it against this closure reservation.
Nested child receipts must not be charged a second time. These are bookkeeping
and regression repairs; no numerical protocol, inference rule or threshold changes.

The 161 original-master receipts sum to 26,974.519950 CPU and 25,240.512583
GPU seconds. The higher opening statement is preserved through the explicit
`surrogate-hmc-all-models-20260922-accounting-reconciliation-01` receipt:
214.712703 CPU and 79.294813 GPU seconds are conservative unresolved bookkeeping
reserves, **not measured new computation**. The total charged balance through
execution 04 remains 28,327.267344 CPU and 26,622.426856 GPU seconds, including
these reserves. Closure-pass balances are 1,261.965309 CPU and 2,297.380540 GPU
seconds. Execution 05 uses only the existing review stage (300 CPU-second
allowance, 240-second wall timeout, deliberate CPU-only execution):

```bash
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages review --output docs/plans/artifacts/surrogate-hmc-closure-20260922-05
```

Skeptical audit of this repair: source snapshots remain frozen during each
execution; historical evidence stays intact; the real original-master dry-run
must deduct closure receipts; closure reservation tests must distinguish other
campaign work; no additional posterior looks or GPU experiments are launched.

Execution 05 exposed twelve tests reading the live campaign directory while
their own review run was still unaccounted. Its 25.037262 CPU seconds are saved;
no numerical job ran. Preserve the runtime check for unfinished attempts and
isolate these unit fixtures from the live directory. The direct receipt-wiring
and discovery tests still exercise the real reader. Repeat only the review
stage under fresh output `surrogate-hmc-closure-20260922-06`, with the same
command and limits. This is a localized test-isolation repair, not a change to
scientific criteria or continuation conditions.

## Question and scope

Test whether the full canonical analytical DCTR force supports reliable
parameter-posterior inference under separately declared endpoint acceptance,
and measure its cost. This is a program for testing surrogate HMC. Passing a
filter fixture, calibrating a diagnostic or completing a subprocess is not a
positive posterior result. LG1 is an accepted practical pilot and is closed.
Its archived formal outcomes, and all other archived outcomes, remain intact.

The owner requested implementation, thorough review and execution on
2026-09-22. The closure driver extends the existing all-model master. It must
execute bounded independent repairs, preserve their results and propagate
missing prerequisites to dependent stages. It must never summarize blocked
model inference as completed campaign validation.

## Evidence contract and execution order

| Stage | Work and comparator | Completion evidence | Failure consequence |
|---|---|---|---|
| reporting | Separate implementation validity, sampling health, practical judgment, formal evidence and next action; read existing archives without new posterior looks | Regression cases preserve LG1 acceptance and original verdicts, distinguish unknown from invalid and reject contradictory evidence | Repair reporting; no scientific promotion |
| uncertainty | Prospectively evaluate the candidate on actual two-parameter posterior function types, using independent IID and stationary dependent draws with exactly known discrete-reference expectations; include failed-mixing controls | Fresh seeds; measured false veto and interval coverage with binomial uncertainty, declared reference/functions/looks; comparison with old rule and IID-only uncertainty | Failed candidate triggers a bounded repair or remains unqualified; blocks formal confirmation, not independent adapter work |
| structural | Trace the actual AR2 canonical consumer and check Chapter 18b deterministic completion and its independent Kalman reference | Same-law complete consumer, canonical analytical score and executable wiring/parity are required for model readiness | Singular transition or missing generative weighting/score support blocks AR2 inference; no process-noise ridge or substitute force |
| wrapped | Implement the normalized wrapped observation law with controlled alias truncation and an analytical directional derivative | Unit mass, periodicity, boundary and narrow/broad-noise comparisons; derivative parity and actual callback wiring | Repair local law; passing does not qualify the integrated likelihood or posterior reference |
| coverage | Reconcile family/variant register with concrete consumers and references; include M09 shared noise explicitly | Separate existing evidence, missing implementations and ready next actions for every registered variant | Unresolved scopes remain unresolved; no generic all-model success |
| inference readiness | Check calibrated uncertainty, exact scope, model law, reference, offline DCTR controls, public tuner and resource forecast | All required prerequisites bound to the same scope before launching a new posterior stage | Missing prerequisite blocks that stage only; LG1 is never relaunched |
| synthesis | Assemble staged execution, practical/formal judgments, repairs, remaining work and exact resource charges | Machine-readable report, readable result, source hashes, commands and run manifest | Errors remain visible; no success inferred from absence of evidence |

Finite-target versus model-posterior comparisons remain separate from
reference-corrected force tests. Construct/refine independent low-dimensional
posteriors before spending on additional finite-endpoint chains. A material
likelihood error triggers fresh scope-specific calibration and untouched
validation; more particles are not assumed to remove all finite-program bias.

## Research intent and assumptions

The proposed mechanism is a frozen position-only DCTR proposal force with
endpoint/kinetic MH correction. Wrong endpoint law, invalid numerical state,
unqualified reference and unavailable required model support veto the affected
inference. A poor candidate or an inconclusive accuracy interval does not stop
an independent repair. Acceptance and raw log-likelihood offsets are explanatory
only. Posterior agreement is judged in physical units, posterior SD and
scientifically relevant probability units with prospective margins. A new
calibration does not authorize rescoring closed chains or selecting thresholds
to make them pass. Practical judgments require explicit evidence and must not
be inferred solely from a small point estimate.

The existing binary/randomization diagnostic is a candidate because its
Gaussian calibration passed; transfer is an unproved hypothesis. Actual
function tables and controlled dependence are the early falsification test.
Discrete-reference expectations are exact for that diagnostic law only;
neither independent synthetic draws nor a reference grid proves actual HMC
mixing. The wrapped alias calculation must state its scale range and tail bound;
finite truncation is an approximation, not an unqualified exact density.
Structural covariance rank and innovation sharing are properties of the model,
not tunable numerical conveniences. Any new structural runtime must preserve
the full canonical method and analytical recursive score.

Constructed diagnostic adversaries: exact IID sampling checks false rejection;
stationary refresh chains test known positive dependence; chain-specific
offsets and drift test failed mixing; the old uncertainty rule and IID-only
standard errors are diagnostic comparators. Evaluate each regime separately.
For a future cost claim, construct exact-gradient HMC, a tuned random-walk
Metropolis baseline and direct reference sampling where feasible, at matched
posterior precision and separately by model/geometry. This repair phase makes
no speed ranking or default-readiness claim.

## Resources and stop conditions

Inherited campaign caps are 54,000 CPU-worker seconds and 226,800 GPU-wall
seconds. Starting charges are 27,189.232653 CPU and 25,319.807396 GPU; available
balances are 26,810.767347 CPU and 201,480.192604 GPU. Reserve at most 2,400 CPU
and 3,600 GPU seconds for this first closure pass, including checks and failed
attempts. This reservation is within the existing total, not an increase or
permission to exceed the LG2 posterior allocation. Before a subsequent pass,
deduct all actual charges and state its scope and reservation. Three numerical
attempts per stage are the maximum in this pass; repair must answer an identified
failure and preserve the previous output directory. Routine regression repairs
remain charged and bounded by the same cumulative resource limits; they do not
consume numerical experiment attempts (see the explicit revision below).

Use unique directories under
`docs/plans/artifacts/surrogate-hmc-closure-20260922-*`. Preserve full logs and
atomic reports. Child timeouts and CPU limits must be enforced; measure parent
and child consumption. GPU numerical work uses the tftwogpu interpreter, trusted
GPU access, memory growth and explicit device/XLA provenance. Tiny mechanics,
reference and diagnostic exceptions may be CPU-only with devices deliberately
hidden. No package mutation, Claude review or external publication is needed.

Stop an affected stage on numerical/model invalidity, missing required reference
or diagnostic, an unrecoverable harness error, or its resource cap. Continue
independent stages when safe. Exhaustion stops execution with an incomplete
report. Missing general structural implementation is not evidence against the
surrogate-HMC mechanism. No unqualified model enters tuning or sampling.

## Skeptical pre-execution review

The original roadmap could misleadingly turn a successful repair script into
an all-model result, assume normalized observation density supplies an integrated
likelihood, or treat a structural callback as canonical consumer support. The
stages above explicitly exclude those substitutions. The old short schedule is
not carried into a formal confirmation. No model is promoted on a proxy, and
cost comparisons are deferred until precision and scope match. The initial
bounded repair/qualification pass is justified; a new retained experiment still
needs its prospective scope/precision/cost design. This is a scientific
prerequisite, not a separate user-approval gate.

Implementation review must cover dependency failure propagation, budget and
process cleanup, archive preservation, report semantics, numerical invariants,
actual consumer wiring and diagnostic calibration. Focused executable checks
are required for those claims. Record commands, findings and final limitations
in the run result, and update the active checkpoint.

## Frozen first-pass protocol and command

The uncertainty stage uses the qualified KSC-T2 reference from
`surrogate-hmc-all-models-20260921-ksc-t2-posterior-01/report.json`, verifies
the saved tensors and forms normalized trapezoidal node masses. This is an
explicit finite diagnostic probability law. Its thirteen function expectations
are computed directly: two standardized means, their three second/cross moments,
six marginal CDF probabilities and two predictive probabilities. There is zero
reference error for this finite law; calibration against continuous posterior
reference error remains a separate prerequisite.

Use four chains, 512 IID replications, 256 stationary refresh replications with
retention probability 0.5, and 128 with retention probability 0.9. The first
two groups have looks at 1,000/2,000 draws; the last has looks at 4,000/8,000.
Three rejection controls each have 128 replications and 1,000/2,000-draw looks:
retention probability 0.99, chain-specific standardized parameter offsets
(-0.8, -0.25, 0.25, 0.8), and drift from -1.5 to 1.5 standardized units.
For offset/drift controls the parameter moments/CDFs are recomputed; predictive
values remain from the original draws, so these are synthetic failed-mixing
controls rather than draws from an alternative physical model.

Seeds are 2026092200 + 10000 times the group index + replication index. The
candidate uses 1,023 randomizations, its previously specified 0.01 rejection
threshold, maximum rank/folded R-hat 1.01, bulk/tail ESS at least 400, and the
maximum MCSE over 40/20/10 batches with sensitivity at most 2. The interval
multiplier is Student-t with 36 degrees of freedom, Bonferroni adjusted for
thirteen functions and two looks at family alpha 0.05. These are hypotheses
being tested, not new runtime defaults. Margins are the archived KSC function
margins. False-equivalence controls shift either a mean or CDF reference by
1.25 times its margin; report any false admission at either look.

Each stationary regime must have at least 64 admitted replications, and
95% Wilson upper bounds at most 0.05 for admitted/unconditional interval failure
and false equivalence. IID any-look false veto must additionally have an upper
bound at most 0.05. Each rejection control must have a Wilson lower bound at
least 0.95 for rejection at both looks. These are per-regime screens, not a
simultaneous all-model guarantee. Preserve every replication, including failures.

The wrapped-law worker integrates 8,192 angular midpoints at standard deviations
0.04, 0.2, 1 and pi, with mass error tolerance 1e-10. It checks a correlated
covariance/residual directional derivative against central differences (step
1e-5, absolute error at most 2e-9). Unit checks independently sum the bivariate
Gaussian aliases and check periodicity, branch boundaries and invalid inputs.
The implementation writes radial variance v, beta=c/v and conditional variance
s²=R11-c²/v, and evaluates N(r;0,v) times the wrapped N(angle-beta*r;0,s²).
The omitted angular mass is at most erfc((2K+1)pi/(sqrt(2)*sigma_max)); this is
a mass bound, not a relative density or derivative bound. The finite sum is
an explicitly bounded approximation to the normalized infinite sum.

Run from the checkout using trusted GPU permissions:

```bash
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --output docs/plans/artifacts/surrogate-hmc-closure-20260922-01
```

The master runs focused CPU tests before numerical workers, with GPUs explicitly
hidden for CPU exceptions. GPU workers verify memory growth and device placement
before the workload. Parent CPU time, worker CPU time, GPU wall time and failed
attempts count toward the reservation. Independent stages continue after another
scientific candidate fails; invalid program checks block dependent execution.

Static review found and repaired tensor-to-JSON materialization, incomplete
retry accounting, setup-failure accounting and an inherited CPU-limit hazard.
Executable regression checks cover those relevant failure paths, archive
preservation and dependency propagation. The calibration and law tests remain
the evidence for their numerical claims; review alone does not establish them.

## Fresh dependent-chain precision confirmation

The first execution is preserved in `surrogate-hmc-closure-20260922-01`.
All 20 unit checks, the wrapped-law checks, IID calibration and three rejection
controls passed. The two stationary dependent groups have estimated any-look
interval failure rates of 2.34%; their 95% upper bounds are 5.018% and 6.664%.
This does not establish a failure rate above 5%. It leaves the declared
calibration screen unresolved. The method, intervals, margins and thresholds
are not changed to obtain a passing result.

Before any new draws, specify one fresh confirmation: 1,024 replications for
each of the two dependent groups, with the same looks and retention probabilities.
Seeds start at 2026192200 and 2026202200, disjoint from all initial seeds. Do not
pool initial and confirmation outcomes, change thresholds or continue adding
looks until passing. Preserve the initial outcome. The candidate qualifies only
for this discrete-law diagnostic if both fresh groups pass and the original
IID/rejection controls passed. A subsequent real-chain comparison still requires
reference-error, precision and scope qualification. A failed fresh confirmation
remains a failed or inconclusive candidate, not grounds to rescore old chains.

This uses the original reservation. Initial charges are 282.045906 CPU-worker
seconds and 279.682331 GPU-wall seconds, leaving 2,117.954094 CPU and 3,320.317669
GPU in that reservation. The initial numerical worker cost 235.666363 CPU and
273.115128 GPU-wall seconds. Scaling by the total draw count ratio 3.076923,
adding 50% headroom and 240 seconds for compilation gives forecasts 1,327.691
CPU and 1,500.531 GPU seconds. These fit the 1,650 CPU/3,000 GPU job bounds and
leave room for a 300-CPU-second expanded master regression suite. Forecasting
by draw count is an imperfect convenience model: the runtime watchdog remains
the stopping rule if sorting, compilation or device costs exceed the forecast.

The follow-up also runs existing master lifecycle and dependency regressions,
since the new stage-selection interface must preserve those behaviors. A fresh
output preserves the first run's source snapshots and calibration decisions:

```bash
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages uncertainty-confirmation --output docs/plans/artifacts/surrogate-hmc-closure-20260922-02
```

Skeptical review: the new confirmation tests the same diagnostic against the
same known law. Additional independent replications address uncertainty in a
failure-rate estimate. They do not repair structural filtering, validate a new
posterior or create an efficiency ranking. The observed chain rejection controls
remain available to falsify the interpretation; all original results are retained.

## Localized regression repair before confirmation

Execution `surrogate-hmc-closure-20260922-02` stopped at the expanded regression
suite: 61 tests passed and six failed. No confirmation draws were generated.
Four failures inherited the real SV20 pilot history despite testing an empty
campaign; two assumed the superseded 1,800-CPU/7,200-GPU family ceiling. The
September 22 SV20 inference plan explicitly increased that ceiling to
13,000 CPU and 14,400 GPU within the unchanged global allocation. Restore test
isolation by running the actual SV history readers against an empty temporary
root, and test both accounting and overrun rejection against the current
ceiling. Do not change the master history, prior outcomes or budget policy to
make these tests pass.

This failed attempt consumed 50.319734 CPU seconds and no GPU seconds. The
closure reservation now has 2,067.634360 CPU and 3,320.317669 GPU seconds left.
The same confirmation, with unchanged seeds, method and statistical criteria,
will run after the complete 67-test suite in a fresh output directory:

```bash
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages uncertainty-confirmation --output docs/plans/artifacts/surrogate-hmc-closure-20260922-03
```

Skeptical audit: the failures concern test setup and an obsolete allocation,
not observed posterior disagreement or the uncertainty candidate. The repair
preserves the tests' independent-history and overrun-rejection assertions. This
is the third and final review attempt allowed by the pass; any new review
failure must be recorded rather than bypassed.

## Regression retry accounting correction

Execution 03 passed 65 tests and stopped at two remaining stale assertions:
the model register now preserves all four comparison statuses, whereas these
tests still expected a scalar `NOT_RUN`. Assert that all four comparisons are
`NOT_RUN`. No runtime or statistical result is changed. This attempt consumed
50.311909 CPU seconds and no GPU seconds; it generated no confirmation draws.

The earlier sentence making this the final review attempt is superseded here.
The driver had counted every successful regression run as an experiment retry
and would now stop a routine, identified test repair despite an unchanged
scientific protocol and ample reserved compute. Apply the repository's campaign
repair policy: keep the complete passing regression suite mandatory, charge
every repair against the same 2,400 CPU /3,600 GPU reservation, and retain the
three-attempt limit for numerical stages. A regression test verifies both sides
of this correction: review can run, while an exhausted numerical stage cannot.
This revises an implementation retry rule, not a scientific promotion criterion.

There are 2,017.322451 CPU and 3,320.317669 GPU seconds left in this reservation.
Execution 04 runs the complete 68-test suite and, only if it passes, the same
single fresh confirmation already specified. There has been no confirmation
attempt and no selection among fresh calibration results.

```bash
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages uncertainty-confirmation --output docs/plans/artifacts/surrogate-hmc-closure-20260922-04
```
