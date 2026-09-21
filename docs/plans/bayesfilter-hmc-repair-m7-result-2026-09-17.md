# HMC master program: M7 execution and remaining work

Date: 2026-09-17; updated 2026-09-18. The resolved development tranche is complete: all 16 GPU
suite jobs finished, the four pipeline pilots were assessed, and the additional
funnel grid diagnosis finished after one preserved configuration failure. No M7
worker is running. Bootstrap, windowed preparation and configuration/budget
translation are also extracted, with their checks and extended budget ledger
below. M7 remains open for the scientific work and historical-code cleanup. The [master program](bayesfilter-hmc-repair-master-program-2026-09-16.md)
is the single active continuation; earlier results retain their original scope.

## Outcome

The energy oracle detects the deliberately reversed Metropolis ratio, but the
fixed-look distributional experiment still has weak observed sensitivity.
Automatic beta-binomial, LGSSM-location and rotated-Gaussian pilots each retained
multiple verified settings and produced one predeclared posterior that passed
its checks. The funnel produced no verified setting. A wider supplied grid
under that same geometry also produced none; this finite experiment does not
establish that every possible setting fails.

R-hat, ESS and MCSE remain posterior diagnostics. None qualifies, ranks, repairs
or delays tuning candidates. All verified settings remain retained; selecting
one identity before posterior draws does not discard its siblings. No numerical
default or statistical ranking was promoted.

## Engineering repairs and verification

Initial geometry lives in `bayesfilter/inference/hmc_geometry.py`, with public
imports and historical `hmc_kernel_tuning` aliases resolving to the same objects.
All 21 definition ASTs (622 moved lines) are unchanged. Eleven before/after cases
match exactly: identity, dense Hessian/covariance, scales, hint precedence,
regularization, fallback and invalid inputs. Historical pickle lookup works.
At that first extraction boundary, bootstrap and windowed preparation still
depended on the historical module. Both were subsequently extracted as
recorded below, followed by public configuration/budget translation.

The broader preparation tests exposed seven pre-existing failures, reproduced
in the isolated pre-extraction package. Six were stale assertions. The real
P4-E compatibility defect now rejects an explicitly requested P4-E stage paired
with an ordinary bank. Archived seed manifests were preserved: the historical
whole `sample_chain` call changed when failure recording added
`parallel_iterations`, but its separate `seed=seed` AST and other named seed
operations remain unchanged. This does not audit arbitrary dynamic seed routes.

The validation implementation composes complete fixed MH transitions for K^s,
with stable TensorFlow signatures, stateless substep streams and XLA on GPU.
Every intermediate state and log ratio is checked for finiteness; a later finite
value cannot conceal an invalid substep. Power one preserves its original
stream. Deadlines are checked between calls. The independent Gaussian energy
oracle checks the actual TFP log ratio and state selection. Host profiles survive
engine failure, and profile-write errors preserve the original failure.

Two further repairs were made during GPU execution:

- Default suite coverage omitted design identity. Completing one epsilon cell
  could incorrectly mark all same-category requirements complete. A saved-result
  counterexample reproduced this. Newly inferred requirements include each
  design ID; explicit category requirements preserve their intended semantics.
  Partial execution and corrupt-result regressions pass. Historical plans retain
  their old meaning and require inspection of every design row.
- Pilot profiles exposed repeated generic container checks on scalar leaves in
  checksum serialization. Exact-built-in fast paths preserve the original
  subclass/mapping precedence, nonfinite rejection, bytes and hashes. All 208
  beta-binomial evidence hashes match the frozen implementation. Every checksum
  validation and checkpoint write remains in place.

| Verification | Result | Evidence under `m7-r1/` |
| --- | --- | --- |
| Geometry, bootstrap, windowed preparation, candidate execution and interface | 209 passed, 1 skipped | `preparation-verified.xml` |
| Validation definitions, mechanics, powers, deadlines and profiling | 84 passed | `validation-verified.xml` |
| Coverage repair and affected execution/definition/rendering tests | 22 passed | `coverage-verified.xml` |
| Serialization repair, controller, artifacts, numerical checkpoint and retained replay | 101 passed | `checksum-verified.xml` |
| Guide/interface after documentation edits | 15 passed | `guide-interface-r2.xml` |
| Distinct cases across these overlapping batches | 336 passed, 1 skipped | `reconciliation-r2.json` |
| Geometry and checksum comparisons | 11 exact cases; 21 unchanged ASTs; 208 identical evidence hashes | geometry JSON files and `checksum-parity.json` |

The inherited tiny-Gaussian bootstrap case exhausted its own repair budget and
remains skipped, not passed. Live windowed-preparation integration tests passed.
Earlier failing attempts remain preserved and charged. These batches span the
recorded engineering stages; they are not 336 independent scientific trials.

The GPU experiments use immutable package identity
`ff0294aa48dd1ced41b00fa26a4cc3523aa32e3571dc3a0e0ed75052e46278ce`, based on Git
`d86dadf68ea57772642c6802990f46a3c6a04c30`. All 473 snapshot Python files were
verified. At the GPU tranche closeout, the live checkout differed in exactly two definitions: `plan_suite`
for coverage and `_stable_payload` for scalar serialization. Neither change is
silently attributed to the frozen GPU execution. The checksum comparison and
affected numerical regressions test those repairs separately. The later bootstrap
extraction has its own source comparison; it does not relabel the frozen GPU jobs.

## GPU results

The [saved summary](artifacts/hmc-repair-master-2026-09-16/m7-r1/gpu-summary-r1.json)
links findings, runtimes and host profiles. All workers used trusted GPU 1,
TensorFlow 2.20.0, TFP 0.25.0, float64 target calculations, XLA and verified
memory growth. TF32 was enabled and recorded. Earlier busy-device probes remain
historical allocation evidence; probes 06 and 07 passed before the later launches.

All six energy jobs completed in 42.80803556606406 GPU worker-seconds. Baseline
and no-op pass at epsilon .3 and .6, with zero observed log-ratio discrepancy.
The reversed-ratio controls fail that independent check at both settings.
Their maximum observed discrepancies were .126590447927712 and
.04237415036744041. All saved energy/ratio values are finite. This is an
engineering oracle; it does not establish distributional power.

All six power jobs completed in 407.04321426688693 worker-seconds. Each has
eight complete experiments per arm, 128 exact anchors and seven rank positions
after the anchor. All 144 executions retained finite intermediate evidence.

| Epsilon | Kernel power | Baseline rejections / 8 | No-op rejections / 8 | Defect detections / 8 |
| ---: | ---: | ---: | ---: | ---: |
| .3 | 1 | 1 | 0 | 0 |
| .3 | 8 | 0 | 1 | 3 |
| .3 | 32 | 0 | 0 | 0 |
| .6 | 1 | 0 | 0 | 1 |
| .6 | 8 | 1 | 0 | 0 |
| .6 | 32 | 0 | 0 | 0 |

The exact 95% binomial intervals are [0,.3694] for 0/8, [.0032,.5265] for 1/8,
and [.0852,.7551] for 3/8. Read each cell separately; these data do not support
a ranking of powers, empirical size calibration or reliable subtle-defect
detection. Longer composition alone did not close the gap.

| Automatic pilot | Worker seconds | Proposed / verified settings | Predeclared posterior |
| --- | ---: | ---: | --- |
| Beta-binomial | 805.911 | 100 / 31 | Passed; 2,000 warmup + 1,000 retained per chain |
| LGSSM location | 689.049 | 100 / 20 | Passed; 2,000 warmup + 1,000 retained per chain |
| Funnel | 161.650 | 7 / 0 | Unavailable; all six requested mean/median intervals remain missing |
| Rotated Gaussian | 622.506 | 100 / 22 | Passed; 2,000 warmup + 1,000 retained per chain |

All four inventory checks pass. The three assessed outputs are within the
predeclared descriptive reference tolerance. The other 70 verified settings
remain unassessed by design. These are four availability/cost pilots, with one
fit per target; they are not broad SBC or stopping calibration. Generated
beta-binomial and LGSSM observations and simulator seeds were saved; generating
parameters were not passed to their fits.

The funnel prepared successfully, but its starting epsilon and doubled repair
encountered nonfinite proposal, score, ratio or momentum evidence. Rejecting
these candidates was correct. The empty result supplies no posterior evidence.

The separate [funnel grid diagnosis](artifacts/hmc-repair-master-2026-09-16/m7-r1/funnel-grid-r2/diagnostic.json)
preserved the exact target, mass, starts and acceptance policy and issued fresh
streams through the same public tuner. It tested all 30 declared pairs at
multipliers (1/4,1/2,1,sqrt(2),2) of the original epsilon across the original
six L values. Forty evidence work items completed; 29 candidates failed
promotion and L=18, epsilon=.2256942307263964 remained inconclusive at its cap.
No pair verified. Smaller tested steps often requested higher epsilon; other
settings produced invalid evidence. This leaves geometry, untested settings
and limited evidence as live explanations rather than proving impossibility.

The first diagnostic attempt failed before transitions because the harness
requested a zero repair reserve. Its script and 3.875301291991491-second charge
are preserved in `funnel-grid-r1/`. After a configuration-only regression, the
retry used one mandatory bookkeeping unit, the same 420 evidence units and the
remaining original time allocation. It completed in 205.24413969699526 seconds.
Neither attempt replaces the failed automatic fit or supplies posterior draws.

## Costs, guide and reproducibility

The four profiled pilots used 2,279.1157282770146 GPU worker-seconds. Their
preparation portions were approximately 45, 61, 56 and 73 seconds. TensorFlow
calls dominate measured costs; those calls combine compilation and execution.
The beta-binomial profile also records 992 checkpoint calls, about 237 seconds
cumulatively in checkpoint writing, and 110,531 checksum calls from that writer.
Cumulative profile rows overlap, and profiling itself adds overhead.

Five alternating unprofiled host timing repeats over the same 208 saved evidence
payloads had medians .0991 seconds for the original serializer and .0815 seconds
for the repaired serializer. A GPU pilot and focused CPU tests were running
concurrently. These are descriptive timings; end-to-end speedup is unmeasured.

The single official book remains [docs/main.tex](../main.tex) and
[docs/main.pdf](../main.pdf). Its 563-page build includes the completion-reporting
repair and refreshed coverage generated from 32 reports. Physical pages 427--429
were visually checked across the builds. Source and PDF hashes were checked
before installation. Dated builds are archival evidence. Three pre-existing
unresolved citations in other material remain (`Afshar2015`, `Gorinova2020`,
`Pakman2014`). The coverage table labels frozen-source observations historical
relative to the later live repairs; it does not relabel them as current executions.

Exact commands, environment, durations, return codes, logs and checksums are in
[m7-r1](artifacts/hmc-repair-master-2026-09-16/m7-r1/). The
[terminal reconciliation](artifacts/hmc-repair-master-2026-09-16/m7-r1/reconciliation-r2.json)
checks every planned suite job, result identity, runtime policy, snapshot and
saved diagnostic attempt. No launched work or reservation remains unaccounted.
CPU reference/tests intentionally hide GPUs. The 300-second conservative CPU
overhead covers unmetered inspection, probes, reporting and configuration smoke
checks; measured builds/tests and failed attempts are charged separately once.

| Worker-second ledger | CPU | GPU |
| --- | ---: | ---: |
| Original allowance | 86,400 | 86,400 |
| Charged before M7 | 83,425.01779734538 | 47,585.583791223704 |
| M7 measured work | 667.194215267722 | 2,938.0864190989523 |
| M7 conservative overhead | 300 | 0 |
| Total charged | 84,392.2120126131 | 50,523.670210322656 |
| Remaining original allowance | 2,007.787987386895 | 35,876.329789677344 |
| Remaining current M7 allocation | 1,432.805784732278 | 6,661.913580901048 |

The M7 allocation is a subset of the original allowance, not extra funding.
No commit, push or external message was performed in this continuation.

## Decisions and remaining work

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep geometry extraction | Exact parity and regressions pass | No engineering veto | Bootstrap/windowed dependencies remain | Continue bounded extraction with preserved aliases and parity | Monolith fully refactored |
| Keep energy/power validation mechanisms | Oracle and finite-substep checks pass | Intended wrong-energy veto detected | Distributional power remains weak | Resolve a larger-anchor/observable sensitivity design using measured cost before replication | Reliable detection or preferred power |
| Keep reporting/serialization repairs | Counterexample, hash parity and affected tests pass | No remaining reproduced defect | Whole-pipeline performance | Measure compiler/steady-state costs separately before larger optimization | End-to-end speedup |
| Preserve three successful pilots and funnel failure | All pilots complete; three declared posteriors pass | Funnel promotion fails; no posterior | Broader target/seed availability and funnel geometry | Investigate target-specific preparation/reparameterization and finite proposal/evidence limits under a new bounded design | Broad calibration or geometry impossibility |
| Keep M7 active | Current development tranche reconciled | No research-direction veto | Powered calibration and consumer references missing | Complete preparation extraction, resolve actual-HMC/stopping designs and bind exact references | Entire repair program complete |

| Inference status | Evidence |
| --- | --- |
| Hard veto screen | Energy defect detected; actual funnel invalid proposals reject promotion; no shared execution failure in the completed suites. |
| Statistically supported ranking | None established. |
| Descriptive-only differences | Per-cell detection counts, candidate counts, pilot errors, host timing and profiles. Binomial intervals quantify the coarse rate uncertainty. |
| Default-readiness | No numerical default promoted. All verified settings remain retained; posterior checks stay separate. |
| Next evidence needed | Better subtle-defect sensitivity, powered real-HMC acceptance and stopping calibration, global exploration, and matched regression/eight-schools/MacroFinance references. |

Post-run skeptical review: successful easy-target pilots and an exact energy
oracle could be mistaken for broad posterior validation. The single-fit
allocation, explicit funnel failure and weak power intervals rule out that
interpretation. The extra grid weakens the explanation that merely adding a few
smaller/interior epsilons would repair this funnel fit; it does not distinguish
all remaining geometry, search and evidence explanations. A geometry-specific
successful experiment could overturn that local failure, but would require
fresh whole-pipeline assessment. The current results reject candidates and
leave statistical claims unsupported; they do not reject the research direction.

## Bootstrap extraction follow-on

Bootstrap preparation now owns its implementation in
`bayesfilter/inference/hmc_bootstrap.py`, with common host-side helpers in
`hmc_preparation_common.py`. Public imports and historical aliases identify the
same definitions; automatic preparation calls the extracted bootstrap directly.
Numerical source closure includes both files. Historical pickle globals still
resolve, serialized adapter identities remain unchanged, and fresh bootstrap
imports do not load the historical tuner. Live seed records preserve their
identifiers and derivations while naming the new physical implementation.

The [follow-on reconciliation](artifacts/hmc-repair-master-2026-09-16/m7-bootstrap-r1/reconciliation.json)
preserves the source baseline, comparison cases, commands, test XML and guide
installation. The move removes 1,584 definition/constant lines from the old
module: 35 definitions, of which 34 have identical ASTs. The screen runner's
only body change is its seed registry owner filename. Another 358 definitions
remain identical. The source-location resolver changes separately, and the
module introduction now accurately distinguishes public and historical routes.
No numerical setting, candidate-retention rule or posterior diagnostic role changed.

| Verification | Result |
| --- | --- |
| Before/after control decisions, payloads, hashes, seeds, errors and scalar/batched affine transforms | All 15 cases equal exactly |
| Original bootstrap tests before extraction | 27 passed |
| Broad extraction/preparation/windowed/retained-execution/warmup/interface batch | 331 passed, one stale-test failure, one inherited skip |
| Corrected historical-wrapper test file, extraction and interface retry | 29 passed |
| Reconciled distinct affected cases | 332 passed, one skipped; overlapping counts are not additive |
| Official guide | 563-page build installed at `docs/main.pdf`; physical page 431 visually checked |

Two measured failures are preserved and charged. First, the diagnostic comparison writer
could not encode a deliberate NaN; it now labels nonfinite values explicitly
without changing the recorded runtime payload or its hash. Second, an old
Phase 7 registry test called today's public candidate-set tuner and expected a
retired `final_kernel_payload`. Its legacy mocks did not intercept that route.
The test now calls the historical wrapper whose registry behavior it actually
tests. The passing numerical integration and new direct-bootstrap wiring test
cover current preparation. The one skipped tiny-Gaussian windowed test exhausted
its bootstrap repair budget; it remains a skip. No test failure was converted
into a numerical success or candidate qualification.

The CPU reference/test workers intentionally hid GPUs. Their seven measured
attempts, including the failures and document build, cost
281.01682444498874 worker-seconds. The additional 120-second overhead covers
inspection, source comparison, reconciliation and reporting, including a metadata
lookup retry for the environment's `tfp-nightly` distribution. The 401.01682444498874
total fits the predeclared 720-second follow-on ceiling. No GPU worker launched.

| Latest worker-second ledger | CPU | GPU |
| --- | ---: | ---: |
| Charged before this follow-on | 84,392.2120126131 | 50,523.670210322656 |
| Follow-on charge, including overhead | 401.01682444498874 | 0 |
| Total charged | 84,793.2288370581 | 50,523.670210322656 |
| Remaining original allowance | 1,606.7711629419064 | 35,876.329789677344 |
| Remaining M7 subset | 1,031.7889602872892 | 6,661.913580901048 |

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Limit |
| --- | --- | --- | --- | --- | --- |
| Keep bootstrap extraction | Exact comparisons and affected regressions pass | No unresolved engineering failure | Windowed/configuration dependencies remain | Extract that next bounded dependency closure | Whole monolith is not yet separated |
| Preserve the scientific findings above | This follow-on changes no numerical policy | Funnel rejection and weak sensitivity remain | Calibration, global exploration and matched references | Resolve their predeclared experiment designs using the recorded costs | Refactoring adds no posterior or default-readiness evidence |

Post-run review: comparing only aliases would have missed a hidden dependency
on the old module. The independent pre-edit capture, fresh-process import test,
automatic-preparation call-chain test, physical seed-source checks and real-chain
regressions address that risk. Historical source manifests remain unchanged and
are compared only across the explicitly recorded file relocation. Timing values
measure cost; no performance improvement or statistical ranking is inferred.
At this bootstrap closeout, windowed extraction was the next engineering step;
its September 18 completion is recorded below. The scientific gaps remain active
in the master program. No worker or reservation remained open at that boundary.


## Windowed preparation extraction, September 18

Windowed preparation, timeout policy, result/configuration types and the frozen
mass/start-bank handoff now own their implementation in
`bayesfilter/inference/hmc_mass_adaptation.py`. Automatic preparation and its
numerical handoff validator call that owner directly. Public imports and old
`hmc_kernel_tuning` globals remain aliases to the same objects; historical
pickle type lookup still works. Fresh imports of the extracted public types and
runners do not load the historical tuner. Its private retry and budget types
appear only in annotations under `TYPE_CHECKING`; public configuration/budget
translation still depends on the historical module.

The [windowed reconciliation](artifacts/hmc-repair-master-2026-09-16/m7-windowed-r1/reconciliation.json)
links the saved pre-edit sources, exact comparisons, commands, test XML and
official guide installation. The bounded move comprises 76 definitions and 22
constants/type aliases. Seventy-five definition ASTs are unchanged; the only
body change in the P4 attempt is its physical seed-owner filename. All moved
assignment ASTs and 282 remaining historical definitions are unchanged. The
historical source resolver maps the relocated seed sites, and numerical source
closure now includes the implementation file. Actual seed derivations and site
identifiers remain unchanged; archived MacroFinance manifests were preserved.

| Verification | Result |
| --- | --- |
| Fixed-clock before/after status, full payload, hash, seeds, callback and error cases | All 14 compare exactly |
| Live rotated-Gaussian reference warmup and frozen handoff | Exact equality of all 16 recorded fields, including numerical traces, metric/transform signatures, final epsilon, start bank and value/score probes; timings excluded |
| Moved source | 75 identical definition ASTs; one physical-owner relocation; all 22 assignment ASTs identical |
| Affected windowed/preparation/candidate/retained/warmup/bootstrap/geometry batch | 326 passed, one new-test error, one inherited skip |
| Corrected extraction, documentation and public preparation budget checks | 26 passed |
| Historical timeout, checkpoint and retained-replay checks | Nine passed |
| Reconciled distinct tests | 352 passed, one skipped; overlapping counts are not additive |
| Official guide | 563-page `docs/main.pdf` installed; physical page 431 visually inspected; all 74 build-source hashes match |

The one new-test error was not a runtime defect: a fixture exercising only the
stage seed requested `complete_payload`, which correctly requires the later P4
seed to have been consumed. The corrected test inspects the consumed stage
entry. The original failing batch and its full cost remain recorded. Existing
P4 lifecycle tests still check the completed registry. The inherited tiny-Gaussian
test remains skipped because its bootstrap exhausted its repair allowance;
the separate live operational warmup and candidate/reload integration passed.
Several existing mocks were retargeted to the implementation they exercise,
including stale geometry/bootstrap mocks in the public preparation budget test.
No failed numerical result was reclassified as a successful tuning candidate.

The guide/reference describe the three extracted owners and the remaining
configuration dependency. `docs/main.tex` and `docs/main.pdf` remain the single
official guide. Its pre-existing unresolved citations (Afshar2015, Gorinova2020,
Pakman2014) are unchanged. The dated PDF is preserved only as build evidence.

Eight CPU-only measured attempts cost 262.70924656209536 worker-seconds,
including the failed batch, successful retry and guide build. With 120 seconds
of predeclared inspection/editing/reconciliation overhead, the charge is
382.70924656209536 seconds, below the reserved 820 seconds. GPU devices were
intentionally hidden in numerical workers; no GPU job launched.

| Latest worker-second ledger | CPU | GPU |
| --- | ---: | ---: |
| Charged before this follow-on | 84,793.2288370581 | 50,523.670210322656 |
| Follow-on charge, including overhead | 382.70924656209536 | 0 |
| Total charged | 85,175.93808362019 | 50,523.670210322656 |
| Remaining original allowance | 1,224.061916379811 | 35,876.329789677344 |
| Remaining M7 subset | 649.0797137251939 | 6,661.913580901048 |

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep windowed extraction | Exact comparisons, source parity and compatibility/integration tests pass | No unresolved engineering veto | Configuration translation and legacy selectors still share the old module | Audit the next coherent translation/budget dependency closure and its test cost | Whole-monolith refactoring complete |
| Preserve prior scientific results | No numerical policy or evidence role changes | Funnel failure and weak subtle-defect sensitivity remain | Calibration, global exploration and matched references | Resolve the target-specific designs recorded in the master | CPU parity establishes GPU performance, adequate burn-in, posterior correctness or new defaults |

| Inference status | Evidence |
| --- | --- |
| Hard veto screen | Extraction checks pass; prior candidate vetoes are preserved. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Test and build timing; live timing fields are excluded from numerical parity. |
| Default-readiness | No numerical default promoted; all verified members remain retained and R-hat/ESS/MCSE remain posterior diagnostics only. |
| Next evidence needed | Target-specific funnel diagnosis, stronger subtle-defect sensitivity, actual-HMC acceptance/stopping calibration and matched consumer references. |

Post-execution review: a mechanical move could appear correct while mocks still
intercept the old owner, source closure misses the executable file, or seed
records cite a dead location. Fresh-process imports, both automatic-preparation
call-chain checks, the live current-source AST map, registry consumption and
real preparation-to-candidate/reload tests address those alternatives. An
unresolved numerical, source or replay mismatch would overturn this engineering
acceptance. Its weakest evidence is device coverage: only CPU reference checks
were run for this extraction, so it adds no GPU performance or posterior evidence.
No M7 worker or reserved launch remains open.


## Configuration and budget extraction, September 18

The active preparation dependency on historical configuration is removed.
`hmc_configuration.py` now owns public presets, geometry/bootstrap/windowed
translation, attempt-budget construction and geometry-scaled timing policy.
The legacy loop configuration type moves with its translation methods; its
executor stays historical. Public dispatch, automatic preparation and candidate
geometry reconstruction import their implementations directly. Historical type
aliases, pickle globals, serialized policies and numeric defaults are preserved.

The [configuration reconciliation](artifacts/hmc-repair-master-2026-09-16/m7-configuration-r1/reconciliation.json)
extends the preceding windowed ledger. All 30 moved definition ASTs, 17 constant
ASTs and 253 remaining historical definition ASTs are unchanged. There are no
unresolved runtime globals in the new owner. Numerical bindings hash the new
file; conservative historical source dependencies remain hashed as before.
This does not assert that the source closure is minimal.

| Verification | Result |
| --- | --- |
| Exact before/after configuration snapshots | 22 records equal, covering ten preset/mass combinations, 160 dimension/attempt budgets, explicit geometry, staged timeouts and invalid inputs |
| Definition and constant comparison | All 30 definitions and 17 assignments identical; 253 historical definitions unchanged |
| Affected configuration/preparation/candidate/replay/dispatch/interface tests | 183 passed |
| Focused public-preset, budget and translation tests | 35 passed |
| Configuration follow-on distinct total | 218 passed; no skip or unresolved failure |
| Combined September 18 distinct test cases | 454 passed and one inherited skip; this union is not the sum of overlapping batches |
| Official guide | Latest 563-page book installed at `docs/main.pdf`; physical page 431 visually inspected; all 74 source hashes match |

One baseline-harness failure is preserved: it supplied an integer to
`staged_timeout_enlargement_rounds`, whose declared type is a mapping. The
fixture was corrected before extraction and the successful baseline saved under
a fresh name. Its 3.773006204981357 seconds are charged. This was a test-input
mistake; the implementation's type check remains unchanged. All post-extraction
checks passed. Mocks for public preparation/configuration were retargeted to
actual owners, preserving error/callback assertions. Fresh subprocesses exercise
public configuration and the automatic preparation call chain without importing
the historical tuner. Real warmup-to-candidate/reload and retained tests pass.
The final import audit found public preparation and policy constants still
using fallback lazy lookup, which could import the historical module. Their
direct exports now name their actual owners. All 23 strengthened public-import
and documentation checks pass; they overlap the 218 distinct follow-on cases.

Seven measured CPU attempts cost 200.20992575306445 seconds. With the predeclared
80-second overhead, this follow-on costs 280.20992575306445 seconds, within its
600-second reservation. Numerical workers intentionally hide GPUs; no GPU job
launched. The guide's three inherited unresolved citations remain unchanged.
The combined charge for both September 18 extractions is 662.9191723151598 CPU
seconds, including failures and 200 seconds of overhead, and zero GPU seconds.

| Latest worker-second ledger | CPU | GPU |
| --- | ---: | ---: |
| Charged before configuration follow-on | 85,175.93808362019 | 50,523.670210322656 |
| Configuration follow-on charge, including overhead | 280.20992575306445 | 0 |
| Total charged | 85,456.14800937325 | 50,523.670210322656 |
| Remaining original allowance | 943.8519906267466 | 35,876.329789677344 |
| Remaining M7 subset | 368.8697879721294 | 6,661.913580901048 |

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep configuration extraction | Exact payload/source parity and all affected checks pass | No unresolved engineering veto | Historical executor remains large and source closure conservative | Preserve the extraction; prioritize the scientific failures before optional historical cleanup | All legacy code removed or whole repository refactored |
| Keep scientific repair active | No numerical defaults or diagnostic roles changed | Prior funnel failure and weak sensitivity remain | Evidence limits, target geometry, calibration and matched references | Resolve the next funnel experiment with fresh streams and its own evidence/cost design | Refactoring demonstrates sufficient burn-in, posterior accuracy or default readiness |

| Inference status | Evidence |
| --- | --- |
| Hard veto screen | No new engineering failure; prior candidate rejections remain preserved. |
| Statistically supported ranking | None. |
| Descriptive-only differences | CPU execution/build costs; no speed ranking inferred. |
| Default-readiness | No numeric default promoted; all verified candidates retained; R-hat/ESS/MCSE stay outside tuning decisions. |
| Next evidence needed | Funnel evidence/geometry diagnosis, useful subtle-defect sensitivity, real-HMC acceptance and posterior-stopping calibration, global exploration and exact consumer references. |

Post-execution review: exact preset payloads alone would miss an import cycle or
public route still reaching old translation. Standalone imports, direct call-chain
tests and numerical preparation/candidate/reload tests cover those alternatives.
The strongest remaining limitation is scientific coverage: all checks here
establish CPU engineering equivalence, not convergence or GPU performance.
M7a's active preparation extraction is complete; the historical executor/readers
and conservative source hashing remain explicit. No worker or reservation remains.
