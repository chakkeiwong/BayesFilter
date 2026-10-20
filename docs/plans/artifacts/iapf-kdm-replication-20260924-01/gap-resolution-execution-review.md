# Gap-resolution execution and terminal review

2026-09-24. The authorized diagnostic tranche is finished at 12/12 launches.
Both original-method replications remain INCOMPLETE. The master in the main
checkout remains the sole planning authority; this note records the execution.

## What changed and ran

The dispatcher now binds reusable outcomes to source/input hashes, interpreter
and installed-package records, reference C++ libraries and loader policy. It
records runtime environment choices and refuses silent reuse after a change.
Legacy checks remain visible but cannot self-upgrade to current PASS.
Explicit --jobs selection avoids spending the remaining budget on unrelated
checks. An ordinary child failure does not prevent an independent selected job.

Attempt 10 failed before training after a successful CUDA probe: the author
reference loaded the older system libstdc++, which lacked CXXABI_1.3.15 required
by Matplotlib. The repair selects the existing reference environment's lib
directory only for that child. No package installation or shared-environment
change occurred.

A focused CPU regression also exposed the author's scheduler call to the
renamed is_better method. The isolated source copy adapts it to _is_better and
drops the removed logging-only verbose keyword. Scheduling state and learning
rates match the installed scheduler across min/max and relative/absolute cases.
Dataset loading explicitly allows the author's locally generated Sensor objects;
this is restricted to the new diagnostic output. The complete source adaptation
is in attempt12/reference-compatibility.patch. Original source is unchanged.

Attempt 11 executed the analytical residual/Jacobian and damped Gauss–Newton
hypothesis on the unchanged joint Equation (15) target. Exact Gaussian input
remained unchanged, the near-Gaussian control recovered its parameters, and
analytical/finite-difference Jacobians agreed. Six of 32 saved failure records
met the diagnostic nomination screen; 26 triggered amplitude collapse.
Post-run input inspection identified only ten distinct inputs: two nominated,
eight collapsed (all five d=20 inputs, three of five d=5 inputs). These are
selected historical failure fixtures, not independent Monte Carlo replications.

Attempt 12 passed the real MDPF source training call chain. It used one weight
pretraining epoch and two full-training epochs, 128 training and 256 validation
sequences, source T=17, N=25 in full training, and batch sizes 64/256. Source
architecture, loss, learning rates and sparse-label/BPTT settings were retained.
The pretraining source cloud size remains its own N=50 setting. The diagnostic
overrides and seeds are in attempt12/diagnostic-configurations.json.

There were two setup_training calls, three train/validation epochs, 96 source
resampling calls, 30 finite-gradient optimizer checks, and 11 successfully read
best checkpoints. Eighteen parameter tensors changed in pretraining and 31 in
full training. Source-generated and reloaded data matched exactly. The actual
reference interpreter used the NVIDIA RTX 4080 GPU; it is an independent eager
PyTorch/NumPy reference, not the BayesFilter TensorFlow/XLA implementation.

## Evidence and remaining limits

The claimed target of attempt 11 is the original Eq15 density-value residual.
The computed residual and analytical Jacobian match the independent reference
on checked controls. The computed fit is a trial local optimization procedure,
not a recovered author optimizer or an established useful twist. Losses around
1e-18 coexist with appreciable relative shape errors on collapsed fixtures.
Thus raw loss/stationarity alone is wrong as a fit acceptance rule. This rejects
the current solver as a general repair; it does not reject iAPF.

The claimed target of attempt 12 is source execution wiring through actual
training. Recorded calls, finite gradients, parameter changes and checkpoint/
dataset round trips support that target. They do not establish IWSG expectation
identities, estimator unbiasedness, learned quality, convergence, paper metrics,
or TensorFlow parity. Those require K1, K2/K3 and P respectively.

The iAPF nomination thresholds are diagnostic hypotheses, not paper tolerances.
The 2/10 result is descriptive and has no success-rate or ranking interpretation.
No comparison among stochastic methods is statistically supported.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| 0R execution repair complete | Dependency/reuse tests and unchanged real resume passed | No budget/provenance/cleanup failure | Future external dependency changes require fresh review | Retain scoped provenance and explicit job selection | Scientific correctness |
| 0M source wiring passes on short reference | Actual training calls, finite updates, round trips | Execution veto cleared in attempt 12 | Full protocol and gradient-estimator correctness untested | K1 and B1 cost calibration | Learned performance or replication |
| I2a current fit is insufficient | Controls pass; only 2/10 distinct inputs nominated | Eight amplitude-collapse vetoes | Source optimizer/floor and useful local solution policy unresolved | I2a.1 source/local-fit reconstruction, then fresh calibration | iAPF invalidity or useful full filter |
| Full experiments remain blocked by prerequisites/budget | Complete protocols not run | Attempt budget exhausted; full cost unknown | Data/splits, uncertainty, comparators and full runtime | Complete K1/I2a.1/B before I3/K2/K3 | Both replications finished |

## Inference status

| Item | Status |
| --- | --- |
| Hard veto screen | Eight of ten distinct iAPF fit inputs collapse; initial MDPF reference execution failed and was repaired |
| Statistically supported ranking | None |
| Descriptive-only differences | Fixture losses, timings, parameter-change counts and nomination counts |
| Default-readiness | Not established for either method or the TensorFlow port |
| Next evidence needed | Independent fit validation, full gradient/value checks, complete repeated paper experiments, conditional heuristics, uncertainty and consumer conformance |

## Cost and attempt accounting

CPU worker seconds used: 18.325333; remaining 1781.674667 of 1800.
GPU worker seconds used: 130.262987; remaining 769.737013 of 900.
Launches: 12/12. Remaining time does not authorize an additional launch.
Attempt 12 took 121.099 seconds. No campaign worker remains running.

The measured mean full-training batch cost was about 0.887 seconds and the
validation batch about 0.586 seconds, with a Python call profiler active.
Pretraining/full-training phase wall times were 27.149/89.925 seconds; respectively
26.065/85.206 seconds lie outside the instrumented epoch functions. That
unattributed time includes other source operations and must be measured rather
than silently omitted or assigned entirely to plotting.

For scale only, linear arithmetic using 79 training and four validation batches,
800 epochs and eleven full trials gives about 177 profiled epoch-component
hours. Assuming one 400-epoch pretraining gives another 4.29 hours. This is
neither a reliable forecast nor a lower bound: profiling, startup/plot/checkpoint
costs, early stopping and scaling can change it. It excludes data generation,
comparators, T=150 evaluation, Appendix B, iAPF and port work. Full budgets remain
NOT_BUDGETED. B1 must measure representative execution without the pervasive
call profiler, with sparse stage timing, before a material compute decision.

## Terminal skeptical review

Twelve focused harness tests passed in development-review09. CPU-only source
imports, both model constructors and scheduler state/cadence checks passed.
Final validation confirmed live/snapshotted source hashes, all 32 preserved RDS
hashes, unchanged author source, and unchanged iAPF R dependencies between the
two manifest revisions. A real unchanged resume consumed zero launches and
returned exit 2 to preserve incomplete scientific status.

Attempt 11 belongs to manifest-repair07; the final training attempt belongs to
manifest-repair08. The conservative dispatcher invalidates a reusable check
when any snapshotted numerical/harness dependency changes. Its machine report
therefore labels the former as legacy/stale for automatic reuse. The preserved
iAPF result remains an inspected diagnostic for its recorded version; it is not
silently reissued as a current passing run.

Strongest alternative explanations: fit nomination tolerances may be stricter
than needed for a useful twist, and the chosen damping/initialization may miss
a good local solution; short source training can execute while its estimator
or objective is wrong for the scientific claim. Fresh source-grounded fit
validation and K1 matched-target derivative checks could overturn those doubts.
The weakest evidence is the tiny, profiled, single-seed training fixture and the
selected, duplicated historical fit inputs. Nothing here ranks methods.

No independent reviewer was used for this bounded continuation. The source and
implementation review, regression results and this terminal review are local
Codex review; they are not a substitute for later source/claim-level scrutiny.

## Evidence paths

- Master: /home/chakwong/BayesFilter/docs/plans/iapf-kdm-replication-master-program-2026-09-24.md
- Actual commands, seeds, source and environment: manifest-repair07.json,
  manifest-repair08.json and ledger.json; per-attempt command.log.
- iAPF: stage0/attempt11/result.json, controls.csv,
  saved-cloud-reconstruction.csv and trajectory-*.csv.
- MDPF: stage0/attempt12/result.json, diagnostic-configurations.json,
  source-training-wiring.json, component-timings.json, environment.json and
  reference-compatibility.patch.
- Diagnostic validity and duplicate-input accounting:
  postrun-gap-review/final-validation.json, observations.json,
  saved-input-identities.csv and inspection-manifest.json.
- Focused development checks: development-review07 through development-review09.
