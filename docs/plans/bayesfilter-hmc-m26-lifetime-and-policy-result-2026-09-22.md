# M26: fit lifetime repair and posterior-policy development results

M26 is complete. Optional process isolation releases framework state after each
complete fit. A separate repair removes bootstrap timing from candidate random
stream identities while preserving the full timing-bearing audit hash. Four
paired complete fits agree exactly across persistent and isolated execution.
Five of six posterior-policy pilots pass their declared checks; the remaining
rotated-Gaussian member mixes too slowly for its allocation. All 117 verified
pilot candidates remain retained. Posterior failures do not change tuning.

The [plan](bayesfilter-hmc-m26-lifetime-and-policy-plan-2026-09-22.md),
[terminal audit](artifacts/hmc-repair-master-2026-09-16/m26-r1/terminal-audit-r1/result.json)
and [next phase](bayesfilter-hmc-post-m26-next-phase-2026-09-22.md) distinguish
engineering completion from the remaining statistical work. The evidence root
is `artifacts/hmc-repair-master-2026-09-16/m26-r1/`.

## Implemented repair and provenance

Numerical search, accuracy and stopping designs can set `isolate_fits=true`
and an explicit `fit_process_timeout_seconds`. The parent stays TensorFlow-free;
each child executes one complete fit, independent assessment and ordinary
framework shutdown. Timeouts, abnormal exits and missing assessments are
preserved separately from numerical results. They cannot count as favorable
coverage. Partial fits can resume within the unused cap under identical source
and design; completed evidence is not silently rerun. An unreconciled launch
requires process reconciliation before another child can start.

The first Gaussian comparison exposed a reproducibility defect: bootstrap
`wall_seconds` entered the geometry hash, preparation identity and tuning seed.
Identical numerical preparation therefore produced different random streams.
Those four executions are preserved as diagnosis, not parity evidence. New
ordinary bindings now use `HMCGeometryInitializationResult.numerical_hash`,
which excludes only the bootstrap probe clocks. The full `artifact_hash` still
binds every timing record. Changed epsilon, seeds and numerical outcomes still
change numerical identity. Old checkpoint scopes are not upgraded.

All repaired numerical experiments use frozen source r2, package identity
`d39688d9c08f9350d673596234c6e688833ce1f87137843874775a65542fe94e`,
from dirty workspace baseline `8f992b205e9a4b8a861db4064cbcedb76af52f1f`.
The snapshot and terminal comparison cover all 506 package Python files; none
differs in the current package at the audit. This corrects the stale
`203fde46...` development-baseline field in the progress record. Each attempt
records its exact command, environment, source, seeds, design, output and wall
time. Later independent audit scripts are archived with their checksums.

These are explicit CPU reference/diagnostic runs in the `tfgpu` conda environment,
with GPU hidden before framework import, memory growth enabled and one thread
per numerical worker. HMC runs use graph mode without GPU/XLA promotion;
some existing diagnostic primitives independently compile CPU XLA kernels.
Neither GPU compatibility nor GPU performance is established by M26.

## Complete-fit parity and resource observations

Two Gaussian and two beta-binomial fit identities were each executed in both
modes: four paired fits, eight executions. Every ordinary search completed and
every process returned zero. Across one side of the comparison, the audit checks
374 candidate records, 81 retained verified members, 596 numerical evidence
records and 386 tensor files. Candidate scopes, seeds, numerical receipts,
acceptance/health decisions, posterior diagnostics and tensor bytes agree.
Full hashes that include elapsed time are checked individually rather than
required to equal. These engineering replays are not eight independent
confirmation replications.

| Target | Persistent RSS after first / second fit, KiB | Persistent registered functions after first / second fit | Isolated RSS after each fit, KiB |
| --- | ---: | ---: | ---: |
| Gaussian | 2014792 / 2609948 | 1287 / 1986 | 2014644 / 1690276 |
| Beta-binomial | 2093848 / 2971748 | 1280 / 2189 | 2094172 / 2132324 |

Each isolated child started near 451000 KiB with zero registered functions and
exited normally. Observed child teardown was 3.1–4.1 seconds. Process isolation
contains accumulation across complete fits in these tested cases. Two fits per
model do not establish the root cause of M21's roughly 90-GiB, 128-fit teardown
failure, an asymptotic memory bound, or a limit on memory within one fit.

## Posterior pilots and the remaining slow member

Each estimator/model cell is one fresh development fit. All use the same
target-specific counts within a model and the predeclared `first_verified`
identity rule, with independent estimator-arm seeds. The rule does not promise
efficient mixing. All unselected verified members remain explicitly unassessed.

| Target / estimator | Verified members | Selected L | Warmup / retained per chain | Posterior result |
| --- | ---: | ---: | ---: | --- |
| Gaussian / lugsail | 25 | 3 | 30000 / 4000 | Declared checks passed |
| Gaussian / autocorrelation | 18 | 3 | 30000 / 4000 | Declared checks passed |
| Beta-binomial / lugsail | 19 | 5 | 2000 / 5000 | Declared checks passed |
| Beta-binomial / autocorrelation | 22 | 3 | 2000 / 5000 | Declared checks passed |
| Rotated Gaussian / lugsail | 17 | 5 | 2000 / 30000 | Declared checks passed |
| Rotated Gaussian / autocorrelation | 16 | 25 | 10000 / 0 | Warmup cap; no health veto |

The five delivered posteriors' declared mean/median intervals cover their known
truths. All six independent fixed arms also cover their declared truths. The
failed stop supplies no retained intervals and remains unavailable in the full
denominator. A single covered interval has a coverage-rate confidence interval
of [.025, 1]; these observations cannot establish calibration. The pilots also
cannot rank estimators: they selected different members under different seeds.

The [saved-chain diagnosis](artifacts/hmc-repair-master-2026-09-16/m26-r1/followup-diagnosis-r1/result.json)
inspects the failed L=25, epsilon=0.7225904034885232 member. Its last
1000-transition warmup window has maximum R-hat 1.21298 and bulk ESS about 17;
the complete 10000 warmup has R-hat 1.01226 and bulk ESS about 196–199. Thus
a longer readiness window merits testing. It does not retroactively pass the
recorded stop.

More seriously, the independent 60000-transition fixed arm has R-hat 1.00114
but bulk ESS only about 985–995. Its two mean MCSEs are .187/.273 under
autocorrelation and .154/.224 under lugsail; median MCSEs are .199/.289.
Both miss the unchanged .05 requirement. The plug-in scaling
`60000*(.28939/.05)^2` suggests roughly 2.01 million retained transitions per
chain for this member, with substantial estimation uncertainty. Increasing the
readiness window alone is therefore an inadequate precision repair. Assess
predeclared siblings before committing to such an expensive chain; do not
delete this member from tuning or infer that its siblings fail.

## Tests, documentation and budget

The [verification record](artifacts/hmc-repair-master-2026-09-16/m26-r1/verification-final.json)
contains **197 distinct passing tests** and 260 passing executions, with no
unresolved failures or skips. The final current-source run passes 44 tests,
including real Gaussian/beta-binomial public pipelines, restart, process
failure/timeout accounting, stream identity and documentation contracts.
Earlier test failures remain recorded: an extension-type diagnostic needed a
string guard; two old bootstrap fixtures needed current payload/probability
and sample-length conventions. Their numerical veto tests remain intact.

The official tuning chapter, agent API reference and validation README explain
the optional isolation and separate numerical identity. `docs/main.tex` builds
without undefined citations/references; the changed rendered pages 411 and 432
were inspected. `docs/main.pdf` was refreshed. The book remains the scientific
guide; the Markdown reference documents the same procedure.

The [terminal ledger](artifacts/hmc-repair-master-2026-09-16/m26-r1/reconciliation-terminal.json)
charges **4877.17 CPU seconds**, zero GPU seconds, including failed attempts,
tests, diagnosis, teardown and a conservative document-build charge. This is
within the 12000-second phase ceiling. **79635.40 CPU seconds (22.12 hours)**
and **84803.52 GPU seconds (23.56 hours)** remain. No M26 worker or reservation
remains live. Nested child receipts are not double charged.

## Decisions and limits

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept optional fit isolation for the tested CPU workflow | Four paired fits agree; ordinary exits and restart checks pass | No unresolved engineering failure | Long-run/GPU behavior and peak within-fit memory | GPU compatibility/pricing before a large campaign | Original memory root cause or universal resource bound |
| Accept clock-independent numerical identity repair | Full paired numerical parity; numerical changes remain identity-sensitive | Full audit hashes remain checked | Scope of other targets/backends | Keep regression and audit evidence | Replay identity for old checkpoints |
| Retain Gaussian/beta count policies as development candidates | Four selected posteriors pass unchanged requirements | No health failure | Delivery/coverage probability | Price statistically adequate fresh confirmation | New defaults or estimator superiority |
| Repair rotated member assessment | One member passes, one has a warmup cap and poor fixed-arm precision | Cap is a posterior promotion veto, not a tuning veto | Unassessed siblings and dependence timescales | Bounded identity-based sibling study and longer readiness window | Failure of tuning, the estimator or all siblings |
| Defer large unchanged confirmation | Exact operating-characteristic/cost calculation completed | No scientifically adequate all-model inventory fits current CPU budget at measured rates | Runtime variance and GPU costs | Measure another backend and resolve member allocation first | Closing coverage from a small pilot |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No pilot numerical-health veto; one selected posterior fails readiness at its cap. All tuning candidates remain recorded. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Runtime, memory, counts, MCSE, per-pilot coverage and estimator differences. |
| Default readiness | No estimator, allocation, member-selection or transport default promoted. |
| Next evidence needed | Fresh complete-fit coverage/delivery studies, GPU scope checks, supplied-map tail and sibling evidence, affordable full-fit power and matching consumer inputs. |

The strongest alternative explanation for the five pilot successes is favorable
seed/member selection. The failure shows that counts adequate for one member
can be grossly inadequate for another on the same model. Independent complete
fits could overturn the nominated policies. The weakest evidence is their
single-fit coverage and runtime estimates, not the checked paired arithmetic.
M26 invalidates neither the target nor the harness: it repairs two engineering
mechanisms and exposes a specific posterior-allocation failure. The next phase
continues that repair within the remaining budget.
