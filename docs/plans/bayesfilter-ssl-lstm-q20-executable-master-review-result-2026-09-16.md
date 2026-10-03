# q20 executable master repair and whole-program review

Date: 2026-09-16. Engineering status: `REPAIRED_REVIEWED_EXECUTED`.
Final master outcome: `UNDER_BUDGETED`, normal exit.
Research status: both q20 GPU/XLA qualification scopes passed on the final
source in r3; no q20 posterior result exists. The final section records the
normal early affordability stop and settled allowance. This is a skeptical
self-review, not an independent endorsement. Earlier review/launch subsections
retain the status at their recorded point in the repair.

The master now runs the complete fixed protocol: trusted device check,
enclosing-graph and public-runner qualification, complete cost measurements,
affordability, independent importance reference, direct and continuation
training, explicit verified kernels, five-method development, frozen choices,
fresh-start reverification and independent confirmation. Every numerical stage
runs in an externally timed process. The cumulative ledger includes failed
attempts, compilation and initialization. Completed stages and copied numerical
checkpoints can be reused only with matching inputs, sources and checksums.

## Review of the complete call graph

| Path | Finding and repair | Evidence |
| --- | --- | --- |
| CLI → coordinator → worker | Replaced three disconnected modes with real stage dispatch, persistent allowance, statuses and owned process groups. A retry cannot renew its cumulative stage cap. | Full supervised known-target campaign: 25 real worker stages, then zero-worker resume with unchanged accounting. |
| Configuration → training | Plain NeuTra incorrectly reused a continuation map. Added separately seeded direct-beta-one and continuation cohorts with explicit cost allocation. | Tiny cohort restart checks both schedules and retained Adam state; complete master selects each baseline separately. |
| Training → frozen map | Interrupted updates could skip the still-missing rung assessment. Resume now checks completed assessments; JSON publication is atomic. Export preserves the actual weighted map. | Exact optimizer restart, interrupted publication, nonzero forward/inverse/logdet/pullback parity. |
| Bridge → public tuner | Scalar probes and retained sample axes were rejected by the batch-only adapter. Reshape through the same batch-native target; no row-mapped training. | Scalar/batch/retained parity plus real identity, classical and fixed-transport consumers. |
| Classical preparation → candidate grid | Preparation tightened the epsilon domain but the materialized candidate grid retained the old value. Rebuild the grid and preserve the issued geometry bound. | Connected classical tuning and fresh-start verification; geometry/domain round-trip regression. |
| NeuTra → public tuner | Ordinary tuner was used for frozen-map coordinates. Dispatch through the public fixed-transport tuner and repository-issued binding. | Actual trained map → public tuning → retained member → posterior test. |
| Kernel → posterior | Positive log-accept threshold was invalid; ensemble incorrectly vetoed large finite energy errors. Use the signed reporting threshold and real endpoint/proposal/status health. | Large finite energy stays diagnostic; invalid status vetoes; both replica variants run. |
| Ensemble coordinates | Affine inverse used host-side Python boolean validation inside a compiled graph. Replace with tensor algebra over already validated geometry. | Classical replica exchange, independent chart mixture and exact checkpoint replay. |
| Reference → comparison | Missing-reference stub replaced by full-support prior importance banks. Comparison checks named quantities, nonzero valid MCSE, source/target/role, start groups and full method inventory. | Known tilted-Gaussian estimates, unvisited sign and collapsed-bank rejection, exact reference replay, complete comparisons. |
| Empty or failed posterior → report | An empty retained archive crashed precision reporting. It now reports missing quantities and cannot pass. | Empty-archive regression and cap-limited classical fixture. |
| Timing → forecast | Batched primitive timing underestimated public serial-chain work; heldout evaluation and process overhead were missing. Measure actual operations and reserve compilation, analysis cadence and startup. | Real pricing dispatch plus complete known-target execution. |
| Qualification → capability | Target-only XLA readiness did not qualify the enclosing public chain. Issue a source-bound receipt only after enclosing-loop and public serial-runner checks. | Small CPU/XLA known-target diagnostic; q20 GPU qualification remains to execute. |
| Resume → resources | Recover uncertain launches conservatively, reject changed artifacts and terminate surviving owned descendants. No budget restoration on resume. | Process-tree timeout, normal-exit cleanup, orphan accounting, artifact mutation and exhausted-budget tests. |

The shared changes are limited to the bridge shape adapter, an opt-in one-step
health result, and repository-issued inverse geometry/new-start binding. The
remaining implementation is in the q20 modules and CLI. No model equations,
data, environment packages, platform route, public release or other research
lane was changed.

## Numerical choices and provenance

The machine-readable [protocol](artifacts/ssl-lstm-q20-executable-master-2026-09-16/protocol.json)
lists every active field. The September 15 parameter ledger and mathematical
audit retain target, objective, uncertainty and hypothesis provenance. The
following clarifies implementation choices rather than treating old proposed
alternatives as the implemented values.

- The target remains q20/T30, four physical parameters, float64, strict principal
  square root, the same Gaussian prior and UKF coefficients.
- Both training schedules use widths 16/32, two tanh stages, LRs 0.0005/0.001,
  three roots, batch 32 and rungs 128/512/2048/8192. These are uncalibrated
  hypotheses. Direct training visits beta one only; continuation visits 0.5
  and 1.0. Beta zero is exact prior initialization. Heldout banks are development
  selection data, not final uncertainty evidence.
- The initial epsilon 0.01, bounded domain [1e-6,2], repair factor 1.5 and at
  most 12 same-L repairs are explicit pilot hypotheses inherited from the
  first production-repair implementation. They differ from alternatives
  proposed in the older ledger and are not claimed calibrated. L remains
  3/5/9/13/18/25. The 200 work-unit and 100-candidate limits are engineering
  ceilings, not justified evidence counts; actual mandatory work and the
  maximum evidence rung determine the measured reserve. Failure to fit or
  verify reports incomplete. Classical preparation retains its stricter
  measured stability bound.
- Sequential physical-coordinate warmup, R-hat, ESS and named MCSE checks use
  the recorded owner policy. Acceptance only qualifies a fixed kernel; finite
  energy magnitude remains explanatory. Independent confirmation rechecks the
  exact frozen pair at new starts without retuning it.
- Eight prior-importance banks with 1024/4096/16384 cumulative rows, ESS 400,
  effective tail count 20 and one-third reference error allocation are initial
  feasibility hypotheses. Rung stability compares the observed difference
  plus marginal-normal uncertainty with twice the allocated reference error.
  This is an asymptotic stability screen, not a certified error bound. Weight
  collapse and unobserved tails cannot acquire zero uncertainty.
- Timing uses two measured invocations and four transitions per chain as
  bounded engineering probes. A factor of two and two termination-grace
  intervals per worker are conservative forecast hypotheses. They cannot
  guarantee completion or change the actual external caps.

The program can conclude that declared confirmation checks passed. It keeps
`production_qualified=false`: shared likelihood code, finite importance banks,
marginal intervals and three fixed-map systems do not establish independent
likelihood correctness, global mode discovery, method superiority or broad
production/default readiness. No method ranking is computed.

Conditional LR/capacity/chart-count/temperature repairs remain experiments
chosen from the recorded failure evidence. The master continues the fixed
training ladder, internal HMC repairs and unchanged-input infrastructure retries.
A candidate requiring a different scientific contrast gets a repair result;
the agent must specify and reprice that contrast within the existing campaign,
not reinterpret the candidate failure as rejection of NeuTra.

## Validation and remaining limitations

| Check | Result | Scope |
| --- | --- | --- |
| Complete supervised Gaussian master | 1 passed, 336.39 s; 25 stages and 331.2275 measured worker seconds | Actual numerical dispatch through confirmation and cached resume. Deliberately loose smoke screens, one confirmation system, no q20 inference. |
| Latest repair/reference/qualification suite | 22 passed, 30.22 s | Training restart, direct/continuation schedules, reference, process supervision and CPU/XLA qualification. |
| Final persistence checks | 18 passed, 20.10 s | Atomic publication, exhausted-budget status, training and supervisor regression. |
| Existing transport/bridge/checkpoint/route policy | 43 passed, 22.92 s | Includes process death at checkpoint boundaries. |
| Earlier connected single-member suite | 4 passed, 87.17 s | Real three-method tuning, retained sampling and fresh-start reverification. |
| Both exchange variants and pricing | 2 passed, 173.92 s | Real numerical stages and exact replica checkpoint replay. |
| Broader shared-HMC regression | 56 passed; 3 legacy test failures, 93.86 s | The three failures inject retired custom fake runners. The base revision already rejects that interface in `fixed_transport_hmc_tuning_tf.py`; its dispatcher was not changed here. |

Counts overlap and must not be summed as unique tests. Logs and compact complete
worker records are preserved under
[the execution artifact root](artifacts/ssl-lstm-q20-executable-master-2026-09-16/).
The full smoke preceded the final atomic-write/exhausted-budget edits; their
focused regressions passed afterward. No later numerical algorithm changed.

The current route doctor reports `READY_USER_APPROVAL`: observed session metadata
and local rules agree. This does not override a managed reviewer. No current
security rejection was observed. The earlier no-idle-GPU result is a resource
availability result and must be rechecked at launch.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Execute the repaired fixed master | Connected engineering checks passed | No known blocking new implementation defect | Real q20 GPU/XLA compile cost and numerical behavior | Commit isolated source; run trusted master with recovered allowance | Real q20 learning or posterior qualification |
| Preserve research hypotheses | No real q20 comparison yet | All posterior/reference/training screens remain active | Reference feasibility, adequate learning, overlap and coverage | Follow the actual stage result and remaining budget | NeuTra failure or a winning method |

| Inference item | Status |
| --- | --- |
| Hard veto screen | Exercised on fixtures; no new real q20 candidate assessed. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Test/pricing runtimes and fixture diagnostics. |
| Default-readiness | Not established by engineering tests. |
| Next evidence needed | Trusted real q20 qualification, full-cost feasibility, assessed training, verified kernels, independent integration, development and fresh confirmation. |

Strongest alternative explanation for successful fixtures is their much simpler
target and loose smoke thresholds. A real target, compilation, health or source
failure would overturn readiness for that execution scope. The weakest evidence
remains the unexecuted q20 GPU path, not the already connected stage dispatch.

## Real execution

Append the committed source, exact command, measured cost, terminal status and
artifact paths after the authorized launch. The recovered allowance subtracts
all saved repair test durations and retains the entire old 7200-second hold.
An explicit startup-overhead hold covers unmeasured process overhead; it is
reserved money/time, not invented measured consumption. No budget is renewed.


### Launch record

The repair is committed on main as `6026b47a`. The numerical source was launched
from isolated commit `98ea3860b9d14bf6c718b8fa4fa0faee869f6aaa`; all 141 integrated
repair/evidence files match their isolated counterparts. Main's unrelated dirty
work is preserved. Raw pytest logs retain original whitespace; code and written
document whitespace checks pass.

Trusted execution admitted this exact command from the isolated checkout:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py campaign --config docs/plans/artifacts/ssl-lstm-q20-executable-master-2026-09-16/protocol.json --budget-record docs/plans/artifacts/ssl-lstm-q20-executable-master-2026-09-16/recovered-allowance.json --output-dir docs/plans/artifacts/ssl-lstm-q20-executable-master-2026-09-16/real-q20-r1
```

The fresh readiness probe passed. The selected device is an RTX 4080 SUPER;
the worker verifies memory growth before initialization. The real target
signature is `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`.
The first qualification attempt is bounded by 600 seconds from process creation,
including native compilation and its termination grace. Its initial status is
running; no qualification or training result is claimed at launch. The active
ledger and receipts live in the isolated checkout at
`docs/plans/artifacts/ssl-lstm-q20-executable-master-2026-09-16/real-q20-r1/`.

Launch `real-q20-r1` consumed 321.0815 seconds before the public-runner
telemetry schema vetoed its receipt. The q20 target emitted one optional
innovation field without its required pair; the repair drops that unpaired
diagnostic at the bridge trace boundary and preserves the required core status
fields. This is a schema normalization repair, not a relaxation of target
health checks. Retry uses a fresh `real-q20-r2` root and subtracts the r1 charge.

The exact partial-telemetry XLA regression and complete supervised master
passed again: 4 tests in 344.71 seconds. Qualification is now dispatched per
positive temperature with a merged receipt that must cover the entire ladder.
The fresh retry uses `retry-protocol.json` and `retry-allowance.json`; the three
remaining initial diagnostic attempts and all prior spending are preserved.


### r2 outcome and reviewed cost-stop repair

The r2 source was `97d203fde1a2f3b795c6eb7e03c432c8fc22a47c`, integrated on
main through `d0ea4fe6`. The exact command used the `campaign` mode with
`retry-protocol.json`, `retry-allowance.json` and output `real-q20-r2` beneath
the execution artifact root. It ran on host GPU 1, an RTX 4080 SUPER, with
TensorFlow 2.20.0, GPU/XLA, float64 and verified memory growth before device
initialization.

| Stage | Outcome | Supervisor wall seconds |
| --- | --- | ---: |
| Trusted readiness | Passed | 3.5035833190 |
| Beta 0.5 enclosing graph and public serial HMC runner | Passed | 320.0745517510 |
| Beta 1 enclosing graph and public serial HMC runner | Passed | 316.5773860420 |
| Full pricing | External timeout, exit 124; partial timing records preserved | 595.6413184400 |
| Total | `MASTER_INCOMPLETE`, initial diagnostic attempts exhausted | 1235.7968395521 |

The remaining balance after r2 was 124709.27452363884 campaign seconds,
including 43732.82154870897 diagnostic seconds. A new focused regression
consumed 58.54711030801991 seconds including startup and exit; 14 checks passed
in pytest's 54.39 seconds. The recovered r3 balance is therefore
124650.72741333082 campaign seconds and 43674.27443840095 diagnostic seconds.

The timing rows cover every configured batch-32 width/positive-beta scope.
Their declared training reservation is 1617749.0854983728 seconds (449.37 h):
63267.25745440868 raw optimizer-floor seconds plus 745607.2852947777 raw
all-rung heldout-reserve seconds, multiplied by two. The saved
`r2-cost-reservation-review.json` identifies each input row by checksum.
This is not a complete campaign price and not a statistical lower bound on
runtime. Early adaptive stopping could reduce work. Nevertheless the existing
reservation rule cannot admit this configuration against about 34.6 h remaining.

Whole-program review found that pricing could continue until its deadline even
once that negative affordability decision was available. The repaired shared
cost calculation now checks each completed row, validates its scope and finite
positive times, and emits a partial-cost deficit before unmeasured downstream
work. The coordinator then reports `UNDER_BUDGETED`. It uses the immutable
initial allowance so resume does not change the cached request. The full
remaining-balance check remains required for positive admission. No numerical
settings, model equations or scientific screens changed.

The actual-worker regression confirms one pricing stage then no downstream
stages, and an unchanged zero-worker resume. The existing complete pricing
fixture still computes every cost and a finite complete forecast. The
supervision and invalid-measurement regressions also passed. Review verdict:
ready for the explicit r3 cost-stop execution under the amended plan, with
fresh source qualification. No independent review endorsement is claimed.

The r2 result invalidated the initial pricing allocation and exposed the cost
reservation deficit; it did not invalidate the target, observed data, numerical
qualification, or NeuTra mechanism. The next scientific work needs a funded
validation/computation plan or measured performance repair. Shortening training
back to an unassessed canary or weakening posterior precision is not justified.


### r3 terminal execution and final review

Executed from `/tmp/BayesFilter-q20-master-repair-20260916`, source commit
`71e0fba399489a8f25fcbdea1185f6bbb600e487`, integrated on main as `04d59bcc`:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py campaign --config docs/plans/artifacts/ssl-lstm-q20-executable-master-2026-09-16/cost-stop-protocol.json --budget-record docs/plans/artifacts/ssl-lstm-q20-executable-master-2026-09-16/cost-stop-allowance.json --output-dir docs/plans/artifacts/ssl-lstm-q20-executable-master-2026-09-16/real-q20-r3
```

The master and all four supervised processes exited with code zero. The
terminal result is `UNDER_BUDGETED`, not a timeout or failed worker. The
same q20 target, observed data, GPU, TensorFlow 2.20.0, float64 and XLA were
used. Every numerical worker records memory growth before initialization;
physical host GPU 1 is the RTX 4080 SUPER. Exact target/bridge/source identities,
seeds, environment, commands and plan/result paths are in the manifests.

| Stage | Outcome | Measured wall seconds |
| --- | --- | ---: |
| Trusted readiness | Passed | 3.0033419160 |
| Beta 0.5 qualification | Enclosing graph and actual public runner passed | 319.5937316370 |
| Beta 1 qualification | Enclosing graph and actual public runner passed | 315.1001293110 |
| Pricing | Normal early reservation deficit | 129.0399081821 |
| Total | `UNDER_BUDGETED` | 766.7371110461 |

The first measured batch-32 scope, width 16 at beta 0.5, implies
268997.48527180706 seconds (74.72 h) under the current reservation rule. This
already exceeds the initial 124650.72741333082-second allowance (34.63 h).
The quote explicitly lists the remaining three training scopes and all
unmeasured downstream costs; none is reported as zero or admitted. Earlier
r2 measurements give the full training-only reservation of 449.37 h. Both
calculations reserve all validation expansions/rungs and use the engineering
factor of two. They are not confidence bounds or proof of the realized cost
of an adaptive run.

The remaining allowance is 123883.99030228473 campaign seconds (34.41 h),
including 42907.537327354854 diagnostic seconds (11.92 h). The original
7200-second unresolved-repair hold and 115-second overhead hold remain
subtracted. `settled-allowance.json` carries the final balance and ledger hash.
The budget is not exhausted; the declared full reservation cannot fit it.

The terminal audit verifies the original source snapshot, the sum of all
attempt charges, 14 saved dependency hashes, three stage-result hashes and
absence of the same owned supervisor processes. Only qualification and pricing
stages ran. The few optimizer updates are explicitly pricing probes; no full
cohort, assessed map, posterior estimate, comparison or confirmation was created.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Repair/review/execute the master | Complete: connected fixture, focused regressions and real committed-source execution | No current code, source-integrity, worker or platform failure in r3 | Real long-run stage behavior beyond qualification remains untested | Preserve this execution and use the current master for a funded plan | Broad production certification |
| Admit the current full research schedule | Failed under the declared reservation rule | Cost admission blocks the full cohort; numerical qualification passed | Few timing samples and conservative all-rung validation reserve | Reprice a concrete equivalent computation repair or reviewed staged allocation within remaining budget/stage caps | Actual adaptive runtime must be 449 h |
| Reject NeuTra or a posterior candidate | No such test was completed | No candidate-level scientific verdict is supported | Learning, coverage, precision and reference feasibility remain untested | Preserve the research question while repairing the executable cost plan | Whitening failure, posterior correctness, or a winning method |

| Inference item | Final status |
| --- | --- |
| Hard veto screen | Graph/proposal/status qualification passed in both scopes; cost admission failed. Posterior vetoes were not evaluated. |
| Statistically supported ranking | None; no method comparison completed. |
| Descriptive-only differences | Short qualification and pricing timings; the reservation extrapolation. |
| Default-readiness | Not established; `production_qualified=false`. |
| Next evidence needed | A funded target-specific training/assessment plan, measured downstream work, qualified reference, verified kernels, development and fresh confirmation. |

Post-run red-team: the strongest alternative explanation for the cost deficit
is over-reservation of adaptive validation, combined with repeated identical
heldout work. A mathematically equivalent cache/batch repair or a reviewed
staged allocation could overturn the planning deficit without changing the
scientific target. It must be measured and preserve the uncertainty/holdout
contract. No speedup is established here. The weakest evidence is the short
cost sample and the absence of actual trained q20 maps and posterior results.
The outcome invalidates the funded reservation plan, not the target, data,
mathematics or NeuTra research direction.

The pasted historical conversation contains a generic request-failure message
and reconnect attempts; it does not identify a security-classification reason.
These current trusted launches were admitted and completed. That distinguishes
the present engineering/cost findings from a platform security rejection but
does not establish the cause of the historical service failure.
