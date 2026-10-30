# Public MacroFinance GPU repair and qualification launch — 29 October 2026

**Terminal update:** the subsequent qualification completed at 11:51:16 China
time on 29 October: 1,248 trials, seven verified/exported/reloaded members and
two movement rejections. See the [terminal result](bayesfilter-modern-hmc-public-qualification-result-2026-10-29.md).
The launch and early-progress descriptions below are historical snapshots.

The public GPU integration repair is installed and passes its scoped engineering
checks. Fresh nine-candidate qualification is running. This is an engineering
result and launch note, not a terminal qualification or posterior result. The
[master](bayesfilter-modern-hmc-implementation-plan-2026-10-07.md) remains incomplete.

## Repair and tested target

The earlier public Phase9 builder required CPU execution while GPU diagnostics
used private composition. MacroFinance's public build_generic_target now accepts
an explicit Phase9SourceBinding, execution_device="GPU" and gpu_evidence pointing
to checked source-matched evidence. Legacy CPU defaults and the historical
global source pin remain unchanged.

The GPU route uses the existing analytic factor-QR likelihood and score, including
its return_health=True path. It exposes checked health for static batches of one
and four chains and preserves invalid-row isolation. It requires bound source,
selected GPU visibility and memory growth configured before framework
initialization. Runtime observation conversion now uses TensorFlow; tensor
materialization for byte hashing remains an artifact boundary.

The GPU-evidence validator checks eight engineering outcomes plus source,
target, adapter, loaded-module, GPU/XLA and memory-policy identity. Validated
evidence enables numerical XLA capabilities. It does not change the provenance
fields gpu_qualified=False and posterior_qualified=False into scientific
admission. The modern retained runner still rejects claim_eligible=True.

The installed MacroFinance delta contains:

- one_country_affine_macro_phase9_hmc.py;
- one_country_affine_macro_source_binding.py;
- the new one_country_affine_macro_gpu_evidence.py;
- tests/test_one_country_affine_macro_public_device.py; and
- tests/test_one_country_affine_macro_gpu_evidence.py.

The scoped patch preserved unrelated dirty work. It is not committed in
MacroFinance. Its exact patch, installed hashes and original hashes are preserved.
The tested target is the actual 48-period, nine-coordinate synthetic model.
The clean BayesFilter checkout is /tmp/BayesFilter-modern-hmc-c2-exact-source-20261028,
commit f9291ee4437aa8c00dabbe331bffc6871b574c5c. This evidence does not cover all
models, all batch sizes or arbitrary source revisions.

## Checked evidence

Engineering root:
artifacts/modern-hmc-implementation-2026-10-07/N5/public-gpu-20261029-01.
Individual manifests preserve exact commands, environment, wall time and paths.

| Check | Result | Directory |
|---|---|---|
| Source, device and evidence tests | 51 passed | unit-02 |
| Actual mixed-row/reference regressions | 10 passed | regression-01 |
| Direct public-builder CPU oracle | 1 passed | cpu-01 |
| Installed public integration, device and evidence tests | 31 passed | installed-cpu-01 |
| Bounded supervisor fake-process tests | 5 passed | supervisor-tests-01 |
| Frozen consumer GPU/XLA checks | All eight passed | gpu-02 |
| Installed consumer GPU/XLA checks | All eight passed | installed-gpu-01 |
| Actual public runner matches checked shared body | Checked outputs exactly equal | installed-runner-01 |

These suites overlap; do not sum their counts as independent tests. Maximum
absolute CPU/GPU differences were 2.1316282072803006e-14 for value,
6.110667527536862e-13 for score and 5.551115123125783e-17 for shared-chain output.
The mixed invalid-row status vector was [0, 1, 0, 0]. The recorded configuration
is TensorFlow/TFP, float64, GPU/XLA, TF32 disabled and verified memory growth.

The frozen-copy public-runner preflight passed. A zero-work controller preflight
reconstructed all nine candidates and its numerical checkpoint. Neither counts
as a completed qualification trial.

## Preserved failures and repairs

The first pricing attempt failed before transitions because a broad wrapper
imported optional hmc_discarded_epsilon_proposals, absent in the clean checkout.
The harness now binds the frozen cohort directly through the public execution
API. No adaptive proposal search is needed for this qualification question.
The failed initialization remains recorded and charged.

The second pricing attempt completed 19 calls: one cold call and two warm
full-horizon calls per pair, each using 64 transitions and four chains. All
recorded pricing health-failure lists were empty. Inclusive pricing time was
540.945659172954 seconds. These observations establish cost and mechanics,
not acceptance qualification, convergence or a ranking of methods.

The user service manager exited before creating a service or worker. A
process/output check found no duplicate launch. The same tested supervisor then
launched in a new process session, with closed stdin and file logs. Heartbeats,
resource checks, deadlines and terminal settlement are unchanged. Trusted
process inspection confirmed both supervisor and worker survived launcher exit.

## Live qualification and accounting

Campaign root:
artifacts/modern-hmc-implementation-2026-10-07/N5/public-qualification-20261029-01.
The [qualification plan](bayesfilter-modern-hmc-public-qualification-2026-10-29.md)
passed skeptical review, full-horizon pricing and frozen-copy preflight. It
retains C1's scientific design without reusing C1's favorable outcomes.

Frozen pairs: L3 at .4 and .8; L5 at .4 and .8; L9, L13 and L18 at .4;
L25 at .2 and .4. The 16/32/64/128 replication ladder, family cap24 and
acceptance floor .55 are unchanged. Every failed and viable candidate remains
recorded. R-hat is reporting-only during tuning. NUTS is excluded. Fresh tuning
seed namespace is (20261029,6101); reload uses (20261029,2001+index).

The frozen consumer's six runtime files matched the installed API at launch.
Its different source path changes provenance identity, so fresh tuning evidence
is required. The immutable launch.json records commands, environment, source
hashes, inputs, design and allowance. Its prepared_not_launched field describes
the prelaunch state; the receipt and live manifest determine current status.

Supervisor PID 524957 and worker PID 524960 started at 01:12:30 Asia/Shanghai
on 29 October 2026 on GPU0, UUID GPU-a1ea1946-07c0-8ed5-2ba1-d96f82c89cd3.
Native trial chunks were being persisted at the first live check. Heartbeats
are recorded every minute; shared GPU capacity is assessed every 30 minutes.
The supervisor does not kill a worker for its own utilization. It records
terminal errors but does not itself perform autonomous code repairs.

The measured maximum native-work projection is 62194.58677121997 seconds.
The inherited 1.5 operational margin plus startup/export/closeout allowances
give 94431.88015682995 seconds, rounded up to **94432 seconds (26.23h)**.
This is a ceiling, not an ETA. The outer deadline is **30 October 2026,
03:26:22 Asia/Shanghai**; numerical work has a cooperative 900-second closeout
reserve. Source/data invalidity, broken required evidence or budget exhaustion
stop dependent work. A candidate rejection does not stop other candidates.

| Entry | Seconds | Accounting |
|---|---:|---|
| Prior upper bound | 235497.7418392606 | Before unmetered overhead |
| Engineering withheld | 3600 | Covers 1190.14167799789 measured attempt seconds |
| Preparation/review/packaging withheld | 7200 | Conservative reserve, not measured charge |
| Active qualification reservation | 94432 | Settle actual inclusive supervisor time once |
| Unreserved upper bound | 130265.74183926059 | Before final overhead settlement |
| Separate SIR balance | 7363.064379416 | Unchanged |

Do not deduct measured engineering twice or add child times to the inclusive
supervisor charge. No new allocation was created.

## Decision and remaining work

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Accept scoped public GPU engineering repair | Public target/runner parity, binding and failure isolation pass | No engineering veto in completed checks | Limited target, shapes and short chains | Continue complete-cohort qualification | General posterior validity |
| Continue current qualification | Terminal audit/export/reload pending | No campaign invalidity at initial live check; candidate screens pending | Stochastic outcomes and machine workload | Monitor checkpoint; audit terminal evidence; settle once | Passing candidates or campaign completion |
| Keep C3 posterior phase open | Independent reference and precision contract unresolved | No qualified posterior evidence | Reference uncertainty and estimator reliability | Specify estimands, uncertainty, fresh streams and sequential assessment | Acceptance proves convergence |

| Inference status | Assessment |
|---|---|
| Hard veto screen | Scoped engineering checks pass; fresh candidate screens are in progress. Historical movement failures remain failures. |
| Statistically supported ranking | None. This run checks integration and qualification, not method superiority. |
| Descriptive-only differences | Pricing, parity residuals and interim candidate counts do not rank samplers. |
| Default-readiness | Not established; numerical capability and scientific admission remain separate. |
| Next evidence needed | Complete native audit/export/reload, independent-reference posterior assessment and estimator/tail reliability work. |

Post-run red-team for the engineering tranche: the strongest alternative
explanation is that fixtures cover only a narrow part of the public route.
Complete-cohort execution reduces that gap but cannot prove posterior validity.
An actual source mismatch, health-isolation failure or discrepancy from the
checked body would overturn the engineering conclusion. Short-chain coverage
and the missing uncertainty-qualified independent reference remain weakest.

C4 arithmetic audits do not repair antithetic squared-coordinate or nonlinear
tail underestimation. C5 independent-fit SIR and C6/N8 integration, examples,
book and release reconciliation remain open. MacroFinance's wider master still
lacks the historical P19 result; do not fabricate or bypass it. N6/N7 conditional
research need not block a scoped optional release. The
[compact evidence index](../validation/modern-hmc-20261029-public-gpu/README.md)
preserves the repair and launch without a large additional raw-data archive.

### Initial live progress check — 29 October, 01:19 Asia/Shanghai

The first two measurement work items returned 32 distinct complete trials.
L=3, epsilon=.4 remains in validation with acceptance evidence still inconclusive;
L=3, epsilon=.8 failed the movement screen (15 of its 16 trials). Both observations
have valid evidence, no engineering-invalidity reasons and no hard vetoes.
R-hat is reporting-only. No candidate is fresh-verified yet. Later partial trial
chunks are saved separately; the native checkpoint and minute heartbeat keep
advancing. This is candidate-local rejection, not campaign failure.

Trusted process inspection confirms supervisor and worker alive. All 629 frozen
source hashes and all 41 archived payload hashes match. The read-only snapshot
is preserved in docs/validation/modern-hmc-20261029-public-gpu/initial-live-check.json.
The refreshed documents have no broken local links in their active sections;
scoped whitespace validation passes. No live numerical source was modified.

