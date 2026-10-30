# SIR frozen-candidate cohort result

20 October 2026. The bounded cohort finished at **01:42 China time**. All four
methods are terminal. The service is inactive; no new numerical run was launched
for this closeout. This completes the SIR evidence phase, not the modern-HMC
master program or its release requirements.

## Results and their meaning

The target was the same full horizon-20, three-parameter SIR finite-filter
posterior, with frozen affine geometry, four starting chains and the existing
fixed-HMC, randomized-HMC, SNAPER-proposed randomized-HMC and fully refreshed
MAMS proposals. All 24 requested roots were created and verified by the shared
tuner. No repair descendants were needed. Tuning membership remained determined
by acceptance and numerical health; posterior diagnostics did not remove members.

| Method | Verified tuning members | Posterior checks passed | Posterior and reference delivered | Failure or incomplete work | Charged worker-seconds |
| --- | ---: | ---: | ---: | --- | ---: |
| Fixed HMC | 8 | 7 | 6 | Member 1 reference failure; member 8 resource-censored | 83767.59229301522 |
| Randomized HMC | 8 | 8 | 7 | Member 1 reference failure | 85925.96169290878 |
| SNAPER-proposed randomized HMC | 6 | 6 | 6 | None in this cohort | 21119.84875757806 |
| Fully refreshed MAMS | 2 | 2 | 2 | None in this cohort | 6502.084065745585 |
| Total | 24 | 23 | 21 | Two reference failures; one resource-censored member | 197315.48680924764 |

Fixed member 1's maximum normalized reference error was
`0.10389830934231056`; randomized member 1's was `0.10192058612266966`. Both
exceed the frozen `0.1` limit. Their posterior diagnostics passed, but agreement
under the declared reference screen was not established. Both remain verified
tuning members. No failed member was selectively rerun, and no threshold was
relaxed. Fixed member 8 was not fully assessed before the original method cap;
its lack of delivery is resource censoring, not a demonstrated sampler failure.

These counts describe this frozen-proposal cohort. They do not establish that
SNAPER or MAMS is superior: candidate sets, trajectory costs and resource
allocations differ, and there are no independent-fit replications supporting a
method ranking. The reference concerns the declared finite-filter target, not
the exact nonlinear state-space posterior. The operational MCSE/stopping screens
retain the calibration limitations documented by the October 18 studies.

## Audit and accounting

The controller's strict terminal audits checked candidate exports, tuning
evidence, warm-up exclusion, posterior decisions, tensor bindings, launch and
source identity, and inclusive numerical costs. The independent closeout
recounted member decisions and checked **5,530 unique SHA-256 bindings** against
the saved evidence. All checks passed. This is a fresh integrity and arithmetic
audit; it does not independently recompute TensorFlow ESS, R-hat or MCSE.

Numerical authority remains `full-model-source-e68cdf1c4`, commit
`e68cdf1c4c6b75c16f92b86a3363a1e24690d22c`; orchestration remains
`continuation-source-03`. Every numerical method recorded GPU 1, XLA, TF32 and
memory growth established before device initialization. Exact commands, data
and geometry identities, seeds and per-method manifests are preserved in the
[closeout run manifest](artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/sir-cohort-20261017-01/closeout-20261020-01/run-manifest.json).

The controller released `22377.993848672195` seconds after numerical, pricing
and engineering charges. Closeout charged the previously outstanding service
submission (`0.04695963021367788` seconds) once, plus the measured closeout
check (`2.3410479170270264` seconds), leaving **22375.605841124954 seconds**
available from this SIR allocation. The timer includes integrity checks and
manifest writing through final receipt preparation; the final receipt write is
excluded. Existing controller charges and pricing were not added again.
The separate diagnostic balance remains `131192.0584563613` seconds. The
separate 3600-second hold and historical 1579.5929542018566-second balance
remain excluded. No new study is funded by this note.

Artifacts are in
[closeout-20261020-01](artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/sir-cohort-20261017-01/closeout-20261020-01/):
`audit.json`, `settlement.json`, `run-manifest.json` and `audit_closeout.py`.
The run manifest preserves the exact Python command and driver path. This audit
used standard-library Python in `tfgpu` with `CUDA_VISIBLE_DEVICES=-1`; GPU
devices were intentionally hidden.
The [closeout plan](bayesfilter-modern-hmc-sir-closeout-2026-10-20.md) records its
evidence contract and skeptical review.

## Decision and next phase

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close bounded SIR execution | Complete 24-root inventory and audited accounting pass | Two reference failures block their deliveries; one fixed member remains censored | Replication, precision calibration and consumer transfer remain untested at release scope | Price a predeclared multi-model/independent-seed confirmation using measured all-member costs; keep failed members in its denominator | Master completion, method ranking or release |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Two reference-screen failures; no new integrity or implementation failure found. One resource-censored member is unresolved. |
| Statistically supported ranking | None. |
| Descriptive-only differences | 6/8, 7/8, 6/6 and 2/2 deliveries and the observed method costs. |
| Default readiness | No new default or estimator is validated. |
| Next evidence needed | Matched independent-fit confirmation, actual-stopping calibration, and actual MacroFinance/DSGE consumer qualification. |

The next N5 phase must be explicitly scoped and priced before using released
funds; it is not a selective retry of these reference failures. Actual consumer
qualification remains N8 work. Conditional N6/N7 alternatives do not block a
properly evidenced scoped N2–N5 release. Refresh the master from these completed
counts and measured costs, preserving the separate funding pools.

Post-run red-team: unequal costs and few independent fits could make the
observed delivery counts misleading as a performance comparison. Passing
operational posterior diagnostics cannot rescue failed reference agreement or
establish nominal MCSE coverage. Independent replications with calibrated
posterior accuracy could change the scientific conclusion; a binding mismatch
would invalidate this engineering closeout. Neither was manufactured here.
