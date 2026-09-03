# Phase 9A fresh-map tuning and replica-exchange preflight result

Date: 2026-08-31  
Subplan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-fresh-tuning-preflight-subplan-2026-08-31.md`  
Status: `CLOSED_PHASE9A_CONTINUATION_VETO_CHART1_BETA0`

## Decision

Phase 9A did not complete.  The q=20 bridge, strict backend, fresh chart
construction, endpoint reliability checks, and the three chart-0 scope
handoffs were exercised successfully.  The required six-scope condition was
not met: the chart-1, beta-0 tuner could not produce a verified handoff after
the declared cap repair.  The shared replica-exchange transition was therefore
not run.  Phase 9B remains closed.

This is a tuning/mechanics-boundary failure for one chart scope.  It is not
evidence that the q=20 target, proper bridge, reverse-KL identity, or learned
chart is mathematically invalid.  It is also not evidence of whitening, mode
discovery, convergence, posterior correctness, HMC readiness, superiority, or
high-dimensional scaling.

## Evidence contract and protocol

The target signature was
`9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`, with
strict `tensorflow_eigh_strict`, float64 target/transport values, TensorFlow
XLA, TF32 enabled at the runtime boundary, two fresh `(16,16)` tanh
two-stage charts, L3 beta ladder `(0,.5,1)`, pure continuation, and fixed
uniform chart-selection probabilities `(0.5,0.5)`.  Training used fresh
batch-native Gaussian batches of size 32.  Each HMC scope used four varied
latent starts, identity z mass, the active fixed-transport tuner, and
leapfrog length 5.  All outputs below are mechanics/preflight outputs.

## Attempt ledger

| Attempt | Result | Evidence and classification |
|---|---|---|
| `attempt-01` | Fail | Runner `NameError` for the omitted checkpoint-restore import.  Infrastructure defect; preserved in `failure.json`. |
| `attempt-02` | Fail | Chart 0, beta 0 was finite and mobile, but the `0.25` cap produced mean acceptance `0.984226394268513` and the tuner requested enlargement.  Tuning-cap repair trigger; preserved tuner JSON. |
| `attempt-03` | Partial pass | Fresh chart-0, beta-0 scope passed with selected epsilon `0.810010108139607`, acceptance `0.8599670269043127`, one trace per reusable HMC graph, and peak allocator `1402670592` bytes (about `1.31 GiB`) in `306.108961227932` seconds.  This was a localization run (`scope_limit=1`), so A3 was intentionally not run. |
| `attempt-04` | Fail | All three chart-0 scopes passed.  Chart 1, beta 0 reached acceptance `0.9989499884228412` at epsilon `0.6289781117350643`; its declared doubled repair `1.2579562234701287` exceeded the cap `1.0`.  Scope-specific cap failure; preserved tuner JSON. |
| `attempt-05` | Fail/continuation veto | Under the final cap `2.0`, chart 1, beta 0 reached acceptance `0.9989499884228412` at epsilon `0.6289781117350643`, then `0.9396178294526476` at epsilon `1.2051892998990203`.  The next repair requested epsilon `2.4103785997980407`, exceeding cap `2.0`; tuner emitted `tune_initial_step_size_exceeds_configured_cap`, `verification_acceptance_outside_pass_band`, and `no_viable_candidate`. |

The attempt directories are preserved under
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/phase9a-fresh-tuning-preflight/`.
The passing partial manifest hash is
`c9da178fbb75cb120724aaf39ba0e583d78cc8de5768a700812de33f4567fe37`.
Attempts 04 and 05 failed before the runner's final manifest write; their
scope tuner JSON and `failure.json` files are the authoritative evidence.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Fresh chart construction | Fresh tensors, checkpoint replay, endpoint reliability | Passed in the completed portions; no chart inverse/logdet/score veto observed | Full-run manifest was not written after the later scope failure | Preserve charts as diagnostic evidence and retest only under a new repair subplan | No chart quality or whitening claim |
| Scope-specific HMC handoff | Six independent verified tuner handoffs required | Failed: 3 chart-0 scopes passed; chart-1 beta-0 failed twice under declared caps; chart-1 beta `.5`/`1` were not reached | Short dual-averaging budgets and chart-specific curvature may explain the cap hits | Write a new target-specific chart-1/beta-0 tuning repair with an explicit candidate grid or longer adaptation budget | No HMC readiness or sampler ranking |
| Shared replica exchange | One controller chunk after all six handoffs | Not run; prerequisite handoff veto fired | Swap and controller behavior remain untested for this fresh bank | Re-run A3 only after every scope has a durable handoff | No transition or mode-travel claim |
| Phase 9A | All A1-A3 primary criteria | Closed by continuation veto | Failure may be bounded tuning policy rather than map failure | Refresh a smallest-scope repair plan; do not open Phase 9B | No posterior, convergence, or production claim |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Supported for the stated preflight: the chart-1/beta-0 tuner emitted a reproducible cap/handoff veto.  No target nonfinite, bridge properness, memory-growth, or learned-map reliability veto was observed in the completed portions. |
| Statistically supported ranking | None.  There was no comparator and no uncertainty analysis. |
| Descriptive-only differences | Chart-specific acceptance and selected-step behavior, checkpoint losses, trace counts, timings, and allocator telemetry are descriptive mechanics diagnostics only. |
| Default readiness | Not assessed and not promoted.  The cap, short tuning schedule, chart, and architecture remain preflight hypotheses. |
| Next evidence needed | A reviewed, fresh-scope repair that explains or resolves chart-1/beta-0 curvature, emits a verified handoff, and then completes the same six-scope controller preflight. |

## Interpretation and red team

The observed pattern is consistent with a chart-specific scale/curvature
mismatch: chart 1 beta 0 remained near-unit acceptance through the first two
repair rounds, so the bounded ladder could not reach the declared acceptance
band.  The same target and protocol produced a valid chart-0 beta-0 handoff,
which argues against treating the failure as a global bridge or backend failure.
That comparison is descriptive, not a statistical ranking.

The strongest alternative explanation is that the four-step dual-averaging
preflight is too short and the fixed cap policy is too restrictive for this
particular fresh initialization.  A seed-specific chart geometry or an
unobserved numerical pathology is also possible.  Evidence that would overturn
the current tuning-boundary diagnosis is a fresh, disjoint chart-1/beta-0
repair using a predeclared longer adaptation or fixed candidate grid that
passes finite/status/movement checks and produces a verified handoff without
changing the target or silently relaxing acceptance criteria.

TensorFlow emitted global retracing warnings while independent trainer and
runner objects were constructed.  The explicit per-instance training and HMC
runner trace checks passed where manifests were completed; the warning is
retained as performance debt and is not used as scientific evidence.

The failure runner did not write a final manifest, so exact process wall time,
allocator telemetry, and full command/environment provenance are unavailable
for attempts 04 and 05.  This is an artifact-quality gap, not a reason to
reinterpret their tuner JSON.  Any future repair must write a progress/failure
manifest before scope execution and include elapsed time, device telemetry,
and the active plan hash.

## Next boundary

Do not launch Phase 9B.  The next action requires a separately audited repair
subplan limited to chart 1, beta 0 (and, if justified by that repair, a fresh
full six-scope replay).  It must state whether it tests longer dual averaging,
a fixed step-size candidate grid, or a revised per-scope cap, give a finite
compute budget, and preserve the current target, map-seed domain, acceptance
band, and nonclaim boundary unless a new plan explicitly changes them.
