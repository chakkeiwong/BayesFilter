# M10 scoped controls and cost result — 2026-09-21

The full canonical DCTR protection ladder selected multiplier0.1 for the M10
T2,N1024 noise-scale scope. All nominal and stress probes were healthy on both
independent tuning partitions. This qualifies experimental numerical controls;
it does not establish reliable surrogate-HMC inference. Both B and C exceed
M10's remaining initial CPU allocation even under the optimistic cost screen.
No M10 tuning or retained chain was launched.

The scientific goal remains testing endpoint-corrected HMC driven by the
imperfect analytical DCTR field. B/R, C/D, D/R and C/R remain separate and
NOT_RUN. A D/R discrepancy would not cancel C/D. The current deferral is a
resource decision, not evidence against that mechanism.

## Checked numerical evidence

[Consumer02](artifacts/surrogate-hmc-all-models-20260921-m10-consumer-02/report.json)
passed all15 checks after the fixture accepted the original calibration and
validation data roles. Enabling filter/reset moment diagnostics preserved the
same values, forces and status bits. Direct canonical-call parity, fixed-bank
replay and row independence remain checked; no reduced DCTR fork was introduced.

[Controls01](artifacts/surrogate-hmc-all-models-20260921-m10-controls-01/report.json)
compared the target value, analytical force, filtering means/covariances and
reset means/covariances. The table reports maximum nominal normalized change
between adjacent protection strengths; the declared tolerance is1e-3.

| Stronger/weaker multiplier | Calibration | Validation | Both pass |
|---|---:|---:|---|
| 1 / 0.1 | 0.001119176857 | 0.000134724804 | No |
| 0.1 / 0.01 | 0.000014612026 | 0.000002501515 | Yes; select0.1 |
| 0.01 / 0.001 | 0.000000201083 | 0.000000198457 | Yes |

Every nominal and stress row was healthy in all16 evaluated scopes. Selection
uses the first adjacent non-harm pass on both partitions, never reference
accuracy, acceptance or speed. This result does not identify an optimal force.
The stored selection is bound to the actual original data, numerical sources,
scope and measured arms. Claim observations were not used.

## Cost and decision

Each arm measured one compilation and four complete four-chain, one-leapfrog
transitions. The screen multiplies compilation plus3250 minimum steady costs
by1.5. It omits nomination, additional leapfrogs, references, heuristics and
analysis. B uses the local513 mechanics endpoint as a cost proxy; a full-prior
B endpoint still needs qualification. C uses the frozen selected DCTR scalar;
an independent D posterior reference remains required.

| Arm | Optimistic CPU seconds | Optimistic GPU wall seconds | Initial-allocation result |
|---|---:|---:|---|
| B | 1046.7254 | 1057.8719 | Deferred |
| C | 1110.7885 | 1121.8631 | Deferred |
| Both | 2157.5139 | 2179.7349 | Deferred |

The forecast saw727.6983 CPU seconds remaining before final bookkeeping.
The completed run left709.7906 CPU and6472.2899 GPU seconds. Subsequent tests
and master reporting leave690.4338 CPU seconds. Each arm exceeds every one of
these remaining CPU values. GPU capacity is not the binding resource. Peak
TensorFlow allocation was33,643,776 bytes, below the4GiB stage cap.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Select scoped controls0.1 | Declared non-harm rule passed | No invalid nominal/stress rows; source/data checks pass | Sparse probes can miss trajectory failures | Preserve selection for this exact scope | Exact score, optimal controls or posterior recovery |
| Defer M10 inference within its initial allocation | Both optimistic costs exceed remaining CPU | Missing B endpoint/D reference also block their dependent claims | Full runs cost more than this screen and may mix poorly | Give remaining ready families a turn; prospectively allocate and forecast any M10 return | Rejection of surrogate HMC or all-model completion |

| Inference status | Evidence |
|---|---|
| Hard veto screen | Mechanics and control validity passed; cost screen blocks a new initial-allocation launch |
| Statistically supported ranking | None |
| Descriptive-only differences | Measured timings and adjacent control changes; no stochastic superiority claim |
| Default readiness | Not established |
| Next evidence needed | Qualified endpoints/references, affordable complete forecast, public tuning, retained comparisons and conditional heuristic checks |

## Master integration and execution record

The master now exposes `--scope M10 --stage P3-M10-controls`, verifies prior
control evidence and reports the per-arm cost screen without a new run.
The controls dependency is consumer mechanics; it does not depend on R passing.
Standalone and nested-master control costs use the same accounting reader.
Missing prior reports require interpretation before another attempt. Reuse
checks numerical source hashes, original tuning roles, wrapper equality,
measured selected controls, all ladder files and the non-harm decision.

Focused regressions passed25/25 in
[tests04](artifacts/surrogate-hmc-all-models-20260921-m10-controls-tests-04/command.log).
Public reuse01 exposed a reporting-order defect: the saved decision preceded
the control-budget update. The repaired public run
[master02](artifacts/surrogate-hmc-all-models-20260921-m10-controls-master-02/decision.json)
is COMPLETED with overall INCOMPLETE_IMPLEMENTATION and blocker
UNDER_BUDGETED_INITIAL_ALLOCATION. Its only job reused preparation, with no
GPU probe, no new sampling and no inference authority. All four comparisons
remain NOT_RUN. Earlier output remains preserved, not overwritten.

Exact commands, environment, seeds, data/source hashes and memory policy are
in each run manifest. Commands used the tftwogpu Python environment. Consumer02
and controls01 ran on escalated GPU1 with verified growth, TF32 and XLA; tests
and checked master reuse deliberately hid GPU. Source snapshots preserve the
executed numerical program and plan before later orchestration repairs.

| Work | CPU seconds | GPU wall seconds |
|---|---:|---:|
| Tests01–02 | 17.683613 | 0 |
| Consumer02 | 77.967253469 | 74.227487337 |
| Controls01 | 255.725082174 | 246.498094814 |
| Closeout tests03–04 | 1.145719540 | 0 |
| Master reuse01–02 | 18.211119000 | 0 |

Global charged totals after master02 are9410.299116665 GPU seconds and
10951.767909164 CPU-worker seconds, below226800/54000. Charges include failed
tests and reporting attempts; nested job accounting is not added twice.

Post-run red team: healthy tuning probes may still hide failures on an HMC
trajectory. The strongest alternative to successful downstream use is an
expensive or poorly mixing force despite benign probes. A fresh qualified,
affordable inference comparison could overturn that concern; this preparation
cannot. The weakest evidence is the optimistic cost extrapolation and sparse
parameter coverage. Heuristic adversaries and tail comparisons remain unrun.
