# M31--M38 execution result

The funded repair and engineering program has reached M38. It repaired two
validation defects: posterior batch/lugsail/readiness controls could not reach
the pipeline, and completed fits could return cached results after a policy
change. Stopped and fixed-count reports now share the declared estimator.
Identity is checked before completed-result reuse and interrupted replay,
including library source, design, data, fit IDs and graph-reuse policy. Historical
directories without this identity remain readable but need fresh directories
for new execution.

This does **not** close every scientific requirement. Full-fit stopped coverage
and defect-power confirmation remain underfunded under the reviewed design;
global exploration and learned-map quality remain open; exact consumer inputs
are absent. No numerical tuning default changed. R-hat, ESS, MCSE and reference
diagnostics remain posterior-only; every verified candidate stays retained.

## Requirement disposition

| Requirement | Result and evidence | Uncertainty / next justified action |
| --- | --- | --- |
| R1: public procedure and lifecycle | Closed for the declared engineering matrix: dispatch, preparation, per-L evidence, replay, unsupported branches, failure/restart and retention tests passed. | Finite route/model coverage, not universal tuning success. |
| R2: equilibration and precision | Policy forwarding and resume attribution repaired; independent batch arithmetic, unavailable batches, fixed/stopped equality and actual posterior call-chain tests passed. | Prior coverage/delivery failures remain evidence. Adequately funded complete-fit fixed/stopped confirmation is still needed. |
| R3: full-fit sensitivity | Exact-normal endpoint diagnostic implemented. Baseline, no-op and quarter/half-SD translated targets executed. Baseline/no-op evidence and draws agree exactly. | Full adaptive-procedure null size and power remain unconfirmed. Missing/capped fits stay in planned denominators. |
| R4: global exploration | Public mixture posterior calls evaluate the mode indicator from single-mode and mode-dispersed starts. Constant/missed-mode controls and independent probability checks passed. | Reporting failure correctly does not make HMC cross modes. The global-exploration failure remains open; supplied-whitened funnel evidence is separate. |
| R5: exact consumer | Fresh bounded inventory and local reply completed. | Both bootstrap paths are absent in both checkouts. The unchanged block-model MIDAS report is not a full-joint reference. Await the exact bundle. |
| R6: execution and structure | Three additional GPU pairs match all candidates, numerical receipts, warmup/retained tensors and posterior decisions. Source-change rejection is tested. | The 20,751-line legacy module remains maintenance debt. Preparation imports its `_json_ready`; mass adaptation imports `_HMCPhaseAttemptState`. No found numerical defect warranted a broad extraction. Float64 and conditional position-field scope remain explicit. |
| R7: learned map | Banana and mixture batch/graph/freeze/reload/retune composition passed on CPU and GPU/XLA. Each GPU map retained two verified tuning members. | Both tiny posterior assessments failed; one update establishes no map quality. Target-specific search adequacy, holdout assessment and serious training price remain unresolved. |
| R8: guide and status | Official chapters 21b/25 and API reference updated; registry checks passed. The 574-page book built without undefined citations/references; changed pages were inspected. | Unrelated chapter 26b/bibliography edits were excluded. Historical evidence retains its original source. |

## Executed evidence

M31 inspected local technical papers, proofs and official statistical code,
then reproduced exact binomial and power arithmetic. The
[design and funding record](artifacts/hmc-repair-master-2026-09-16/m31-r1/design-and-funding.md)
and `design-arithmetic.json` preserve assumptions. The chosen 384-fit coverage
design per model costs approximately 36.97 Gaussian plus 30.45 beta-binomial
GPU hours at descriptive M30 prices, exceeding the remaining allowance. This
is that design's price, not a universal lower bound. The normal-endpoint law
is a diagnostic alternative, not an achieved finite-HMC null/power result.
No token confirmation was launched.

M32's baseline suite passed **220 tests**. One optional ArviZ reference module
was skipped because matplotlib is absent. Six fresh GPU/XLA fits completed on
rotated Gaussian, constrained Dirichlet and supplied residual-whitened funnel.
Each static/dynamic pair retained two members and matched all four numerical
receipts, warmup/retained arrays and posterior decisions exactly. See
`m32-r1/gpu-parity-audit.json`. Their predeclared small counts and broader test
acceptance band establish mechanics, not production qualification or speed.

M33 exposed `posterior_precision_settings` for batch size, minimum batches and
lugsail r/c, and `posterior_assessment_settings` for ESS floors and readiness
persistence. Tests compare lugsail arithmetic to independent batch formulas
and actual stopped/fixed reports to the runtime policy. Changed estimator,
readiness, source and legacy unbound fits are rejected before cached returns;
the same-policy call preserves its saved result.

M34 uses one frozen simulated normal-conjugate dataset and shared streams in
four controls. These are paired, **not four independent replications**. The
zero-mean prior/noise scales are tau=2, sigma=1 with six observations. Ordinary
preparation uses the native broad L grid and original acceptance policy.
Baseline/no-op retain 11 members each; quarter/half-SD translations retain 13
and 16. One predeclared member per arm passed local posterior checks after
500 warmup and 500 retained draws; all siblings remain recorded. The endpoint
is the last retained draw of chain zero from the first verified candidate by
ID. Complete no-op numerical evidence and posterior draws equal baseline.
See `m34-r1/activation-audit.json`. Deliberately wrong targets passing local
checks illustrate why those checks alone cannot establish accuracy. These
outputs give no supported detection-rate estimate or candidate ranking.

M35 ran actual public mixture calls from both starting regimes. The indicator
reaches shared posterior assessment; the independent reference probability is
`w*Phi(a)+(1-w)*Phi(-a)`. Constant indicators cannot supply usable MCSE. This
tests wiring and failure reporting, not discovery of unknown modes.

M36's exact-input inventory and response are in `m36-r1/input-inventory.json`
and [the local MacroFinance reply](bayesfilter-hmc-m36-macrofinance-reply-2026-09-23.md).
The consumed bootstrap was not rerun; no external message was sent.

M37 executed `NeuTraReverseKLTrainer.train_step` through its fixed-shape batch
graph, frozen export/reload and public fixed-map tuner on banana and mixture.
The trusted GPU canary verified XLA, memory growth, batch size eight, one
update, static signatures, frozen forward/logdet equality and inverse agreement.
It used 110.40 outer GPU seconds including compilation and retuning. This
cannot price adequately trained maps. The
[target-specific protocol](bayesfilter-hmc-m37-training-protocol-2026-09-23.md)
constructs simple geometry comparators and conditional regimes. Search ranges
remain hypotheses; no inherited recipe was promoted to a quality result.

M38's affected suite initially passed 120 tests; two new global-wiring tests
read the wrong report nesting and were corrected. The subsequent guide/global
suite passed 66 tests, including those two. The final source/resume suite
passed 53 tests. **These counts overlap and must not be summed.** The final
book used the committed baseline plus owned edits and five committed figures.
PDF pages 430, 431 and 474 were inspected; `m38-r1/guide-verification.json`
records the checksum and reference check.

## Provenance, failures and budget

Baseline commit is `e9fee584704a465fc6ab984e3a8cc53980f335d0`. Attempt manifests
record exact commands, environment, seeds/design, source hashes, device,
elapsed time and outputs. CPU workers hide GPUs. GPU workers used trusted
execution, memory growth and XLA. Unrelated dirty Q20/training, chapter 26b,
bibliography and governance files were excluded from numerical snapshots.

M32/M34 GPU fits use `m38-r1/source-final-r2`. The only later library change is
the source-identity check before validation resume in `procedures.py`; it
changes no transition, diagnostic or fresh-fit output. That guard was tested
on r4; M37 and final documentation use r5. Older evidence is not relabeled as
new scientific validation. Full numerical archives/snapshots remain local;
compact designs, audits, manifests and receipts are published with the runners.

Failed attempts remain charged and preserved. The first snapshot omitted a
script needed for collection. The first arithmetic fixture correctly met a
negative per-chain lugsail estimate, while its new cached-policy test exposed
the real bug. Later failures were obsolete error-message assertions and wrong
report nesting. The first book snapshot omitted five committed figures. Each
was repaired and its affected check passed. The optional ArviZ module remains
unexecuted; its explicit check reports a missing matplotlib module, not a
numerical failure.

The ledgers reconcile **2,480.699 CPU worker-seconds** and **1,167.560 GPU
worker-seconds**, including failed attempts and conservative bookkeeping.
Remaining authorized allowance is **71,946.615 CPU** and **74,041.939 GPU
worker-seconds** (about 19.99 and 20.57 hours). Mixed integration tests are
charged once to their outer receipt phase. No worker remains running.
`m38-r1/program-reconciliation.json` gives the compact total.

## Decision and skeptical audit

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Publish validation repairs | Public call-chain and restart tests pass | Known engineering failures repaired | Finite model matrix | Use explicit policies and fresh identities | Nominal stopped coverage |
| Accept added reuse cells | Exact evidence and array equality | No parity failure | Other targets/shapes | Preserve optional scoped reuse | Speed superiority |
| Accept endpoint development | Formulas, denominators and activation pass | Incomplete outputs cannot count as detections | Adaptive output law | Fund null/power confirmation | Achieved power |
| Finish bounded roadmap | Every package has evidence or a dependency | Missing inputs block only R5 | Confirmation and learned geometry | Address named gaps under an explicit funded amendment | All scientific gaps closed |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No unresolved engineering veto in executed scope. Tiny learned-map posterior failures veto promotion, not the roadmap. |
| Statistically supported ranking | None sought or established. |
| Descriptive-only differences | Times, endpoints, member counts and tiny posterior outcomes. |
| Default-readiness | No default-policy promotion. |
| Next evidence | Complete denominators/uncertainty for R2/R3; validated learned geometry/global checks; matching consumer inputs. |

Finite-sample and shared-stream behavior, especially after adaptive stopping,
remain the strongest alternative explanations for apparently successful
posterior outputs. Adequately powered independent failures would overturn a
calibration claim; none is made here. The one-update map canary is the weakest
evidence and establishes composition only. Engineering, sampler validity and
scientific interpretation remain separate. M38 ends this roadmap; there is no
automatic M39 or further one-experiment phase sequence.
