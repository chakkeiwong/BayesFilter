# State-space campaign preparation completed

The K0–K7 implementation and CPU preparation are complete. The frozen package
is ready for GPU preflight after C1 finishes and its enclosing receipt is
settled. No state-space GPU work has started; C1 was still active at the final
read-only check. This completes preparation, not the 32-fit experiment.

The [active plan](bayesfilter-hmc-state-space-48h-plan-2026-09-25.md#launch-after-c1-settlement)
contains the exact frozen launch command and evidence contract. The
[master program](bayesfilter-hmc-repair-master-program-2026-09-16.md) and its
machine checkpoint record the same next steps. The official tuning guide
remains `docs/main.tex`, its tuning chapter, and the public interface reference.

## Implemented scope

All eight cases call actual BayesFilter filtering implementations: QR Kalman
for location, persistence and noise-scale models; SVD Kalman for the existing
four-state, 18-parameter, T=120 fixture; and the batched SVD sigma-point filter
for the two-parameter, T=16 nonlinear case. The scalar stationary initial law
and its parameter derivatives are included. The new persistence chart reaches
0.97. Priors are defined on the declared raw coordinates; no extra chart
Jacobian is inserted into those priors. Synthetic observations come from the
declared model simulations.

The 32 main slots retain the original datasets and sampler seeds. They use
the broad L grid `(3,5,9,13,18,25)`, native ordinary search, candidate-specific
repair and fresh verification. All verified members are retained. The
predeclared posterior assessment chooses two distinct-L members for K0/K7 and
one for other cases, without using posterior results to choose them. R-hat,
ESS and MCSE remain posterior checks. The serious warmup minimum/window/cap
remain 2,000/1,000/10,000; retained sampling has a 10,000 cap and per-quantity
mean/median precision requirements with lugsail mean MCSE.

The controller provides CPU reference preparation, source freezing, two GPU
engineering preflights, eight complete-fit pilots, affordability decisions,
bounded main execution and a report preserving all 32 slots. It checks C1
settlement, the separate NeuTra reservation, frozen source/data, the selected
GPU UUID, memory growth, actual GPU value/score placement, and matching pilot
workloads. Empty searches, missing assessed members, warmup-only results,
censored runs and retries cannot provide a cheap full-fit price. Enclosing
receipts are charged once; an unconfirmed shutdown preserves its reservation.
A failed stage now returns a failing CLI exit status and still writes a report.

## Verification and evidence

All numerical checks used the `tfgpu` Python environment with GPUs explicitly
hidden and CPU thread limits of two. CPU XLA equivalence is useful engineering
evidence, but cannot establish GPU compatibility or placement.

| Check | Result | Durable receipt |
| --- | --- | --- |
| Broad SSM regression selection | 142 passed, zero failed/skipped; 707.664 enclosing seconds | [r5 execution](artifacts/hmc-state-space-preparation-2026-09-26/prepared-r5/regression-execution.json), [JUnit](artifacts/hmc-state-space-preparation-2026-09-26/prepared-r5/regression.xml) |
| Final controller and strict K0 ordinary/prepared/isolated checks | 24 passed; 106.813 enclosing seconds | [final execution](artifacts/hmc-state-space-preparation-2026-09-26/final-checks/execution.json), [JUnit](artifacts/hmc-state-space-preparation-2026-09-26/final-checks/regression.xml) |
| Actual model arithmetic and total scores | All eight model cases passed independent likelihood and two-scale finite-difference checks | Included in the broad selection |
| CPU XLA against graph execution | All five distinct execution shapes passed, with bounded tracing checks | Included in the broad selection |
| Independent posterior references | All 14 oracle-backed datasets passed the declared sensitivity checks; K6 explicitly unavailable | [frozen references](artifacts/hmc-state-space-preparation-2026-09-26/prepared-final-r1/references) |
| Frozen package, inputs, dependencies and stage planning | Passed; 8 mechanics cells, 4 bridges, 8 pilots and 32 original main slots | [launch readiness](artifacts/hmc-state-space-preparation-2026-09-26/prepared-final-r1/launch-readiness.json) |
| Official book | PDF built and relevant rendered pages inspected; unresolved citations elsewhere remain | [built book](artifacts/hmc-state-space-preparation-2026-09-26/book/main.pdf), [build log](artifacts/hmc-state-space-preparation-2026-09-26/debug-logs/bayesfilter-ssm-book.log) |

These test selections overlap; they are not 166 unique tests. The only package
file changed after the broad selection was `ssm_campaign.py`. Its final quota,
price, GPU identity and command-failure behavior is covered by the final
focused selection. The final package identity is
`6aa8915e05ccaf3d0ab4ad03ff265777e4786aaace83113ab3f792ace16cde0f`, with Git
baseline `de80aaff5812ebfbed551977476c0868551a2c88` and preserved dirty-source
hashes/diff. Live unrelated work was neither reverted nor committed.

All eight broad ordinary/prepared fixtures actually produced verified
candidates. An intermediate test allowed an empty search to count as a
positive lifecycle check; the final test removes that allowance, and the
saved r5 results show that branch was never taken. The strict K0 cases were
rerun after increasing their test-only work quota from 8 to 32. Most tiny
fixtures reached posterior assessment but exhausted their eight-transition
warmup cap with zero retained draws. K0 prepared produced eight retained
draws per chain and exercised archive/reload and warmup exclusion. These are
mechanism checks, not successful posterior estimates. See the
[public-case inventory](artifacts/hmc-state-space-preparation-2026-09-26/broad-public-inventory.json).

The independent K0 fixed-kernel diagnostic used the predeclared 512 anchors,
L=3, epsilon 0.12 and existing rank/two-sample tests. Baseline and no-op had no
detected discrepancy; deliberately wrong energy was detected by both checks.
It consumed 23.374 CPU worker-seconds. Its
[receipt](artifacts/hmc-state-space-preparation-2026-09-26/prepared-r3/stationarity-execution.json)
and native results are preserved separately from full-procedure fits. Lack of
detection is not proof of stationarity or calibrated detector power.

K3 dataset A needed the planned 321-point grid after its 161-point sensitivity
check failed. No tolerance was relaxed. The K7 T=2 true-model Gauss–Hermite
diagnostic changed by approximately `7.19e-6` from order 15 to 21, below its
declared `0.005` refinement tolerance. Its sigma-point likelihood differs from
that latent integration result. K7's sampler reference is the same sigma-point
approximation it samples; this diagnostic does not establish the approximation's
long-horizon accuracy. K6 has no independent joint parameter-posterior oracle.

## Repairs and preserved failures

The preliminary fixture-only suite was rejected before launch. The replacement
implements the actual K0–K7 definitions, generated data, independent references
and serious sample-count policy. Failed development attempts remain in the
preparation directory; their logs are retained under `debug-logs/`.

| Failure | Repair and subsequent check |
| --- | --- |
| Omitted-data control was rejected before reaching its discrepancy detector | Register and activate the actual-filter control; verify density disagreement specifically |
| Preparation and candidate target-status policies differed | Use the same `per_chain_step` policy through preparation, export, reload and isolated execution |
| Nonlinear module constants were created inside an initial trace | Import numerical helper modules before tracing target instances; repeated-instance and public-route checks passed |
| Nonlinear covariance telemetry omitted the paired condition field | Report the scalar innovation covariance condition consistently; numerical health and both public routes passed |
| Shared QR tracing erased the static event dimension across targets | Restore explicit value/score/status shapes in campaign and legacy adapters; ordered mixed-model regression passed |
| Changing live source invalidated a running fit's identity | Freeze source for broad tests and avoid package edits during live focused tests; do not bypass identity checking |
| Frozen tests lacked a cited legacy QR evidence dependency | Include that dependency and K6's complete data/configuration files in the snapshot |
| K0 short test exhausted its search quota | Increase only the mechanics fixture quota; keep the verified-candidate assertion and acceptance criteria |
| Warmup-only assessed members could yield a cheap runtime price | Require the declared member count and at least 1,000 retained draws per member in complete-fit pilot receipts |
| Failed stage returned a successful launcher exit status | Return nonzero, stop later stages, preserve reporting; focused CLI regression passed |

The final audit checked wrong numerical baselines, hidden GPU fallback,
parameter/data/source mismatches, pilot/main workload mismatch, statistical
claims from tiny tests, shrinking denominators and reuse of C1's reservation.
The corrected preparation passes this engineering audit. GPU evidence and
runtime prices remain execution tasks with their original criteria.

## Budget and continuation

The [CPU accounting record](artifacts/hmc-state-space-preparation-2026-09-26/cpu-execution.json)
preserves failed and successful subreceipts. Early calls and the book build
lack complete enclosing timing, so exact total worker use is not known.
Conservatively charge the full predeclared six-CPU-hour preparation allocation,
once, instead of inventing a precise total or adding nested timings twice.
The [derived CPU reconciliation](artifacts/hmc-state-space-preparation-2026-09-26/cpu-reconciliation.json)
leaves 45,738.430 worker-seconds from the inspected CPU balance. The preparation
allocation is part of the 16-hour CPU campaign ceiling; at most ten CPU hours
remain for its post-C1 work. No GPU time was charged or reserved by preparation.

The frozen launch root is
`artifacts/hmc-state-space-preparation-2026-09-26/prepared-final-r1/`.
Checked datasets and reference receipts were copied unchanged, and the suites
were rebuilt with the final tested controller; its manifest records that
derivation. The frozen launcher imports its own package. Later live-checkout
edits therefore cannot change the campaign silently.

After C1 settles, check the selected GPU's availability and execute the command
in the active plan. Preflight plus full-fit pricing has a 10,770-second ceiling
including enclosing allowances. The main ceiling is 22 GPU hours, with a
four-hour reserve, all constrained by the actual grant remaining after C1 and
the separate 1,200-second NeuTra reservation. These are ceilings, not additional
grants. Full 32-slot affordability is unknown until measured pilots and C1's
terminal charge are available. Unfunded slots remain explicit. The 48-hour
run clock begins at the first GPU stage; new work stops by hour 42 and
computation by hour 46. Preparation time is reported separately.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Implementation ready for GPU preflight | CPU arithmetic, lifecycle, persistence and accounting checks passed | Known engineering failures repaired and retested | GPU compilation, device placement and full-fit costs not measured here | Settle C1, then execute frozen preflights | GPU qualification or full campaign success |
| References usable for declared scopes | 14 checked dataset references | K6 oracle unavailable; failed sensitivity never accepted | Finite refinement is not a rigorous error bound; K7 is approximate | Use matching references after real retained sampling | True-model nonlinear accuracy or K6 posterior accuracy |
| Preserve C1 and bounded funding | No competing launch; original 32-slot report preserved | Active C1 blocks launch | Remaining grant and number of affordable main cases | Price complete unchanged workloads after settlement | A new 36-hour GPU grant |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Deliberate score/data/energy defects detected; known infrastructure failures repaired; failed warmup remains explicit |
| Statistically supported ranking | None |
| Descriptive differences | Reference sensitivities, short-run diagnostics and observed test times only |
| Default readiness | No new default promoted |
| Next evidence needed | Trusted GPU preflights, complete-fit prices, retained posterior checks and independent-reference results per funded slot |

The strongest alternative explanation for later poor posterior delivery is
geometry or weak identification, rather than an epsilon/L controller defect.
The weakest current evidence is long-run GPU behavior: it has deliberately not
been measured while C1 runs. A numerical/reference discrepancy would invalidate
the affected lane and trigger repair; a failed candidate or posterior check
alone would not reject state-space inference. C1/C2 calibration, learned-map
quality, difficult global exploration and exact MacroFinance inputs remain
separate gaps.
