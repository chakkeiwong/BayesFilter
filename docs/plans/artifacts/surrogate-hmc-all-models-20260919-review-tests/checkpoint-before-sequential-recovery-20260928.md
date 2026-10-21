# Active surrogate-HMC checkpoint — 2026-09-27

Goal: test surrogate HMC: canonical analytical DCTR position-only proposal force
plus separately frozen full-support MH endpoint. No exact endpoint-gradient
claim. Master incomplete; LG1 closed. FP32/TF32/GPU/XLA force default;
immaterial replay differences warn only. Claude review deferred.
Checkout /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917;
branch ledh-graceful-failure-execution-20260917;
HEAD 8a4cffa2fd085f43ab5710c24d6e7b48770aea0d. Preserve unrelated dirty files.

Scope: M13 data1704/T20/state4/obs2/parameters2/N4096/design9.
Controls multiplier.001; FP64 endpoint4096x8. Data3701 untouched.
Reference/coverage02 qualified. Reference/assessment source files are hash-bound.

Public tuning03 failed candidate: first prior-scale chain accepted nothing and
used fallback500 warmup/1000 verification; other chains had zero fallback.
Localization01: initial force(276,426) is finite but sends the zero-momentum
proposal from(-.693,-.693) to(-1.621,-2.020); filter invalid and endpoint
potential rises from-17.06 to583.97. Covariance-scale starts from discarded RWM
warmup remain valid with lower endpoint potential. Initialization sensitivity
is unresolved until fresh verification and retained assessment pass.

RWM resume06 COMPLETE, healthy at2000/chain: Rhat1.006840/1.003476, practical
agreement true, formal simultaneous equivalence INCONCLUSIVE. Its metric
warmup alone supplies four repaired starts. Exact interrupted/uninterrupted
RWM parity regression passes. Final initialization-tests-03:70 passed.
Logs under docs/plans/artifacts/surrogate-hmc-m13-initialization-tests-20260927-03.

Global caps198000CPU/226800GPU; continuation144000CPU/126000GPU unchanged.
Reallocated12000/12000 from public to repair: public60000/50000,
repair22000/22000, baseline6000/6000, retained56000/48000.
Receipt discovery now includes newer M13 jobs without duplicate charges.

Active plan: docs/plans/surrogate-hmc-m13-initialization-repair-2026-09-27.md.
TERMINAL CHECK 2026-09-27 14:31 HKT: no campaign processes remain.
Initialization repair worker completed with M13_PUBLIC_TUNING_REPAIR_REQUIRED.
Fresh max Rhat1.137819/1.046165 >1.01. All other fresh checks pass;
zero warmup/verification fallback. Selected epsilon.07105464358,L2.
Retained sampling did not start. No posterior qualification.

Execution failure: parent report/accounting missing; worker run-manifest stale
RUNNING. Internal tuning21927.65seconds; artifact span22407.85seconds,
exceeding the18000-second attempt cap. Parent-loss cause not established.
All188 source hashes match. Previous prelaunch charges77250.058277CPU/
59597.896030GPU exclude this run; remaining budget is unresolved, not unchanged.
No second attempt authorized by the single-attempt repair plan.

Warm-up audit: this run discarded only500 transitions/chain, then collected
1000 verification draws/chain (4chains;4000draws). The typed verified tuner
has no pre-verification warm-up Rhat/readiness gate or extension loop; health
and verification Rhat were evaluated after both phases. It used the public
tuner but not the shared NeuTra sequential warm-up readiness protocol.
Sufficiency of500 is unestablished; failed verification does not distinguish
unfinished warm-up from slow stationary mixing. See result-note clarification.

Sampling-stage audit: shared sequential policy checks cumulative retained
draws in1000-draw increments up to10000/chain. M13's later retained pilot
already uses250-draw health chunks and cumulative assessment at1000/2000,
with a2000 cap and calibrated two-look uncertainty procedure. It never ran.
The earlier public verification is fixed at1000; its missing bounded extension
must be repaired alongside warm-up readiness. Rhat too HIGH triggers more
draws while health and budget permit; a failed first look is not a terminal
method rejection. Do not silently expand the calibrated retained looks/cap.

Exact next action (existing master): repair parent-loss/watchdog and accounting,
reconcile charges, then implement/test bounded warm-up readiness and cumulative
public-verification continuation for the typed surrogate-force route. Keep
retained assessment's calibrated looks/cap unless explicitly revised with
matching uncertainty calibration and available budget.
No new research launch before those execution/budget conditions are resolved.
Result:docs/plans/surrogate-hmc-m13-initialization-repair-result-2026-09-27.md.
Evidence:initialization-repair-20260927-01/status-reconciliation-20260927.json.

Approvals remain as documented in
docs/plans/surrogate-hmc-command-approvals-2026-09-27.md. Use exact approved
prefixes. Claude review deferred. User campaign authorization persists within
scope and budget; neither approval prompts nor a review gate caused this stop.
