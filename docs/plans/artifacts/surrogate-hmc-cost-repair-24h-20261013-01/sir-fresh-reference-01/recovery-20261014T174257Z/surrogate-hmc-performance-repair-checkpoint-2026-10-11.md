# Active surrogate-HMC computation repair checkpoint
Active 2026-10-14: compute an independent SIR reference at dataset 26101401,
T=50, position 2, theta=(-0.0175,-0.0175,-0.0175). Plan:
docs/plans/sir-fresh-point-reference-2026-10-14.md. Executor:
scripts/run_sir_fresh_reference.py. Output under the campaign root:
sir-fresh-reference-01/. Exact stored observation tensor hash matches. Five CPU
input/score/reference tests passed. GPU/XLA bootstrap Fisher reference running;
particle ladder 131072/524288/1048576 with 8/16/32 independent replications.
Four-hour/six-launch cap within the existing remaining 15.2207 GPU hours and
16 attempts; budget.json and reference state.json track actual consumption.
Exact next action: inspect ladder uncertainty and compare both saved OT arms.
This diagnoses model-likelihood/score accuracy; it does not admit HMC or change
FP32/N4032 defaults. Existing posterior target/prior decision stays pending.

Previous point clarification (superseded next action):
Owner clarification, 2026-10-14: the 1482.484% SIR maximum is fresh dataset
26101401, position 2, log-nu score -1.325783 -> -20.980306. It is not the saved
nominal point shown in the score/reference table (maximum 7.613219%).
Small denominator amplifies the maximum, but the same row has a kappa sign
change and 113.77435% vector change. No oracle or posterior ranking follows.
Details: ot-epsilon1-steps5-01/score-reference-comparison.md under the root below.
No GPU work; budget unchanged. Exact next action: explain the scope distinction.
Existing master target/prior decision remains unresolved.

Updated 2026-10-14T13:09:05.441796+00:00.
Checkout/branch: BayesFilter-ledh-graceful-failure-execution-20260917 /
ledh-graceful-failure-execution-20260917; HEAD 10884528ca8b1a7463895256d30b9655d69c0955.
Preserve unrelated dirty work. Owner authorized 24 additional GPU hours.

Current owner question COMPLETE: epsilon=1/five Sinkhorn iterations compared with
each model's fresh selected settings, identical inputs and frozen other controls.
Plan: docs/plans/surrogate-hmc-ot-five-iterations-2026-10-14.md.
Root: docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/.
Results: ot-epsilon1-steps5-01/report.md, report.json and verification.json.
All 90 paired rows (3 datasets x 6 positions x 5 models) checked: finite values
and scores, complete horizons, all existing validity checks, source snapshots
and current source hashes match. Eight focused CPU tests passed before launch.
N4032/K2016, GPU/XLA, TF32 off. FP32 except SIR FP64 recursion/FP32 pairs.
Actual TensorFlow device RTX 4080 SUPER with verified memory growth.

Maximum scaled score changes: LGSSM 0.0264%, KSC 7.29%, predator-prey 5.58%,
SIR 1482.48%, range bearing 18.62%. All log-likelihood percentage changes are
below 0.1%, but SIR centered log-likelihood changes reach 1.40324 log units.
LGSSM alone passes both 1% screens on inspected inputs. Other models show
material score differences, not proof of worse model accuracy. Both epsilon
and iterations change except predator-prey. No selected setting/default changed.
Measured runtime ratios are descriptive; no HMC or posterior claim.

Diagnostic charged 1489.346674 worker-wall GPU seconds
(24.8224 minutes); campaign used
31605.508454/86400 seconds, with 54794.491546
remaining (15.2207 hours), 104/120 attempts.
No RUNNING attempts; all five workers and their parent exited. Next-phase plan
refreshed without marking the overall master complete.

Exact next action: report completed diagnostic to owner. Existing master work
still requires the pending posterior target/prior decision before public-authority
HMC tuning and cumulative posterior sampling. That is existing master scope,
not a new direction. Do not repeat the completed computation or OT diagnostic.
Current force remains optional covariance-proposal, not canonical admission.

Previous completed-stage checkpoint follows:
Updated 2026-10-14T10:37:12.110652+00:00.
Checkout/branch: BayesFilter-ledh-graceful-failure-execution-20260917 /
ledh-graceful-failure-execution-20260917; HEAD 10884528ca8b1a7463895256d30b9655d69c0955.
Preserve unrelated dirty work. User authorized 24 additional GPU hours.
Root: docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/.
Budget: 30116.161780219758 of 86400 worker-wall seconds used; 56283.83821978024
remain (15.6344 h), 99/120 attempts. No RUNNING attempts or live GPU workers.

Current status: COMPUTATION_EVIDENCE_COMPLETE_POSTERIOR_NOT_RUN.
Result: fixed-design-computation-result-01/result.json and result.md.
The master finalize command executed successfully; 158 CPU tests +3 subtests pass
(computation-terminal-regression.log). Eleven new tests cover result failure paths.
Fresh source-bound tuning, two-transition mechanics, actual-consumer parity/cost,
and selected-setting precision passed all five models. 32 dependency hashes match.
Four local positions pass 1% scaled likelihood/score/force against finite-program
FP64; binding reconstruction exactly matches. No model-oracle or posterior claim.
N4032/K2016/TF32off/XLA; uniform FP32 except SIR mixed FP64 recursion/FP32 pairs.
Fixed-design ancestor indexing repaired. Rejected transport rewrites and cache/
tile candidates remain diagnostic-only. No pfor/autodiff claim score.

Actual four-chain force costs LG25.94s/KSC9.37s/pred16.26s/SIR26.72s/range6.38s;
long-chain practicality unresolved. Few-repeat runtime ratios are descriptive.
Full posterior is existing master work and is NOT complete. Its actual prior is
unresolved: saved independent standard normals in affine q are mechanics fixtures.
Owner question sent: retain them as benchmark posterior targets vs scientific
model-specific priors. Do not assume an unanswered required target decision.
Exact next action: resolve that response from the active conversation; audit and
bind the selected target to the public tune_hmc_kernel authority through the
repository typed position-force binding, then budget actual adaptive tuning and
cumulative posterior runs. Do not rerun completed computation phases or relaunch
old sessions. The 24h ledger remains available; no new compute approval required.
Current force is optional covariance-proposal, not canonical Contract-E admission.
