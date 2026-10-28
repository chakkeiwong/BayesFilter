# Active checkpoint

Question: likelihood and analytical score accuracy for one configuration each of
LGSSM, KSC SV, predator–prey and range–bearing.
Checkout: .localresources/worktrees/ledh-iapf-execution-20260922; branch
ledh-refactor-with-policy-fix. Prior observation-parent work committed as
14e6fbd59. Preserve unrelated dirty files.

Stage: COMPLETE. All 96 evaluations across four launches passed finite,
full-horizon and every recorded validity check. N=1008, eight paired seeds;
T=50 for LGSSM/KSC/predator–prey, T=20 for range–bearing. GPU FP64/XLA,
TF32 off, verified memory growth, tftwogpu environment.

Findings: weighted parents have higher mean ESS in 8/8 pairs against uniform
for both supported models. Accuracy is not consistently better: predator–prey
weighted has higher likelihood RMSE and all six score RMSEs than original.
LGSSM differences are small; KSC and range–bearing conditional proposals are
unsupported, so their explicit original arms were checked. PP uses an adapted
Zhao–Cui reference with unresolved systematic error; RB a finite-particle Fisher
reference. No statistical ranking, scientific promotion or HMC claim.

Plan: observation-parent-four-model-check-20261028.md.
Result: observation-parent-four-model-results-20261028.md.
Artifacts: artifacts/observation-parent-four-model-check-20261028/.
Runners: scripts/run_observation_parent_four_model_check.py and
scripts/summarize_observation_parent_four_model_check.py.
Source/input snapshot: all 119 distinct payloads match original manifest hashes,
including pre-existing dirty dependencies and the earlier LGSSM runner version.

Budget: 364.186 of 3600 execution wall seconds used, four of eight launches,
no experimental retry. This requested diagnostic is closed; do not spend the
remaining allowance automatically. Delivery: scoped campaign commit and componentwise result note; use the latest
commit touching the result note for its commit identity. Further tuning or cross-model support
requires its own scientific plan; no implementation extension was requested.
