# SQMC completed checkpoint — 2026-09-28

Worktree: /home/chakwong/BayesFilter-SQMC; branch sqmc-development;
Expanded comparison commit 479a4616. Preserve all existing work and evidence.
Effective writes: repository and /tmp. GPU commands require escalation.
No package/environment change, publication or HMC was part of the research.
Owner authorized merge/push and branch synchronization on 2026-09-29.

Status: REQUESTED EXPANDED COMPARISON COMPLETE. Do not rerun completed research.
All 8 scopes × 4 routes completed, with 128 valid final cells and all 8,480
actual score coordinates/errors: P44 d3 T10/120,N1008; full A/full SPD Q d3/d10
T2/10/120,N1020. Preserve original 8 P44 units from run-01 and 24 full-model
units from repair-run-01. No campaign or monitoring process remains active.

Authoritative result: docs/benchmarks/sqmc-expanded-results-20260926.md.
Detailed tables, CSVs, paired SVG/PNG plots and terminal review:
docs/plans/artifacts/sqmc-expanded-20260928/renewal-01/final-evidence-02/.
report/audit.json and terminal-checks.json: 0 engineering failures/missing units;
all scores and likelihoods verified; 1,008 source files archived and checked.
Terminal disposition: accept completed descriptive comparison; withhold promotion.
Review is Codex plus executable checks, not independent external review.

Finding: IID has smaller observed score error in 27/96 SQMC comparisons;
none loses to zero-score or first-only baselines. All final configurations are
numerically viable, but no statistical ranking is supported (2 independent
final datasets/scope). Selected epsilon and flow steps differ across routes.
FP64 GPU/XLA, TF32 off is reference evidence, not production/default/HMC evidence.
Q scores are lower-Cholesky coordinates, not raw symmetric covariance entries.

Repair: original full-d3 mass failures preserved. Matched-input CPU/GPU replay
agreed; additional balancing failed. Existing epsilon ladder .4/1.6/6.4/25.6/102.4
calibrated on fresh full-model partitions; then all-coordinate flow2/8 tuning,
validation and frozen final evaluation. No shared numerical-kernel change during
repair and no relaxed validity guard. Static/dynamic diagnostic parity passed.
Original rejected-calibration tables and every retry/check/log remain preserved.

Budget ledger: renewal-01/budget.json. Charged 21,377.329993s = 5.938147 GPUh,
including conservative 1,918s earlier pilots. Remaining 21,822.670007s = 6.061853h.
Window began 2026-09-27 17:25:28.470267UTC; numerical completion 23:48:25.469708UTC;
6.3825 elapsed hours, before deadline 2026-09-28 07:25:28.470267UTC. No retry-limit
breach; no repair-run infrastructure retry. Remaining budget is not an instruction
to expand the completed campaign. Any new scientific scope needs a new plan.

Known reporting limitation: repair manifests inherited a generic pilot-reuse
sentence; actual fresh seeds and row labels are correct. Final reports clarify
that pilot reuse applies only to P44. Preserve original manifests.

Next action: read the completed result and terminal review for any user-directed
follow-up. No outstanding work remains under the requested expanded comparison.

Completion update, 2026-09-29: the subsequent KSC stage and full-mixture
correction are also complete. The final master-program campaign summary is
`docs/benchmarks/sqmc-master-program-final-summary-20260929.md`.
The final KSC checkpoint is
`docs/reset-memos/sqmc-ksc-sv-checkpoint-20260928.md`. This completed Kalman
checkpoint and its linked evidence remain preserved.

Completion update, 2026-09-30: the KSC reset-repair campaign and stalled-session
recovery are complete. The active KSC checkpoint above links the final 38-unit,
525-evaluation result, with only three SQMC T10 repair nominations.
