# Active R iAPF validation checkpoint

2026-09-20. /home/chakwong/BayesFilter, surrogate-hmc,
HEAD6fbcf3147660c40d5d5644bbcbcc9fadbcb06aef. Preserve unrelated changes.
Owner requested continued execution; bounded floor-diagnosis/repair complete.
Plans: docs/plans/iapf-r-log-fit-validation-2026-09-20.md and
docs/plans/iapf-r-d80-positive-floor-repair-2026-09-20.md. Terminal result:
floor-result.md, result.md and summary.json in this root. Prior closed result:
../iapf-r-log-fit-repair-20260920-01/result.md.

Question: does frozen log_quadratic R iAPF retain Kalman likelihood accuracy on
fresh linear-Gaussian datasets and higher dimensions? Independent R/CPU
reference; different fitting objective from paper Eq.15. No paper replication,
score, HMC, KDM, LEDH or TensorFlow production promotion.

Preflight: append-only CSV reporting fixes quadratic overhead. Real runner
before/after parity passes all numerical CSV fields,1200 fits,800 prefixes.
Reporting validator rejects missing fit, duplicate prefix and changed data.
Seven existing pytest cases pass. Core frozen. Skeptical audit passed.

Terminal campaign: attempts01/02 d20,32 repeats each, data seeds70000020 and
71000020, means .996404/1.009890 with bootstrap95 [.976202,1.016881] and
[.985510,1.033173]; all38,400 fits pass and no heuristic veto. Attempt03 d40
pilot 4 repeats mean .875362, descriptive only. Attempt05 fresh d40 16 repeats
mean1.004931 bootstrap95 [.961774,1.051352], no veto, all fits pass.
Attempt04 d80 pilot failed the frozen20-iteration controller cap at N=8000;
all2,000 QR fits admissible, last-six likelihood CV1.401898 above .5 threshold.
Attempt06 exact replay reproduced status/history/counts/fits and exact-twist
oracle matched Kalman with recorded zero log error. This rejects the fitted d80
scope for controller/weight instability; it does not reject Gaussian algebra.

Campaign charged1474.773800/1800 worker seconds;325.226200 remain. Eight planned
launches plus one localized infrastructure retry; no further launch allocated,
no process running. No d80 cap extension was made. Final pytest9/9 (3.09sec),
including88 R conformance/33 alternative checks, mutations, captured sources,
child timeout cleanup and phase log/exit-code behavior. Engineering checks only.
Completed attempt07-floor-diagnostic in90.618141sec: all3 baseline replays exact.
At time93 all4000/8000 draws come from the floor component in two bad passes,
mean probability>.99999. Removing only that floor changes log errors from
-37.69775/-36.04595 to-1.20514/-1.89285. Local floor calibration problem confirmed;
no claim about original-author choices. Saved positive-floor calibration supports
optional power8 candidate, with max probability8.316e-14 versus1 for power2.
Attempt08 had a logging/status harness fault; preserve it, no candidate
inference. Attempt09 retries under the same600-second repair allowance;
combined repair time373.687453sec. All four phases complete. Healthy equality7/7,
tail checks1700/1700 (minimum relative margin.307997), fits9000/9000,
conditional heuristic and d20 non-harm screens pass. Ordinary d20 MSE rises
25.6-36.3% descriptively, within factor2 screen; no equivalence claim.
Two d80 pilot and four untouched d80 repetitions each use8 adaptive passes,
N2000. Fresh ratios1.08194,.97580,1.10588,1.18958; mean1.088298,
descriptive bootstrap95%[1.008317,1.162667]. Four repetitions do not establish
accuracy. Default paper_eq15/power2 unchanged; power8 remains optional.

Exact next action: write/review a bounded frozen power8 validation plan with
32 repeats on each of two untouched d80 datasets plus a d40 regression control.
Use disjoint seeds; require each d80 mean-ratio95% interval wholly in[.9,1.1],
numerical/tail validity and conditional heuristic screens. No tuning on these
datasets. Original-author objective/solver/floor/data gaps remain open. TF
comparison and published-scale replication follow qualification; no score/HMC,
KDM or LEDH promotion. Final checks/provenance: checks-floor-final.json.
