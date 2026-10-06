# Active LEDH/Zhao–Cui checkpoint

Question: marginal versus ancestor LEDH likelihoods/scores for PP and SIR d18,
T=10/20/40/50, with original author TT and bootstrap references. Branch
sqmc-development; correction commit b3f4ca646; campaign/evidence commit
8f0ebe955. User authorized commits, scoped permissions, matched comparisons
and remaining reference-harness repairs. No main merge/push this stage.
Plan: docs/plans/ledh-zhao-horizon-comparison-20261006.md.
Root: docs/plans/artifacts/ledh-zhao-horizons-20261006-01/.

All 64 main GPU FP64/XLA LEDH, 32 covariance-only and bootstrap checks complete.
Three PP rank20 fits and all PP scores complete. SIR rank20 fit17 has all four
likelihoods and T10/20/40 scores. Missing T50 radius is running in auxiliary
session4604, score-repair-002, since08:57:47UTC. It watches later rank40 score
timeouts too. Do not relaunch it or write score-repair-attempts.json. Primary
session15473 runs rank40 fit2 (T10 complete), cap115200sec, then four score jobs.
Do not create another attempts.json writer. Ranks and independent fits remain
separate; two radii/retries of one proposal are not independent fits.

Budget172800 aggregate job-seconds, including failures and derivative checks.
09:07UTC: used50632sec, reserved148424 including active caps, remaining24376.
Auxiliary watcher additionally reserves9600sec for future primary score slots.
Future parent launches use shared accounting. Current primary retains its
launch-time code; the auxiliary reservation protects its remaining launches.

47 focused tests pass (61.21sec). Latest repair: generated checkpoint writer
accepts nonlinear pre_sol without legacy_mean_log_weight, recording NaN and
an availability flag. Actual Octave writer tested for both return formats;
no source solver or running fit changed. Commit this follow-up next.

Report008 has all completed values and scores, separate rank references and
conditional/between-fit uncertainties. SIR likelihood error remains about255
ancestor/233 mixture afterT20; even bootstrap N1008 is much closer. PP value
differences small; references disagree on some scores. No ranking, oracle,
default or HMC claim. GPU derivative check002 passes both SIR policies, maximum
normalized errors7.44e-6/6.04e-5; exact trace value/coordinate0 score parity.
Moment-safety branch changes under perturbation; no global smoothness claim.
Monograph built614pages; new pages402–404 visually inspected. Compact frozen
evidence is committed under committed-evidence-01; large MAT/source trees stay
local and ignored. Latest result note is still explicitly interim.

Next: commit checkpoint-writer repair and its tests, finish T50 rank20 score,
refresh numerical tables and result note; retain rank40 convergence check within
budget. Preserve all prior evidence and unrelated changes.
