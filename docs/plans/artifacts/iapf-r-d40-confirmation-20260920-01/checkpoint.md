# Active R iAPF reference checkpoint

2026-09-20; /home/chakwong/BayesFilter; surrogate-hmc;
HEAD 6fbcf3147660c40d5d5644bbcbcc9fadbcb06aef. Preserve unrelated edits.
Owner requested refresh and continuation; no new permission gate is needed.

Stage COMPLETE: docs/plans/iapf-r-d40-confirmation-2026-09-20.md.
Question: does the prior d40 upward likelihood discrepancy recur with fresh
randomness, and does the frozen optional log_quadratic/power8 reference pass
on fresh d40 data? T100; 32 repetitions per dataset; exact Kalman comparator.
Predeclared bootstrap95 wholly inside [.9,1.1], separately; no pooling/tuning.

Result: all 64 new repetitions pass accuracy, fit, tail and conditional
heuristic screens. Same data: mean .967538, CI [.932404,1.002977].
Fresh data: mean 1.022861, CI [.982292,1.061802]. The previous upward
discrepancy did not recur. All 38,600 fits and 6,400 Gaussian-limit tail checks
pass. No ranking, exact unbiasedness, Eq.15 replication or default promotion.
Evidence: docs/plans/artifacts/iapf-r-d40-confirmation-20260920-01/
result.md, summary.json, likelihood-audit.md, attempt*/manifest.json.

Core unchanged: c7152fea96d917bf90218bbbbc2bb31c01b0831b439625ff922454b01d748fd0.
The audit derives E[Z_hat_final | learned guides, N, stopping history]=Z for
the idealized algorithm and traces the executed proposal/weight/controller.
Eleven focused tests pass, including returning-the-stopping-value mutation.
Strengthened test source preserved separately in audit-sources/audit-manifest.

Both planned launches complete; no retry, no running worker. Deliberate CPU
R4.1.2 reference, GPU hidden, one BLAS/OpenMP thread, at most two R workers.
Filtering 997.694811s + post-run 22.571156s = 1020.265967/1500 summed seconds;
479.734033s unused in the closed stage. Test logs report 7.85s total within
the separate 120s mechanics allowance. All raw results and source snapshots
are preserved; no numerical settings or defaults changed during confirmation.

Exact next action: bounded official supplementary/author-source investigation
to reconcile Eq.15, actual optimizer/constraints, floor and early doubling.
Record sources or unknowns and specify the smallest discriminating fitting
check. Written Eq.15 has an amplitude escape; public R uses a different
scaled objective, with author provenance unverified. Current log_quadratic
fit is also different. Do not relabel it as Eq.15 or scale to a paper-
replication claim by repetition count alone. Later TF parity must identify
the qualified R comparator; LEDH/KDM/score/HMC gates remain open.
