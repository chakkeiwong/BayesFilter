# Nonlinear covariance pilot: results and next repair

All three LEDH covariance providers fail the conditional heuristic promotion screen at the tested settings. EKF and UKF have lower observed score error in the weak regime; the locally linear adapted PF has lower observed error in the curved and concentrated regimes. The latter concentrated-regime differences remain poorly resolved. This rejects promotion of these candidates, not further work on the score problem.

The run completed all 1,440 numerical rows without a failed attempt in 1,037.961 seconds: 432 calibration/validation rows and 1,008 untouched evaluation rows. All nine provider/regime selections were issued before evaluation opened. The source-frozen GPU process completed despite the editor interruption. The subsequent CPU comparison also completed successfully.

## Target and comparison

The claimed comparison target is the score of the scalar nonlinear state-space model specified in [the run plan](../../../younis-score-nonlinear-calibration-2026-09-15.md), in coordinates `(A, log_sigma_Q, log_sigma_R, H, m0, log_sigma_P0)`. The LEDH arms compute analytical total derivatives of their declared finite particle programs. Those quantities are compared to an independently refined FP64 numerical model-score reference; they are not asserted equal to that reference or unbiased for it. Gaussian filters compute their approximation's derivatives. Particle comparators retain their stated sampling and derivative semantics.

Each row uses N=128 and T=4. Each regime has 24 new observation datasets and two particle streams per method. The summed squared score error weights the six coordinates equally, with per-coordinate results retained in `nonlinear-calibration-01/comparison.json`. This criterion depends on the declared parameterization. All methods use the same data and reference within a regime. This is an equal-particle-count comparison, not a matched-cost study.

| Method | Weak regime MSE | Curved regime MSE | Concentrated regime MSE |
|---|---:|---:|---:|
| LEDH with UKF covariance | 0.197775 | 0.845628 | 0.464271 |
| LEDH with SGQF covariance | 0.190184 | 0.731296 | 0.453713 |
| LEDH with persistent-mixture covariance | 0.203901 | 0.850318 | 0.473756 |
| EKF | 0.104062 | 4.511172 | 11.803098 |
| UKF | 0.061222 | 2.575795 | 8.740324 |
| Bootstrap PF | 2.239920 | 3.704394 | 6.834585 |
| Locally linear adapted PF | 0.217133 | 0.459693 | 0.361371 |

The values above are observed means, not a certified ranking. The report bootstraps whole dataset pairs, keeping both particle streams together, with 20,000 resamples. Its descriptive 95% intervals for MSE differences illustrate the uncertainty:

| Comparison (candidate minus comparator) | Weak | Curved | Concentrated |
|---|---:|---:|---:|
| LEDH minus UKF | 0.1366 [0.0740, 0.2167] | See full report | See full report |
| LEDH minus locally adapted PF | See full report | 0.3859 [0.0932, 0.7417] | 0.1029 [-0.1323, 0.3018] |
| SGQF minus locally adapted PF | See full report | 0.2716 [0.0200, 0.5713] | 0.0923 [-0.1432, 0.2923] |
| Mixture covariance minus locally adapted PF | See full report | 0.3906 [0.1295, 0.7065] | 0.1124 [-0.1160, 0.3151] |

These intervals have neither certified small-sample coverage nor a simultaneous ranking interpretation. In particular, the concentrated-regime intervals include zero. The predeclared screen vetoes promotion on observed heuristic losses while preserving this statistical qualification.

## Engineering and numerical evidence

All 1,440 rows completed on the RTX 4080 SUPER with FP32, TF32, XLA enabled and verified memory growth before initialization. The recorded maximum trace count is one. The numerical reference used CPU FP64. Maximum reference domain error was 1.213e-14, mesh-refinement error 2.323e-15 and tail mass 6.662e-16, below the predeclared 1e-7, 1e-7 and 1e-9 tolerances. These checks support the numerical reference on these datasets; they are not a proof of every nonlinear model or horizon.

The comparison validated source fingerprints, result digests, frozen selections, model settings, data/reference identity, candidate identity, and exact dataset/replicate coverage before aggregation. The 16 focused uncertainty, join and execution-order tests passed before launch, and their main-checkout counterparts passed after integration. No new code change was needed to assemble the completed report.

The frozen selected controls were:

| Scope | Flow substeps | Sinkhorn / terminal balance | Provider control |
|---|---:|---:|---|
| Weak LEDH | 2 | 3 / 3 | UKF |
| Curved LEDH | 2 | 30 / 30 | UKF |
| Concentrated LEDH | 2 | 3 / 3 | UKF |
| SGQF, each regime | 2 | 3 / 3 | Level 2 |
| Mixture, weak and curved | 2 | 3 / 3 | Within-component fraction 0.75 |
| Mixture, concentrated | 2 | 3 / 3 | Within-component fraction 0.25 |

Other numerical controls remain explicit warm-start hypotheses: OT epsilon 2, reset ridge 1e-5, correction strength 0.2, LM damping 0.01, LM scale floor 1e-4, trust radius 0.5, coordinate cap 0.4 with power 8, and pairwise strength 0.02 with RMS cap 2. Their incomplete calibration prevents a claim of exhaustive tuning. The change from the earlier N=32 pilot combines more particles, calibration and different datasets; its effect cannot be attributed to any one of these changes.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep the implementation available; reject promotion of the tested provider candidates | Oracle score MSE measured; each candidate loses a heuristic in each regime | Numerical/source/coverage checks pass; heuristic promotion veto fires | Modest dataset count, incomplete numerical-control calibration and unequal compute | Perform the planned control-safety and scope-calibration repair using fresh partitions; finish outstanding estimator and twist prerequisites | No default change, covariance-to-score improvement theorem, method superiority or rejection of the research direction |

| Inference item | Status |
|---|---|
| Hard veto screen | No failed numerical/reference/source invariant; promotion veto from observed heuristic losses |
| Statistically supported ranking | None issued under the pilot's predeclared contract |
| Descriptive differences | SGQF's observed errors are below the other LEDH arms in this pilot; this is not selection or promotion evidence |
| Default-readiness | Not established; broad control calibration and matched-cost evidence are missing |
| Next evidence needed | Independent scope-specific calibration, more datasets/streams, matched-cost replication and a predeclared ranking protocol |

The strongest alternative explanation for the losses is the inherited numerical-control configuration rather than the covariance provider itself. A second is finite-particle variance at this short horizon. The weakest evidence is the modest number of datasets and streams, especially in the concentrated regime. A fresh, adequately powered comparison after complete control calibration could overturn the candidate verdict. Numerical or derivative parity failures would instead invalidate the corresponding implementation evidence and require repair before interpretation.

## Execution record and refresh

Source: `.localresources/worktrees/younis-score-calibration-20260915`, commit `6de94b54f980c1a5d8f01f530080f66a812a53ce`. Interpreter: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`. GPU UUID: `GPU-68251639-fe82-8f81-3ccc-2953c32e805b`. The exact 21 study specifications, commands, per-row seeds, data versions, memory policy and environment are in `nonlinear-calibration-01/campaign.json` and each study's `initial-manifest.json`, `state.json` and result files. Base data seed is 92167; calibration datasets 300–303, validation 310–311, evaluation 400–423; particle stream IDs 0 and 1. Bootstrap seed is 95231.

The run command was `scripts/run_younis_score_campaign.py --action run --campaign <this directory>/nonlinear-calibration-01/campaign.json`, with the environment and 5,400-second timeout recorded in the plan and command log. The completed CPU report used the same interpreter/source and `--action compare`, with GPU explicitly hidden. Full logs are `campaign.log` and `comparison.log`; machine-readable outcomes are `progress.json` and `comparison.json` under that run directory.

This used launch 11 of the initial 12-launch engineering envelope, leaving one repair launch; the 17.30 GPU minutes are within both the 90-minute slice and eight-hour parent time budgets. The current process is finished. No run should be described as still active.

Between-phase refresh: preserve these evaluation datasets as opened holdout evidence. Do not select new controls on them or silently continue an unchanged comparison. Complete the outstanding iAPF optimizer/objective specification and bounded identity tests, and the dedicated numerical-control calibration prerequisite already owed by the master. The iAPF density-objective issue is documented in `iapf-density-fit-audit.md`; it does not invalidate this covariance comparison. Wider model/horizon coverage, normalization/consistency calibration, estimator combinations, matched-cost replication and terminal review remain incomplete tasks inside the master.
