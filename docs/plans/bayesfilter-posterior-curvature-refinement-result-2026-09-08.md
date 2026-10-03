# Fixed-center posterior curvature refinement: result

Date: 2026-09-08
Status: implemented and bounded synthetic verification passed
Authority: `/home/ubuntu/python/BayesFilter` → `/home/ubuntu/workspace/BayesFilter`
Base HEAD: `d2124d425b0ea0ae0e3e5f4246bd6b03ff8a2170`

## Outcome

The reviewed opt-in BayesFilter helper is implemented as
`refine_posterior_local_curvature`. It estimates a fixed-center regional score
quadratic in pilot coordinates, validates independent selection/audit/proposal
partitions, and returns a position covariance/factor only after raw SPD and
finite-factor checks. It does not recenter, clip eigenvalues, shrink the pilot,
build an HMC mass artifact, train NeuTra, run HMC, or alter existing defaults.

The shared TensorFlow dense fit preserves the historical unrestricted
least-squares-then-symmetrize calculation. The new path uses triangular solves
for `Sigma = F K^-1 F.T`, checks all pairwise replicate generalized eigenvalues,
accounts for failed/padded attempts, and emits strict JSON diagnostics.

## Executed verification

All commands ran through the approved exact prefix
`/bin/bash /tmp/bayesfilter-curvature-refinement-20260908/bundle.sh`, from the
BayesFilter root, with `CUDA_VISIBLE_DEVICES=-1`,
`TF_FORCE_GPU_ALLOW_GROWTH=true`, fixed CPU thread limits, and
`PYTHONPATH=/home/ubuntu/workspace/BayesFilter`. GPU was deliberately hidden;
this is not GPU readiness or production-scale evidence.

- `unit.NWWWZB`: 50 passed in 6.55 seconds; one unrelated h5py warning.
- `regression.llzfNL`: 40 passed in 20.35 seconds; existing h5py/TFP warnings.
- `integration.wedkgs`: 13/13 standalone cases passed in 1.76 seconds active
  (5 seconds wrapper wall time).
- `audit.fXanFl`: recursive AST/import closure, MacroFinance/local-HMC exclusion,
  syntax, whitespace, and scoped Ruff correctness checks passed.

The standalone integration is independent of MacroFinance and includes shifted
1D, correlated, too-small/too-large pilot, six-order correlated, 142D correlated,
mild quartic, banana, mixture saddle, flat, bounded finite-sentinel, and paired
host-XLA/eager Gaussian cases. Exact Gaussian covariance errors met the declared
`2e-7` tolerance. Target-only XLA value error was
`1.1102230246251565e-16`, score error `2.220446049250313e-16`, refined covariance
error `0.0`, and fixed-batch trace count `1`.

Expected rejections occurred as declared: banana and mixture saddle
`curvature_fit_rejected`, flat direction `curvature_fit_rejected`, and bounded
sentinel `ineligible_target_row`. These validate gates, not a general claim
about all non-Gaussian targets.

## Evidence and limits

The full JSON report is retained at the staging path
`/tmp/bayesfilter-curvature-refinement-20260908/integration.wedkgs/result.json`.
The test JUnit/log artifacts are in the same staging root. The final source
hashes are recorded below for the applied implementation and test/doc files:

```text
b8d005fb61aece006f44cb81a9b2d45d0e5cacd9190ba672b704317c86c420e6  bayesfilter/inference/score_curvature_tf.py
35454dce7a64ffd87396330bc34dd31fd441239fecda5bb81f6bee7008a5e5f3  bayesfilter/inference/posterior_curvature_refinement.py
6cd276b2fb4d40948c6acd21209b3310ad67d00daa261b4fe95f4e613f1ae567  tests/test_posterior_curvature_refinement.py
87185466d6538be4548ddb1593a3cad11903063be184e064b365ebf130d142c4  tests/test_posterior_curvature_refinement_regressions.py
63e76729d40c309f421dd731d2495a1432be94b00f80ef07b118ad0ec42c63eb  scripts/run_posterior_curvature_refinement_integration.py
49d50aa01241a2c995f876f4858ae933c032f88b045f5818f8cd654d19350d7a  docs/reference/posterior-local-geometry.md
3302ca3f32d2ed54fca131f3072f13a58349beceeb57d11d372afb1f384959bb  docs/plans/bayesfilter-posterior-curvature-refinement-plan-2026-09-08.md
```

This evidence is CPU-only synthetic verification. It does not establish
performance on MacroFinance, filter likelihood correctness, MAP accuracy,
NeuTra whitening, HMC stability, posterior convergence, GPU/XLA production
readiness, or any universal choice of thresholds. Before using the helper in a
claim-bearing pipeline, run a separate reviewed target-specific identification,
finite-support, factor-update, and downstream HMC qualification plan.
