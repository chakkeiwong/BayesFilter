# PP-UKF Remote Tuning Reconciliation Audit

Date: 2026-07-21

## Scope

This note records the reconciliation of the local PP-UKF tuning work with
`origin/main` at commit `41f2aa4`. The pre-merge local work remains preserved
in stash `pre-origin-main-merge-20260721-local-tuning-worktree`; it must not be
dropped until the user explicitly confirms the merged tree is the desired
working state.

No merge conflicts remain. The ignored complete-highdim source snapshot was
not modified. No PP-UKF long or claim-bearing run was launched.

## Research Intent And Evidence Contract

Question: does the updated tuning machinery provide a valid, bounded,
scope-specific route for the PP-UKF NeuTra/HMC lane without silently changing
the target or mass identity?

Comparator: the existing PP-UKF target adapter and its plain dense-IAF/HMC
contract tests, with the remote HMC route and budget contracts as the
operational baseline.

Promotion criterion: the PP-UKF contract path must construct, validate, and
replay the selected tuning artifacts while preserving target status, route
identity, fixed-identity mass semantics, and failure provenance.

Hard vetoes: non-finite target or kernel values, stale or caller-stamped route
identity, a changed fixed-identity mass signature, missing bootstrap evidence,
invalid artifact schemas, cross-scope tuning artifacts, or a failed focused
contract test.

Explanatory diagnostics: acceptance, runtime, proposed-step tails, candidate
losses, and rejected metric-boundary details. These do not establish posterior
correctness, convergence, superiority, or production readiness.

Nonclaims: this reconciliation does not certify PP-UKF posterior correctness,
HMC convergence, statistical superiority, or default/production readiness.

## Remote Tuning Changes Relevant To PP-UKF

- The remote `hmc_kernel_tuning.py` adds explicit mass-policy propagation,
  fixed-identity invariants, operational route contracts, fixed-metric grid
  search, bounded budget accounting, process-parallel execution, and retained
  bootstrap hard-veto diagnostics.
- Route identity is selected from `algorithm_id`; runner, XLA, and timeout
  settings cannot silently select another algorithm.
- Bootstrap failures are preserved as artifacts and stop later handoff
  construction. A failed candidate remains evidence and a repair trigger; it
  is not silently converted into a successful handoff.
- The remote warmup now supports externally qualified epsilon payloads. The
  artifact validator accepts both the historical probed `passed` schema and
  the strict `externally_qualified` schema with provenance and no probes.
- Metric-boundary failures are represented as `candidate_metric_rejected` and
  retain the incumbent metric. This is an explicit no-update decision, not an
  exception that authorizes fallback to an unrelated route.
- Fixed-metric and operational grid machinery remains applicable to PP-UKF
  only after the PP-UKF scope, target data partition, particle/parameter
  dimensions, dtype/backend, and route controls are bound in a fresh tuning
  artifact. Settings from another horizon or model are warm starts only.
- The remote `neutra_artifacts.py` inverse/pullback and SSL-LSTM topology
  guards are retained.

## Compatibility Repairs

- Added `bayesfilter/inference/neutra_training_legacy.py` for the preserved
  plain dense-IAF PP-UKF/end-to-end API while retaining the newer named-family
  trainer.
- Unified `NeuTraTrainingError` so both trainer surfaces raise the same
  exported exception class.
- Extended `bayesfilter/inference/hmc_tuning_artifacts.py` narrowly for the
  remote externally qualified epsilon payload, new step-ceiling telemetry,
  and the explicit `candidate_metric_rejected` outcome. Existing provenance,
  finite-value, lineage, and nonclaim checks remain strict.

## Verification

Passed:

- `102 passed` in the PP-UKF/NeuTra trainer, campaign, and end-to-end contract
  suite.
- `356 passed, 1 skipped` in the focused HMC tuning, fixed-mass, grid,
  budget, route-contract, bootstrap, and regression suites.
- `46 passed` in the HMC tuning artifact construction, replay, corruption,
  provenance, and schema-compatibility suite.
- `git diff --check` and syntax/import checks for the reconciled modules.

The historical artifact fixture was updated to handle either a probed or an
externally qualified initial epsilon and either an accepted or rejected metric
boundary. Its mutation checks now remain deterministic under both valid remote
producer outcomes, and the complete artifact suite passes.

## Decision And Next Action

The reconciled tuning implementation is ready for a new PP-UKF *offline,
scope-specific tuning* test, not yet for a claim-bearing PP-UKF run. The next
run must create fresh disjoint calibration/validation partitions, a repository
issued tuning artifact, a bounded attempt budget, and an untouched claim
partition. It must use the fixed-identity route explicitly and retain every
failed candidate and bootstrap diagnostic in a new versioned output directory.
