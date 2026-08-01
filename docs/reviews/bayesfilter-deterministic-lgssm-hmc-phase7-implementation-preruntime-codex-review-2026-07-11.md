# Deterministic LGSSM HMC Phase 7 Implementation Pre-Runtime Review

Date: 2026-07-11

Review type: fresh Codex substitute implementation review. Claude remained
unavailable because the bounded one-path call was rejected by the managed
external-disclosure policy before execution.

## Scope

- `bayesfilter/inference/hmc_convergence.py`
- `bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py`
- `docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py`
- `docs/benchmarks/configs/multidim_lgssm_phase7_burnin_sampling_2026_07_11.json`
- focused tests and private/public ignore rules

## Findings Repaired

1. Private artifact preflight initially trusted embedded artifact hashes. It
   now recomputes fixture, XLA, geometry, mass, kernel, private-replay, and
   private-loop hashes and verifies the public replay reference against the
   private file SHA-256 and byte count.
2. The machine wall-time cap was initially checked only between chunks. Worker
   futures now receive the remaining wall time; timeout forcibly terminates
   worker processes and writes a structured blocker.
3. A nonfinite diagnostic payload could initially be treated like a failed
   promotion check and extended. Diagnostic hard vetoes now stop immediately.
4. Persistent workers initially returned state through the parent for the next
   request. Workers now retain current state in their process-local cache;
   parent state copies are artifact/inspection data only. Worker PID stability
   and XLA trace-count stability are checked each round.
5. The private retained archive was initially written and hashed without a
   reopen check. It is now written through an atomic replacement, reopened,
   shape/finite/provenance checked, and only then referenced publicly.
6. The discarded compile probe initially reused the first burn-in seed. It now
   uses the predeclared burn-in `check_index=9999`, separate from executed
   chain chunks.

## Verified Boundaries

- Existing Phase 6 artifact hashes recompute exactly from their payloads.
- Public HMC mechanics remain redacted.
- Private replay uses BayesFilter's checked retained-kernel replay API.
- Exactly two mass transforms are applied before raw-parameter diagnostics;
  unknown transform depth is a hard veto.
- Rank-normalized and folded split R-hat, split-chain bulk ESS, and split-chain
  q05/q95 tail ESS are fixed before runtime.
- CPU hiding, two-worker/two-chain partition, thread allocation, seeds, XLA,
  caps, and no-resume policy are pinned in config.
- Failed tuning cannot emit private replay, and failed runtime writes a
  structured blocker without private exception content.

## Checks

- Public API collection: `36 tests collected`.
- Public API execution: `36 passed`.
- Broad Phase 7/replay/chunk/checkpoint focused suite: `164 passed`.
- Independent SciPy average-rank/Blom reference check: passed.
- Python compile: passed.
- `git diff --check`: passed on in-scope files.
- Forbidden active-runtime scan for `GradientTape`, `batch_jacobian`,
  `tape.`, and `jit_compile=false`: no matches.

## Skeptical Verdict

The implementation is ready for the deterministic Phase 6 private-replay
refresh. This is not yet Phase 7 evidence: exact refreshed kernel hashes and
the actual-target multicore XLA smoke must still pass before serious sampling.

VERDICT: AGREE
