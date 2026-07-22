# LGSSM N=5000 Seed-Batch Capacity Plan

Date: 2026-07-20
Status: `CLOSED_MEMORY_FEASIBLE_BATCH8_SEMANTIC_PARITY_FAIL`
Parent result:
`docs/plans/bayesfilter-lgssm-particle2000-particle5000-kalman-bias-ladder-result-2026-07-20.md`

## Engineering Intent And Evidence Contract

| Field | Declaration |
| --- | --- |
| Question | Can the canonical `T=50,N=5000,K=2500` Contract E--Chol value-and-total-score program execute eight estimator seeds concurrently within the 8192 MiB GPU limit, and how much faster is it than the preserved singleton execution? |
| Candidate | One compiled seed microbatch containing claim seeds `82220..82227`. |
| Exact baseline | The same eight seeds from `n5000_repair_scope_attempt01`, executed as eight size-one microbatches with matching current source hashes and selected controls `(sinkhorn_steps=20,balance_steps=5)`. |
| Pass criterion | GPU/TF32/XLA and exact `N=5000,K=2500,2 x 2` scope; canonical total score; replay; finite/chart/reset/marginal/work gates; peak TensorFlow allocator use below 8192 MiB; and per-seed value/score agreement with the source-matched singleton baseline to absolute tolerance `1e-4`. |
| Veto | OOM/resource failure, stale baseline source, scope/control/seed mismatch, failed hard gate, missing replay, allocator peak at or above the cap, or parity failure. |
| Explanatory diagnostics | Cold compile-plus-first execution, warm replay time, total measured trace/cold/replay time, peak allocator bytes, seeds/second, and paired speedup relative to the preserved singleton timings. |
| Artifact | `docs/benchmarks/artifacts/lgssm_n5000_seed_batch_capacity_20260720/attempt01/`. |
| Nonclaims | This test does not retune controls, change the closed Kalman-bias result, rank algorithms, prove `N=10000` capacity, or establish that size 16 fits. |

## Defaults And Assumptions

| Choice | Provenance/status | Failure mode | Early diagnostic |
| --- | --- | --- | --- |
| Eight concurrent seeds | User-requested capacity candidate | XLA temporaries exceed the fixed GPU limit | One fresh size-eight launch, fail closed on resource error |
| Seeds `82220..82227` and controls `(20,5)` | First half of the final untouched `N=5000` claim | Comparing different stochastic work or controls would invalidate speed/parity | Validate baseline metadata and exact ordered seeds before GPU setup |
| Preserved singleton baseline | Source-matched final campaign artifact | Code drift makes old timings or values incomparable | Verify all recorded source SHA-256 values before execution |
| `1e-4` absolute parity tolerance | Float32/TF32 engineering comparison | A larger difference could hide batch-dependent behavior | Report maximum value and score discrepancies separately |
| One replay | Same canonical claim execution structure | Cold-only timing overstates steady-state cost | Require bitwise replay within the size-eight compiled graph and report both timings |
| Timing context | Equal-work speed comparison ideally uses an uncontended GPU | Another GPU process can depress throughput while allocator checks still pass | Record compute-process contention at launch. A contended run may establish capacity/parity and report observed throughput, but not a clean speed ratio; clean timing remains a follow-up diagnostic. |

## Skeptical Plan Audit

Verdict: `PASS`.

- The baseline uses the same eight seeds, controls, target, horizon, particle
  count, chunks, dtype, TF32, XLA route, and current source hashes.
- Memory and speed come from the canonical value-and-total-score program, not a
  value-only or short-horizon proxy.
- The comparison separates cold compile/first execution from warm replay and
  compares equal seed counts.
- Hard scientific admission is not inferred from a resource benchmark. The
  prior Kalman screen remains failed.
- The run has one bounded GPU attempt with an 8192 MiB logical-device limit and
  a 45-minute wall-time cap. A resource failure ends this candidate test; it
  does not trigger altered chunks, disabled XLA, or weaker score semantics.
  If another process owns the GPU after bounded waiting, the attempt records
  `externally_contended`: its memory/parity evidence remains usable, while its
  speed is descriptive observed throughput rather than a clean performance
  ratio.

## Active Execution Policy

The old fixed size-one `N=5000` choice was conservative for memory. Batch eight
fits comfortably, but this test found batch-size-dependent value and total-score
drift relative to the same singleton seeds. Future claim-bearing runs must keep
size one as the correctness fallback until a larger microbatch passes same-seed
parity. Do not describe size one as a memory requirement, and do not promote
batch eight from capacity evidence alone.
