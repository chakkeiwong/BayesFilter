# LGSSM NeuTra Sequential HMC Repair Phase R2 Subplan

Date: 2026-07-15  
Status: `READY_AFTER_SUITABILITY_REVIEW`

## Objective And Entry Conditions

Run fresh confirmatory HMC for both R1-admitted fixed kernels and decide the
exact-fixture NeuTra claim using the parent plan's full convergence, posterior
agreement, and truth-recovery gates. R1 admitted both candidates with immutable
kernel hashes and left both original confirmation seed families unused.

## Evidence Contract

| Item | Frozen value |
| --- | --- |
| Main question | Does either admitted frozen NeuTra candidate produce a confirmatory HMC sample that passes convergence, tuned plain-HMC agreement, and truth recovery on this exact 18D LGSSM fixture? |
| Kernel | Per-candidate R1 kernel: step size `0.8`, 10 leapfrog steps; no retuning |
| Baseline | Immutable tuned plain-HMC summary bound by its existing SHA-256 and identical 18-parameter order |
| Warm-up | Fresh sequential chunks; minimum 2,000; latest 1,000 raw-coordinate modern R-hat `<=1.05`; maximum 10,000 per chain; archive all and exclude from posterior |
| Retained sampling | Fresh cumulative chunks; minimum 4,000; maximum 10,000 per chain |
| Full convergence pass | all-parameter modern R-hat `<=1.01`, bulk ESS `>=1000`, and tail ESS `>=400` |
| Posterior pass | every mean within 4 combined MCSE of tuned plain HMC and every truth-recovery distance `<=3` candidate posterior SD |
| Hard vetoes | identity/artifact drift; nonfinite state, target, log acceptance, diagnostic, MCSE, or posterior quantity; invalid target status; energy-error divergence; no movement; warm-up/retained cap failure; comparator mismatch |
| Explanatory only | acceptance, runtime, number of chunks, individual R-hat/ESS values, and between-candidate differences |

The retained controller continues when the full convergence screen is not yet
sufficient, not only when R-hat misses. R-hat and ESS are all sample-size-
repairable convergence gates, so rejecting a candidate at the 4,000 minimum
without using the remaining declared cap would repeat the fixed-budget error.

## Fresh Seeds And Commands

The original Phase 3 confirmation seed is the warm-up root. The retained root
is deterministically separated by adding 10,000 to the second component; this
predeclared derivation prevents warm-up/retained stream overlap.

| Candidate | Warm-up root | Retained root |
| --- | --- | --- |
| `dense_seed1201` | `(20260715,3101)` | `(20260715,13101)` |
| `dense_seed1202` | `(20260715,3301)` | `(20260715,13301)` |

Run in order with `CUDA_VISIBLE_DEVICES=-1`, TensorFlow intra/inter-op threads
`8/1`, and OpenMP/BLAS threads `8/1`:

```bash
python docs/benchmarks/run_lgssm_neutra_gap_closure_2026_07_15.py \
  confirm-candidate --job-id dense_seed1201
python docs/benchmarks/run_lgssm_neutra_gap_closure_2026_07_15.py \
  confirm-candidate --job-id dense_seed1202
python docs/benchmarks/run_lgssm_neutra_gap_closure_2026_07_15.py \
  confirm-finalize
```

## Required Artifacts And Checks

Write new per-candidate and aggregate results under
`sequential-repair-attempt-01/confirmation-attempt-01/`. Preserve separate
per-chunk and cumulative TensorFlow archives for warm-up, latent retained, and
raw retained draws. Bind target, candidate payload, adapter, R1 kernel hash,
seeds, command, Git commit, environment, CPU-hidden status, XLA, wall time,
plan/result paths, comparator hash, complete parameter diagnostics, posterior
rows, archive hashes, and nonclaims.

Before launch: focused compile/tests, TensorFlow-only closure, and R1 artifact
validation. After each candidate: verify output hash, archive hashes/shapes,
full convergence fields, posterior fields, and budget. After aggregate:
terminal result note, reset memo, local checks, and bounded read-only terminal
result review when Claude is available.

## Forbidden Claims And Actions

Do not use R1 retained draws, include warm-up in posterior summaries, retune,
retrain, weaken thresholds, overwrite artifacts, select only the descriptively
better candidate, or claim superiority, calibration, robustness, generality,
production readiness, or a repository default. A candidate pass supports only
the exact frozen candidate and favorable LGSSM fixture under recorded gates.

## Handoff And Stop Conditions

Process both candidates independently. A candidate passes only if all health,
full convergence, comparator agreement, and recovery gates pass. At least one
pass supports the limited exact-fixture positive claim. No passes close the
current fixed candidates as negative without rejecting the research direction.

Stop a candidate at a true hard veto or 10,000 cap. Stop the campaign only for
common target/harness invalidity, corrupted evidence, six-hour aggregate budget
exhaustion, or a boundary-changing repair. A localized serialization/XLA/schema
defect may be repaired once under the unchanged contract in a fresh attempt.

## Suitability Review

The subplan uses the correct tuned comparator, does not promote warm-up or
acceptance proxies, extends every sample-size-repairable convergence criterion,
preserves fresh seeds and artifact separation, and has explicit caps and
nonclaims. Environment and commands match the passing R1 route. Verdict:
`PASS`; continue after focused implementation checks.
