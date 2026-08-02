# Austria Growth Versus Particle Count: N=4032 Comparison Plan

Date: 2026-08-01
Status: `EXECUTED_DIAGNOSTIC_COMPLETE_N_INCREASE_DID_NOT_REDUCE_GROWTH`

Terminal artifact:
`docs/benchmarks/artifacts/genut_score_variance_repair_validation_20260801_n4032/attempt03/`

Terminal plot:
`docs/benchmarks/artifacts/genut_score_variance_repair_validation_20260801_n4032/derived_physical_vs_full_20260801_v2/`

Terminal result:
`docs/plans/bayesfilter-genut-score-growth-n4032-comparison-result-2026-08-01.md`

## Research Intent

| Item | Predeclared answer |
|---|---|
| Question | Does the post-step-7 full particle-filter tangent growth contract as particle count increases? |
| Comparator | Existing Austria `N=1008` attempt-08 artifact versus a fresh same-route Austria run at `N=4032` |
| Primary diagnostic | Per-step and cumulative finite-time directional log growth for diagonal and pairwise correction arms |
| Success criterion | Fresh route remains hard-valid and its post-step-7 growth is lower/non-positive relative to `N=1008` |
| Vetoes | Invalid particle design, non-finite route, reset/marginal failure, score-increment mismatch, or missing artifact |
| Statistical status | Three seeds and eight probes are descriptive; no ranking or causal claim |
| Nonclaims | No proof that particle count alone causes contraction; no bias, HMC, default, or superiority claim |

## Scope And Controls

- Austria SIR only: `d=18`, `T=20`, diagonal and pairwise arms.
- Particle count: `N=4032`, the nearest legal exact replicated-cubature count to
  the requested `N=4000`; `N=4000` is rejected because `4000 % (2*18) != 0`.
- Seeds: particle seeds `98201..98203`; eight fixed tangent probes per seed.
- Controls are copied from the completed `N=1008` diagnostic and are not
  retuned. This is a mechanism comparison, not claim-bearing tuning evidence.
- Backend: TensorFlow GPU/XLA, float32, TF32 enabled, memory growth required.
- Output root:
  `docs/benchmarks/artifacts/genut_score_variance_repair_validation_20260801_n4032/`.

## Skeptical Audit

- `N=4032` changes the finite-particle program and its computational scale;
  it is not an exact `N=4000` result.
- Reusing controls across `N` is a deliberate diagnostic choice. It cannot
  establish the best achievable behavior at either particle count.
- Positive or negative growth remains finite-horizon directional evidence. It
  is not an asymptotic Lyapunov estimate.
- The physical-transition curve and full-filter curve use different tangent
  spaces. Their timing and sign can be compared, not their absolute norms.
- Three seeds do not support a statistical ranking.

## Evidence Contract

- The run must preserve `result.json`, `result.md`, and a manifest in a fresh
  output directory; prior attempt-08 evidence is immutable.
- The result must record the exact legal particle count, design divisibility,
  controls, seeds, device, memory policy, TF32/XLA, wall time, and attempt
  status.
- A derived plot must use the same physical-transition calculation and the
  immutable `N=1008` result as the baseline.

## Pre-Mortem And Stop Conditions

- OOM or XLA compilation failure can occur because Sinkhorn uses dense
  `N x N` arrays. Stop and preserve the failed attempt; do not silently reduce
  `N` or change the route.
- If only one arm fails, retain the other as a failed candidate diagnostic and
  do not infer particle-count improvement.
- If all hard gates pass, interpret the comparison descriptively and continue
  only to the derived plot/reporting step.

## Planned Commands

```text
CUDA_VISIBLE_DEVICES=-1 python - <<'PY'
from bayesfilter.highdim.cubature_genut_candidate import cubature_design
try:
    cubature_design(dim=18, num_particles=4000)
except ValueError:
    pass
else:
    raise SystemExit('N=4000 unexpectedly accepted')
assert cubature_design(dim=18, num_particles=4032).shape == (4032, 18)
PY

CUDA_VISIBLE_DEVICES=-1 python -m py_compile docs/benchmarks/run_genut_score_variance_repair_validation.py

TF_FORCE_GPU_ALLOW_GROWTH=true python docs/benchmarks/run_genut_score_variance_repair_validation.py \
  --particles 4032 \
  --arms austria_diagonal austria_pairwise \
  --growth-only --probe-batch-size 1 \
  --output-root docs/benchmarks/artifacts/genut_score_variance_repair_validation_20260801_n4032/attempt01

CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter python docs/benchmarks/plot_austria_physical_vs_particle_growth.py \
  --input docs/benchmarks/artifacts/genut_score_variance_repair_validation_20260801_n4032/attempt01/result.json \
  --output docs/benchmarks/artifacts/genut_score_variance_repair_validation_20260801_n4032/derived_physical_vs_full_20260801
```

## Execution Closeout

The requested `N=4000` preflight failed closed because the exact replicated
cubature design requires `N` divisible by `2d=36`. The nearest legal count,
`N=4032`, was used. Attempt 01 exposed a noise-shape harness mismatch; attempt
02 reached the route but exhausted GPU memory in the eight-probe dense
Sinkhorn/JVP workspace. Attempt 03 repaired memory by running the same eight
probes sequentially with batch size one and passed all hard validity checks.
