# N=4096 TF32 Score Displacement Result

Date: 2026-07-30
Plan: `docs/plans/bayesfilter-zhao-cui-moment-teacher-score-mcse-transfer-n4096-plan-2026-07-30.md`
Classification: canonical-LGSSM transfer evidence, not moment-teacher score evidence

## Outcome

At `T=2`, `N=4096`, TF32 produces a resolved systematic score displacement
relative to the identical FP32/XLA program with TF32 disabled in four of five
coordinates. The absolute displacement is nearly unchanged from the earlier
`N=1024` scope, while the reference MCSE is smaller. Consequently, the worst
drift increases from 0.483 MCSE at `N=1024` to 0.759 MCSE at `N=4096`.

Both precision arms passed finiteness, chart, reset, marginal, replay, graph,
work-count, and exact-chunk checks. They used identical source, commit, FP32
storage, controls, seeds, and prepared tensor hashes. The exact transport
policy selected `K=2048`, giving a `2 x 2` block grid. Eight paired seeds were
run as two fixed-shape batches of four per arm.

## Score Results

TF32 displacement means `TF32 - FP32-no-TF32`.

| Coordinate | Mean displacement | Paired MCSE | Signs | Sign-test p | Reference MCSE | Displacement / MCSE | Systematic? |
|---|---:|---:|---:|---:|---:|---:|---|
| `phi1` | +0.003025 | 0.0000995 | 8+/0- | 0.0078125 | 0.017983 | 0.168 | yes |
| `phi2` | +0.001211 | 0.0000308 | 8+/0- | 0.0078125 | 0.007588 | 0.160 | yes |
| `phi3` | +0.0000439 | 0.0000306 | 7+/1- | 0.0703125 | 0.012319 | 0.00356 | no |
| `q_scale` | -0.022910 | 0.0002977 | 0+/8- | 0.0078125 | 0.030204 | 0.759 | yes |
| `r_scale` | -0.010346 | 0.0002688 | 0+/8- | 0.0078125 | 0.038723 | 0.267 | yes |

The strict negligible-error screen of 0.1 MCSE fails in four coordinates. The
proposed practical screen of 0.5 MCSE also fails because `q_scale` is at 0.759
MCSE. The `q_scale` mean displacement is about 77 paired-difference MCSEs, so
its direction is not explained by uncertainty in the paired precision
comparison.

## N=1024 Comparison

| Coordinate | N=1024 displacement | N=4096 displacement | N=1024 ratio | N=4096 ratio |
|---|---:|---:|---:|---:|
| `phi1` | +0.003114 | +0.003025 | 0.159 | 0.168 |
| `phi2` | +0.001128 | +0.001211 | 0.0995 | 0.160 |
| `phi3` | +0.0000964 | +0.0000439 | 0.00753 | 0.00356 |
| `q_scale` | -0.022727 | -0.022910 | 0.483 | 0.759 |
| `r_scale` | -0.010241 | -0.010346 | 0.245 | 0.267 |

The stable absolute displacement across particle counts is consistent with a
deterministic precision effect rather than Monte Carlo noise. Increasing the
particle count reduces stochastic MCSE but does not remove this displacement.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Reject the 0.5-MCSE practical screen for this transfer scope | maximum ratio 0.759 > 0.5 | both execution arms valid | long-horizon and moment-teacher behavior | localize or protect TF32-sensitive score operations | no exact mathematical bias claim |
| Classify four score coordinates as systematically displaced relative to FP32-no-TF32 | one-sided exact sign p=0.0078125 and drift > 2 paired MCSE | no comparison-validity veto | only eight paired seeds | retain raw paired evidence; replicate only if a narrower effect estimate is needed | no universality across models |
| Keep the moment-teacher final score not checked | complete finite program does not exist | integration remains absent | its precision propagation may differ | repeat this test after canonical integration | no moment-teacher admission |

## Inference Status

| Item | Status |
|---|---|
| Hard veto screen | both arms pass |
| Systematic displacement | supported for `phi1`, `phi2`, `q_scale`, and `r_scale`; not supported for `phi3` |
| 0.1-MCSE negligible screen | fail |
| 0.5-MCSE practical screen | fail due to `q_scale` |
| Default readiness | no |
| Next evidence | selective precision repair, then exact-scope paired rerun |

## Post-Run Red Team

The comparator is FP32-no-TF32, not an FP64 mathematical oracle. Therefore the
checked claim is systematic TF32 displacement relative to that comparator, not
exact score bias. Batching could in principle affect execution order, but both
arms used the identical two-batch structure and each displacement is computed
within paired seeds. The weakest generalization is the short `T=2` horizon;
longer horizons could amplify, cancel, or otherwise change the displacement.

Artifact:
`docs/benchmarks/artifacts/zhao_cui_moment_teacher_score_mcse_transfer_20260730/n4096_attempt01/aggregate/result.json`.

