# NeuTra reverse-funnel architecture and tuning result (2026-08-15)

## Outcome

Raw architecture capacity was not the blocker. All four tested families were
proven by explicit construction to contain the exact funnel map. After separate
learning-rate/schedule tuning, however, all eight cold-start confirmations
failed the untouched exact proposal-law gate. Increasing width from 100 to 200
and preserving the root coordinate between stages did not repair the failure.

The staged repair succeeded. Training only the exact-compatible root-to-child
scale coefficients first produced a proposal-law pass. Starting joint training
from that state, independently tuning its learning rate, and running two fresh
5,000-update continuations from the same selected warm state preserved the pass
under two fresh joint-training and audit seeds.

## Cold-start campaign

Every architecture selected peak learning rate `2e-3` with piecewise multipliers
`1`, `0.1`, and `0.01` at 60% and 85% of its update budget. The observed
calibration losses are descriptive only; they do not support ranking.

| Architecture | Exact-map inclusion | Confirmation seeds passed | Mean `E[y^2]` | Mean lower/upper tail | Mean importance ESS fraction |
|---|---|---:|---:|---:|---:|
| One stage, width 100 | proved | 0/2 | 0.8928 | 0.01625 / 0.01795 | 0.9738 |
| Three stages, full reversal | proved | 0/2 | 0.9516 | 0.01854 / 0.01727 | 0.9756 |
| Three stages, root-preserving | proved | 0/2 | 0.9396 | 0.01916 / 0.01991 | 0.9792 |
| Three stages, root-preserving width 200 | proved | 0/2 | 0.9383 | 0.01902 / 0.01987 | 0.9781 |

The exact root second moment is one and each exact tail probability is
`0.0227501`. Conditional residual second moments were near one in all arms, so
the recurring defect was root compression rather than failure to learn the
standardized child residual scale. The explicit scale coefficients remained
far below the analytic value one in cold-start fits.

## Staged repair

The restricted root-scale fit selected update 2,250. Its 99 exact-path
coefficients averaged `0.99912`, with mean absolute error `0.00092`, and its
untouched proposal screen passed.

The joint warm-start calibration tested peak rates `2e-4`, `5e-4`, and `1e-3`
with the same piecewise schedule. All three passed their calibration screens;
`1e-3` was retained as the representative arm by the predeclared observed-loss
selector. That observed difference is descriptive, not a statistically
supported ranking.

| Joint confirmation | Selected update | `E[y^2]` | Lower/upper tail | Importance ESS fraction | Log-ratio SD | Gate |
|---|---:|---:|---:|---:|---:|---|
| Seed 5 | 5,000 | 0.99599 | 0.02279 / 0.02260 | 0.99808 | 0.04375 | pass |
| Seed 6 | 5,000 | 0.99469 | 0.02245 / 0.02244 | 0.99814 | 0.04312 | pass |

Neither confirmation clipped a gradient. Both selected the terminal checkpoint,
so the campaign establishes a viable staged procedure, not a universal
convergence theorem or a minimal update count.

## Decision table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Do not enlarge the generic IAF further for this funnel | width-200 failed the same exact-law gate | no engineering veto | only two confirmation seeds per arm | retain width 100 as the representative diagnostic architecture | width can never help other targets |
| Reject cold-start joint reverse-KL for this target | 0/8 exact-law passes | all runs finite and XLA-valid | tested tuning grid is bounded | use structured warm-up when target structure supplies it | reverse KL is generally invalid |
| Accept staged warm-up as a viable funnel repair | 2/2 fresh joint continuations from one selected warm state passed | no nonfinite values or clipping | exact funnel supplies privileged structure; end-to-end warm-up seed robustness not tested | replicate the full warm-up plus joint pipeline, then investigate a general target-derived warm-up before SSL-LSTM transfer | SSL-LSTM or HMC readiness |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | No implementation, finite-value, GPU-memory, or XLA veto fired. Cold candidates were rejected by the exact scientific gate. |
| Statistically supported ranking | None. The campaign supports rejection/pass classifications, not superiority rankings among viable schedules. |
| Descriptive-only differences | Calibration losses, tail distances, importance ESS, log-ratio SD, and runtime differences. |
| Default-readiness | Not established. The repair uses exact funnel structure unavailable in a generic posterior. |
| Next evidence needed | Replicate the end-to-end staged fit, design a target-available staged or continuation objective, validate it on further known-law models, then test NeuTra HMC only after a proposal-law pass. |

## Run manifest summary

- Git commit: `3030d86d` with a dirty worktree preserved.
- Environment: `/home/ubuntu/anaconda3/envs/tfgpu`.
- Hardware: GPU 0, RTX 4080 SUPER; TensorFlow memory growth verified in each cell.
- Backend: TensorFlow float64, XLA JIT, TF32 disabled, batch 4,096.
- Cold campaign: 32 calibration cells plus eight confirmations, wall time
  `1330.60` seconds.
- Repair: one restricted fit, three warm-start calibration cells, and two joint
  confirmations; per-cell manifests contain exact commands, seeds, device
  receipts, wall times, and artifact hashes.
- Artifact root:
  `docs/plans/artifacts/neutra-reverse-funnel-architecture-tuning-2026-08-15/`.

## Post-run red team

The strongest alternative explanation is that a still-unsearched cold-start
optimizer schedule could reach the same basin. The current evidence does not
exclude that possibility. It does show that more width and root-preserving
permutation did not solve the problem under their own reviewed tuning grids,
whereas the structured warm-up did so reproducibly. A cold-start procedure that
passes fresh exact-law confirmations would overturn the need for this warm-up.
