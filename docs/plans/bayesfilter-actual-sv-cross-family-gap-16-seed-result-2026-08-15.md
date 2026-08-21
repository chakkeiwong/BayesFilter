# Result: actual-SV cross-family gap 16-seed replication (2026-08-15)

Plan: `docs/plans/bayesfilter-actual-sv-cross-family-gap-16-seed-replication-plan-2026-08-14.md`

## Run manifest

- Git commit (dirty tree, uncommitted benchmark work): `18cfe609`
- Command: `python docs/benchmarks/run_actual_sv_cross_family_gap_seed_sweep.py
  --output docs/benchmarks/artifacts/actual_sv_cross_family_gap_seed_sweep_20260814/attempt01/result.json
  --markdown-output .../result.md`
- Environment: conda `tf-gpu`, Python 3.11.14, TensorFlow 2.19.1, float64,
  CPU-only (GPUs deliberately hidden via `CUDA_VISIBLE_DEVICES=-1`)
- Seeds: 16 bases `83120 + 20000*k`, k=0..15; k=0 reproduces the attempt01
  paths of the three-route benchmark exactly. Dims 1/2/3, horizon 20.
- Fitted mixtures (7/14/28) fitted once, identical across seeds; dense order
  401, radius 8; deterministic, no runtime randomness.
- Wall time: 2.4 min. Hard veto (non-finite value): did not fire.
- Artifact:
  `docs/benchmarks/artifacts/actual_sv_cross_family_gap_seed_sweep_20260814/attempt01/{result.json,result.md}`

## Seed distributions of paired raw-`y` differences (16 seeds)

`d = route - exact dense`, log-likelihood units; t95 = two-sided 95%
Student-t interval for the mean.

| dim | route | mean | sd | t95 | range |
|---|---|---|---|---|---|
| 1 | KSC-7 dense | -0.0386 | 0.1370 | [-0.112, +0.034] | [-0.335, +0.122] |
| 1 | fitted-28 Kalman | +0.0009 | 0.0079 | [-0.0033, +0.0051] | [-0.0122, +0.0187] |
| 2 | KSC-7 dense | +0.0195 | 0.3370 | [-0.160, +0.199] | [-0.579, +0.485] |
| 2 | fitted-28 Kalman | +0.0004 | 0.0100 | [-0.0049, +0.0058] | [-0.0208, +0.0177] |
| 3 | KSC-7 dense | -0.1487 | 0.3828 | [-0.353, +0.055] | [-0.802, +0.582] |
| 3 | fitted-28 Kalman | -0.0024 | 0.0142 | [-0.0099, +0.0052] | [-0.0296, +0.0184] |

(fitted-7 and fitted-14 rows in the artifact; fitted-14 ~= fitted-28
everywhere, consistent with the earlier refinement-stability result.)

## Answer to the question

The suspect dim-2 value (-0.0144, fitted-28, original path) sits well inside
its 16-seed range [-0.0208, +0.0177], about 1.4 sd from the seed mean
(+0.0004). The seed-0 originals for dims 1 and 3 are likewise inside their
ranges. Verdict per the plan's interpretation rule: **path-level Monte Carlo
variation — `correct`, nothing is wrong with that path or route.** The
apparent dim-2 anomaly in the single-path benchmark was seed scatter.

## Decision table

| Item | Status |
|---|---|
| Decision | Dim-2 single-path gap classified as seed scatter; no repair or follow-up warranted |
| Primary criterion (original value within seed distribution) | Passed at every dim |
| Veto diagnostics | None fired (all values finite; refinement stable across seeds) |
| Main uncertainty | 16 seeds bounds the fixture distribution at these parameters/horizon only |
| Next justified action | None required; more seeds only if a mean-level family bias claim is ever wanted |
| Not concluded | No family ranking; no bias claim for fitted-28 (its t95 includes 0 at every dim) |

## Refinement of the earlier interpretation

The three-route result note attributed the single-path cross-family gap to
"KSC-7 mixture bias". The sweep sharpens this: across seeds the KSC-7 route's
deviation from exact is dominated by per-path **scatter** (sd 0.14-0.38, t95
for the mean includes 0 at dims 1-3), not by a resolved mean bias. The
correct statement is that fitting the mixture to the exact log-chi-square
density shrinks the per-path deviation scale by roughly 20-30x (sd 0.14-0.38
-> 0.008-0.014); whether a nonzero mean bias remains is `not checked` at this
seed budget and is not claimed.

## Post-run red team

- Strongest alternative explanation for the dim-2 original value: none needed
  — 1.4 sd events are expected in 16 draws.
- What would overturn: a larger sweep showing the fitted-28 dim-2 mean
  significantly nonzero with the original path in its tail; nothing here
  suggests that.
- Weakest evidence: seed bases share the generator family; independence
  across seeds rests on the generator's seeding quality, as everywhere else
  in the repository.
