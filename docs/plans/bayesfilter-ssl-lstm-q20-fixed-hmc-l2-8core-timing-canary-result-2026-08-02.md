# q=20 Fixed-HMC L=2 Eight-Core Timing Canary Result

Date: 2026-08-02  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-fixed-hmc-l2-8core-timing-canary-plan-2026-08-02.md`  
Status: `CANARY_COMPLETED`

## Outcome

The exact Chart A public-API `L=2` ladder completed on eight pinned CPU cores
in `2262.530864966975 s` (`37.7088 min`). The matching prior 16-core run took
`1806.6523481040495 s` (`30.1109 min`). The eight-core observation was
`1.25233x` the 16-core wall time: eight cores were `25.23%` slower, not twice
as slow.

This is evidence that eight cores are a workable per-`L` allocation for this
specific `L=2` workload. It is not evidence that cost is linear in `L`, so it
does not establish the duration of `L=25` or of the full five-point grid.

## Timing Comparison

| Work | 16 cores (s) | 8 cores (s) | 8/16 ratio |
| --- | ---: | ---: | ---: |
| DA budget 8 | `216.0568` | `263.1484` | `1.2189` |
| Screen after budget 8 | `252.6826` | `326.1597` | `1.2908` |
| DA budget 16 | `298.5412` | `382.6788` | `1.2818` |
| Screen after budget 16 | `264.5950` | `324.9576` | `1.2281` |
| DA budget 32 | `504.0184` | `635.8849` | `1.2616` |
| Screen after budget 32 | `269.8643` | `328.7887` | `1.2183` |
| Full harness | `1806.6523` | `2262.5309` | `1.2523` |

All six numerical calls executed in both runs. Their deterministic tuned step
sizes and screen acceptances match exactly, showing that the eight-core result
did not become shorter through a different ladder branch or early exit:

| DA budget | Tuned step | Screen acceptance |
| ---: | ---: | ---: |
| `8` | `0.6472725572` | `0.7816461580` |
| `16` | `0.5403124786` | `0.6377098886` |
| `32` | `0.5899598740` | `0.8436936092` |

The acceptance values are descriptive workload-identity checks here. They do
not admit an HMC kernel; no screen was inside the predeclared `[0.65,0.75]`
band.

## Evidence Contract Result

| Evidence role | Result |
| --- | --- |
| Engineering question | Passed: exact full Chart A `L=2` ladder completed with eight pinned cores. |
| Timing comparison | Observed eight-core wall was `25.23%` slower than the matching 16-core wall. |
| Continuation vetoes | None: exit code zero; finite required telemetry; target status valid; affinity and threads correct; GPU hidden; XLA enabled. |
| Explanatory only | Per-call timing, acceptance, tuned steps, retracing warnings, and shared-host load. |
| Not concluded | `L=25` timing, linear-in-`L` scaling, full-grid duration, kernel admission, convergence, posterior validity, or default-readiness. |

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Treat eight cores as a feasible per-`L` execution size | Passed for the exact `L=2` workload | No continuation veto | Single shared-host timing observation | Use this value for resource sizing; time a bounded representative `L=25` call before relying on linear extrapolation | That eight cores are optimal |
| Do not admit the `L=2` kernel | No acceptance screen in `[0.65,0.75]` | No hard numerical/status veto | Screens are short | Run the intended multi-`L` tuner only under a separate reviewed grid plan and sufficient budget | Fixed HMC or the chart is invalid |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | No hard numerical/status veto in the canary. Native divergence remained unavailable, not zero. |
| Statistically supported ranking | None; this is one timing observation per core allocation. |
| Descriptive-only differences | Eight-core wall and per-call times were `21.8%` to `29.1%` slower. |
| Default-readiness | Not applicable and not established. |
| Next evidence needed | A bounded `L=25` timing canary or a reviewed argument before using an `L=2` linear projection for the full grid. |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `b370dc89e6e79f3853e0fccd5ab5b4fa2cb9065d` |
| Worktree | Dirty with concurrent agent work; unrelated changes were preserved. |
| Command | Exact command recorded in the plan. |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu`, TensorFlow `2.20.0`, TFP `0.25.0` |
| CPU/GPU | CPU-only by explicit diagnostic exception; `CUDA_VISIBLE_DEVICES=-1`; `physical_gpus=[]` |
| CPU affinity | Logical CPUs `32..39` |
| TensorFlow threads | Intra-op `8`; inter-op `1` |
| XLA | Enabled; runtime emitted `Compiled cluster using XLA!` |
| Dtype | `float64` |
| Chains | Four, batched in one rank-2 `[4,4]` chain bank with one shared scalar step |
| Leapfrog grid | Diagnostic override `(2,)` only |
| DA budgets | `(8,16,32)`; tune results `8` per round |
| Screen | `16` results with `4` burn-in steps per round |
| Seeds | Tune bases `(20260625,100..102)`; screen API-owned deterministic offsets from `(20260625,200)` |
| Chart checkpoint | `checkpoint-1500.json`, SHA-256 `c87ee24874705bb12296cc05b82310326579694cc04c2a3682792f9bf18fb9ff` |
| Target signature | `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278` |
| Transport hash | `caf6c9ec1a46d04253b2ae3922d83e619f38c824cea955d5da8ac419d2dfed7f` |
| Harness SHA-256 at run | `b288274860c1b8ec122c80ad7b8e67527d34eb0a741dd84f91ab96b202366d1a` |
| TF tuner SHA-256 at run | `8a627c283084e1d90908b5a3bf731f3e4e862c9189179c3c1da371779c9cbd42` |
| Wall time | `2262.530864966975 s` |
| Summary artifact | `docs/plans/artifacts/ssl-lstm-q20-fixed-hmc-l2-8core-timing-canary-2026-08-02/r1/chart-a/summary.json`, SHA-256 `8c61462645e8b9af6abaa90e17422b228f319e7b165538e314a272f68f30c526` |
| Tuning artifact | `docs/plans/artifacts/ssl-lstm-q20-fixed-hmc-l2-8core-timing-canary-2026-08-02/r1/chart-a/tuning-result.json`, SHA-256 `a52d8bf7024037c1948b424ca75f8d12eea0d40ef3b0f138f558c7de42bce5de` |
| Plan | `docs/plans/bayesfilter-ssl-lstm-q20-fixed-hmc-l2-8core-timing-canary-plan-2026-08-02.md` |
| Result | `docs/plans/bayesfilter-ssl-lstm-q20-fixed-hmc-l2-8core-timing-canary-result-2026-08-02.md` |

The older artifact records the same commit, public-API configuration, chart
provenance, environment, and deterministic outputs, but it did not record
hashes for the then-untracked harness and tuner. Exact historical source-byte
equality is therefore unsupported, even though file mtimes and matching outputs
are consistent with an unchanged path.

## Campaign Accounting

| Charge | Seconds |
| --- | ---: |
| Prior campaign charge | `4178.2646` |
| Eight-core canary | `2262.5309` |
| Cumulative | `6440.7955` |
| Authorized cap | `20000.0000` |
| Remaining | `13559.2045` |

## Post-Run Red Team

Strongest alternative explanation: the `25.23%` slowdown reflects a mixture
of core count, CPU placement, cache/memory effects, and concurrent host load,
not core count alone. The machine was shared with unrelated CPU-heavy jobs.

What would overturn the resource conclusion: repeated eight-core runs that
time out or show materially larger walls under representative load, or an
`L=25` canary showing strongly nonlinear cost.

Weakest evidence: one observation at each core allocation and no isolated-host
replication. The result answers the requested canary timing but does not support
a statistically defensible performance ranking.

