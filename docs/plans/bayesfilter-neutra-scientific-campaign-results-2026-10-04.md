# Scientific NeuTra campaign: executed, training not yet qualified

The executable campaign exists and has been run. The corrected calibration has
not produced a qualified training recipe. Exact target samples pass the teacher
screen, but all eight tested forward-KL plus reverse-KL student recipes across
two development targets fail the distribution screen. These are four recipe
settings tested on each target, not eight independent seeds. No stochastic
ranking is supported. Unseen-target generalization remains untested under the
corrected procedure because its training prerequisite failed.

Plan: `bayesfilter-neutra-scientific-campaign-plan-2026-10-04.md`, including its
revision 2 and 3 amendments. Recovery: `bayesfilter-neutra-scientific-campaign-reset-2026-10-04.md`.
Active program: `scripts/run_neutra_scientific_campaign_master.py`; numerical
consumer: `bayesfilter/testing/neutra_scientific_campaign.py`. Both use the
shared `NeuTraTransport` and canonical author IAF core. This campaign does not
introduce another transport implementation.

## What ran and what the audit repaired

R1 produced 108 final decisions: 54 executed SMC/AFT/CRAFT trials and 54
prerequisite failures for FAB/Gabrié/AIS, plus six exact-teacher final controls.
The subsequent scientific audit failed. Teacher and map discrepancies had been
divided by reference uncertainty alone, omitting uncertainty in the evaluated
population. For equal population variances with 1,024 teacher versus 4,096
reference draws, the statistic was inflated by sqrt(5). It rejected an exact
iid teacher. The implementation also used only 128/256 updates where its plan
said 512/1,024, selected a native profile using one favorable development target,
and did not reload frozen checkpoints before assessment. No RKL finish ran.
These findings invalidate the original scientific failure classifications.
Preserved audit: `artifacts/neutra-scientific-2026-10-04/campaign-r1/terminal-scientific-audit-r1.json`.

Launch/runtime repairs made along the way include the absolute-path import
environment, the preflight action, missing scripts in the source closure,
unequal-sized reference/probe finiteness checks, and an overwritten teacher
failure status. Failed attempts remain in the compute ledger. R1 is debugging
and pricing evidence, not evidence that the algorithms failed scientifically.

R2 introduced four independent complete teacher populations and estimated
uncertainty across their feature means, added map-draw uncertainty, assessed
component responsibilities, local first/second moments and radial tails, and
kept separate references for teacher qualification and map assessment. It
implemented 512/1,024 forward updates followed by 128/256 RKL updates, reloaded
both frozen endpoints and ran all 1,000 diagnostic points at each endpoint.
The four exact-teacher banks passed. The four student endpoints failed.

R3 tested the next explicit capacity/effort rungs: width 32 with 4,096 forward
updates, and width 64 with 8,192, both followed by 256 RKL updates. The remaining
architecture is unchanged: three IAF stages, author masks/ELU/initialization,
full reversals, free scale bias and conditional cap 2. Forward learning rate
was 0.001, RKL learning rate 0.0003, batch 64, Adam (0.9,0.999,1e-8), emergency
clipping 1000. These are tested hypotheses, not new defaults. All gradients,
parameters and optimizer states remained finite, and **zero updates were
clipped**. R3's four exact-teacher banks passed and four student endpoints
failed. Each failed recipe is preserved with a heldout learning history.

R2 and R3 terminal artifact audits passed. Both are honestly recorded as
`scientific_terminal_under_calibrated`. R3's 138 prerequisite decisions comprise
24 native calibration cells, six final exact-teacher controls and 108 final
method cells; none was silently described as executed. The program does not
interpret an artifact audit pass as a map-quality pass.

## Current numerical evidence

These values describe single fit seeds 9101 and 9201 on the separate two- and
three-center calibration targets. The maximum feature discrepancy is measured
in combined standard errors for independent map/reference draws. Its threshold
5 is an exploratory screen, not a calibrated test or global correctness bound.
The responsibility-mass discrepancy threshold is 0.15. KL estimates use the
normalized exact benchmark density and independent assessment samples.

| Development target | Width / forward updates | Forward KL after forward fit | Forward KL after RKL | Final max feature discrepancy | Final max responsibility discrepancy | Final screen |
|---|---:|---:|---:|---:|---:|---|
| Two centers | 32 / 4,096 | 0.3749 | 0.5086 | 17.40 | 0.1660 | Failed |
| Two centers | 64 / 8,192 | 0.1072 | 0.1362 | 9.45 | 0.0359 | Failed |
| Three centers | 32 / 4,096 | 0.5065 | 0.5359 | 14.83 | 0.0652 | Failed |
| Three centers | 64 / 8,192 | 0.4298 | 1.1422 | 32.07 | 0.2410 | Failed |

The forward-fit histories show continuing descriptive progress. At width 64,
the two-center forward KL was 0.4340, 0.4120, 0.4011, 0.3011 and 0.1072 at
512, 1,024, 2,048, 4,096 and 8,192 updates. The three-center sequence was
1.0883, 0.6476, 0.6078, 0.5892 and 0.4298. These reused development assessments
are not independent confirmations or proof that a further rung will pass.

RKL can reduce the mass assigned to a component whose current fit has high
energy. In the width-64 three-center run, the observed maximum responsibility
discrepancy grew from 0.0509 after forward training to 0.2410 after RKL. This
supports preserving both endpoints and investigating the RKL continuation.
It is not a theorem that RKL must fail or a statistically supported comparison
of objectives across targets and seeds.

The required 1,000-point test ran on every fitted forward and RKL checkpoint
in r2 and r3: 16 complete probes with 1,000 valid rows each. For the width-64
r3 endpoints:

| Target | Median score-residual norm | p95 score-residual norm | Range of log-density residual |
|---|---:|---:|---:|
| Two centers | 0.3444 | 43.0105 | 8.3126 |
| Three centers | 1.7290 | 26.0747 | 11.2718 |

The small two-center median does not establish whitening: its tail residuals
are large and its independent distribution screen failed. All probe magnitudes
are explanatory, including their maxima and extreme quantiles. The probe draws
from the standard-normal base, not the posterior.

## Coverage and method status

The program contains fixed unwarped/warped mixtures and fresh random two- and
three-center targets. Generalization distances are Uniform[6,10], variances
Uniform[0.5,2], with random orientation/translation and weights at least 0.1.
The fresh final target seeds are 1103/1104 and 2103/2104; fit seeds are 51/52/53.
Their exact definitions are frozen in r3's `target-catalog.json`. Learners get
the density/score interface, not component centers, labels or variances.
These final targets have not been evaluated by the corrected procedure.

| Method | Current executable route | Evidence limit |
|---|---|---|
| FAB | Explicit applicability prerequisite | Nonlinear alpha-two integrability unresolved; no FAB scientific trial executed |
| Gabrié | Fixed-map global-MH then local-MALA control | Full adaptive author controller not reproduced by this arm |
| AIS | Fixed temperature ladder, no resampling, MALA mutation | Corrected native calibration not reached after student failure |
| SMC | Matched ladder with resampling and MALA mutation | Corrected native calibration not reached after student failure |
| AFT | Existing disjoint train/validation/test stage adaptation | Full upstream controller equivalence not established |
| CRAFT | Existing frozen-pass/stage-update adaptation | Full upstream controller equivalence not established |

This is an executable bounded testing program with honest prerequisite rows.
It is **not** a completed scientific comparison of six faithfully reproduced
upstream algorithms. R1's native samples cannot substitute for the corrected
calibration. No method ranking, default change or q20 transfer is supported.

## Validation, resources and recovery

The final source gate passed **75 tests**. Tests include derivative/target
references, randomized-target bounds, teacher uncertainty under duplicated
particles, zero particle weights, both-sample map variance, actual forward/RKL
checkpoint reload with full probes, chunked-versus-uninterrupted Adam/RNG
equivalence, profile admission on both targets, cumulative accounting and
terminal resume without new workers. Tiny CPU training tests are explicitly
debug/reference exceptions. Trusted GPU preflight passed with XLA and memory
growth verified before logical-device initialization. Actual training used
one trace per compiled update block, native batches, GPU 1 and no sample-wise
Python loop or NumPy training path.

The study deliberately used the existing configured FP64 benchmark trainer.
TF32 was enabled but does not accelerate FP64 operations. This is an explicit
reference-arithmetic exception, not production precision/performance evidence.
Sequential workers keep resource attribution simple; this launcher does not
claim to use all three GPUs concurrently.

| Revision | GPU process seconds | CPU core seconds |
|---|---:|---:|
| r1, including failed/superseded attempts | 1,724.1972 | 2,604.4830 |
| r2 | 139.7154 | 302.8175 |
| r3 | 240.5863 | 357.8355 |
| Total | **2,104.4988** | **3,265.1359** |
| Remaining of the same 6,000 / 12,000 allocation | **3,895.5012** | **8,734.8641** |

GPU-process time is worker lifetime, including compilation and host work; it
is not measured kernel occupancy. CPU cost includes child shutdown, measured
by the parent. No historical grants were added. Manifests retain exact command,
Python environment, source/git hash, target/seed/profile, TensorFlow version,
memory policy, allocator peak, times, checkpoint and reference hashes. All
workers used `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`.

Executed wrapper verbs were `check`, `preflight`, `price`, `run`, `resume` and
`status`. The already installed narrow allowlist covers direct/shell forms;
trusted GPU launch approvals were saved and reused. No broad interpreter rule
was added. Local rule matches do not override a managed reviewer. No rejection
blocked the final repaired execution.

The final `resume` was also exercised: it performed audit r2 of campaign-r3,
changed no attempt count or compute cost and launched no worker. It will not
secretly rerun a rejected campaign. To inspect: use the exact wrapper `status`.
The active `scientific-result.json`, `matrix.json`, `next-phase.json`, `pricing.json`
and `audit-r2/result.json` are under
`docs/plans/artifacts/neutra-scientific-2026-10-04/campaign-r3/`.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Reject r1 scientific interpretation | Wrong uncertainty denominator/protocol mismatch | Harness/protocol veto | Invalid failure labels | Preserve and use corrected procedure | Algorithm failure |
| Keep corrected harness | 75 tests and terminal source/artifact audits pass | No unresolved execution veto observed | Limited test coverage | Use it for the next explicit training repair | Scientific validity from test passage alone |
| Reject tested student recipes | Local distribution screen fails on both targets | Finite/GPU/batching checks pass; clipping absent | Optimization, finite training-bank effects, capacity and RKL coverage regression | Calibrate continuation/optimizer/capacity on development targets, retaining both endpoints | NeuTra impossibility or successful training |
| Hold generalization admission | No single student recipe passes both calibration targets | Promotion prerequisite failed | Unseen targets untested | Resume generalization only with a newly frozen qualified recipe | Cross-target transfer |

| Inference class | Status |
|---|---|
| Hard veto screen | r1 harness invalid; r2/r3 numerical/source checks pass and candidate distribution screens fail |
| Statistically supported ranking | None |
| Descriptive-only differences | Learning curves, KL estimates, residual quantiles, runtime and RKL mass changes |
| Default readiness | Not established; no default changed |
| Next evidence needed | Passing target-specific training protocol, independent fit/target replications, then native-teacher qualification and frozen generalization trials |

The observed result invalidates the current candidates, not the target, shared
transport mathematics or research direction. Both planned repair rungs were
executed. No longer calibration rung is silently selected or promoted; further
changes require an explicit extension of the numerical plan within the remaining
allocation. The strongest alternative explanation is insufficient optimization
or parameterization/capacity rather than teacher quality. The weakest evidence
is one initialization per development target and a small recipe search. A
coverage-preserving continuation that passes independent assessments would
overturn the present candidate-level rejection. A clean scientific success
must still be measured downstream before any HMC or posterior claim.
