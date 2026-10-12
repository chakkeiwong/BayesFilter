# Priced continuation execution record

12 October 2026. This continues the
[priced comparison plan](bayesfilter-modern-hmc-priced-comparison-2026-10-12.md)
and the [MCSE reference study](bayesfilter-mcse-stopping-calibration-2026-10-12.md).
Numerical source remains the committed `full-model-source-e68cdf1c4` snapshot.
No numerical tuning, transition, posterior or reference default changes here.

Current checkpoint, 14:09 UTC: pricing and its audit are complete; the MAMS
consumer is running fresh qualification on GPU 1. The active service is
`bayesfilter-modern-priced-continuation-20261012-02.service`, with artifacts in
`continuation-02` under the output root below. Earlier waiting/pricing entries
are historical checkpoints. Five 64-transition measurement trials across four
chains have completed and been checkpointed. No candidate is verified yet.
The next check is the first completed measurement stage, followed by fresh
verification, any planned repairs and posterior outcomes within the recorded
34.17-hour child allocation.

The active output root is
`docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01`.
The original GPU price process (supervisor 2238414, worker 2238416) and CPU-only
MCSE diagnostic (2243564) were recovered from their manifests and live process
records. No duplicate campaign was launched. Both use their original allocation:
7200 GPU-worker seconds and 1800 CPU-reference seconds, respectively, with the
latter inside the existing 3600-second local reserve.

The fixed-HMC cell has completed all eight roots and sixteen native calls.
Every completed call passed its recorded numerical health screen. A separate
read-only audit checked every receipt, input stream, target-status trace shape,
sample shape/dtype, native metadata and charged work. Initial states are
identical in the fixed and randomized cells. These are short cost diagnostics;
they grant no membership, posterior or method-ranking authority. The remaining
method prices are still active as this record is opened.

The terminal reporter permits only the documented fixed-HMC repair-schema
correction when comparing original and corrected designs. The other three
designs are identical. Changed starts, seeds, targets, geometry, trajectories,
posterior criteria or unrelated search controls cannot reuse the prices.
The report retains numerical vetoes and both first/repeated native call costs.
It checks unchanged numerical source and compiled operator, the runtime memory
policy and complete method/root inventory before deriving an envelope.

The measured fixed-HMC envelope is already too large to fund the full
comparison: at load one the conservative minimum/full-count scenarios are
56.06/278.45 hours. This overprices repeated cold calls and repair growth and
must not be described as an observed completion time. It supports the plan's
scoped MAMS feasibility branch. The prior audited deployment's total-minus-native
cost is 860.163968 seconds across 32 trials, used as the explicitly conservative
26.880124-second amortized overhead sensitivity.

The continuation's allocation rule is recorded before launch in the plan.
It waits for settled, audited prices, preserves all four adapted MAMS roots
and four reserved repairs, and reports the minimum-count scenario against the
existing 123019.624951-second unallocated balance, less any pricing overrun.
That conservative forecast is not a continuation veto; the actual remaining
budget bounds a feasibility attempt with unfinished outcomes preserved. It
does not spend held reserves again. It writes the actual allocation before launching the
snapshot's `run_modern_hmc_frozen_full_model.py`. A capacity shortage triggers
bounded resource waiting; a missing price, source inconsistency or unexpected
probe failure stops dependent work. Scientific failures remain candidate or
posterior outcomes under the existing contract.

Validation command, with GPUs deliberately hidden:

```sh
env CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
  TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=2 OMP_NUM_THREADS=2 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest --noconftest -q \
  tests/test_modern_hmc_priced_continuation.py \
  tests/test_modern_hmc_price_report.py tests/test_modern_hmc_comparison_prices.py \
  tests/test_hmc_stopped_mcse_calibration.py
```

All **53 tests passed in 1.08 seconds**. Earlier focused verification passed
40 tests in 0.96 seconds. Both are local engineering charges within the reserve.
The tests include altered design fields, corrupted native bytes, missing status
traces, full candidate/error-budget accounting, resource censoring, supervisor
overruns, double spending, omitted methods/roots and actual orchestration of a
stub child at the immutable-source launch boundary. The stub does not establish
numerical equivalence; the numerical consumer's prior isolated checks remain
the authority for that question. Actual completed price receipts were also
inspected without rerunning a transition.

Skeptical post-implementation review: the main remaining cost uncertainty is
shared GPU load and long-trajectory/assessment behavior. The price model does
not resolve that uncertainty; the launch remains bounded and may be censored.
The strongest alternative explanation for a later precision failure remains
inadequate geometry or an unreliable uncertainty estimator. The parallel MCSE
diagnostic addresses the latter under known stationary controls, without
changing the current posterior rule. No comparative or release claim follows
from the engineering checks or short prices.

## Completed MCSE study and continued execution

The CPU study is now terminal. The first allocation completed 717 records;
the checked continuation completed the remaining 51 under the original seeds,
preserving every completed result unchanged. All six conditions have the
planned 128 replications. Total calibration cost is 1940.794223 CPU-worker
seconds. The continuation repair is committed as `9075f1a42` and passed 21
focused tests in 0.74 seconds; combined with the earlier tests, 64 distinct
focused regressions pass. The separate full-result audit passes all source
hashes, identities, summaries, parent hashes and cumulative accounting.

The [result note](bayesfilter-mcse-stopping-calibration-result-2026-10-12.md)
records the checked finite-covariance derivation, unavailable estimates,
operational stopping counts and pointwise interval uncertainty. The unmasked
lugsail formula underestimates variance in the antithetic Gaussian example.
Neither the even-batch variant nor the autocorrelation alternative is supported
as a general replacement default. Stationary controls cannot test warm-up or
global posterior exploration. The current MAMS posterior rule remains frozen.

The GPU continuation was launched under trusted execution from committed
`a049a50a6`. Its systemd unit is
`bayesfilter-modern-priced-continuation-20261012-01.service`; its durable
manifest is `continuation-01/manifest.json` under the output root above.
It is waiting for the original price supervisor, with periodic status updates.
At the 13:09 UTC inspection, fixed HMC had completed all eight roots and
randomized HMC had seven complete roots plus the final root's first price.
No observed price had a numerical-health veto. SNAPER and MAMS remain in the
declared queue. This is continuing cost measurement, not a completed posterior
run. The 7200-second GPU allocation and held local reserve are unchanged.

Continuation self-audit: the result note agrees with the complete numerical
summaries; the replication unit, covariance target and stopped-versus-fixed
interpretation remain explicit. No diagnostic has become a tuning membership
gate, no unavailable interval has been dropped, and no stochastic ordering has
been promoted. Price/launch accounting remains separate from the completed
CPU study. An incomplete price inventory requires a fresh bounded continuation;
it cannot be filled by an estimate and reported as measured.

The finite-covariance check is now a persistent regression against the actual
TensorFlow kernel: 14 additional CPU/XLA tests pass, including negative-chain
pooling and batch remainders. The immutable snapshot supplied the numerical
code; its precision source agrees byte-for-byte with the current repository.
The check cost 3.553224 seconds including startup. This brings the distinct
focused-test total to 78 without repeating the completed stochastic study.
`local-accounting-01.json` records measured local charges and keeps the full
3600-second reserve held, including unmeasured short-command overhead; it does
not release or spend that reserve a second time.

By 13:15 UTC, both classical cohorts had completed: 16 roots and 32 prices.
SNAPER preparation was active, followed by the four adapted MAMS roots.
There was no numerical-health veto in the completed calls. The price supervisor
and its dependent continuation service remained active.

## Continuation rule repaired before dependent launch

At 13:48 UTC, review found that the first service could refuse to attempt the
next phase solely because a conservative all-member forecast exceeded the
balance. Its formula includes uncreated reserved repairs and repeated cold
costs. It is not a lower bound on actual work. The plan now distinguishes that
forecast from actual allocation exhaustion: the unchanged full MAMS cohort
can be attempted within the remaining budget, with partial outcomes retained.
It still cannot establish full coverage, comparative ranking or release from
an unfinished cohort.

The waiting `continuation-01` service was stopped with no child launched and
zero child-worker cost; its separate settlement preserves the original
manifest. The GPU price worker continues. The replacement writes to the fresh
`continuation-02` output. Thirty-three focused tests pass in 0.616564 seconds
including startup, covering oversized forecasts, exact budget exhaustion,
overruns, all-member preservation and resource-stopped child outcomes. The
actual remaining ceiling is unchanged at 123019.624951 seconds before any
price overrun. No source, seed, model, statistical gate or posterior check
was changed. This is a planning-policy repair within the existing campaign.

## Terminal pricing and actual MAMS launch

Pricing finished at 13:56:11 UTC: eight fixed-HMC roots, eight randomized-HMC
roots, six SNAPER roots and four adapted MAMS roots, each with two native calls.
All 52 calls returned without a recorded numerical-health veto. The supervisor
charged 6851.941203 seconds; the unused 348.058797 seconds stay held, preventing
double spending. `continuation-02/price-audit.json` verifies all 52 distinct
streams, source and compiled-operator hashes, native receipts, complete cohort,
common starts and memory policy. The final price JSON SHA-256 is
`e74d6cac0c283898fd76df27862231446ccdfe71f8a30600cb3686884957c5c5`.

The continuation wrote its allocation and launched the immutable consumer at
about 13:56:12 UTC. Controller, supervisor and worker PIDs were 2304792,
2306449 and 2306450 at inspection. The child's manifest records source commit
`e68cdf1c4c6b75c16f92b86a3363a1e24690d22c`, host `CUDA_VISIBLE_DEVICES=1`,
verified memory growth on the visible GPU, TensorFlow 2.20.0, TFP 0.25.0,
TF32 and XLA. The log records successful XLA compilation. Preparation completed
and the manifest entered `qualifying` at 14:00:50 UTC; the search design was
written at 14:03 UTC. At 14:09 UTC, five native measurement trials (ordinals
0--4) had completed, each with sample shape `[64,4,6]`. The launch audit checked
their binding, compilation, device and durable receipts, along with all 197
declared source/input hashes. It does not replace the public controller's
statistical and numerical-health assessment. The intermediate tuning checkpoint
can say `partial_budget` while the worker is still qualifying; this is not a
terminal resource stop. No duplicate child was launched.

The allocated ceiling is 123019.624951 seconds (34.17 hours). The MAMS
minimum-count scenario is 142574.539112 seconds (39.60 hours), including all
reserved repairs and conservative cold-call/overhead costs. That forecast is
not an expected finish time or a measured lower bound. A positive actual
balance funds a useful attempt; full completion remains uncertain. All four
roots, four reserved descendants and unfinished denominators remain in scope.
The supervisor enforces the actual allocation and the consumer preserves
partial outcomes. No acceptance, convergence, precision or reference threshold
was relaxed.

The parent manifest's `price_status` and `last_checked_utc` still reflect its
last prelaunch poll. Its top-level `status=running`, saved price audit and
child manifest establish the current phase; do not interpret the stale poll
field as an active price worker or restart the numerical child to refresh it.
`launch-audit-01.json` preserves a separate timestamped observation without
editing live manifests. `local-accounting-02.json` adds the measured
0.616564-second regression charge while retaining the local reserve.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Limit |
| --- | --- | --- | --- | --- | --- |
| Continue bounded MAMS feasibility | Complete pricing inventory and audit pass; actual allocation positive | No price health veto; source, device, receipt and design checks pass | Shared load, qualification/repair cost and posterior sufficiency | Monitor the full frozen cohort and preserve every outcome | Pricing establishes no tuning membership or posterior validity |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | None in the 52 short prices; qualification and posterior vetoes are assessed afresh |
| Statistically supported ranking | None |
| Descriptive-only differences | Native call times and planning cost scenarios |
| Default-readiness | Not established; no default changed |
| Next evidence needed | Fresh qualification and posterior/reference results, followed by funded matched methods/models and independent replications for broader claims |

Post-launch skeptical review: observed process activity and successful compilation
establish execution, not sampler validity. The cheapest prices cannot justify
omitting slow roots. The allocation decision survives the prior forecast flaw
because partial outcomes are explicit and the actual cap is enforced. Geometry,
mixing and the documented antithetic MCSE weakness remain alternative explanations
for a later posterior failure; none is repaired by a passing price audit.
