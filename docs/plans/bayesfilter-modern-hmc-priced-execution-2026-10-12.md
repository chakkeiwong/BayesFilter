# Priced continuation execution record

12 October 2026. This continues the
[priced comparison plan](bayesfilter-modern-hmc-priced-comparison-2026-10-12.md)
and the [MCSE reference study](bayesfilter-mcse-stopping-calibration-2026-10-12.md).
Numerical source remains the committed `full-model-source-e68cdf1c4` snapshot.
No numerical tuning, transition, posterior or reference default changes here.

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
and four reserved repairs, and checks the minimum scenario against the existing
123019.624951-second unallocated balance, less any pricing overrun. It does not
spend held reserves again. It writes the actual allocation before launching the
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
