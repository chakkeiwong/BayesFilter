# Range-bearing fixed 16-step continuation

Terminal update, 2026-10-16: fresh 16-step tuning and GPU mechanics passed.
Inference received the final1029.17 seconds of the campaign and timed out
while XLA compilation was still reported in progress, before any saved
sequential samples. The86400-second allocation is exhausted; no worker is
running. See `artifacts/surrogate-hmc-range-bearing-flow16-20261016-01/result.md`.
The planned work below was executed to its budget stop, not completed to its
posterior criterion. Further inference remains part of this master and needs
additional compute authorization.

Owner instruction: "I agree. use 16 flow steps."

Use 16 fixed Euler flow steps for range bearing, T20, N4032/K2016,
FP32 with TF32 off, GPU/XLA, literature-model-priors-v1, and the already
accepted epsilon=1/five-step OT family. The objective remains testing
surrogate HMC against its unchanged independent likelihood endpoint.
This is a repair phase of the existing five-model master. Other model
settings and active frozen selections remain unchanged.

The eight-step baseline failed its force-validity guard in warmup. Full
CPU replays at 16 and 32 steps pass both the failing input and its preceding
healthy input (range-bearing-warmup-fallback-20261015-01/result.md).
Sixteen steps is now the owner's selected candidate. It is not a proof of
global stability: the left-endpoint Euler resonance moves to information
eigenvalue 32. Keep the numerical guard and the zero-fallback health check.

## Execution and evidence

1. Add an explicit offline flow-step override to the preparation master;
   preserve it in the fresh tuning selection consumed by mechanics and HMC.
2. Test argument propagation, unchanged controls, and selection consumption.
3. After the existing master exits, perform fresh range-only calibration and
   validation with seed offset 400, separate from earlier offset 200 and the
   failed warmup inputs. Require the existing independent-reference criteria.
4. Run actual GPU/XLA mechanics, then adaptive four-chain HMC. Exclude warmup;
   retain cumulatively until the existing R-hat/ESS/health criteria or caps.
   Do not add a terminal validation chain or reuse the failed warmup.

The primary criterion is the existing adaptive HMC criterion, not the replay
or mechanics result. Invalid forces, nonfinite targets, and declared energy
or divergence failures veto a candidate; exhausted budget or inconsistent
source/data/selection stops execution. Acceptance and timings are explanatory.
Passing does not establish an exact nonlinear posterior or superiority to
other filters. Independent endpoint approximation uncertainty remains.

The existing 86400-second cumulative GPU worker-wall budget is unchanged.
Wait for the active master to finish before writing its shared ledger; do not
spend its reservations twice. Up to three new worker attempts are needed
(tuning, mechanics, HMC); increase the administrative attempt ceiling only
by three if necessary, preserving the compute cap. Tuning cap 1800 seconds,
mechanics cap 1800 seconds, inference cap 32400 seconds, each clipped to the
then-available budget. Preserve all old evidence and use a new output root:
docs/plans/artifacts/surrogate-hmc-range-bearing-flow16-20261016-01.
If the existing queue consumes the allocation, record BUDGET_EXHAUSTED.

Skeptical audit: the baseline and saved-input failure are checked, the
override affects offline tuning only, and inference loads the resulting
selection. Changing the integration count changes the finite filter, so the
old selection is a warm start only. New calibration/validation and the actual
consumer checks are mandatory. The local replay is not promoted to posterior
evidence. No numerical-source change is needed, so the live campaign's
source-bound selections remain valid. The cheap diagnostic is selection
flow_steps=16 plus actual consumer mechanics; a later recurrence failure
triggers another repair rather than silent guard relaxation.

## Implementation review and execution

The preparation master now accepts `--flow-steps` only for fresh offline
tuning of one explicitly selected model. The advance controller passes 16
to tuning, then mechanics and inference consume the saved fresh selection.
The fixed epsilon=1/five-step OT candidate family preserves 16 flow steps.
Tests exercise this call chain, reject an inference-time override, verify
selection loading, and check that the continuation cannot spend an active
budget reservation. No numerical kernel or other model selection was changed.

CPU-only verification (CUDA intentionally hidden):

```sh
env CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q tests/test_surrogate_hmc_performance_repair.py tests/test_surrogate_hmc_five_model_master.py tests/test_surrogate_hmc_advance_continuation.py tests/test_surrogate_hmc_range_flow16.py --disable-warnings
```

Result: 194 passed, two warnings, 24.29 seconds. An initial test fixture
omitted its output directory; creating that fixture directory fixed the
failure. This was not a numerical failure. The supervisor CLI import/help
also passed with CUDA hidden. These checks establish wiring and budget
behavior; GPU numerical and posterior evidence remain pending.

The following supervisor waits for current master PID 493895, identified
by its process start time, before launching the three repair phases:

```sh
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_range_flow16.py --wait-pid 493895
```

It writes its own state while waiting and does not modify the campaign ledger.
After that master exits, unresolved RUNNING reservations require reconciliation.
Every numerical worker uses the existing GPU/XLA master with memory growth;
the supervisor contains no numerical implementation. The actual commands,
phase results, remaining-budget decisions, and launch provenance are recorded
under the new versioned output root. Remaining budget may be insufficient
for a completed posterior run; queuing is not a claim of completed HMC.
