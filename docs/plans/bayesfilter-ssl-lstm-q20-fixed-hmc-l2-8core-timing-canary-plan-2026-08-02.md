# q=20 Fixed-HMC L=2 Eight-Core Timing Canary

Date: 2026-08-02  
Status: `COMPLETED`

## Research Intent Ledger

| Field | Predeclared value |
| --- | --- |
| Main question | How long does the unchanged Chart A public-API `L=2` tuning workload take when constrained to eight CPU cores? |
| Candidate under test | CPU affinity `32..39` and TensorFlow intra-op thread count `8`. |
| Exact baseline | The prior Chart A 16-core run: `1806.6523481040495 s`, CPUs `0..15`, otherwise the same harness configuration. |
| Expected failure mode | Eight cores may be slower, and unrelated concurrent CPU work may inflate or destabilize the observed wall time. |
| Primary timing result | Completed process wall time from the harness summary; report its ratio to the 16-core baseline descriptively. |
| Promotion criterion | None. This is a resource-sizing canary, not kernel or default promotion evidence. |
| Promotion veto | Any attempt to use acceptance or completion to admit an HMC kernel or select a leapfrog length. |
| Continuation veto | Timeout, crash, nonfinite required telemetry, invalid target status, CPU affinity/thread mismatch, visible GPU, XLA disabled, source/config mismatch, or artifact collision. |
| Repair trigger | A continuation veto or missing manifest field requires harness diagnosis before extrapolation. |
| Explanatory diagnostics | Acceptance, tuned step sizes, compile messages, CPU utilization, host load, and finite log-accept/energy tails. |
| Must not be concluded | Full five-point grid time, chart ranking, kernel admission, sampler convergence, posterior validity, GPU comparison, or default-readiness. |

## Evidence Contract

- Engineering question: measure the exact full `L=2` ladder on Chart A with
  eight pinned CPU cores.
- Comparator: the Chart A 16-core artifact at
  `docs/plans/artifacts/ssl-lstm-q20-fixed-hmc-api-cpu-xla-validation-2026-08-02/r4-tuning/chart-a/summary.json`.
- Primary result: wall time and the derived ratio `eight_core / sixteen_core`.
- Hard vetoes: process/runtime failure, nonfinite required target or sampler
  telemetry, invalid target status, GPU visibility, XLA disabled, or execution
  metadata inconsistent with eight cores.
- Explanatory only: short-screen acceptance, tuned steps, energy/log-accept
  tails, and the single observed timing ratio.
- Artifact: unique output root
  `docs/plans/artifacts/ssl-lstm-q20-fixed-hmc-l2-8core-timing-canary-2026-08-02/r1/`
  plus a result note beside this plan.

## Default And Assumption Audit

| Choice | Provenance and status | Justification | Failure mode | Early diagnostic |
| --- | --- | --- | --- | --- |
| Chart A | Measured comparator baseline | Holds checkpoint, transport, target, and chart fixed against the `1806.65 s` run | Does not prove Chart B timing | Report Chart A-only nonclaim |
| `L=2` | User-requested diagnostic candidate | Directly answers the timing request and matches the baseline | Not representative of larger `L` if cost is nonlinear | Do not extrapolate from this artifact alone |
| Eight cores | User-requested candidate | Tests proposed per-`L` allocation | TF/XLA may not scale monotonically | Record affinity and configured threads |
| CPUs `32..39` | Convenience-chosen disjoint affinity range | Avoids the earlier campaign's `0..31` ranges | Other unpinned host work may still contend | Capture pre/post host process snapshots |
| Chart A 16-core baseline `1806.6523481040495 s` | Measured | Same harness and chart | Dirty source or host-load drift can weaken comparison | Bind commit, harness/tuner hashes, and note drift |
| `(8,16,32)` DA budgets, 8 tune draws, 16 screen draws | Inherited unchanged harness baseline | Exact workload comparability | Scientifically too short for general tuning conclusions | Acceptance remains descriptive only |
| Four chains in one rank-2 batch | Inherited unchanged public tuner route | Exact topology comparability | Does not measure independent per-chain processes | Record chain count and execution mode |
| CPU-only, FP64, XLA on | Repository policy plus baseline | Required path under test | Silent device/JIT drift invalidates timing comparison | Manifest and XLA log check |
| `3600 s` cap | Derived: approximately two times the 16-core baseline | Bounds the canary while allowing slower 8-core execution | A slower valid run may time out | Preserve timeout as under-budgeted timing evidence only |

## Skeptical Plan Audit

- Wrong baseline: avoided by comparing only Chart A with its own prior full
  `L=2` ladder, not Chart B or a sequential-sampling canary.
- Proxy promotion: wall time and short acceptance are not kernel-promotion
  evidence.
- Stop conditions: explicit timeout, finite/status, device, XLA, affinity, and
  artifact-integrity vetoes are present.
- Fairness: harness, checkpoint, transport, seeds, ladder, chains, dtype, and
  XLA configuration are held fixed. Core allocation is the intended change.
- Hidden assumption: system load differs because unrelated agents have active
  CPU workloads. This makes the ratio a measured shared-host observation, not
  an isolated scaling law.
- Environment mismatch: the command pins the existing `tfgpu` environment but
  deliberately hides CUDA before TensorFlow import.
- Artifact adequacy: the harness writes the full tuning artifact and summary,
  which contain the fields needed to verify the workload and timing.

The plan passes for a bounded shared-host timing canary. It does not pass as a
plan for multi-`L` tuning, kernel admission, or an isolated CPU scaling study.

## Exact Command

```bash
CUDA_VISIBLE_DEVICES=-1 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
TF_NUM_INTRAOP_THREADS=8 \
TF_NUM_INTEROP_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
timeout 3600s taskset -c 32-39 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_fixed_hmc_api_cpu_xla_validation_2026_08_02.py \
--mode chart-tuning \
--chart chart-a \
--threads 8 \
--output-root \
docs/plans/artifacts/ssl-lstm-q20-fixed-hmc-l2-8core-timing-canary-2026-08-02/r1/chart-a
```
