# BayesFilter PP-UKF true-HMC restart reset memo

Date: 2026-07-22 (Asia/Hong_Kong)

Branch at reset: `main`

Pre-reset base: `9303ed7`

Campaign plan: `docs/plans/bayesfilter-pp-ukf-true-hmc-validation-plan-2026-07-22.md`

## Restart state

The detached session `pp_ukf_hmc_24h_20260722` was stopped before repository
cleanup. Terminal `Ctrl-C` did not interrupt the compiled TensorFlow workload,
so the verified Python PID was sent `SIGTERM`. The wrapper recorded exit code
`143`. The tmux server and Python process are no longer running.

The last durable checkpoint is:

```text
docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-08/progress.json
```

Its SHA-256 remained unchanged before and after termination:

```text
8590b64b48581d3bb13f8a8c02aa2dee323d6d842a87f83909a8063d3a5c391d
```

The progress file is mutable restart state and is intentionally ignored by
Git. This memo preserves its identity and interpretation. Do not delete or
overwrite attempt 08.

## Completed candidates

| `L` | Decision | Warm-up per chain | Retained per chain | Hard veto |
|---:|---|---:|---:|---|
| 5 | admitted | 2,000 | 1,000 | none |
| 9 | rejected: retained convergence screen | 2,000 | 3,000 | none |
| 12 | rejected: retained convergence screen | 2,000 | 3,000 | none |
| 13 | admitted | 2,000 | 1,000 | none |
| 14 | admitted | 2,000 | 2,000 | none |

These rows are durable and must not be rerun. The in-flight `L=17` evaluation
did not reach a checkpoint and therefore must be rerun. Remaining candidates
are `L=17,18,19,24,25`.

## Budget accounting

The authorized aggregate campaign cap is 86,400 seconds (24 hours).

| Charge | Seconds |
|---|---:|
| Attempts before attempt 08 | 8,891.885516 |
| Attempt 08 through five-row checkpoint | 6,676.976188 |
| Post-checkpoint work through termination | 1,855.849831 |
| Aggregate charge to carry forward | 17,424.711535 |
| Remaining campaign budget | 68,975.288465 |

The post-checkpoint charge is the interval from the checkpoint modification at
`2026-07-22 15:40:43.078972 +08:00` to the recorded exit-code modification at
`2026-07-22 16:11:38.928709 +08:00`. It is charged even though the in-flight
candidate was not durable. This accounting is deliberately conservative: the
harness checkpoint counter and filesystem interval imply an attempt-08 charge
about 266 seconds larger than the output-root-to-exit wall interval. The larger
charge is carried forward rather than silently spending beyond the cap.

## Diagnostic correction

Finite extreme log acceptance is explanatory only. It is not a divergence or
energy veto. Hard vetoes are non-finite state, target, or log acceptance;
invalid target telemetry; no movement across every chain; and native
divergence only when the kernel exposes native divergence telemetry. The
correction and focused test evidence are recorded in
`docs/plans/bayesfilter-pp-ukf-energy-veto-correction-2026-07-22.md`.

The claimed target is true fixed-transport PP-UKF HMC validation. The current
checkpoint contains sequential HMC convergence screens for five frozen tuning
candidates. It does not rank viable candidates statistically, prove posterior
correctness, establish sampler superiority, or support default readiness.

## Restart command

Run only after checking that no old PP-UKF validation process or tmux session
exists. GPU access must be trusted/escalated and memory growth must be set before
TensorFlow import.

```bash
tmux new-session -d -s pp_ukf_hmc_24h_20260722_restart \
  "cd /home/chakwong/BayesFilter && \
  TF_FORCE_GPU_ALLOW_GROWTH=true \
  python docs/benchmarks/run_pp_ukf_true_hmc_validation_20260722.py \
    --output-root docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-09 \
    --resume-progress docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-08/progress.json \
    --prior-elapsed-seconds 17424.711535 > \
    docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-09-launch.log 2>&1"
```

The driver identifies completed candidates by candidate ID, so resuming from
attempt 08 skips `L=5,9,12,13,14`. Attempt 09 must be a fresh output root.
The driver was repaired during this reset so both incremental and terminal
progress writes preserve resumed rows and retain the global ten-candidate
planned count; focused regression tests cover that behavior.

## Repository hygiene boundary

Track authored source, tests, benchmark harnesses, plans, result notes, compact
public results, run manifests, selection decisions, verification evidence, and
local paper/author-code copies required to ground scientific claims. Ignore raw
sampler/training tensors, NumPy state, event streams, mutable checkpoints and
progress, private payloads, logs, local databases, and complete source
snapshots. `.localresources` is tracked because project policy requires a local
copy of papers and official code that materially support implementation or
source-faithfulness claims.

## Decision and inference status

| Question | Status | Next justified action |
|---|---|---|
| Engineering restart | ready | resume from the hash-identified checkpoint in fresh attempt 09 |
| Primary convergence criterion | mixed after five candidates | run the five remaining frozen candidates |
| Hard-veto screen | no hard veto in the five durable rows | continue under unchanged criteria |
| Statistically supported ranking | none | do not rank passing candidates from descriptive diagnostics |
| Default readiness | not established | requires completed validation and the plan's downstream evidence |

Strongest alternative explanation: current admissions and rejections may be
short-chain stochastic outcomes rather than stable candidate differences. A
completed campaign with declared uncertainty-aware comparison is needed before
any ranking. The reset changes no target, data, frozen controls, thresholds,
hardware class, or scientific claim.
