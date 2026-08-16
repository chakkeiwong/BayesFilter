# q=20 CPU-XLA 32x1 HMC Tuning Repair

> Superseded 2026-08-02: this historical plan uses forbidden `L=1`
> candidates, a custom tuner instead of the fixed-transport API, and an
> unsupported finite log-accept hard veto. It must not be rerun or used as
> claim-bearing evidence.

Date: 2026-08-01
Tier: serious local CPU/XLA tuning repair
Status: `READY_TO_EXECUTE`

## Research Intent And Evidence Contract

| Role | Contract |
| --- | --- |
| Main question | Can shorter identity-mass HMC trajectories remove the rare extreme log-accept events seen in the first exact-payload tuning campaign while retaining a usable acceptance range for either trained chart? |
| Candidate mechanism | `L=1` reduces exposure to unstable multi-step trajectories; smaller `L=2` step sizes test whether the Chart A confirmation failure was a boundary-scale problem. |
| Exact baseline | Failed r1 grid and confirmation in `ssl-lstm-q20-cpu-xla-32x1-distributed-hmc-validation-2026-08-01/r1`; no r1 arm or sample is reused as repair evidence. |
| Prospective grid | `L=1` with steps `{0.75,0.875,1.0}` and `L=2` with steps `{0.50,0.525,0.55}`. Each arm receives two fresh independent 32-transition replications. |
| Promotion criterion | Same as r1: both replications finite/moving, no hard veto, and pooled mean acceptance per replication in `[0.55,0.85]`; deterministic selection targets `0.70`; independent 16-chain 64-transition confirmation requires every-chain acceptance in `[0.35,0.95]` and all hard screens. |
| Hard vetoes | Unchanged from r1, including `abs(log_accept_ratio)>1000`, invalid target status, nonfinite target/state/log-accept, exposed native divergence, unmoved chain, non-XLA, visible GPU, affinity drift, archive failure, aggregate RSS over 64 GiB, crash, or cap. |
| Repair trigger | No confirmed kernel for a chart triggers metric/mass adaptation. It does not justify more local step-size fishing or relaxing a veto. |
| Continuation veto | Shared target/checkpoint/transport/harness invalidity, resource cap, or missing diagnostics. |
| Artifact | `docs/plans/artifacts/ssl-lstm-q20-cpu-xla-32x1-hmc-tuning-repair-2026-08-01/r1/`. |
| Nonclaims | No convergence, retained posterior, posterior correctness, chart ranking, CPU default, GPU equivalence, or scientific-validity claim. |

## Default And Numerical Audit

| Choice | Provenance/status | Justification | Failure mode / early diagnostic |
| --- | --- | --- | --- |
| `L=1` | Repair hypothesis derived from r1 multi-step extreme energy errors | Tests shorter trajectories without changing target, chart, mass, or veto | Random-walk behavior; later ESS gate prevents promotion even if tuning passes |
| `L=1` steps | Derived bracket from r1 `L=2` acceptance at `0.75` and `1.0` | A larger step offsets the shorter trajectory while testing whether one leapfrog step avoids multi-step energy extremes | May still cross invalid regions; log-accept veto detects it |
| `L=2` steps | Derived refinement below r1 selected `0.565685` | Tests whether Chart A's rare confirmation extremes disappear before changing the metric | Chart B may remain above acceptance band; this is a clean metric-repair signal |
| Two replications | Inherited reviewed r1 requirement | R1 showed pooled means can hide replication-specific extreme events | Short chains still estimate event rates poorly; independent confirmation remains required |
| 7,200 s cap | Derived from measured r1 3,328.5 s for eight arms including `L=4`; reviewed convenience ceiling | Six `L<=2` arms plus confirmation should fit with ample compile margin | Cap fires before sequential sampling; no gate is relaxed |

Numeric provenance: all new step sizes are derived refinement hypotheses, not
promoted defaults. `0.50`, `0.525`, and `0.55` partition the interval below the
failed/fragile midpoint; `0.75`, `0.875`, and `1.0` span the r1 region where
two-step acceptance crossed the target while testing a one-step trajectory.

## Skeptical Audit And Pre-Mortem

- Wrong baseline: no; r1 exact-payload identity-mass tuning is the comparator.
- Proxy promotion: no; acceptance selects only a kernel candidate, and this
  repair stops after confirmation. It cannot make a retained-HMC claim.
- Hidden threshold relaxation: no; every r1 hard veto and acceptance band is
  unchanged.
- Local optimization drift: bounded. This is one prospectively frozen repair
  grid; failure ends step-size refinement and triggers metric adaptation.
- Misleading pass: `L=1` may pass acceptance but mix poorly. No sequential or
  posterior claim follows until later R-hat/ESS validation.
- Misleading failure: identity mass may be the problem. A failed repair rejects
  this kernel family, not the learned transports.
- Artifact adequacy: exact arm, replication, per-chain telemetry, confirmation,
  checkpoint identities, XLA/CPU affinity, RSS, command, hashes, and wall time
  are recorded.

Audit decision: `PASS_FOR_ONE_BOUNDED_SHORT_TRAJECTORY_REPAIR`.

## Command

```bash
CUDA_VISIBLE_DEVICES=-1 taskset -c 32 timeout 7200 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_cpu_xla_32x1_distributed_hmc_validation_2026_08_01.py \
  --tuning-profile short-trajectory-repair-v1 \
  --campaign-cap-seconds 7200 \
  --output-root \
  docs/plans/artifacts/ssl-lstm-q20-cpu-xla-32x1-hmc-tuning-repair-2026-08-01/r1
```
