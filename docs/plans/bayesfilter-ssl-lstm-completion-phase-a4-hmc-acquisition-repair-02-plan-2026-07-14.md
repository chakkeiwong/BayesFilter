# Phase A4 HMC Acquisition Repair-02 Plan

Date: 2026-07-14 (Asia/Shanghai)

Status: `AUTHORIZED_PROSPECTIVE_ADAPTATION_REPAIR`

## Authorization

The owner authorized one fresh repair after the repair-01 smaller-step tuning
candidate was rejected for over-acceptance. This repair uses the repository's
reviewed fixed-mass dual-averaging path with target acceptance `0.70`, the
original four starts, fresh artifacts, and the remaining shared A4 GPU budget.
Forecast calibration remains conditional on a fresh admitted retained archive.

## Research Question

Does fixed-mass dual averaging, targeting acceptance `0.70` during warmup,
produce a finite four-chain HMC path whose frozen post-warmup step passes a
broad repair screen and then an unchanged A4 retained-admission gate?

## Mechanism And Frozen Settings

| Field | Value |
| --- | --- |
| Target | Locked A1 `ssl_lstm_completion:a1:masked_svd_ukf_four_parameter` |
| Starts | Original four dispersed latent starts from A0 |
| Initial step | `0.3925`, the prior balanced candidate |
| Leapfrog steps | `4`, unchanged during adaptation and retention |
| Initial trajectory length | `1.57`; the retained trajectory becomes `4 * frozen_step_size` after adaptation |
| Adaptation | TFP `DualAveragingStepSizeAdaptation`, fixed mass, `256` adaptation steps |
| Warmup | `320` transitions; adaptation ends before retained screen |
| Target acceptance | `0.70` |
| Adaptation screen | `64` post-warmup draws; diagnostic/private only |
| Frozen-step safety interval | `[1e-4, 2.0]`; finite scalar only |
| Repair acceptance band | Aggregate in `[0.55,0.85]`; every chain in the broad safety interval `[0.20,0.95]`; target `0.70` is descriptive, not a post-hoc ranking rule |
| Retained gate | Fresh `250` burn-in plus `250` retained draws; existing A4 movement, acceptance `[0.20,0.95]`, R-hat, ESS, MCSE, finiteness, and divergence gates unchanged |
| Execution | TensorFlow/TFP, `float64`, GPU/XLA, trusted managed session |

Adaptation-screen samples are private diagnostics and are not A4 calibration
draws. The retained run starts from the final adaptation-screen state only;
no failed repair-01 state, sample, or seed is read.

## Evidence Contract

| Field | Requirement |
| --- | --- |
| Primary question | Does adaptation produce a usable frozen kernel and an admitted retained archive? |
| Exact baseline | The locked A1 target and original starts; repair-01 is failure context and budget lineage only |
| Adaptation promotion | Finite moving chains; finite scalar frozen step; step trace constant after warmup; aggregate acceptance `[0.55,0.85]`; every chain acceptance `[0.20,0.95]`; finite target/log-accept telemetry; GPU/XLA placement |
| Retained promotion | Existing A4 gate: all chains move; acceptance `[0.20,0.95]`; max rank-normalized split R-hat `<=1.05`; bulk/tail ESS `>=100`; mean MCSE/SD `<=0.10` in latent and free coordinates |
| Hard vetoes | Target/source/geometry drift; malformed/nonfinite artifacts; unmoved chain; non-scalar or unsafe frozen step; changed step after warmup; positive native divergence if exposed; budget exhaustion; archive collision; failed receipt/hash lineage |
| Explanatory diagnostics | Target acceptance, realized acceptance, final step, step trace spread, runtime, finite log-accept extrema, target range, and device placement |
| Nonclaims | No posterior correctness, convergence proof, sampler superiority, predictive equivalence, NeuTra readiness, model adequacy, or default readiness |
| Preservation | Public JSON receipts plus private TensorFlow shards under `phase-a4/hmc-acquisition/repair-02/` |

## Sequential Execution

1. Run CPU-hidden focused tests and source/hash checks.
2. Run exactly one trusted GPU/XLA dual-averaging adaptation screen.
3. If adaptation is not selected or any hard veto fires, write a blocker and stop.
4. If selected, run exactly one fresh `250/250` retained archive from the final adaptation-screen state using the frozen step.
5. If retained admission passes, refresh the A4 forecast-calibration live plan with exact runnable counts and seed separation before opening calibration data; then run the smallest calibration nomination rung only.
6. Stop HMC immediately on retained admission. Stop all work on any hard veto, invalid artifact, or budget projection failure.

## Budget And Lineage

The shared cap is `28800s`. Prior trusted GPU receipts consume
`1556.734474526951s`, leaving `27243.26552547305s` before repair-02. The
repair runner verifies exact SHA-256 values for all prior receipts, refuses
collisions, and charges every trusted wall time. The failed repair-01 tuning
shard is diagnostic only and is never reused as state or samples.

## Artifacts

| Artifact | Path |
| --- | --- |
| Runner | `docs/benchmarks/run_ssl_lstm_a4_hmc_repair_02_2026_07_14.py` |
| Tests | `tests/test_ssl_lstm_a4_hmc_repair_02.py` |
| Adaptation receipt | `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/repair-02/adaptation.json` |
| Retained receipt | `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/repair-02/segment-0.json` |
| Result | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-hmc-acquisition-repair-02-result-2026-07-14.md` |
| Review | `docs/reviews/bayesfilter-ssl-lstm-a4-hmc-repair-02-native-review-2026-07-14.md` |

## Exact Commands

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a4-repair02-pycache \
  TMPDIR=/tmp/bayesfilter-a4-repair02-tmp \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_ssl_lstm_a4_hmc_repair_02.py

PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a4-repair02-pycache \
  TMPDIR=/tmp/bayesfilter-a4-repair02-tmp \
  CUDA_CACHE_PATH=/tmp/bayesfilter-a4-repair02-cuda-cache \
  XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_a4_hmc_repair_02_2026_07_14.py adapt

PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a4-repair02-pycache \
  TMPDIR=/tmp/bayesfilter-a4-repair02-tmp \
  CUDA_CACHE_PATH=/tmp/bayesfilter-a4-repair02-cuda-cache \
  XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_a4_hmc_repair_02_2026_07_14.py segment
```

The final command is eligible only when `adaptation.json` is `SELECTED` and
passes its full replay/hash contract.

## Skeptical Audit

| Challenge | Disposition |
| --- | --- |
| Wrong baseline | Locked A1 target and original starts remain fixed; prior failures are not scientific comparators |
| Proxy promotion | Adaptation acceptance only admits the frozen kernel to the retained gate; it cannot admit calibration |
| Missing stop | Non-selected adaptation or any retained hard veto stops the repair; no seed search or threshold relaxation |
| Confounded mechanism | Leapfrog count remains fixed at `4`; dual averaging changes step size and therefore trajectory length by design. The receipt records both the frozen step and resulting trajectory length |
| Hidden state reuse | Final adaptation state is a fresh private artifact; repair-01 state/shards are never read |
| Resource risk | Prior receipt wall times and conservative projections are checked before each GPU call |
| Misleading pass | A retained pass admits only A4 calibration input; it does not establish posterior or sampler correctness |

Audit decision: `PASS_FOR_CPU_CHECKS_AND_ONE_TRUSTED_ADAPTATION_SCREEN`.

## Forbidden Claims And Actions

- Do not call adaptation acceptance a convergence or posterior-correctness result.
- Do not pool adaptation-screen draws into calibration or retained diagnostics.
- Do not alter target acceptance, warmup length, leapfrog count, starts, or
  retained thresholds after seeing the adaptation receipt.
- Do not reuse repair-01 samples/final state or select another seed after a
  failed adaptation screen.
- Do not run forecast calibration, A5, NeuTra, or NeuTra-HMC without retained
  admission.
