# Corrected q=20 Fresh Theta Pilot Phase 28 Subplan

Parent: `bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Entry receipt: Phase 27 `PASS_CORRECTED_PARAMETER_MEASURE_CONTRACT`  
Status: `READY_TO_EXECUTE`  
Local cap: 14400 s from the remaining campaign pool

## Question

Can a fresh paired C0/M0 pilot maintain an explicit proposal and target in the
declared `theta in R^4` measure through tempering and resampling?

## Design

- Draw fresh stateless particles from a two-component local Gaussian proposal
  plus, for M0 only, a defensive broad Gaussian component. The prior geometry
  file is a calibration warm start and is hash-bound; its particles are not
  reused.
- Evaluate the batch-native q=20 target directly on `[N,4]` theta rows.
- Use a fixed, predeclared beta schedule for this pilot. It is a hypothesis,
  not an adaptive theorem or promoted default.
- Apply systematic resampling at nonterminal stages. The identity mutation is
  the invariant reference kernel; no finite-run mixing claim is made.
- Store `target_log_theta` and `proposal_log_theta` without a chart Jacobian.
  A chart is sampling metadata only.

## Evidence contract

Primary gates are finite/status-valid rows, finite theta-density terms,
protocol hash, beta-one reachability, and unique ancestry/weight receipts.
The M0 label remains a candidate role and is not an SMC-U proof. ESS, mode
occupancy, and log-mass are descriptive. A failure triggers a fresh repair
root, never a relabeling of the old state-space LEDH evidence.

## Command

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase28_2026_08_25.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase28-fresh-theta-pilot \
  --particles 64 --calibration-particles 16 --arms both
```

## Repair and refresh

Classify failures as harness, target, density/measure, numerical, or
candidate. Repair only the smallest cause under the unchanged target and
budget. The next phase is Phase 29 fresh-theta ETPF and must consume the
versioned M0 receipt, not a parent bank. If support or mass diagnostics are
poor, retain the run and test a declared proposal repair; do not call it a
whole-direction blocker.

