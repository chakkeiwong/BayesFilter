# Corrected Parameter-Authority Phase 37 Support Ladder Subplan

Parent: `bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Entry gate: Phase 36 adjudication retained the theta measure and identified
finite-bank support as the smallest discriminating hypothesis  
Status: `PASS_SUPPORT_LADDER_HARD_GATES_NEUTRA_SUPPORT_STRESS_ROLE_LIMITED`
Local cap: 1800 s

## Question

Does increasing the fresh theta-bank size from 64 to 128 and 256 improve
proposal support, mode occupancy stability, and the corrected NeuTra boundary,
or do held-out moment residuals persist despite more particles?

## Evidence contract

Use the same q=20 target, theta measure, proposal family, beta schedule,
defensive component, target signature, and no-replay rule as Phase 28. Generate
fresh banks at `N=64,128,256` with seeds `20260825 3701,3702,3703`; preserve
each root. For each bank record finite/status gates, ESS, weighted mode
fractions, proposal/target log terms, support diagnostics, and the fixed
measure hash. Aggregate the three roots with the size-aware reporter, then run
the batch-native NeuTra boundary screen on the nominated N=256 M0 bank. Do not
pool incompatible banks or call N=256 superior.

The N values and seeds are a reviewed ladder hypothesis, not promoted defaults.
Metrics are descriptive except for hard shape/finite/status/support gates. No
ranking is claimed from one seed per size.

## Pre-mortem

- A larger bank can repeat the same mode-biased proposal; retain support and
  ancestry diagnostics rather than interpreting ESS as coverage.
- Runtime may grow without adding useful support; record wall time and stop if
  the local cap is reached.
- A candidate NeuTra screen can pass finite gates while held-out whitening
  remains poor; preserve the no-IID and no-posterior nonclaims.
- A size-dependent protocol hash or proposal change would invalidate the
  comparison; fail closed on signature mismatch.

## Commands

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase28_2026_08_25.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/n64 \
  --particles 64 --calibration-particles 16 --arms both --seed 20260825 3701

CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase28_2026_08_25.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/n128 \
  --particles 128 --calibration-particles 32 --arms both --seed 20260825 3702

CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase28_2026_08_25.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/n256 \
  --particles 256 --calibration-particles 64 --arms both --seed 20260825 3703

python docs/benchmarks/aggregate_ssl_lstm_q20_parameter_authority_corrected_phase37_support_ladder_2026_08_25.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/aggregate \
  --pilot-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/n64 \
  --pilot-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/n128 \
  --pilot-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/n256

TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase31_2026_08_25.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/n256 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/neutra-identity \
  --precondition identity --steps 200 --seed 20260825 3711

TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase31_2026_08_25.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/n256 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/neutra-affine \
  --precondition affine --steps 200 --seed 20260825 3711
```

## Repair and refresh

Repair a harness or signature failure in a fresh root under unchanged criteria.
Low ESS or mode variation is descriptive. If all sizes fail common support,
that is a continuation veto for this proposal family and triggers a new
proposal-design plan; it is not a theorem about the q=20 target.

Execution receipt: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase37-result-2026-08-25.md`.
The aggregate first failed because the reporter used the wrong calibration
field (`particles` instead of `particle_count`); the repaired attempt was
written under `phase37-support-ladder/aggregate-attempt2/`. All three pilot
roots and both N=256 boundary screens then passed their hard gates. Persistent
held-out moment residuals are a tuning/representation repair trigger, not a
continuation veto.
