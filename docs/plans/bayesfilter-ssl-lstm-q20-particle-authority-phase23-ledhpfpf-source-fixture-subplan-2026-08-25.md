# Phase 23 Source-Faithful Invertible LEDH-PFPF Fixture

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `PASS_SOURCE_FAITHFUL_LEDHPFPF_FIXTURE`  
Budget cap: `3600 s` within the unchanged `64800 s` campaign cap  
Output root: `docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase23`

## Objective

Implement the bounded linear-Gaussian LEDH-PFPF map from Li--Coates: repeated
affine pseudo-time steps, the product of step determinants, an inverse map, and
the proposal-density change-of-variables identity. This fixture tests the
density/Jacobian lifecycle that the q20 M3 scaffold lacks.

## Source contract

Use the local Li--Coates text at equations (6)--(16), (19)--(20), and Algorithm
1. The fixture binds a positive-definite prior covariance, positive-definite
observation covariance, fixed observation, and a declared ten-step pseudo-time
schedule. The schedule and matrices are fixture hypotheses, not q20 defaults.

## Evidence contract and skeptical audit

Hard gates: finite/positive-definite inputs; every affine step has finite
nonzero determinant; inverse round trip; product determinant agrees with the
composed map; and `log q_T(T(x)) = log q_0(x) - log|det DT|` for the recorded
proposal. Target log density and importance weights must be finite, but equality
to the posterior is not required or claimed for a discretized flow.

The strongest misleading pass is a flow that has a good-looking transformed
cloud but omits the determinant or uses a later reset. The fixture records both
the exact product and the density identity, and has no reset operation.

## Exact command

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_particle_authority_ledhpfpf_fixture_2026_08_25.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase23-attempt1-ledh-fixture
```

## Refresh

On pass, refresh Phase 24 to an actual q20 affine-flow/proposal identity probe
with explicit target terms. On failure, repair the smallest step, inverse, or
determinant equation. No q20 authority replacement, HMC, or default promotion
is in scope.
