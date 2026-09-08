# Phase 20 Source-Faithful GenUT Sigma-Point Fixture

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `PASS_SOURCE_FAITHFUL_GENUT_FIXTURE`  
Budget cap: `3600 s` within the unchanged `64800 s` campaign cap  
Output root: `docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase20`

## Objective

Implement the bounded Ebeigbe et al. `2d+1` GenUT construction using the
standardized diagonal skewness and kurtosis equations (13)--(34) and verify
mean, covariance, diagonal third, and diagonal fourth moment receipts on a
finite weighted fixture. This is a quadrature/representation candidate only.

## Fixture and numeric provenance

The fixture is a 5-by-5 tensor-product grid with axis values `[-2,-1,0,1,2]`
and axis weights `[0.04,0.22,0.40,0.25,0.09]`. This is a bounded diagnostic
hypothesis chosen after the first attempted cloud failed the paper's positive
central-weight feasibility condition. The measured standardized skewness is
about `0.05031` per axis and kurtosis about `2.56191`, yielding positive central
weight (about `0.21856`). These values are recorded, not promoted as defaults.

## Evidence contract and skeptical audit

Hard gates: finite points/weights; positive discriminants, offsets, and central
weight; weights sum to one; and all four selected moment residuals at or below
`1e-8` in float64. A feasibility failure is a mathematical/fixture repair
trigger. Moment success cannot establish a density, global mode coverage, or
IID samples.

The audit specifically checks that the asymmetric offsets differ when measured
skewness is nonzero. A symmetric rule passing covariance would not pass this
source-identity gate.

## Exact command

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_particle_authority_genut_fixture_2026_08_25.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase20-attempt1-genut-fixture
```

## Refresh

On pass, refresh Phase 21 to a q20 GenUT proposal/status probe only after
checking feasibility on the actual bank. On failure, preserve the measured
discriminant/weight evidence, repair the smallest equation or fixture choice,
and rerun in a fresh directory. No HMC or default promotion is in scope.
