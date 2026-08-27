# Phase 18 Source-Faithful Second-Order LETF/ETPF Fixture

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `PASS_SOURCE_FAITHFUL_ETPF_FIXTURE`  
Budget cap: `3600 s` within the unchanged `64800 s` campaign cap  
Output root: `docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase18`

## Objective

Implement and test the smallest source-faithful Acevedo et al. second-order
LETF/ETPF slice on a deterministic eight-particle, two-dimensional fixture:

1. regularized Sinkhorn transport with the paper's row/column marginal roles;
2. finite-iteration row-mass correction;
3. explicit-Euler solution of the paper's dynamic Riccati equation; and
4. analysis ensemble `Z_f (D + Delta)` with first/second moment receipts.

This is a fixture candidate only. It does not change the q=20 authority, does
not create IID samples, and does not authorize a q=20 or HMC route.

## Source and numeric provenance

The operation order and equations are anchored to the local Acevedo text at
equations (16), (20), (26), (42)--(44), and (48)--(57). The explicit Euler
step `0.1` and stopping increment `1e-3` are source-reported implementation
values (equations/algorithm summary around (56)), not newly tuned defaults.
The fixture uses eight particles, two dimensions, regularization `10`, and
400 Sinkhorn iterations as bounded diagnostic hypotheses chosen to expose
finite-iteration behavior; they are not q=20 defaults.

## Evidence contract and skeptical audit

**Primary gates:** finite tensors; converged Riccati iteration; corrected
transport column sums `1` and row sums `Nw`; equal-weight analysis mean equal to
the weighted forecast mean; covariance residual no larger than `1e-3`, matching
the source stopping tolerance. The source base transport's nonnegativity and
the corrected transform's negative fraction are recorded.

**Vetoes:** missing source/hash, non-finite values, marginal residual failure,
or a result that calls the fixture IID/posterior evidence. A fixture failure
triggers equation-level repair; it does not reject the ETPF idea.

**Pre-mortem:** Sinkhorn marginal drift may be mistaken for a Riccati failure;
the artifact records both. The correction may leave the convex hull, which is a
known source behavior and must not be hidden. Passing moments may still not
identify a density, so no density claim is allowed.

## Exact command

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_particle_authority_etpf_fixture_2026_08_25.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase18-attempt2-etpf-fixture
```

## Exit and refresh

On a pass, refresh Phase 19 toward a bounded q=20/N-small integration only
after checking memory and source-operation scope. On failure, preserve the
unique artifact, repair the smallest failing equation/receipt, and retry with
a new directory. No method ranking or HMC is in scope.
