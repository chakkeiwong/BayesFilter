# Phase 26 Reduced-Coordinate LEDH Boundary and Density Fixture

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `REDUCED_MECHANICS_PASS_TARGET_BINDING_VETO`  
Budget cap: `3600 s` within the unchanged `64800 s` campaign cap  
Output root:
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase26`

## Objective

Verify the smallest mathematically valid repair after Phase 25: represent the
20-dimensional innovation measure with an explicit invertible coordinate map,
including its density and determinant, and then test whether that reduced
measure binds to the four-dimensional q=20 parameter posterior. The fixture is
mechanics-only. It must not be promoted to a q=20 LEDH proposal.

## Evidence contract

| Item | Rule |
|---|---|
| Comparator | Phase 25 q=20 target and its exact innovation covariance |
| Primary criterion | reduced-coordinate inverse round trip and change-of-variables density identity |
| Promotion veto | nonfinite Cholesky/determinant, failed inverse, or omitted determinant |
| Binding veto | no explicit map from the four-parameter target to the reduced innovation proposal |
| Explanatory diagnostics | covariance spectrum, log determinant, target value/score finiteness |
| Nonclaims | no source-faithful q=20 LEDH, posterior correctness, mode discovery, whitening, HMC, or default promotion |
| Artifact | unique JSON/Markdown receipt with source hashes and binding decision |

## Procedure

1. Instantiate the q=20 target in the CPU/reference lane and obtain its
   positive-definite 20-dimensional innovation covariance `Q`.
2. Use the Cholesky factor `L` as a declared reduced-coordinate map
   `x = L u`, with `u` standard Gaussian. Compute the inverse, log determinant,
   and both sides of `log q_x(x) = log q_u(u) - log|det L|`.
3. Evaluate the aggregate four-parameter target separately and record that no
   target-to-innovation proposal callback exists.
4. Classify the fixture as a mechanics pass but a target-binding failure. Do
   not silently reinterpret the innovation flow as parameter-space LEDH.

## Assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
| Cholesky map | covariance field in the actual target | only represents innovation measure | dimension/binding ledger | diagnostic fixture |
| standard Gaussian base | explicit fixture definition | not the q=20 posterior | target-binding boolean | hypothesis, not default |
| fixed 20D points | deterministic mechanics check | misses state dependence | finite/inverse receipts | diagnostic only |
| CPU-hidden TensorFlow | reference-lane policy | no GPU claim | device receipt | reference exception |

## Pre-mortem

- A valid reduced density could be reported as a valid q=20 parameter proposal.
  The binding veto and separate dimensions prevent that.
- A determinant could be omitted or given the wrong sign. The inverse and
  change-of-variables residual are hard checks.
- The covariance could be positive definite only at one point. The target
  covariance and its eigenvalue receipt are recorded; no global claim follows.

## Exact command

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_particle_authority_ledh_reduced_fixture_2026_08_25.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase26-attempt1
```

## Refresh/stop rule

If mechanics fail, repair the fixture and rerun under the same scope. If
mechanics pass but target binding is absent, close direct q=20 LEDH as wrong
relative to the stated parameter-particle target and retain the fixture as
historical diagnostic evidence. This route-specific blocker does not invalidate
the ETPF, SMC, GenUT, or NeuTra results.

## Executed receipt

The prescribed command completed in `7.6 s`. The reduced Cholesky map passed
its inverse and density identities (`5.55e-17` and `0` residuals), but the
target-binding veto fired because the map is 20-dimensional innovation space
and the declared target is a four-dimensional aggregate parameter posterior.
Direct q=20 LEDH is closed relative to this target.
