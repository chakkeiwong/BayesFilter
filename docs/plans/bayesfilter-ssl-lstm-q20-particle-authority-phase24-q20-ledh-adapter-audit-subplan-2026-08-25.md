# Phase 24 q=20 LEDH State-Space Adapter/Lifecycle Audit

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `ADAPTER_NOT_READY_REPAIRABLE`  
Budget cap: `3600 s` within the unchanged `64800 s` campaign cap  
Input: q20 target implementation, Phase 23 LEDH fixture, and Phase 3 M3 scaffold  
Output root: `docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase24`

## Objective

Determine whether the SSL-LSTM q20 target exposes enough explicit state-space
callbacks to bind the Li-Coates LEDH-PFPF proposal contract: pre-flow proposal,
transition, observation, covariance lifecycle, pseudo-time step matrices,
determinant product, and post-flow target terms. Inspect the target's actual
public/private interfaces and record a source/code hash-bound decision.

This phase is an interface/admissibility audit. It must not substitute a
parameter-space affine map and call it LEDH.

## Evidence contract

For each required term, record an exact local symbol/path or `missing`, the
quantity available, and whether it can be evaluated batch-natively for q20.
The primary decision is `ADAPTER_READY` only if every required term is bound to
the same target and a fixture-compatible density lifecycle. Otherwise return
`ADAPTER_NOT_READY_REPAIRABLE` and define the smallest adapter work.

Hard vetoes are source/hash mismatch, contradictory target signatures, or an
attempt to promote an extension as source-faithful. Missing public callbacks
are a repair trigger unless the target/source identity is unavailable under the
master program's real-blocker definition.

## Pre-mortem

- Internal state-space code may exist but not expose the proposal law or
  covariance lifecycle; presence alone is not enough.
- A parameter gradient may be mistaken for a state Jacobian. Record dimensions
  and total/partial derivative roles explicitly.
- A successful affine fixture cannot close a q20 adapter gap.

## Exact command

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_particle_authority_ledh_adapter_audit_2026_08_25.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase24-attempt1
```

## Executed receipt

The prescribed CPU command completed in `7.1 s` at:

`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase24-attempt1/`

The target exposed finite batched structural callbacks with dimensions
`batch=2`, `points=3`, `state=60`, `innovation=20`, and `observation=1`.
Initial, innovation, and observation covariance tensors were bound. The audit
found no explicit transition or observation log-density callback, no frozen
pre-flow proposal density, no returned per-step covariance lifecycle, and no
LEDH pseudo-time matrix/determinant API. The aggregate value/score method does
not decompose those terms. The result is
`ADAPTER_NOT_READY_REPAIRABLE`; no LEDH admission was granted.

## Refresh

Phase 25 must test the smallest explicit density decomposition against the
actual target measure. It must check whether the structural transition is
singular in the 60-dimensional state coordinates and whether a reduced
innovation-coordinate proposal can be bound without changing the q=20
parameter target. Do not label a private structural callback or a
parameter-space affine map source-faithful LEDH. This is not a reason to launch
HMC.
