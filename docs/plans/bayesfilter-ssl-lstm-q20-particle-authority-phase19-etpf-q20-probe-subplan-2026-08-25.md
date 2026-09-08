# Phase 19 Small q=20 ETPF Integration Probe

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `PASS_ETPF_Q20_PROBE_ROLE_LIMITED`  
Budget cap: `3600 s` within the unchanged `64800 s` campaign cap  
Input: Phase 8 audited N=300 bank, deterministic 32-row subset, and Phase 18 fixture implementation  
Output root: `docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase19`

## Objective

Apply the source-faithful second-order LETF/ETPF map to a small deterministic
subset of the actual q=20 authority bank and check target/status validity of
the transformed rows. This is an integration probe, not a claim-bearing q=20
filter. The subset is a computational hypothesis chosen to keep the Riccati
matrix bounded; it cannot estimate q=20 posterior mass or mode probabilities.

## Frozen input and selection

Load the Phase 8 metadata-bound bank, verify protocol hash/target signature and
`mode_axis=2`, then select 32 rows at deterministic evenly spaced indices and
renormalize their retained weights. Record the index list and tensor hash. The
raw N=300 bank remains immutable and is not replaced.

## Evidence contract

Primary gates are source transform constraints (finite, converged Riccati,
row/column marginals, mean/covariance residual), target value/score/status
finite and valid for every transformed row, and complete input/output hashes.
Support-range excursions, mode occupancy, target values, covariance residual,
runtime, and negative correction fraction are explanatory only.

Vetoes are stale input metadata, non-finite values, transform constraint
failure, target/status failure, or missing artifact fields. A q=20 probe failure
triggers an implementation/scale repair; it does not reject ETPF or the fresh
authority and does not authorize a larger run.

## Pre-mortem and assumptions

- The selected subset may be atypical; no population claim is allowed.
- Second-order corrections may produce invalid target rows or bridge points;
  those are recorded rather than clipped.
- The target call may be expensive or unsupported on transformed points; this
  is a target-integration diagnostic, not a reason to silently change the
  target.
- N=32 and regularization `10` are inherited fixture hypotheses, not defaults;
  the earliest check is the source fixture and metadata hash.

## Exact command

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_particle_authority_etpf_q20_probe_2026_08_25.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase6-attempt9-metadata-n300-seed2401 \
  --riccati-tolerance 1e-5 --riccati-max-steps 5000 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase19-attempt2
```

## Refresh

If the probe passes, refresh Phase 20 to a separately audited GenUT or LEDH
source-faithful fixture, not an automatic N=300 ETPF run. If it fails, preserve
the exact transformed rows and classify target/integration versus equation
failure before any retry.
