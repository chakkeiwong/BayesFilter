# Phase 1 Contracts and Known-Density Fixtures Subplan

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `PASS_GATE`  
Budget cap: `7200 s`  
Output root:
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase1`

## Objective

Implement the smallest machine-checkable contracts that distinguish a valid
particle authority from a finite moment transform, then run CPU/XLA reference
fixtures before touching q=20.

## Required implementation

Create a TensorFlow/TFP diagnostic/reference lane with no NumPy runtime math.
It must validate:

1. affine change of variables: known map, inverse, determinant, and proposal
   log density agree row by row;
2. fixed/frozen protocol hash: stages, resampling triggers, mutation controls,
   defensive mixture, and proposal-law version hash identically before and
   after draws;
3. known-density unnormalized mass identity on a tractable target, including
   a two-mode mixture and a deliberately mode-missing input cloud;
4. mutation invariant-target diagnostics on a tractable bridge target;
5. defensive mixture positivity and a finite score-class second-moment check;
6. replay metadata parity: historical proposal logs recompute exactly from
   retained metadata, or the block is rejected.

The fixture runner must emit a JSON receipt and a Markdown result note. A
fixture pass is evidence for its tested quantity only; it is not q=20 evidence.

## Gates

| Gate | Role | Required result |
|---|---|---|
| affine identity | promotion criterion for M3 contract | finite residual below declared reference tolerance |
| frozen protocol hash | hard veto | exact hash parity |
| mass identity | promotion criterion for M0 candidate | known normalizer/functional agreement with uncertainty receipt |
| mutation check | promotion veto | no status/nonfinite failure and declared invariance diagnostic |
| mode-missing fixture | explanatory diagnostic | bridge/mode behavior recorded without mode-discovery claim |
| metadata parity | hard veto for replay | recomputation equality or explicit rejection |

## Stop/repair

Repair implementation or fixture defects in place under the same scope. A
failed candidate contract blocks only that arm. A mathematical contradiction on
the exact affine/mass identity is a continuation veto unless the claimed role is
narrowed explicitly. Refresh Phase 2 with the measured fixture tolerances and
which M0 controls are actually frozen.

## Planned command

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_particle_authority_contracts_2026_08_25.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase1
```

## Required artifacts

- runner and focused unit tests;
- JSON receipt and Markdown result;
- phase1 repair/refresh note;
- run manifest containing commit, environment, seeds, device policy, and
  protocol hash.

## Executed receipt

The first wrapper attempt failed before package import because it omitted the
repository root from `sys.path`; that failed artifact is preserved as a
harness repair trigger. The repaired run is
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase1-attempt2`.
All seven fixtures passed under CPU-hidden XLA in `0.726 s`.
