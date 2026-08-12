# Zhao-Cui Austria SIR Gap-Closure Result

Date: 2026-07-30

Status: `SUPERSEDED_HISTORICAL_SOURCE_REPLICA_RESULT_NOT_ACTIVE`
Historical terminal status: `BLOCK_T1_SOURCE_REPLICA_FIT_OR_PROPOSAL_GATE`

> Superseded by
> `docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-parameter-extension-master-plan-2026-07-30.md`.
> Preserve the measurements below as source-replica evidence. The proposed
> author TT-cross/ALS repair is forbidden in the current fixed-variant plan.

## Decision

The missing Algorithm-3 mechanics gap is closed: the code now preserves the
full 36D affine covariance in a block-upper author-order frame and implements
the suffix-conditioned reverse KR sampler for `x_t | x_{t-1}, y_1:t`. The
generic reordered `(x_previous,x_t)` compiler is not used.

The fixed-parameter T1 continuation gate still fails. The resolved rank-4
candidate passes finiteness, numerical conditional consistency, roundtrip, and
memory checks, but its corrected proposal ESS is `1.0000000000313847 / 8`.
This blocks T2, T20, parameter score, GPU/XLA, and HMC-facing execution. It is
evidence against the current local Adam-fitted candidate, not against the
Zhao-Cui algorithm or the parameter extension.

## Evidence

| Item | Result | Role |
|---|---|---|
| active observation identity | `cd794ad6e90a74f7cf6dc06b33550bff4bef6fbf66bb0917846d0691b5910f07` | hard validity |
| source settings bound | `Lagrangep(4,8)`, `AlgebraicMapping(1)`, rank baseline 40/20/5, ALS 8/2 | source baseline |
| upper conditional adapter | block-upper full-covariance frame plus reverse suffix KR | engineering pass; `fixed_hmc_adaptation` plus extension |
| fit residual | `0.19576338979370758` | explanatory only |
| same-frame holdout residual | `1.3244289311520887` | current candidate diagnostic |
| normalizer | `0.012582664786933229` | hard finiteness pass |
| conditional roundtrip max | `1.51676346677454e-05` | pass against `1e-4` |
| numerical vs exact TT conditional log density | max abs `0.007612825749294672` | explanatory consistency check |
| corrected proposal ESS fraction | `0.12500000000392308` | hard continuation veto; threshold `0.5` |
| corrected log-weight spread | `310.44405295738187` | explanatory diagnostic |
| transition log-density range | `[-369.38846253821714, -49.730936493720044]` | explanatory failure localization |
| KR working-set peak | `609664` bytes | bounded-memory pass |

Historical terminal artifact:
`docs/benchmarks/artifacts/zhao_cui_austria_sir_source_replica_gap_closure_20260730/attempt07_t1_source_replica_final/result.json`.

Attempt history:

- `attempt01`: JSON `mappingproxy` serialization harness failure; preserved.
- `attempt02`: superseded because holdout used an independently fitted frame.
- `attempt03`: corrected same-frame holdout exposed residual
  `3.4338869950437804e+56`; adapter was still absent.
- `attempt04`: first adapter execution; coarse grid and rank-1 candidate failed
  holdout, ESS, and roundtrip.
- `attempt05`: rank-4/resolved-grid repair passed roundtrip and finite holdout,
  but ESS collapsed to `1/16`.
- `attempt06`: correction audit confirmed numerical conditional consistency and
  isolated the remaining failure to proposal/fit mismatch.
- `attempt07`: exact final-code rerun with explicit holdout-veto provenance;
  metrics reproduced attempt 6 and this is the historical terminal artifact.

## Decision Table

| Decision | Primary criterion | Veto | Main uncertainty | Next justified action | Nonclaim |
|---|---|---|---|---|---|
| upper conditional mechanics | passed | none | finite-grid KR is not production closure | keep as tested adapter | not source-faithful `computeL` square root |
| T1 continuation | failed | ESS fraction `0.125 < 0.5` | local Adam fitter differs from author TT-cross/ALS | implement/audit the author training route or a reviewed target-specific fit protocol | no T2/T20/score |
| memory policy | passed | none | paper-scale rank/particles not run | retain microbatch budget checks | no GPU capacity claim |
| Zhao-Cui direction | remains viable | no target/math/harness invalidity | author training not locally reproduced | treat this as a repair trigger | no algorithm rejection |

## Inference Status

| Inference field | Status |
|---|---|
| hard veto screen | T1 ESS veto only; finiteness, identity, adapter, roundtrip, and memory pass |
| statistically supported ranking | none; deterministic bounded diagnostics with small particle counts |
| descriptive-only differences | residuals, density ranges, runtime, and log-weight spread |
| default-readiness | no |
| historical next evidence proposed by this campaign | author TT-cross/ALS reproduction or reviewed target-specific training, then untouched T1 ESS/roundtrip; not current authority |

## Run Manifest

| Field | Value |
|---|---|
| command | `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python scripts/run_zhao_cui_austria_sir_source_replica_t1.py --cpu-reference --output-root docs/benchmarks/artifacts/zhao_cui_austria_sir_source_replica_gap_closure_20260730/attempt07_t1_source_replica_final --fit-rank 4 --fit-sample-count 64 --holdout-sample-count 32 --train-steps 8 --optimizer-batch-size 16 --cdf-grid-size 33 --cdf-bisection-steps 16 --particle-count 8 --seed 8615` |
| environment | `/home/chakwong/anaconda3/envs/tf-gpu/bin/python`, TensorFlow 2.19.1, intentional `CUDA_VISIBLE_DEVICES=-1` |
| hardware | CPU reference/debug lane; GPU deliberately not used |
| git | commit `74f7aa9bd151969f79393db508cb828e951a9a30`; dirty worktree preserved |
| seed | fit `8615`; proposal `8715` |
| wall time | `45.153338137999526` seconds |
| tests | `41 passed, 2 warnings` |

## Ledgers

Engineering correctness: source/paper identities, algebraic-map Jacobian,
full-covariance block-upper refactor, suffix conditioning, immutable cores,
correction assembly, and memory budget are tested.

Numerical validity: the resolved grid passes roundtrip; exact TT versus
numerical grid conditional density differs by at most `0.0077` log units on the
audit draw. The proposal nevertheless produces a hard ESS failure.

Scientific interpretation: the current local P86 Adam fit is an
`extension_or_invention`, not author random TT-cross/ALS. Its failure cannot be
promoted to evidence that Zhao-Cui cannot work.

## Red Team

Strongest alternative explanation: the proposal mismatch may be repaired by
the author's adaptive TT-cross/ALS training, larger source-shaped training
clouds, rank/L1 tuning, or another target-specific protocol. The present run is
too small to rank alternatives, but it is sufficient to veto continuation of
this candidate. A result overturning this decision must pass an untouched T1
ESS fraction `>=0.5` and roundtrip `<=1e-4` under the same exact target and
correction, before T2 is attempted.

Nothing here establishes exact likelihood, pseudo-marginal unbiasedness,
posterior correctness, source-faithful assembled execution, HMC readiness,
GPU performance, or statistical superiority.
