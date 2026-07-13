# LEDH Shared Compact Score Memory Root-Cause Repair Result

Date: 2026-07-09

Status: `BLOCKED_REQUIRES_REDUCE_ONLY_SCORE_RECURRENCE`

## Objective

Trace and repair shared compact score memory causes before another full
`N=10000` score admission attempt, while preserving the same realized finite-`N`
LEDH `observed_data_log_likelihood_estimator` / `log_likelihood` scalar.

## Review

Plan reviewed by a fresh Codex read-only reviewer after Claude review was not
used for this repo-derived packet in this run.

Review packet:
`docs/reviews/bayesfilter-ledh-shared-compact-score-memory-root-cause-repair-plan-review-packet-2026-07-09.md`

Verdict: `VERDICT: AGREE`

Reviewer found no material flaws: same finite-`N` scalar preserved, shared
hotspot coverage adequate, Phase 1 value-preserving, score-only GPU rungs
diagnostic rather than admission, and stop conditions sufficient.

## Implementation

Changed:

- `experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py`
- `tests/test_ledh_compact_transport_jvp.py`

Repair made:

- Added `_half_pairwise_squared_cross_jvp` to compute the JVP of
  `0.5 * _pairwise_squared_cross` without materializing the old
  `[B,row_chunk,col_chunk,state_dim,param_dim]` broadcast product.
- Replaced the old `d_diff`/`d_cost` broadcast product in:
  - `_filterflow_streaming_softmin_jvp`;
  - `_filterflow_streaming_column_log_normalizer_jvp`;
  - `_filterflow_streaming_transport_from_potentials_jvp`.
- Replaced the old transport `d_weighted` 5D product with two direct
  contractions:
  - `tf.einsum("brc,bcdp->brdp", transport_block, d_particle_block)`;
  - `tf.einsum("brc,bcd,brcp->brdp", transport_block, particle_block, d_log_transport)`.

This is a value-preserving tensor-lifetime repair. It does not change the
public helper signatures or returned tensors.

## Local Checks

Passed:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_ledh_compact_transport_jvp.py -q
```

Result: `5 passed`.

Passed:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_ledh_compact_transport_jvp.py \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result: `53 passed, 2 warnings`.

## GPU Rungs

All GPU commands were trusted/escalated because GPU/CUDA commands must not be
interpreted from the sandbox.

### Rung 1: `N=256,T=3` LGSSM Score-Only

Artifact:
`docs/plans/artifacts/ledh-shared-compact-score-memory-phase3-lgssm-score-only-n256-t3-2026-07-09.json`

Result:

- emitted successfully;
- `score_status = blocked_score_only_diagnostic_not_admitted`;
- `score_admission_status = blocked_score_diagnostic_stage_not_admitted`;
- `score_gpu_memory_info_after.peak = 2472704` bytes, about `2.36 MiB`;
- explicitly non-admitting because same-scalar FD was not run in score-only
  stage.

### Rung 2: `N=1000,T=10` LGSSM Score-Only

Artifact:
`docs/plans/artifacts/ledh-shared-compact-score-memory-phase3-lgssm-score-only-n1000-t10-2026-07-09.json`

Result:

- emitted successfully;
- `score_status = blocked_score_only_diagnostic_not_admitted`;
- `score_admission_status = blocked_score_diagnostic_stage_not_admitted`;
- `score_gpu_memory_info_after.peak = 10085888` bytes, about `9.62 MiB`;
- explicitly non-admitting because same-scalar FD was not run in score-only
  stage.

### Rung 3: `N=10000,T=50` Single-Seed LGSSM Score-Only

Artifact:
`docs/plans/artifacts/ledh-shared-compact-score-memory-phase3-lgssm-score-only-n10000-t50-seed81120-2026-07-09.json`

Result:

- no artifact emitted;
- trusted `nvidia-smi` during the run showed approximately `15740 MiB / 16376
  MiB` used with nonzero GPU utilization;
- this exceeds the reviewed score memory budget of `14000 MiB`;
- the run was interrupted after the memory gate had already failed;
- traceback showed execution inside
  `_filterflow_streaming_softmin_jvp` called from
  `_filterflow_streaming_finite_sinkhorn_potentials_jvp_total` inside the LGSSM
  compact score pass.

## Interpretation

The contraction repair is correct and useful, but insufficient for the full
LGSSM admission blocker. The remaining root cause is not the old time-history
storage and not the removed `d_weighted` block. The remaining blocker is the
current full forward-sensitivity API:

- the score route still carries `d_particles` with shape roughly
  `batch x N x state_dim x param_dim`;
- the Sinkhorn potential JVP still computes and stores full tangent potentials
  with shape roughly `batch x N x param_dim` many times per time step;
- the transport step still returns full `d_transported`;
- at `N=10000,T=50,sinkhorn_iterations=10`, this keeps the full score pass over
  the memory budget and prevents artifact emission.

The next repair cannot be another procedural rerun. It needs a reduce-only or
checkpointed score recurrence that avoids carrying full parameter-axis particle
sensitivities through the entire filter except in tiny equivalence tests.

## Decision Table

| Decision | Status |
| --- | --- |
| Shared 5D contraction patch | Implemented and locally verified |
| Tiny and medium GPU score-only rungs | Passed as diagnostics |
| Full `N=10000,T=50` LGSSM score-only rung | Blocked: memory budget exceeded, no artifact |
| Score admission | Not admitted |
| Next justified action | Implement reviewed reduce-only score recurrence subplan |

## Nonclaims

- No LGSSM score artifact is admitted by this result.
- No leaderboard score row should be promoted from the score-only diagnostics.
- This does not establish HMC readiness, posterior correctness, exact Kalman
  score equality, runtime ranking, or cross-model full-scale admission.
