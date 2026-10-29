# Main synchronization and score-fix assessment

Completed 2026-10-04. Main at a925f67a19d8c5d265b8ff71bfd2c78ac1964ff4
is an ancestor of this branch's merge commit
a55c6be7a16e9401242ee3449248fa347eb58d90.

The user requested synchronization and an assessment of whether the score repair
would matter for surrogate HMC. The synchronization is complete. The explicit
parameter-score correction identified in main is for KSC stochastic volatility.
The active M13 campaign uses the range-bearing model, so that particular callback
correction does not directly repair its force.

## Preserved work and integration

The original tracked work remains recoverable in stash@{0}
(main-sync-preserve-20261003-local-tracked) and tracked-local-diff.patch.
The stash has deliberately not been dropped. A post-merge inventory check found
all 25,092 pre-existing untracked entries still present (25,077 files and 15
directory/symlink entries). This was an existence/size check, not a second complete
content-hash audit; the original SHA-256 inventory remains available.
Only two inventoried files changed size during the sync:
hmc_tensorflow_verified.py and its test, for evidence-role/XLA-metadata compatibility.
The original campaign archives remain in place.

Overlapping changes required semantic integration:
- retain this branch's numerical failure status, structural innovation support,
  small-matrix operations, and exact-divisor transport chunks;
- retain main's SQMC ancestry selection/gather and stage diagnostics;
- use identical active/inactive batched correction output schemas, including
  optional stages and moment-safety diagnostics;
- retain main's displacement-based moment-safety check before whitening, with
  keyword arguments and compatible two-result normalization; local callers can
  explicitly request the additional validity result;
- retain main's public candidate-set tuning dispatch plus the branch's explicitly
  typed campaign continuation configuration, with its actual initial chain bank;
- normalize legacy evidence-role names and align the local test with main's
  honest XLA metadata (requested compilation is not external qualification).

The merge commit is recorded, while restored local campaign edits and these
compatibility repairs remain in the working tree. No unrelated campaign files
were committed or deleted. post-sync-working-tree.patch preserves the resulting
tracked difference.

## What the score fix changes

Commit b7ed96ec7e35e1bb7c68f41ed18b358f97f714db changes
bayesfilter/highdim/ledh_canonical_models_tf.py::ksc_sv_canonical_model.
The flow's observation mean is h + 2*log_beta + mixture_mean. Its total tangent
is therefore dh + 2*d(log_beta). The old callback returned only dh; the new
callback includes the missing direct parameter term.

That old callback was wrong relative to the total derivative of this finite KSC
LEDH program. The repaired callback now passes the independent finite-difference
regression in tests/highdim/test_sqmc_ksc.py for both parameter coordinates,
including the observation and observation-density tangents. This is evidence
about the implemented derivative, not proof that the approximate filtering
likelihood equals the true stochastic-volatility likelihood.

M13 follows a separate checked source call chain:
surrogate_hmc_m13_endpoint_mechanics.py::force_components evaluates the
range-bearing target from ledh_range_bearing_target_tf.py, which calls
canonical_batch_fused_value_score. Its potential force is -score plus the
Gaussian-prior gradient. The acceptance potential is supplied separately by
the frozen importance-likelihood endpoint. M13 does not call the KSC model
factory or its repaired observation tangent.

A force repair can change proposals, energy errors, acceptance and mixing if the
consumer uses the repaired path. A correct LEDH score can still differ from the
gradient of M13's separate importance-likelihood endpoint. Shared numerical and
control-flow updates from main are included, but their quantitative effect on
M13 was not measured against the old source. No improved R-hat, ESS, acceptance
or whitening is established by this sync, and saved posterior draws have not
been reinterpreted as new evidence.

## Validation and repair evidence

Final tests ran in tftwogpu with CUDA_VISIBLE_DEVICES=-1, deliberately on CPU.
The two final batches report 134 passed and 1 skipped:
- safety-and-ksc-score-03.log: 42 passed, 1 skipped, 32.37 seconds;
- final-integration-01.log: 92 passed, 422.71 seconds.

The latter covers the M13 public tuner, cumulative retained sampling, consumer
mechanics, structural innovation, batched canonical score, oracle contract,
and main's tuning dispatcher. The oracle check exercises its nested tests;
those are not added again to the reported count. The skipped test is the
inherited TensorFlow test_session placeholder, not a failing scientific test.
Both git diff --check and git diff --cached --check pass, and there are no
unmerged index entries.

Earlier attempts exposed and repaired an active/inactive output mismatch,
a missing SQMC gather/trace in conflict resolution, incompatible safety-helper
calls, a normalization return-shape mismatch, and stale evidence-role/XLA
metadata expectations. Complete logs are preserved; they were implementation
integration failures, not evidence against surrogate HMC. The final batches
passed after those repairs.

| Decision | Primary criterion | Veto status | Main uncertainty | Action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Main synchronization complete | Main ancestry and focused regression checks pass | No unresolved merge/test veto | Focused CPU coverage does not certify every GPU route | Preserve changes, logs, stash and report completion | All-model campaign completion |
| KSC derivative repair is relevant to KSC forces | Direct missing term restored; finite differences pass | No observed derivative-regression failure | Downstream sampling effect not measured | Use repaired source in future authorized KSC work | Higher acceptance or faster convergence |
| No direct KSC-fix explanation for M13 | M13 call chain uses a different model | No M13 consumer regression in the final checks | Shared-update effect and surrogate/endpoint force mismatch remain unmeasured | Keep prior M13 diagnostics unchanged | M13 mixing or whitening solved |

Post-run audit: the strongest unsupported alternative would be to attribute all
M13 mixing trouble to the identified KSC bug. The inspected model call chains
rule out that direct mechanism. A measured before/after M13 force comparison
could reveal an effect from other shared changes; this task did not perform one.
The weakest coverage is GPU execution and broad nonlinear campaign behavior,
which focused CPU regression tests do not establish.

The synchronization request is finished. It did not restart the sampling
master. Its previous time allocation expired on 2026-10-01T16:00:00Z; the saved
12,750 retained draws per chain and 20,000-draw ceiling remain recorded in the
campaign checkpoint. No new sampling allocation was inferred from this request.
