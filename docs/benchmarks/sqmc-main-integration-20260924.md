# SQMC integration validation — 2026-09-24

The owner authorized closing the SQMC documentation, merging the branch into
main, merging remote main, pushing main, and bringing main back to the branch.
The closeout commit is `c0e2c57227d0fe34b866a2915d0cce25efb4cd21`; starting
main is `89065bc6354801cb368d5e163ba49fd9c4372d10`.

## Conflict resolution

The numerical conflict combined two independent changes. Main's TensorFlow time
loop, moment providers and independent schedules, shared batched correction,
callback support, and invalid-result convention remain in place. SQMC ancestor
selection runs inside the loop and gathers the corresponding state, covariance,
weight and tangent rows. Cumulative validity is a scalar loop state. Diagnostic
traces include ancestor identities, Hilbert ties, state-map saturation and
cumulative validity. A fixed particle shape is restored inside the loop for
SQMC ordering. Invalid runs retain main's rejection behavior: assertions where
supported, or a negative-infinite value with zero score.

The completed SQMC characterization runner and its compatibility entrypoints
replace the older copies. The master program links the corrected transfer
closeout. The 10D transfer runner is explicitly labeled historical diagnostic
code; its old zero scores do not support a score claim.

## Focused validation

All runs use conda environment `tftwogpu` with `CUDA_VISIBLE_DEVICES=-1` and
`BAYESFILTER_TEST_DEVICE_SCOPE=cpu`: GPUs were intentionally hidden. These are
CPU reference checks, not GPU/XLA or scientific promotion evidence.

- The SQMC closeout commit's required oracle hook passed 3 tests in 105.03 s.
- The numerical integration selection passed 9 tests in 84.70 s: analytical
  score parity, identity/Hilbert ancestry, invalid shared correction and reset
  rejection, covariance/tangent carry, and post-reset callback trace parity.
- Two additional tests passed in 18.22 s. Both Hilbert ancestry modes match
  eager values/scores through a fixed-signature TensorFlow graph across two
  parameter values, use a While operation, and do not retrace.
- Changed Python files parse; resolved source and test whitespace checks pass.
  Historical Markdown hard line breaks remain untouched.

The numerical selection excludes the three annealed tests because main's
executor already requires one annealing stage. This integration preserves that
restriction and makes no claim about multi-stage annealing.

The first integration attempt failed collection because a fresh worktree lacked
its ignored native-op binary. Native sources had no difference between the two
parents. The existing SQMC binary was copied into the integration worktree;
its checksum and source are recorded with the local validation logs. No native
source or installed environment was changed.

Exact commands, timings, logs, native-op checksum and the final Git checkpoint
are under `/tmp/bayesfilter-sqmc-integration-20260924/`. The merge commit also
runs the repository's required oracle hook. Remote synchronization is verified
against the final Git refs; this note records the pre-commit validation.

The campaign's scientific limits remain those in the
[corrected closeout](sqmc-campaign-final-summary-20260924.md): transfer score
accuracy, scope-specific tuning and production eligibility are unresolved.
No research campaign was launched during integration.
