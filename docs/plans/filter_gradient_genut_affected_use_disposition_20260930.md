# GenUT affected-use disposition for the XLA rewrite

The reduced `dual_cap_genut_primal_tf` routine is used only as a value/reset
operation in the inspected active callers. Its failed reverse-AD derivatives
are unsupported; they are not the analytical score returned by the registered
score route. No precision redesign of that reduced derivative is required to
close the analytical-score rewrite when those failures cannot enter that score.
This does not qualify reduced reverse AD, dismiss a reachable value regression,
or confer canonical status.

The only runtime call to `dual_cap_genut_primal` is the reduced branch of
`genut_guided_proposal_tf._restore_cloud_primal`, selected when
`dual_cap_enabled=True` and `trust_region_enabled=False`. The registered value
owner can reach that optional branch. Its actual disabled/reduced/trust call
counts and unchanged ordinary outputs are witnessed on CPU/GPU in04936/04938.
The production configuration requests dual cap with trust region and therefore
uses the shared full correction instead. Initialization and ordinary guided
standard-score callers omit dual_cap_enabled, retaining False. The RQMC caller
can explicitly request the reduced value branch, while its standard-score
marks are accumulated by the independent declared score recursion, not by
reverse differentiation of that reset.

The registered analytical score calls
`ledh_unified_correction_tf.batched_higher_moment_shape_jvp`. Its tested
CPU/GPU call chain never reaches the reduced primal when that routine is
blocked in the diagnostic. Neither GradientTape nor a reverse derivative of
the reduced value routine substitutes for the analytical authority. The
incoming annealed analytical implementation retains this separation.

The reduced cap fraction feeds the reset diagnostics and RQMC history. The
registered value program does not consume it; it uses reset_valid/marginal_valid
and the existing moment checks. The two RQMC report consumers read the stored
fraction for reporting. They do not nominate, admit or switch a numerical
candidate from it. The saved mismatch already occurs between original graph
and XLA; it remains a failed diagnostic comparison, not a repaired fraction
or a new threshold. Functions in other correction authorities that use the
same field name do not establish use of this reduced routine's result.

The explicit module marker CANONICAL_LEDH_ADMITTED=False and admission
requirements stay in force. A canonical-looking module/function or registry
role is not an artifact admission. The incomplete canonical rebuild remains
excluded, v1 scalar artifacts remain ineligible, and analytical/canonical score
claims cannot use a reduced autodiff diagnostic. Saved FP32 weight-gradient and
GPU source/reset reverse-gradient failures04245--04261 remain unchanged and
must be cited if anyone proposes that use. Such a proposal requires its own
repair and admission evidence; the present execution qualification cannot
supply it.

This conclusion is limited to inspected callable wiring, saved value/reference
checks and the preserved diagnostic evidence. Reopen it if a new active caller
uses a reduced reverse gradient or cap fraction for a numerical/status/selection
decision, or if the optional value branch shows an unexplained before/after
regression. It does not excuse a defect merely because its route is optional.
Source-bound terminal run05305 passes120 checks, including the affected-use
readback and existing campaign/policy tests. This closes F01's inspected
execution/affected-use disposition. All saved derivative and cap-report
failures remain failed and unsupported; no canonical rebuild claim follows.

Bounded terminal qualification: run the source-bound readback and existing
policy/campaign checks in one CPU worker, with one localized harness retry
reserved (two workers,600 CPU seconds total,300 each). No numerical campaign
or GPU work is needed because applicable CPU/GPU wiring already exists.
Record source hashes, original witness hashes, manifest and JUnit in the
next numbered campaign directory. Stop on changed numerical dependencies,
new active reduced derivative use or loss of admission blocks. No tolerance,
method, unsupported-use classification or source allowance may be relaxed.

Primary-agent skeptical review confirms the registry is a literal dictionary,
the sole direct runtime reduced call and12 source identities match, and the
saved live witnesses distinguish reduced-value and analytical authorities.
A source hash cannot prove arbitrary external callback purity; the result
retains that boundary. Changed analytical source from the remote integration
needs its existing05202 qualification and current source separation check.
