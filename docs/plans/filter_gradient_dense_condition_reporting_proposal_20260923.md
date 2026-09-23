# Dense-attempt ill-conditioned diagnostic disposition

Proposal only. The complete D3 attempt still fails its original comparison;
no comparison exception or numerical-policy change has been installed.

GPU03450 compares every cloud and fit record against the frozen external cloud
loop and original3582b4ac fitter. It fails only the scalar
`fits[1/3].diagnostics.prediction_jacobian_condition_number`. All cloud arrays
match bit for bit. All accepted covariance/precision, selected family, rank,
optimizer counts, errors, statuses and other numerical fields pass1e-10.
The return-to-original check reproduces the same values.

03454 establishes that the current standalone fitter and the composed attempt
match completely, so this discrepancy predates attempt composition. It also
evaluates the original reverse-Jacobian/SVD and current forward-Jacobian/SVD
diagnostics at identical raw optimizer states. In the second replicate, the
original condition is1.2037467664403725e14 while XLA gives1.2042554344578561e14
at that same original state. Both retain rank6. The fitted loading is within
approximately5e-12 of its allowed magnitude. With27 score equations,
`27 * eps * condition` is about0.72: ten-digit relative agreement of this
condition estimate is not a meaningful accuracy claim. This observation is an
explanatory resolution indicator, not a rigorous error certificate.

03451 preserves the unsuccessful multiplication-rounding trial; it was removed.
03452/03453 preserve diagnostic harness errors (field name and attempted tracing
of an original host-only reference);03454 uses the unmodified original in its
supported eager diagnostic role. No optimizer or target method changed.

The proposed disposition is restricted to the recorded D3 fixture: retain both
raw estimates and label the discrepancy `ill_conditioned_diagnostic_unresolved`.
Omit strict scalar equality only at those two condition fields when both values
are finite and above1e12, their relative discrepancy is below1e-3, both fits are
`factor_1`, and both Jacobian ranks are exactly6. These bounds describe this
fixture's evidence envelope; they are not runtime safety thresholds. A resolved
rank5 condition (about73 for the changed-input second replicate) keeps its
ordinary1e-10 comparison. Every other field and every decision retains its
original gate, including covariance, precision, final loss, chosen family,
optimizer counts, rank and target-call order. No accurate/equivalent condition
estimate, scientific readiness, or new parameter-identification claim follows.

The post-run proposal checker accepts the three saved records and rejects seven
adverse mutations: changed rank, changed status, changed usable covariance,
missing condition, well-conditioned replacement, nonfinite condition and an
unexplained large difference. Its result is
`artifacts/filter-gradient-repair-20260917/dense-condition-reporting-proposal-03454.json`.
The checker is diagnostic only; this is not a passing runtime test. If approved,
encode this precise scope in the tests, add mutation checks and renew full
CPU/GPU attempt comparisons. New failures remain repair triggers.

The earlier owner instruction to report ill-conditioned errors motivates this
proposal. The already qualified02622--02626 exception concerned a rejected,
unused precision matrix; this new field belongs to an accepted fit and is a
different comparison boundary. Explicit agreement is therefore requested under
AGENTS.md's campaign repair rule requiring reapproval when promotion criteria
change. The separate E2 clipping-count proposal remains pending.

| Decision | Primary criterion | Veto | Uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Keep D3 complete comparison open | Only ill-conditioned condition-report scalars fail | Equality gate has not changed | Precision of an almost singular Jacobian diagnostic | Owner decision on the bounded reporting treatment; independent work continues | No full initializer qualification or main merge |
| Reject enclosure as the source of this failure | Standalone and composition match every field | No composition-specific numerical mismatch in03454 | Other shapes/targets remain | Continue controller preparation and healthy qualification | No general correctness claim |

Review: the strongest alternative explanation is a Jacobian defect hidden by
ill-conditioning. Same-state checks show the original/current graph values can
agree while GPU XLA differences are amplified; this does not prove either
condition estimate accurate. Keeping every actual geometry and decision field
checked, preserving the raw diagnostic, and limiting the proposed allowance to
the exact reproduced fixture avoids making that unsupported claim. This is a
primary-agent review, not independent numerical certification.
