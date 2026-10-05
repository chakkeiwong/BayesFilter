# Dense controller, seeded clouds and SVD correctness checkpoint

The prepared-cloud dense initializer now executes its complete attempt loop,
locator, cloud evaluation and curvature fit in one TensorFlow/XLA call. Its
NumPy-free formatter and NPZ writer preserve completed records, partial archives
and original exception ordering. Full CPU/GPU counts and cumulative charges
are recorded in the accompanying numbered checkpoint receipt.

The numerical authority is the frozen external initializer at `2f386f75`,
SHA-256 `7c4d5598959fcd9605bed01d16d416f2002fd442301e82264ff396b1aa050f75`,
with the pinned original BayesFilter locator and fitter. Prepared clouds replace
only the reference RNG; the separate RNG checks execute its actual TensorFlow
draw statements. This checkpoint does not yet compose seeded clouds with the
whole controller or qualify the real 23-parameter DZ5 target.

Controller cases cover D1/D3 healthy, invalid locator, invalid cloud, score veto,
fit rejection, second invalid cloud, invalid rank and copied partitions. D1
polynomial cases use the real locator/fitter to exercise retry-to-success and
exhaustion. Every case compares complete results, every archive member, ordered
target positions and exact callback/row counts. Three original/changed/original
calls check reuse, unchanged HLO, one trace and frozen result derivatives.
Weak references verify collection of the owner, callback and dependency graphs;
this proves Python collection only, not native executable eviction.

CPU controller renewal passes runs 03491–03508. Seed-stream qualification passes
D1/D3/D23 CPU 03488–03490 and GPU 03515–03517, including current cloud sizes
68/46/46 at D23. Raw Philox words and uniform doubles agree exactly; normal and
cloud calculations retain the existing 1e-10 comparison. GPU completion is
appended after the source-frozen matrix ends.

The controller investigation found an additional real numerical defect. Run
03469 failed the trace-normalized precision-difference operator norm despite
fitted matrices agreeing to rounding. On identical small inputs, run 03470
showed raw XLA SVD underestimating that norm by about 28%; magnitude normalization
and binary64 convergence repair its evaluation. The unchanged 45 norm,
stability and both-sides-of-cap checks pass CPU 03487 and GPU 03514. Run 03471
preserves a test adapter error that supplied a list where the reference expected
a shaped precision array; only that adapter was corrected.

Run 03486 then reproduced the same compiler problem in actual Kalman/SRUKF
consumers at condition about 6. Kalman condition telemetry was inaccurate;
SRUKF reconstructed covariance and gain errors were about 54% and 51–61%, and
likelihood discrepancies reached about 0.44. This diagnostic worker passed
execution, not numerical qualification. The repaired three consumers normalize
SVD inputs and request binary64 convergence, retaining all rank cutoffs,
likelihood formulas and TensorFlow's existing real thin-SVD derivative rule.
The reproduced CPU covariance/gain errors are below 3e-15 after repair.

The 7 primitive and 32 endpoint/analytical-consumer checks pass CPU 03509–03510
and GPU 03512–03513. They include scale-relative reconstruction, square/tall/wide
and batched matrices, zero/deficient/repeated singular values, reference
pullbacks and finite differences, complete linear SRUKF/Kalman comparisons and
existing analytical-gradient consumers. The 28-call source inventory is a
discovery list; other SVD consumers and matched costs remain open.

All 129 policy checks pass 03511. The partial source guard covers 238 files and
1,340 exact exceptions. Five added exceptions permit only completed JSON/history
and archive-member traversal; no numerical loop or NumPy runtime exception was
added. Diff whitespace checks pass. Touched/new-file lint passes except three
confirmed pre-existing diagnostics in `rectangular_factor_tf.py` (unused local
names and unsorted exports). This is not a globally clean lint claim.

Serious runs use the stable campaign runner, numbered output directories,
recorded environment and exact source/policy hashes. GPU runs use the selected
non-desktop device with verified memory growth. Foreign GPU load prevents a
clean timing claim from these correctness workers. No changed scientific
threshold or comparison tolerance was needed for the SVD repair.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain prepared-cloud controller candidate | Full source records and archives pass the completed CPU/GPU matrix | No remaining mismatch in that matrix | Seeded composition and actual consumer are separate | Qualify enclosing seeded program, then public adapter | Full initializer or DZ5 readiness |
| Retain SVD correctness repair | Independent residuals and complete affected consumers pass | Original numerical failures preserved; repaired reproduced cases pass | Other reachable SVD callers and cost remain | Complete caller dispositions and matched measurements | General numerical safety or scientific validity |
| Keep final merge closed | F01–F20 terminal criteria remain open | Public initializers/staged wiring, actual DZ5, memory/cost and integration gates remain | Two reporting proposals await owner decisions | Continue independent E5/E6 work and preserve pending scope | Completed master program |

Review: the strongest misleading interpretation would promote component tests
to whole-consumer qualification or treat Python collection as memory release.
Neither inference is supported. The SVD finding concerns well-conditioned
matrices and cannot use the unrelated ill-conditioned-reporting proposal as an
exemption. This is a primary-agent review; no independent reviewer was invoked.
The live dirty MacroFinance tree and completed CDF retained-r5 are preserved.

GPU controller completion: runs 03518–03535 pass all 18 cases. The checkpoint
receipt therefore records 18 CPU / 18 GPU controller cases, plus three cloud
shapes on each backend. The fetched remote main is now `622d9a9ed`; it is unmerged.

A subsequent exploratory seeded trial (outside the registered runner) passed D1
and failed D3/D23 on an isotropic target. It lacked the already declared
execution-field normalization, and an import-only lint edit occurred while it
ran. It is not qualification evidence. Its JUnit, source snapshot and original
archives are preserved under `seeded-initializer-exploratory-20260924-r1`; a
conservative 320 CPU seconds is charged by a supplemental receipt. D23 showed a
9.59e-8 off-diagonal precision difference; separate RNG probes showed cloud
differences below 2.8e-17. Identical-cloud fitter attribution is required before
any disposition. No criterion has been relaxed. Seeded composition is installed
as an internal candidate only; public wiring remains unchanged.

Reference audit follow-up: the original-reference helper used current
`fit_dense_score_precision_tf` as a dependency. The old public/factor modules
were pinned, but that dense dependency was not. The recorded controller passes
therefore establish parity against a partially isolated baseline, and need
renewal with the third numerical module pinned. No complete original-closure
claim is made for 03491–03535. The same repair applies to validated-fitting
and attempt-composition renewals. This limits the reference evidence, not the
independently checked Kalman/SRUKF SVD result.
