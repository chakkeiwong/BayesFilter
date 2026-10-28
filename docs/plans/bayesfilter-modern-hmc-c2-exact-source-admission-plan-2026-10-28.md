# C2: exact-source MacroFinance admission

28 October 2026. This plan follows completion of the source-isolated C1b
qualification and identifies the actual blocker to the next public-consumer
check. It uses the existing combined allowance; no additional compute is
requested. The terminal C1b settlement is in
`artifacts/modern-hmc-implementation-2026-10-07/N5/budget-resume-repair-20261027-01/settlement.json`.

## Question and evidence contract

Can MacroFinance Phase 9 execute its normal public target entry point against
the exact BayesFilter source bytes tested in the resumed qualification, while
preserving MacroFinance's clean-source and commit-identity checks?

The observed baseline is the current public loader. It calls
`one_country_affine_macro_governance.audit_bayesfilter_authority` before
importing TensorFlow. That function requires both the literal
`LIVE_BAYESFILTER_COMMIT` and an empty `git status --porcelain` result. The
consumer currently pins `43a7f113ddbadf13c6421b7b6d7b2d26f5a4f1e9`; the shared
BayesFilter checkout is at `d0705f7956a6e224b46cb478667ddd82b0953d32` and has
extensive unrelated dirty changes. Consequently `_load_runtime` raises the
existing admission error before numerical evaluation. The historical
public-entry failures therefore diagnose a source-identity mismatch; they do
not show that the target, score, or sampler failed.

**Pass criterion:** the consumer's expected commit identifies a clean
BayesFilter tree whose bytes equal the explicitly reviewed source set; its
normal `_load_runtime` and public target builder load only that tree; Phase 9
value/score, support and row-isolation integration tests pass with the stated
CPU-only environment; and source hashes remain unchanged throughout.

**Vetoes:** bypassing or mocking the consumer's source gate in an integration
result; an unreviewed pin change; a dirty or source-mismatched admitted tree;
imports from another checkout; failure of target/reference/score or row
isolation checks; or executing with a different device contract. Do not repair
these vetoes by relaxing target support or numerical thresholds.

**Diagnostics only:** existing direct target and GPU/XLA checks can localize
numerical behavior, but they cannot substitute for public admission. Runtime
and acceptance are descriptive. R-hat remains reporting-only for tuning.

The integration result will not establish posterior convergence, burn-in or
MCSE sufficiency, method ranking, default promotion, full empirical-model
validity, GPU readiness for this Phase 9 lane, or release readiness. Phase 9's
existing `_load_runtime` requires `CUDA_VISIBLE_DEVICES=-1`; preserve that
CPU-only contract for C2 rather than changing it to a GPU campaign.

## Source and default audit

| Choice | Provenance and status | Failure mode | Earliest check |
|---|---|---|---|
| Exact source identity | Hard-coded MacroFinance commit and clean-tree checks; binding is an owner-controlled consumer default | A commit hash alone can omit dirty numerical dependencies; changing the constant without a source diff admits unreviewed bytes | Compare the consumer pin, current commit, frozen C1 source inventory and clean proposed tree |
| Numerical baseline | C1b frozen snapshot at Git base `e38348bee2b321d695181ec74e7486a611f67df4` plus its exact copied working-tree bytes; not equivalent by assertion to current `d0705f7` or the older `43a7f1` pin | Testing another tree can pass while the campaign source remains unadmitted | Hash the actual consumer dependency closure against a clean source candidate before public test execution |
| Runtime | Existing Phase 9 CPU-only loader contract with memory growth configured before TensorFlow import | Eagerly importing TensorFlow or exposing a GPU changes the tested path | Check environment and `tensorflow` absence before invoking `_load_runtime` |
| Test tolerances and support | Existing Phase 9 and BayesFilter regression criteria | Reusing one local point or loosening tolerances can hide score or support errors | Run the existing parameterized target, independent covariance-reference, derivative and mixed-row tests unchanged |
| Compute allocation | Approximately 236,027.809 combined seconds after the latest 84.50-second regression run; separate SIR funding is unchanged | Spending from the SIR pool or counting nested worker time twice | Record each C2 diagnostic's measured inclusive wall time once; create a fresh C2 output root |

The C1 numerical run used the immutable source snapshot and dependency hashes,
not the current dirty worktree as a clean Git revision. Do not claim that the
upcoming repair-plumbing commit alone represents those numerical bytes. First
produce a source-diff inventory and a clean, reproducible BayesFilter commit or
tree containing precisely the dependencies that C2 intends to admit. If that
cannot be assembled without importing unrelated concurrent edits, stop and
report the exact missing files or history rather than staging the shared tree.

The MacroFinance lock lives outside this repository. Any change to its pin
must be reviewed against the exact proposed BayesFilter commit and the
consumer's own change history. This plan does not silently make that external
change. The unit-test-only repair commit in BayesFilter is not, by itself, an
admission candidate.

## Ordered execution

1. Preserve the C1 terminal records and commit only the verified BayesFilter
   resume-repair files. Keep all other shared-worktree changes unstaged.
2. Compare commit `43a7f113ddbadf13c6421b7b6d7b2d26f5a4f1e9`, current base
   `d0705f7956a6e224b46cb478667ddd82b0953d32`, and the C1 frozen dependency
   inventory. Identify the exact source files required by Phase 9 and row
   isolation. Do not infer source equivalence from commit ancestry or matching
   API names.
3. Build an isolated clean source candidate using only reviewed files and
   dependencies. Run the relevant BayesFilter tests against that candidate,
   record its commit and source hashes, and ensure the original shared
   worktree and all concurrent edits remain untouched.
4. At the MacroFinance repository boundary, update the hard-coded admission
   commit only after the consumer's source owner accepts the exact clean
   candidate. Preserve existing requirements for a clean worktree, import
   origin and environment. Do not replace these with a diagnostic bypass.
5. Run the unmodified Phase 9 public target/integration tests from the
   MacroFinance repository under `CUDA_VISIBLE_DEVICES=-1` and
   `TF_FORCE_GPU_ALLOW_GROWTH=true`, after confirming TensorFlow is not already
   imported. Include the source-authority check, value/score and independent
   covariance-reference tests, analytic derivative coverage and the
   mixed-row eager/graph/XLA-CPU isolation checks. Save the actual command,
   environment, commit, hashes, test output and inclusive time in a new
   versioned C2 result directory. Do not run HMC sampling in this phase.
6. Audit the imported module paths and terminal statuses. If all C2 criteria
   pass, refresh the master to C3 posterior/reference work; otherwise classify
   the exact gate, source, target, score or row failure and plan the smallest
   discriminating repair. Candidate tuning outcomes do not serve as posterior
   evidence.

The focused recovery suite just completed is the current test baseline:
50 passed in 84.50 seconds with GPUs intentionally hidden. The inherited
300-second test-invocation ceiling in the C1 repair plan is an operational
debugging limit, not a numerical criterion; record actual times and revise the
limit if observed setup or compilation makes it insufficient. C2 itself has
not launched while the public admission check is known to fail.

## Skeptical audit before execution

The plan passes the skeptical audit only with the source distinction above.
The wrong baseline would be the dirty BayesFilter checkout or the C1 snapshot
treated as though it were already the commit named by MacroFinance. A proxy
would be the passing direct score test treated as proof that `_load_runtime`
admits the public route. The clean-source gate is a real source-integrity
condition, not removable ceremony: the exact dependency closure and imported
paths must match the reviewed commit. Test tolerances, data, target and support
remain fixed. The C1 result establishes finite-horizon candidate qualification
for its recorded snapshot only. It cannot make a different consumer source
version numerically equivalent. The earliest discriminating test is source
identity at `_load_runtime`, before TensorFlow import; only after it passes do
the downstream numerical integration tests answer their question.

The immediate implementation scope is the BayesFilter commit and source-diff
proposal. Editing MacroFinance's pin is an external repository mutation and
must follow the applicable workspace approval route. If that approval is not
available, retain the prepared patch and continue only with BayesFilter-local
source comparison and tests. No GPU run, public release, or posterior claim
is needed to resolve this C2 admission question.
