# C2: exact-source MacroFinance admission

## Current execution: clean preview, 28 October 2026

Candidate `f9291ee4437aa8c00dabbe331bffc6871b574c5c` is committed in
`/tmp/BayesFilter-modern-hmc-c2-exact-source-20261028`. Its commit subject uses
the word “Admit”; this is an untested integration candidate, not admission.
The 11 changed modules are byte-for-byte copies from C1. All 1,063 frozen
source files and the four current consumer dependencies have been rechecked.
The runner checks all 70 C1-loaded BayesFilter modules before importing TF,
executes the actual clean-source loader with explicitly process-local preview
inputs, then audits additional imports and source stability. Existing tests are
copied unchanged outside the clean worktree to avoid cache/source pollution.
No source gate is mocked, no numerical tolerance changes, and no HMC runs.

Execution manifest: cwd `/home/ubuntu/python/BayesFilter`; interpreter
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python`; runner
`scripts/diagnostics/run_modern_hmc_c2_source_preview.py`. Arguments:

```text
--candidate /tmp/BayesFilter-modern-hmc-c2-exact-source-20261028
--consumer /home/ubuntu/python/MacroFinance
--snapshot docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/rejection-consistency-20261027-01/isolated-source.json
--dependencies docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/c2-source-admission-20261028-01/dependency-comparison.json
--row-tests tests/test_macrofinance_row_isolation_integration.py
--output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/c2-source-admission-20261028-01/preview-01
```

Environment: `CUDA_VISIBLE_DEVICES=-1`, `TF_FORCE_GPU_ALLOW_GROWTH=true`,
`PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`,
`TF_NUM_INTRAOP_THREADS=4`, `TF_NUM_INTEROP_THREADS=2`, `OMP_NUM_THREADS=4`.
Thread caps are convenience resource-sharing choices, not numerical defaults.
A 1,800-second external debugging ceiling bounds the test invocation; it is a
conservative operational cap allowing compilation beyond the preceding
84.50-second suite, not a measured completion forecast or scientific veto.
Record actual elapsed time once; timeout or a failed test triggers diagnosis.
Outputs are confined to the new preview directory and its sibling launch log.
The exact process command and environment are saved in `manifest.json`.

Skeptical review: unchanged numerical tests compare the factor-native score
against independent covariance/finite-difference and reverse-mode oracles,
with support and mixed-row checks in eager, graph and CPU XLA modes. The test
uses exact C1 sources instead of the dirty shared tree. Its important limitation
is that process-local authority injection proves compatibility, not official
consumer configuration. A pass cannot promote posterior convergence or release.
No active MacroFinance-local filtering/HMC implementation is selected; NumPy
use is confined to the consumer's explicitly diagnostic reference tests.

The earlier full-inventory “absent” counts below are provisional: the first
comparison classified failed `git show` calls as absence without distinguishing
lazy-object retrieval errors. Positive hash matches and the 11 copied-source
checks are established; these provisional absence totals are not admission
evidence. They require a Git-tree inventory before reuse.

### Preview-01 findings and localized repair

The unchanged preview completed in 57.1971583 inclusive seconds: 28 passed,
three failed. All 70 C1-loaded dependencies match, all 1,063 frozen hashes
remain valid, actual new BayesFilter imports match C1, and both real clean
source checks pass. All ten row-isolation/reference checks passed. Two tests
fail before comparing scores because eager GradientTape asks an XLA Kalman
function to return its internal TensorList across the compilation boundary.
The third inspects the public wrapper after the analytic callback moved into
`_build_generic_target_with_runtime`. This is an oracle execution defect and
a stale structural test, not observed evidence of a wrong analytic score.

Repair the diagnostic oracle by compiling its value and reverse-mode gradient
together in one fixed-signature XLA function. Execute the existing strict
chart support checks outside that numerical function; the compiled function
evaluates the same chart/likelihood without duplicating assertions. Preserve
the original reference value, gradient tolerances and target. Keep a small
bounded cache of functions keyed by the observation length to avoid recreating
the graph per call. Update the structural test to inspect both wrapper and
implementation and verify delegation. Add an invalid-support oracle regression
and record the enclosing-XLA metadata. First test the exact proposed consumer
patch in a temporary overlay; only a passing patch is installed with platform
permission. The official consumer pin is not changed as part of this repair.

Skeptical repair audit: do not replace the independent covariance oracle with
the analytic QR score itself, remove support validation, reduce tolerances, or
change the runtime target to make the test pass. An XLA compile error means
this repair hypothesis failed, not that the analytic sampler failed. The same
31-test matrix plus new oracle checks and explicit source-delta reporting
distinguish the oracle repair from any change to C1 runtime semantics. This
changes only execution boundaries and a test's source lookup, not mathematics.

Preview-02 rejected the full-XLA oracle hypothesis: 30 passed, three oracle
tests failed in 61.8538303 inclusive seconds. TensorFlow's matrix-exponential
reverse pass requires an unbounded TensorList. The failed patch is preserved.
Use an explicitly diagnostic non-XLA fixed-signature graph for the independent
oracle, calling the existing BayesFilter covariance filter's undecorated body
to avoid nested XLA boundaries. No filter equations are copied or changed.
Record `jit_compile=false` and the concrete compiler failure in its metadata.
This reviewed reference-only exception has no effect on the analytic target
or the GPU/XLA sampler. Validate reference parity, support rejection and graph
reuse; do not describe this reference as GPU or default-readiness evidence.

The external one-country master's read-only `recover --brief` and
`next --brief` both fail because a historical P19-SYNTHETIC evidence file
`control01/result.md` is missing. That is a distinct consumer-program issue;
no consumer master dispatch is started or bypassed. The BayesFilter-owned
isolated integration diagnostic can continue without altering that program.

Preview-03 passes all 33 checks in 63.1957277 inclusive seconds, including
cached reference-graph reuse and strict support rejection. All source gates
and imported BayesFilter hashes pass. Install only the tested two-file patch:
`git -C /home/ubuntu/python/MacroFinance apply` with the absolute
`N5/c2-source-admission-20261028-01/consumer-oracle-repair.patch` path, after
`apply --check` and original-file hash checks. This reversible mutation needs
platform permission because MacroFinance is outside the writable root. It
preserves pre-existing edits and changes neither the hard-coded runtime root
nor commit. Re-run the same preview against installed consumer files in
`preview-04`, retaining the explicit preview-only clean-source binding.

Also run the clean candidate's existing stationary-derivative, stacked-QR,
QR-score and candidate-set artifact regressions under the same CPU/thread
environment, with cache disabled and JUnit/log/timing files outside the clean
worktree. Use the same 1,800-second diagnostic ceiling. Core regressions can
expose implementation incompatibility; they cannot replace official consumer
configuration or authorize posterior execution. No consumer pin update or
consumer campaign dispatch is included in this installation.

Core-regression-01 stopped in collection after 3.5275275 seconds: the general
test conftest preloads an optional Sylvester custom-op binary not present in
the isolated source tree. C1 and the Phase 9 route do not import that op. Retry
these same selected tests with `BAYESFILTER_PRELOAD_CUSTOM_OP=0`, recording the
exception explicitly. This changes test bootstrap only; if any selected test
actually requires that binary, diagnose it rather than replacing its result.
The tests cannot establish custom-op readiness. No unavailable binary is
silently copied from the shared checkout or treated as a qualified dependency.

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
| Numerical baseline | C1b frozen snapshot records Git base `c41c38941ab2344a5993ab81c797a0dba91d1375` plus exact copied working-tree bytes; it is not equivalent by assertion to current HEAD or the older `43a7f1` pin | Testing another tree can pass while the campaign source remains unadmitted | Hash the actual consumer dependency closure against a clean source candidate before public test execution |
| Runtime | Existing Phase 9 CPU-only loader contract with memory growth configured before TensorFlow import | Eagerly importing TensorFlow or exposing a GPU changes the tested path | Check environment and `tensorflow` absence before invoking `_load_runtime` |
| Test tolerances and support | Existing Phase 9 and BayesFilter regression criteria | Reusing one local point or loosening tolerances can hide score or support errors | Run the existing parameterized target, independent covariance-reference, derivative and mixed-row tests unchanged |
| Compute allocation | Approximately 236,027.809 combined seconds after the latest 84.50-second regression run; separate SIR funding is unchanged | Spending from the SIR pool or counting nested worker time twice | Record each C2 diagnostic's measured inclusive wall time once; create a fresh C2 output root |

The C1 numerical run used the immutable source snapshot and dependency hashes,
not the current dirty worktree as a clean Git revision. A full comparison of
the 1,059 files in the frozen BayesFilter copy against the consumer pin and
the current branch is saved in
`artifacts/modern-hmc-implementation-2026-10-07/N5/c2-source-admission-20261028-01/full-dependency-comparison.json`.
At the pin, 400 files match the frozen copy, 103 differ, and 556 are absent.
At the current committed head, 736 match, 51 differ, and 272 are absent.
Among 70 loaded BayesFilter modules in the resumed run, 58 match current HEAD
and 12 differ; the changed loaded modules include QR/Kalman derivatives,
stationary LGSSM derivatives, candidate-set execution/tuning, verification,
target building and status handling. Therefore neither the pin nor the
current HEAD identifies the C1 runtime bytes. A clean candidate must be built
from the frozen manifest itself, with every dependency included and verified;
do not stage the dirty shared checkout or assume an intervening commit is
equivalent. A second comparison against the C1 snapshot's recorded Git base
`c41c38941ab2344a5993ab81c797a0dba91d1375` finds 737 identical files, 50
different files and 272 absent files; 11 of the 70 loaded modules differ from
that base. The C1 runtime therefore includes exact working-tree dependency
bytes beyond its Git base. These counts and per-file hashes are preserved in
`c41-comparison.json` beside the full comparison.

The MacroFinance lock lives outside this repository. Any change to its pin
must be reviewed against the exact proposed BayesFilter commit and the
consumer's own change history. This plan does not silently make that external
change. The unit-test-only repair commit in BayesFilter is not, by itself, an
admission candidate.

## Ordered execution

1. Preserve the C1 terminal records and commit only the verified BayesFilter
   resume-repair files. Keep all other shared-worktree changes unstaged.
2. Use the committed comparison to identify Phase 9's actual numerical
   dependency closure and all local imports. In particular, the old pin lacks
   the factor-native score and stationary-row isolation code exercised in C1.
   Do not infer source equivalence from commit ancestry or matching API names.
3. Build an isolated clean source candidate from the immutable C1 copy and
   hash manifest, rather than from whichever files happen to be in the shared
   worktree. Include required runtime/custom-op dependencies and MacroFinance
   target inputs. Validate all 1,059 source hashes and the complete imported
   module closure before testing. Run relevant BayesFilter tests there, record
   its clean commit and source hashes, and leave concurrent edits untouched.
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

The immediate implementation scope is a clean candidate based on the recorded
C1 Git base `c41c38941ab2344a5993ab81c797a0dba91d1375`, with the 11 loaded
BayesFilter source files whose exact bytes differ in the frozen C1 snapshot
copied from that snapshot. The intermediate pass criterion is exact identity
for every one of the 70 modules loaded by the resumed qualification, followed
by the existing Phase 9 source/target tests in a clean isolated worktree. If a
test imports another source file, add it only after verifying its C1 snapshot
hash and record that expanded closure. This candidate is an integration
preview, not public admission: its temporary test process may point the
consumer's authority check at the isolated clean candidate, but must clearly
record that this uses a test-only expected-commit injection. No test result
from that preview will be described as the official MacroFinance route passing.

Skeptical pre-execution review: the comparator is the exact locked Phase 9
source check; the changed implementation bytes come from the immutable C1
snapshot, not the dirty checkout. The pass criterion is module hash equality
plus the existing numerical tests, not a proxy such as import success alone.
The primary risk is incomplete transitive closure; the first diagnostic
compares every module actually loaded, then tests record any additional import.
No acceptance, timing or posterior metric is a promotion criterion here.
Failure, source drift or unrecorded imports stops this candidate; it does not
authorize changing the consumer pin. Even a pass establishes only local source
compatibility. The MacroFinance owner must accept the clean candidate before
the external source pin is changed. No GPU run, public release, or posterior
claim is needed to resolve this C2 source-authority question.
