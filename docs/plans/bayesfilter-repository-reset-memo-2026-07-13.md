# BayesFilter Repository Reset Memo

Date: 2026-07-13

Status: `CLEAN_MERGED_BASELINE_READY_FOR_RESTART_AFTER_SYNC`

## Restart Baseline

The content baseline for this reset is merge commit
`12a88d96f980672fe51e7df10591404a114f4c2e`, which combines the local
HMC/Kalman/SSL-LSTM checkpoint with the previously published LGSSM/LEDH/HMC
lane. The reset-memo commit is expected to be a direct descendant of that
merge. On restart, use the current `origin/main` head as the repository
baseline and verify that it contains `12a88d9`.

The merge was semantic rather than a side-selecting conflict resolution. It
preserved both lanes, repaired duplicate Kalman call keywords introduced by
the merge, and retained the HMC minimum-retained promotion gate.

## Repository State

At reset preparation:

- `main` was clean and three commits ahead of the then-fetched `origin/main`;
- `git ls-files --others --exclude-standard` returned no paths;
- `git diff --check` passed;
- no unresolved merge conflict remained; and
- ignored local outputs were preserved in place rather than deleted or added
  to Git.

The final synchronization checks must be rerun after this memo is committed
and pushed. The required restart state is a clean `main`, no untracked
non-ignored path, and local `HEAD` equal to `origin/main`.

## Tracked-Versus-Ignored Contract

Every file currently present in the worktree is either tracked or matched by
an ignore rule. The policy is:

- track source, tests, authored plans/results/reset notes, reconstruction
  tools, compact manifests, and artifacts needed to support a retained gate,
  promotion, or claim;
- ignore caches, build products, logs, local tool state, private diagnostic
  sidecars, raw reproducible payloads that have a retained compressed or
  summarized counterpart, and iterative review packets/transcripts;
- ignore generated JSON by default, while keeping bounded benchmark evidence
  and exact claim-bearing contract/fixture paths visible through narrow
  exceptions; and
- never infer that an ignored local artifact is published evidence.

The seven absent LEDH claim fixtures are explicit exceptions. If regenerated,
they must appear as untracked until validated and committed; they must not be
silently ignored. Existing tracked files remain tracked regardless of ignore
rules.

## Evidence Ledgers

| Ledger | Current state | What the state supports | What it does not support |
| --- | --- | --- | --- |
| Engineering implementation | Merged code compiles, focused merged regression suites passed, and retained JSON parses. | The merged implementation and focused contracts are internally usable for continued engineering. | Whole-repository correctness, API/release readiness, or absence of untested regressions. |
| Numerical/runtime evidence | Several bounded CPU/XLA and focused GPU-era artifacts are retained, with the lane-specific limits below. | Reproduction of the exact bounded gates recorded by their result notes. | General memory improvement, full GPU coverage, HMC convergence, or posterior correctness. |
| Scientific/default claims | No new scientific or default-policy promotion is made by this merge or reset. | Existing owner directives and previously retained claims only, within their original evidence contracts. | Superiority, statistical ranking, broad validity, or readiness inferred from green unit tests. |

## Verification Preserved At Closeout

The merge closeout recorded these focused results:

- merged HMC regression: `210 passed`;
- multidimensional triangular LGSSM suite: `9 passed`;
- deterministic LGSSM HMC driver suite: `11 passed`;
- LEDH source-only wiring checks: `12 passed`;
- generic LGSSM and target-builder checks: `23 passed`;
- every changed Python file compiled;
- every changed retained JSON file parsed; and
- `git diff --check` passed.

These are engineering checks. They are not promotion criteria for memory,
posterior, HMC, GPU, product, or scientific claims. No fresh whole-repository
test suite was run during merge closeout.

## Kalman QR Batched XLA

Authoritative result:
`docs/plans/bayesfilter-kalman-qr-batched-xla-lean-repair-result-2026-07-13.md`.

Status: `CLOSED_NO_REPAIR_CANDIDATE_PROMOTED`.

All three bounded counterfactual constructions preserved the checked Kalman QR
value/score semantics, but none established a memory repair. The value-only
construction passed its structural nomination screen and then failed the
prospective first-pair RSS promotion trigger. No tested construction showed
lower memory, no timing ranking is supported, and there is no fresh GPU or
historical `T=120,B=16` repair evidence.

The next justified research action is a new, concise discriminator comparing
the current reverse-mode time scan with a mathematically equivalent
checkpointed/custom-gradient or forward-sensitivity implementation. Derive and
test equivalence before a bounded XLA memory comparison. Do not rerun the old
full grid or claim that memory/performance is fixed.

## SSL-LSTM Completion

Authoritative ledger:
`docs/plans/bayesfilter-ssl-lstm-completion-visible-execution-ledger-2026-07-11.md`.

Status: `PHASE_A3_TRACE_REPAIR_REVIEW_AGREED_SIGNED_CHAIN_REFRESH_ACTIVE`.

The exact CPU-hidden focused suite passed `65/65`. Its trace then exposed
three parser false positives; the narrow parser repair passed focused checks
and bounded review. That repair made the prior signed review-anchor, boundary,
and fixture backlink chain stale. Therefore the passing test outcomes remain
engineering evidence, but no subsequent A3 evidentiary runtime is authorized
from the stale chain.

On restart, refresh the acyclic review-anchor -> boundary -> fixture chain,
require both loaders to accept it, and then run the reviewed CPU-hidden oracle
artifact command and independent verifier. A fresh full SSL-LSTM suite was not
run during repository closeout.

## HMC

### Scalar Filtering Reference Lane

Authoritative result:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ae-reference-method-expansion-decision-result-2026-07-09.md`.

Status: `CURRENT_SEQUENTIAL_REFERENCE_BRANCH_BLOCKED_EXPANSION_REQUIRED`.

The current fallback-resampling sequential-reference branch trades beta
completion against retained ancestor diversity and did not nominate a valid
reference. HMC-versus-reference agreement remains unassessed. This is a
reference-construction failure, not evidence that the target, HMC mechanics,
or broader research direction is invalid.

The next action is either a materially different reference-method design with
predeclared validity gates or an explicit closeout preserving unresolved
reference agreement. Do not claim posterior correctness, convergence, HMC
readiness, or sampler ranking.

### Deterministic Multidimensional LGSSM Lane

Authoritative result:
`docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6aa-svd-score-wiring-retry-result-2026-07-10.md`.

Status: `PASSED_KERNEL_HANDOFF_PHASE7_APPROVAL_REQUIRED`.

The SVD/eigh XLA value/score gate and fixed-kernel tuning handoff passed with
confirmed JIT and no recorded hard veto. Phase 7 long burn-in and retained
sampling were not run. The handoff is engineering/tuning evidence only, not
posterior convergence, recovery, HMC readiness, GPU readiness, or scientific
validation. Preserve the existing Phase 7 approval boundary.

The multidimensional LGSSM contract, synthetic data, and manifest were
reconstructed from the documented Phase 1 parameters and deterministic
`np.random.default_rng(20260708)` stream. Retained extrema provided
reconstruction fingerprints. The reconstruction deliberately does not
reassert unavailable historical hashes or stale empirical-standard-deviation
prose. The reproducible reconstruction tool is
`docs/benchmarks/reconstruct_multidim_triangular_lgssm_phase2_fixture_2026_07_13.py`.

## LEDH

The source-only wiring checks passed, but the broader claim-test suite is not
hermetic because these claim-bearing JSON fixtures are absent:

- `docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.json`;
- `docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json`;
- `docs/plans/ledh-phase5-actual-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.json`;
- `docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json`;
- `docs/plans/ledh-phase6-generalized-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.json`;
- `docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json`; and
- `docs/plans/ledh-phase7-ksc-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.json`.

Do not reconstruct these files from Markdown summaries. Regenerate each with
its reviewed harness/command, validate the schema and directly relevant tests,
then track the JSON if the associated claim is retained. Until then, report
the broad suite as non-hermetic and do not use it for claim promotion.

## Restart Priorities

1. Verify the published reset baseline: clean `main`, `HEAD == origin/main`,
   and no output from `git ls-files --others --exclude-standard`.
2. Restore LEDH test hermeticity by reviewing the exact generators, regenerating
   the seven missing claim fixtures, and running their artifact tests. This is
   the first repository-integrity gap; do not fabricate fixtures.
3. Finish the SSL-LSTM A3 signed-chain refresh, then run and independently
   verify the bounded oracle artifact before making any A3 claim.
4. Choose the HMC research branch explicitly: design a new scalar reference,
   close that lane with agreement unresolved, or separately approve the
   deterministic LGSSM Phase 7 runtime. These are independent decisions.
5. Start a new Kalman research plan only for the checkpointed/custom-gradient
   versus forward-sensitivity discriminator. Require correctness equivalence
   before memory evidence and GPU/historical-scale evidence before claiming a
   repair.
6. Run a broader repository suite after the artifact gaps are closed; record
   failures by lane rather than weakening gates to obtain a green aggregate.

## Restart Commands

```bash
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
git merge-base --is-ancestor 12a88d96f980672fe51e7df10591404a114f4c2e HEAD
git ls-files --others --exclude-standard
git diff --check
```

Expected result: clean synchronized `main`, successful ancestry check, no
untracked non-ignored output, and no whitespace error.

## Final Boundary

This reset publishes an engineering baseline and a truthful gap ledger. It
does not state that the original Kalman memory/performance problem is fixed,
that SSL-LSTM A3 evidence is complete, that either HMC lane is posterior-valid,
that the absent LEDH fixtures exist, or that any method is scientifically
superior or default-ready.
