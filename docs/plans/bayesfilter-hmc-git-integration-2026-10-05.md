# HMC repair Git integration, 2026-10-05

The owner requested committing the HMC repairs, merging remote changes,
resolving conflicts, and pushing to `main`. Integration uses
`/tmp/bayesfilter-hmc-v7-merge-20261005` on
`integrate/hmc-v7-20261005`, starting at
`70a6d7e9611a9f859a2519f37f236b61cac35c1e`. The fetched comparator is
`origin/main` at `a925f67a1`. The shared development checkout and the frozen
confirmation/monitor services continue independently.

## Scope and evidence

Commit the candidate-set acceptance repairs, retained-runner bridge, state-space
validation and supervision, their regression tests, the official tuning
chapter and agent reference, and the HMC plans. Include compact manifests and
results, with hashes of copied files. Raw simulation output, build products,
archived source trees, and unrelated NeuTra/q20 edits remain outside this
commit. Live ledgers are explicitly point-in-time snapshots, not terminal
results. The original campaign files remain the authority for the active run.

The engineering question is whether the combined source preserves both the
local repairs and remote HMC execution functionality. The baseline is the
recorded local commit plus the explicit HMC overlay, compared with the fetched
remote commit. Promotion to the Git branch requires resolving every conflict,
passing focused merged-source regressions and guide checks, and a clean
integration index. A failed behavioral regression or missing dependency blocks
the push until repaired. Test duration is explanatory only. These checks do
not establish GPU parity, statistical calibration, release qualification, or a
new default; the live campaign uses its own unchanged scientific contract.

## Execution and skeptical audit

1. Snapshot the scoped files into the isolated worktree and commit them.
2. Merge `origin/main`, preserving remote chain execution, TensorFlow migrations,
   and the local acceptance-policy versions and retained-candidate semantics.
3. Run CPU-only engineering regressions with the `tfgpu` interpreter,
   `CUDA_VISIBLE_DEVICES=-1`, growth enabled, and bounded threads/affinity.
   Validate the rendered route tables and official tuning chapter linkage.
4. Record the actual commands, receipts, conflicts, and limitations. Fetch again,
   incorporate any additional remote commit, then push `HEAD:main` without force.

The validation cap is 2,000 enclosing CPU wall seconds, drawn from the already
authorized release balance (2,235.56 uncommitted seconds at planning), with no
GPU allocation and no new campaign. This is a convenience safety cap, not a
scientific threshold; use focused failures to choose any follow-up within it.
Charge enclosing receipts once, without adding nested test timings. Git I/O
does not consume an experiment allocation. Stop validation on budget exhaustion
and report the unresolved check rather than infer a pass.

Audit passed before execution: the baseline and explicit overlay are identified;
neither short tests nor a successful push will be treated as release evidence;
the environment is deliberately CPU-only; an isolated checkout avoids changing
the running source or unrelated work; and the remote overlap is tested rather
than resolved by choosing one entire version. The main risks are missing
untracked dependencies, interaction with remote TensorFlow migrations, stale
generated documentation, and accidental inclusion of active/raw artifacts.
The snapshot inventory, clean-tree checks, behavioral tests, and documented
artifact selection address those risks. The live run's original deferrals are
preserved and cannot be replaced by recovery outcomes for the release criterion.

## Result

The scoped repairs were committed as `3c8176d7e`. Merging `a925f67a1`
produced three conflicts, resolved by inspecting individual changes:

- `hmc_candidate_set_execution.py`: retain the replicated v7 policy decoder
  alongside the remote v5/v6 decoder, with strict payload roundtrip checking.
- `ssm_targets.py`: retain rejection of unknown fixture parameters; the rest
  of the added target implementation agrees with remote.
- `simple_nonlinear_generic_target_adapter_tf.py`: retain bounded executable
  graph reuse with current theta, observations, and model constants passed as
  inputs. The underlying likelihood and remote fixture interfaces are retained.

The first test collection exposed the fresh checkout's unbuilt native library;
it was then built from the merged source. The second collection exposed an
untracked dependency omitted from the initial snapshot,
`bayesfilter/runtime/execution_budget.py`. That dependency is now included.
An AST import inventory found no other missing local module dependencies.
Neither collection failure is a numerical or sampler result.

The remote chain-execution suite passed all 12 tests. The acceptance, health,
identity, timeout, monitoring, capability-registry and documentation suites
passed all 562 tests. The larger replay and model suite passed 208 tests before
stopping on three stale assertions that expected the generic LGSSM/nonlinear
adapters to lack their now-implemented XLA capabilities. The numerical tests
passed; the old capability expectations were wrong for the changed adapters.

The assertions now reflect the supported XLA path. The nonlinear adapter test
also checks both explicit JIT settings, requiring full-chain diagnostic
readiness to remain false for the non-XLA configuration. The existing
full-chain regression now additionally compiles the admitted cubature route
and compares its trajectory calculation against a graph execution with
identical explicit momenta. All three generic targets (QR LGSSM, nonlinear UKF,
nonlinear cubature) passed this compilation/parity check. This is bounded
engineering evidence, not posterior or GPU qualification.

The focused adapter/remaining-test run passed 51 tests; two bootstrap consumer
checks initially lacked tracked provenance scripts omitted by the sparse
checkout. Materializing those scripts resolved both checks, without changing
the consumers. A premature retry before all scripts were present is preserved
alongside the other failed receipts. The final two consumer checks passed.
All selected tests therefore have passing evidence on the final runtime;
counts across these runs overlap and should not be summed without deduplication.
The only changes after the larger numerical run were the three test files.

Receipts, exact commands, environment, source hashes, and failed attempts are
under `docs/plans/artifacts/hmc-git-integration-2026-10-05/`. Total enclosing CPU
validation time, including native builds and failed attempts, was
**1,087.305 seconds**, within the 2,000-second cap. Each enclosing receipt was
charged once; the unused reservation was released. The release CPU balance is
2,948.257 seconds, including the unchanged 1,800-second monitoring reservation.
GPU allocation and the running frozen source were unchanged.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action |
| --- | --- | --- | --- | --- |
| Ready for the requested Git integration | Conflicts resolved; focused source, replay, model, monitoring, and guide checks pass | No unresolved selected-test failure or merge conflict | This is scoped CPU validation; no whole-repository or new GPU qualification | Commit the merge, refresh remote state, and push without force |

The final review checked missing dependencies, both sides of the conflicts,
source immutability during the tests, and preservation of unrelated shared
edits. It found no additional blocker to the requested Git operation. The
generated guide tables were checked by the documentation suite; a full book
PDF was not rebuilt in this integration.

No release or default-promotion verdict is made here.
