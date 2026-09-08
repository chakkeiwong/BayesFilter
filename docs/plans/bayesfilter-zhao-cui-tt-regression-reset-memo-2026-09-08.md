# Zhao-Cui TT Regression Workspace Reset Memo

Date: 2026-09-08  
Status: `RESET_COMPLETE_FOCUSED_INTEGRATION_PASS_PHASE8E_NOT_LAUNCHED`  
Workspace: `/home/chakwong/BayesFilterZhaoCui`  
Branch: `zhao-cui-tt-regression-20260908`  
Remote: `git@github.com:chakkeiwong/BayesFilter.git`  
Current-main base: `d2124d425b0ea0ae0e3e5f4246bd6b03ff8a2170`  
Recovery source: `21d5870f8320ba03ef017315ef3c62638e789fef`  
Scope: Zhao-Cui/C2 tensor-train regression and observation-informed proposal work only

## 1. Reset decision

The previous working branch accumulated two unrelated programs:

1. canonical LEDH, Contract-E, Younis/KDM, and RQMC work; and
2. Zhao-Cui/C2 TT-regression, mixture-UKF/APF, exact-likelihood Laplace,
   recursive-map, and DMIS work.

That arrangement was wrong for continued development. It made branch status
ambiguous, allowed a cleanup for one program to remove files needed by the
other, and made apparently unrelated tests and policy documents look like
dependencies. Contract-E is not part of the Zhao-Cui TT-regression problem.

The repair is a real repository boundary, not a naming convention:

- `/home/chakwong/BayesFilter` remains the LEDH-side checkout;
- `/home/chakwong/BayesFilterZhaoCui` is a separate clone for Zhao-Cui/C2;
- the Zhao-Cui clone has its own branch, index, working tree, and remote;
- only a named Zhao-Cui/C2 file manifest is imported into the new branch; and
- no whole mixed-branch merge is performed.

A wholesale merge of `ledh-refactor-with-policy-fix` into `main` would copy the
same organizational error into the shared branch. The correct operation is to
preserve the old work, start from current GitHub `main`, and restore only the
Zhao-Cui/C2 files whose provenance is understood.

## 2. Preservation of the old checkout

Before creating this workspace, the then-present LEDH-side changes were
committed and pushed as:

```text
94d073324596ee6c3931dc56ef562ce3d07c31c9
Repair Contract-E covariance carry and KDM diagnostics
branch: ledh-refactor-with-policy-fix
```

That commit is an archival checkpoint for the old checkout. It is not a
Zhao-Cui dependency and is not merged here. Additional unrelated changes
appeared in the old checkout after that checkpoint. They were not made,
staged, committed, restored, or copied as part of this reset.

The two inherited Codex subagent sessions visible at reset time were already
interrupted. No live subagent is assigned to this workspace.

## 3. Base correction

The first local clone was made from the source checkout's local `main` at
`d26edcdfee29474913a82f9e8c15705cd6d0da1d`. A remote check showed that this
was stale and divergent from GitHub. The new clone was therefore pointed at
the actual GitHub remote, fetched, and reset before import to:

```text
d2124d425b0ea0ae0e3e5f4246bd6b03ff8a2170
Clarify canonical HMC tuning routes and handoff
```

The branch is therefore based on current GitHub `main`, not the stale local
branch and not the mixed LEDH branch.

## 4. Zhao-Cui recovery provenance

The imported program is reconstructed from named commits rather than from the
dirty old working tree.

| Commit | Role in this reset |
| --- | --- |
| `665d8ce06ab0c0b753818c4cb479d035f02d16d9` | Earlier C2 generic-DMIS checkpoint and base-mass-aware evaluator provenance |
| `21d5870f8320ba03ef017315ef3c62638e789fef` | Selected recovery tree containing the restored Phase 8 call chain, tests, plans, and fixtures |
| `167e10ded87c3a07cdb1722c93207f97efae6b12` | Historical memo recording the prior root restoration; useful provenance, not imported authority |
| `94d073324596ee6c3931dc56ef562ce3d07c31c9` | Old mixed-branch preservation checkpoint; explicitly excluded as an import source |

The selected tree is not merged as history because its ancestry contains
unrelated work. Instead, 84 named source paths were read from `21d5870f` and
restored onto current `main`. Of those paths, 68 differ from current `main` and
form the branch delta; the remainder were already identical on current main.

Two additional ignored C2 fixtures required by the imported tests were copied
from the old checkout and force-added as ordinary test inputs:

| Fixture | SHA-256 |
| --- | --- |
| `docs/benchmarks/fixtures/c2_sv_n4_seed52_obs42_t20_frozen_v1.json` | `459b5033321e6878a966a61e897efb0f9e3b1b977497e88860b12842e2f0cfce` |
| `docs/benchmarks/fixtures/c2_mixture_ukf_lgssm_phase0_v1.json` | `388e32c9f50b01a35fcb9cf3c6d3b2dda9501e7a48356ade1be56e0759be4927` |

Unlike the historical attempt05 outputs, these files are executable test
dependencies. Tracking them makes the branch's focused contracts reproducible
from Git rather than dependent on another checkout.

The imported categories are:

- `bayesfilter/highdim/c2_*.py`;
- the C2-required Hermite, Gaussian-TT, exact-likelihood Laplace, DMIS,
  RBF/hybrid-basis, recursive-map, and frozen-proposal modules;
- C2 benchmark and diagnostic programs;
- frozen C2 Phase 8E calibration fixtures and bin boundaries;
- C2 plans, memos, and Lean files; and
- focused C2 and directly supporting tests.

The import does not contain a staged path named for LEDH, Contract-E, RQMC,
Younis/KDM, or surrogate-force work. References to LEDH inside C2 scientific
documents remain only where they explain a comparator or a scope boundary.
They do not make the LEDH implementation part of this branch's candidate call
chain.

## 5. Historical attempt05 evidence

The four documents that began the n=4 investigation are ignored by the
repository-wide artifact rule, so they cannot be reconstructed by cloning Git
alone. They were copied byte-for-byte from the old checkout into the same
relative path in this workspace:

```text
docs/benchmarks/artifacts/c2_completion_20260824/attempt05/
```

Their reset-time SHA-256 values are:

| File | SHA-256 |
| --- | --- |
| `attempt05_n4_failure_analysis.pdf` | `a6f6230892551caf6bd8163187ac16c0c04f3ab1dc09a1387cd0e539b18f4747` |
| `attempt05_n4_failure_analysis.tex` | `536681032a2758687816667d7e5b5f9ade3d8a30c796afca261889694d20c517` |
| `n4_diagnostic_handoff_memo_20260828.md` | `a2c7d80b0b2795c986f206eb725d92689d1283e02fa7ebcd672fd96767473654` |
| `n4_diagnostic_t20_result_addendum_20260828.md` | `ad20891f2e8ced29761b812f3c96f6de2fc2d5bfd9286536a4d1ea5c9c4c6a48` |

These files remain ignored local evidence. They are not staged in the reset
commit and their presence must not be inferred from a future Git checkout.

## 6. Scientific question retained by the reset

The program is about the Zhao-Cui/C2 filtering failure, not Contract-E.

The original GH9/squared-TT route attempted to fit and recursively propagate a
density representation whose direct normalizer and tails became unreliable in
the n=4 fixture. The current candidate direction asks whether an
observation-conditioned proposal can localize sampling while exact target and
complete-mixture importance weights preserve correctness:

1. construct per-ancestor observation-informed proposals with UKF or exact
   likelihood/Laplace guidance;
2. retain a full-support defensive proposal channel;
3. evaluate the exact transition-observation target;
4. divide by the complete proposal mixture, not the selected component;
5. use a lagged, recursively updated moment map for TT coordinates; and
6. preserve the declared frozen finite program for the analytical gradient.

The TT approximation can guide proposals or represent a local residual. It is
not allowed to replace exact importance correction merely because its fitted
Gram or direct normalizer is finite.

## 7. Research intent ledger

| Field | Reset declaration |
| --- | --- |
| Main question | Can the restored observation-informed C2 proposal program avoid the n=4 localization failure while preserving exact complete-mixture weights and the declared analytical gradient? |
| Candidate mechanism | Per-ancestor UKF/APF and exact-likelihood Laplace proposals, a defensive component, and a lagged recursive TT coordinate map |
| Expected failure mode | Weak observation response, invalid curvature/Newton rows, proposal-target mismatch, collapsed ESS, unstable recursive map, or a finite but wrong TT normalizer |
| Exact baseline | The C2 frozen finite-program evaluator with exact transition and likelihood factors and complete proposal denominator |
| Cheap heuristic adversaries | Bootstrap conditional, per-ancestor UKF K=1, transformed Student proposal, and stationary Gaussian independence; retained-TT remains a diagnostic/reference arm |
| Promotion criterion | The active phase's predeclared validity and heuristic screens, followed by fixture-cluster uncertainty evidence for the exact downstream filtering quantity |
| Promotion veto | Any invalid exact-weight branch, analytical-gradient mismatch, or loss to a declared cheap heuristic in a salient active-time condition |
| Continuation veto | Corrupt target/fixture, missing call-chain dependency, irreconcilable source mismatch, invalid comparator, required GPU unavailable after trusted probe, or exhausted phase budget |
| Repair trigger | Candidate-only loss, localized adapter/schema failure, fixed-schedule failure, or bounded infrastructure failure that leaves the scientific target intact |
| Explanatory only | Held-out fit residuals, shell residuals, Gram condition, TT rank, ESS without uncertainty, runtime, and one-seed differences |
| Must not be concluded | No posterior correctness, general-model validity, unbiased-likelihood, HMC-readiness, production-readiness, default-readiness, or superiority follows from the reset or focused tests |

## 8. Current evidence status

The imported master program labels Phase 8C as a fresh-path validity pass with
the candidate unpromoted. The later recovery plan records a Phase 8D schedule
repair and says Phase 8E is ready after clean integration. Those statements
were produced in a recovery checkout, not this new branch.

Accordingly, reset-time operational status is deliberately stricter:

```text
engineering integration: focused CPU-only verification passed
Phase 8C evidence: historical descriptive evidence
Phase 8D evidence: historical recovery evidence pending call-chain tie-out
Phase 8E: not launched from this branch
candidate promotion: no
default change: no
```

Passing import, compile, and focused tests will establish only that the
recovered call chain is present and mechanically coherent on current main. It
will not reproduce Phase 8C or Phase 8D numerical evidence.

## 9. Skeptical pre-execution audit

This audit covers the reset and focused integration verification only. It does
not authorize a Phase 8E statistical campaign.

| Audit question | Finding |
| --- | --- |
| Wrong baseline | Found and repaired: stale local `main` was replaced by current GitHub `origin/main` before import |
| Proxy promoted as evidence | Prevented: compile/tests are engineering checks only; ESS and fit diagnostics remain explanatory unless a phase declares otherwise |
| Missing stop conditions | Repaired: boundary contamination, missing source files, call-chain failures, or invalid fixtures stop integration; scientific vetoes remain those of the active phase plan |
| Unfair comparison | No scientific comparison is run during reset; later arms must share observations, exact target, complete denominator, and declared particle/time budgets |
| Hidden source assumption | Exposed: `21d5870f` is the selected recovery tree; mixed-branch working files are not treated as authority |
| Environment mismatch | Focused verification is CPU-only with GPU hidden and must be labeled as such; later GPU/XLA runs require trusted GPU access and memory-growth recording |
| Artifact mismatch | The reset memo, Git commit, test command/results, and future versioned run directories answer distinct questions; old ignored artifacts are not silently promoted |
| Model-specific patch risk | The restored candidate uses the generic exact-likelihood/Laplace and complete-mixture construction; no small-observation threshold patch is introduced by this reset |

Verdict: the reset plan is adequate for bounded integration verification after
the explicit source boundary and current-main correction. A Phase 8E launch
still requires the Phase 8E evidence contract, clean branch commit, trusted
GPU probe, versioned output root, and its stated compute budget.

## 10. Default and assumption audit

| Choice | Provenance | Status | Failure mode | Earliest check |
| --- | --- | --- | --- | --- |
| GitHub `main` at `d2124d42` | Current remote on 2026-09-08 | Repository baseline | Stale or force-moved base | Record base hash; fetch before integration or merge |
| Recovery tree `21d5870f` | Prior Phase 8 integration recovery | Recovery source, not scientific oracle | Missing dependency or stale interface on new main | Import, compile, call-chain tests |
| Base-mass-aware frozen evaluator | `665d8ce0` and recovery memo | Required finite-program boundary | Silent target/measure change | Exact denominator and APF identity tests |
| `quarter_long_12` schedule | Phase 8D recovery result | Frozen candidate control, not universal default | Path-specific calibration or Newton failure | Fixed-schedule validity on untouched fixtures |
| Phase 8E `N=4096`, `T=20`, two branches | Phase 8E plan | Planned research budget | Underpowered or high Monte Carlo variance | Fixture-level uncertainty and active-time table |
| Hermite/RBF/hybrid representation arms | Prior C2 plans | Diagnostic candidates | Tail mismatch, ill-conditioned Gram, rank inflation | Held-out/shell residual, Gram, rank, recursive error |
| CPU-only focused verification | Reset procedure | Engineering exception | Cannot establish GPU/XLA behavior | Later trusted GPU/XLA smoke before campaign |

## 11. Integration verification checklist

The following checks must be completed before the reset status changes to
`RESET_COMPLETE`:

- [x] Separate clone created at `/home/chakwong/BayesFilterZhaoCui`.
- [x] Branch `zhao-cui-tt-regression-20260908` created.
- [x] Remote corrected to the GitHub repository.
- [x] Branch based on current GitHub `main` at `d2124d42`.
- [x] Explicit C2 manifest restored from `21d5870f`.
- [x] Staged path scan contains no unrelated program path.
- [x] Attempt05 historical files copied with matching SHA-256 values.
- [x] Required C2 test fixtures are tracked with recorded hashes.
- [x] `git diff --check` passes.
- [x] `HermiteBasis1D` and required C2 adapters resolve from this checkout.
- [x] Changed Python sources compile.
- [x] Focused CPU-only integration and call-chain tests pass.
- [x] Reset result and exact commands are added below.
- [ ] Reset commit is pushed to the dedicated remote branch.

## 12. Focused verification record

### Environment

```text
conda environment: tftwogpu
execution mode: deliberate CPU-only verification
CUDA_VISIBLE_DEVICES: -1
Python bytecode/cache: redirected to /tmp or disabled
pytest cache provider: disabled
GPU/XLA evidence: none claimed by this verification
```

### Compile check

The first compile command attempted to write `__pycache__` inside the sibling
clone and failed with read-only-filesystem errors. It did not expose a syntax
error. Redirecting bytecode to `/tmp` repaired the harness:

```bash
mkdir -p /tmp/bayesfilter-zhaocui-pycache-20260908
env CUDA_VISIBLE_DEVICES=-1 \
  PYTHONPYCACHEPREFIX=/tmp/bayesfilter-zhaocui-pycache-20260908 \
  /home/chakwong/anaconda3/bin/conda run -n tftwogpu \
  python -m compileall -q bayesfilter/highdim docs/benchmarks tests/highdim
```

Result: pass, no compiler output.

### Fresh-clone custom-op bootstrap

Initial pytest collection failed because the ignored TensorFlow custom op
`bayesfilter/ops/_symmetric_sylvester_ops.so` is not present in a Git clone.
Two source-build attempts then identified pre-existing environment/build
limitations:

1. CMake found `cuda_runtime_api.h` but omitted the companion
   `nvidia/cuda_nvcc/include` directory containing `crt/host_defines.h`.
2. Supplying that include directory reached TensorFlow's XLA headers, which
   require unavailable standalone `llvm/ADT/ArrayRef.h` headers.

This is a build-bootstrap defect, not a C2 numerical failure. The C++ op source
and custom-call stub were byte-identical in both checkouts:

```text
symmetric_sylvester_op.cc  151ff416363e2b75c7050fb8143a34c9f6c319c5892056c20974642cb045ff7d
custom_call_status_stub.cc  50860d1454fa31c962d946a1c200b2b8613333b425a7ba00706b9039d944a3be
```

The existing same-environment binary passed a direct import smoke and was
copied into the new checkout as an ignored local bootstrap artifact:

```text
_symmetric_sylvester_ops.so  661f11b9db1f6e9ab9ce4aae8ae86591779cdbdfe6fda777a7c87956ec2686fa
```

A future fresh machine still needs either a compatible binary or a repaired
toolchain with the CUDA-NVCC and LLVM headers. This does not block work in the
present clone, but it remains explicit engineering debt and must not be
mistaken for a portable source-build pass.

### Focused pytest progression

The first run after loading the custom op completed collection and reported
`74 passed, 25 failed`. Every failure was a `FileNotFoundError` for one of the
two ignored C2 fixtures now tracked in this branch. No numerical assertion
failed. After adding those fixtures, the unchanged suite passed:

```bash
env CUDA_VISIBLE_DEVICES=-1 \
  PYTHONDONTWRITEBYTECODE=1 \
  TF_CPP_MIN_LOG_LEVEL=2 \
  MPLCONFIGDIR=/tmp/bayesfilter-zhaocui-mpl-20260908 \
  /home/chakwong/anaconda3/bin/conda run -n tftwogpu \
  python -m pytest -q -p no:cacheprovider \
  tests/highdim/test_c2_hermite_basis.py \
  tests/highdim/test_c2_coherent_plan_math.py \
  tests/highdim/test_c2_sv_frozen_proposal_apf_tf.py \
  tests/highdim/test_c2_exact_likelihood_laplace_adapter.py \
  tests/highdim/test_exact_likelihood_laplace_apf_tf.py \
  tests/highdim/test_c2_mixture_ukf_apf_phase0.py \
  tests/highdim/test_c2_mixture_ukf_apf_phase1.py \
  tests/highdim/test_c2_mixture_ukf_apf_phase4.py \
  tests/highdim/test_c2_mixture_ukf_apf_phase4_repair.py \
  tests/highdim/test_c2_mixture_ukf_apf_phase5a.py \
  tests/highdim/test_c2_mixture_ukf_apf_phase7.py \
  tests/highdim/test_c2_phase2_generic_dmis_repair_wiring.py \
  tests/highdim/test_c2_phase8b_comparator_contract.py \
  tests/highdim/test_c2_phase8d_aggregate_contract.py \
  tests/highdim/test_c2_phase8d_schedule_repair_contract.py \
  tests/highdim/test_c2_phase8e_launcher_contract.py \
  tests/highdim/test_c2_phase8e_statistical_aggregate.py \
  tests/highdim/test_recursive_moment_map_tf.py \
  tests/highdim/test_rbf_basis_tf.py \
  tests/highdim/test_hybrid_basis_tf.py
```

Result: `99 passed, 2 deprecation warnings in 20.51s`.

The warnings come from TensorFlow Probability's use of
`distutils.version`. They are not test failures and do not affect the tested
C2 quantities.

### Interpretation

The focused result establishes that the selected C2 basis, evaluator,
exact-Laplace adapter, mixture-UKF phases, recursive map, representation
mechanics, and Phase 8 contract/analysis helpers integrate on current main.
It does not reproduce Phase 8D, exercise the GPU/XLA campaign route, or provide
new statistical evidence about proposal quality.

## 13. Next scientific action after reset

Do not restart from Contract-E and do not re-run every historical C2 phase.
After the focused integration checks pass:

1. reconcile the master-program header with the later recovery record;
2. verify the Phase 8E runner reaches the restored general implementation
   through executable wiring tests;
3. review the Phase 8E evidence contract against current code and current
   repository policy;
4. run the smallest GPU/XLA smoke with memory growth and structured manifest;
5. repair localized harness defects within the unchanged phase budget; and
6. launch the predeclared Phase 8E ladder only if no continuation veto fires.

Every serious run must use a new versioned output directory and record commit,
command, conda environment, CPU/GPU mode, TensorFlow memory policy, XLA/TF32
settings, data/fixture hashes, seeds, wall time, plan, and result path.

## 14. Branch discipline from this point

- Zhao-Cui/C2 work is edited, tested, committed, and pushed only from
  `/home/chakwong/BayesFilterZhaoCui` on its dedicated branch.
- LEDH, Contract-E, Younis/KDM, RQMC, and surrogate-force implementation work
  is not added to this branch.
- A comparison may be documented without importing the compared program.
- Current `main` may be merged or rebased only after inspecting the incoming
  paths and rerunning the focused C2 contracts.
- This branch is not merged to `main` merely because mechanics tests pass.
  Scientific promotion and repository integration are separate decisions.
- Unknown changes in another checkout are preserved and reported, not swept
  into this branch or silently reverted.

## 15. Reboot entry point

The authoritative starting documents are:

1. this reset memo for workspace and evidence boundaries;
2. `docs/plans/bayesfilter-c2-mixture-ukf-apf-master-program-2026-09-02.md`
   for the scientific program;
3. `docs/plans/c2-phase8-recovery-20260907.md` for recovery provenance; and
4. `docs/plans/c2-phase8e-statistical-replication-20260907.md` for the next
   proposed research campaign.

If these documents disagree about a completed numerical result, preserve the
disagreement until the cited artifact and executable call chain are checked.
The reset memo governs workspace separation; it does not retroactively certify
historical scientific claims.
