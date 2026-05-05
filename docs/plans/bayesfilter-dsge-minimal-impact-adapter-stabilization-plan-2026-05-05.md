# Experiment plan: DSGE minimal-impact adapter stabilization

## Question

Can the DSGE project finish the BayesFilter structural adapter pilot with
minimal effect on existing DSGE economics, filters, HMC code, and tests?

## Mechanism being tested

This plan isolates adapter metadata and legacy-path gating only.  It does not
move DSGE equilibrium logic into BayesFilter and does not promote the existing
DSGE SVD sigma-point path to a structural nonlinear filter.

Current read-only preflight found that `/home/chakwong/python` commit
`8645623 Expose DSGE structural metadata for BayesFilter` already added:

- `bayesfilter_state_names`;
- `bayesfilter_stochastic_indices`;
- `bayesfilter_deterministic_indices`;
- `bayesfilter_innovation_dim`;
- deterministic-completion bridges for Rotemberg NK and SGU;
- `tests/contracts/test_structural_dsge_partition.py`.

Therefore the remaining DSGE work is stabilization, audit, and integration
evidence, not a broad implementation from scratch.

## Scope

- Variant: DSGE client-repo adapter stabilization handoff.
- Objective: prove that DSGE models expose explicit BayesFilter structural
  metadata while keeping existing DSGE behavior stable and claim labels
  conservative.
- Seed(s): none unless existing DSGE tests require them.
- Training steps: none.
- HMC/MCMC settings: none; no sampler promotion in this pass.
- XLA/JIT mode: only existing DSGE contract tests and tiny metadata checks.
- Expected runtime: focused DSGE contract tests should be CI-scale; extended
  HMC tests remain out of scope.

## Success criteria

- SmallNK remains all-stochastic and passes the BayesFilter DSGE gate.
- Rotemberg NK and SGU expose mixed structural metadata and pass the
  BayesFilter gate only because completion bridges exist.
- EZ or other unaudited models remain fail-closed until timing and structural
  partition metadata are audited.
- Legacy full-state SVD sigma-point paths for mixed DSGE models are blocked by
  default or require an explicit approximation label.
- No DSGE economics, priors, parameter transforms, HMC samplers, or solver
  equations are moved into BayesFilter.
- No `converged`, `production-ready`, or unlabeled structural-filter claim is
  added.

## Diagnostics

Primary commands for the `/home/chakwong/python` agent:

- `pytest -q tests/contracts/test_structural_dsge_partition.py`
- `pytest -q tests/contracts/test_model_contracts.py`
- `pytest -q tests/contracts/test_svd_lgssm_reference.py tests/contracts/test_svd_generic_nonlinear_ssm.py`

Secondary commands:

- `python -m py_compile src/dsge_hmc/models/structural_metadata.py src/dsge_hmc/models/small_nk.py src/dsge_hmc/models/rotemberg_nk.py src/dsge_hmc/models/sgu.py src/dsge_hmc/filters/_svd_filters.py`
- `rg -n "converged|production-ready|sampler-usable|legacy full-state|approximation" docs src tests`
- `git diff --check`

BayesFilter-side confirmation after the DSGE commit:

- Import the DSGE models with `/home/chakwong/python/src` on `sys.path`.
- Run `bayesfilter.adapters.dsge.dsge_structural_adapter_gate` on SmallNK,
  Rotemberg, SGU, and an unaudited model such as EZ.
- Confirm metadata regimes are `all_stochastic`, `mixed_structural`,
  `mixed_structural`, and fail-closed respectively.

## Expected failure modes

- A mixed DSGE model passes through legacy full-state SVD filtering without an
  approximation label.
- A deterministic-completion bridge overwrites stochastic coordinates.
- Metadata disagrees with `n_states()`, `n_shocks()`, or nonzero rows of
  `eta`.
- Tests instantiate expensive solvers or HMC runs by default.
- An unaudited model accidentally inherits metadata from a base class and passes
  before a timing audit.

## What would change our mind

- If Rotemberg or SGU completion bridges fail identity tests, keep those models
  blocked and only promote SmallNK adapter metadata.
- If legacy SVD gating breaks many unrelated DSGE tests, keep the new default
  blocked but add explicit approximation labels to the affected tests rather
  than relaxing the guard.
- If EZ timing metadata is unclear, do not infer it; create a separate EZ
  timing audit plan.

## Command

```bash
cd /home/chakwong/python
pytest -q tests/contracts/test_structural_dsge_partition.py
pytest -q tests/contracts/test_model_contracts.py
pytest -q tests/contracts/test_svd_lgssm_reference.py tests/contracts/test_svd_generic_nonlinear_ssm.py
git diff --check
```

## Interpretation rule

- If focused DSGE metadata tests pass and BayesFilter gates classify the models
  correctly, the DSGE adapter pilot can be marked client-side metadata-ready.
- If only SmallNK passes, the next BayesFilter phase may proceed only for
  all-stochastic DSGE evidence; mixed DSGE particles/HMC remain blocked.
- If mixed models need unlabeled full-state SVD paths to pass tests, stop and
  keep DSGE mixed structural filtering blocked.

## Path dependency

The dependency path is:

```text
DSGE metadata stabilization
  -> BayesFilter DSGE gate confirmation
    -> DSGE structural particle evidence
      -> DSGE derivative/backend certification
        -> DSGE HMC sampler diagnostics
```

MacroFinance provider evidence and generic BayesFilter particle evidence can
continue independently, but DSGE-specific particle or HMC claims should not
advance until this metadata stabilization gate passes.

## Minimal-impact instructions for the `/home/chakwong/python` agent

1. Treat `8645623` as the baseline.  Do not rewrite the DSGE adapter work
   unless tests prove it incorrect.
2. Keep metadata as class attributes or tiny helper functions near model
   classes.  Do not add a new adapter framework unless repeated code becomes a
   real maintenance problem.
3. Keep `src/dsge_hmc/models/structural_metadata.py` lightweight and
   dependency-free except NumPy.
4. Do not alter canonical DSGE residuals, parameter transforms, priors,
   perturbation solvers, or HMC orchestrators.
5. For SmallNK:
   - keep `state_names=("a", "v")`;
   - keep all states stochastic;
   - require no deterministic-completion map.
6. For Rotemberg:
   - keep `state_names=("R", "g", "z", "dy")`;
   - keep stochastic indices `(0, 1, 2)` and deterministic index `(3,)`;
   - test that completion preserves `(R,g,z)` and makes `dy` satisfy the
     documented output-growth identity.
7. For SGU:
   - keep stochastic indices for exogenous states `(a,zeta,mu)`;
   - keep deterministic indices for endogenous/predetermined states;
   - test that completion preserves exogenous coordinates.
8. For EZ and future models:
   - leave fail-closed until a model-specific timing audit exists.
9. Preserve the legacy SVD sigma-point path only as an approximation:
   - mixed DSGE full-state path blocked by default;
   - explicit opt-in plus nonempty approximation label required for old smoke
     tests.
10. Commit the `/home/chakwong/python` work separately and record the commit
    hash in the BayesFilter reset memo before BayesFilter proceeds.

## Phase D0: verify current DSGE metadata baseline

Plan:
- Confirm that the current `/home/chakwong/python` HEAD includes the metadata
  commit and that imports expose the expected fields.

Execute:
- Read model files and contract tests.
- Run a cheap import preflight from BayesFilter.

Test:
- Metadata fields exist on SmallNK, Rotemberg, and SGU instances.

Audit:
- Existing work is already minimal-impact: model-local attributes, a helper
  module, focused contract tests, and legacy SVD path guards.

Interpretation:
- D1 is justified; broad implementation from scratch is not.

## Phase D1: focused `/python` test execution

Plan:
- Run only tests that validate metadata and guard legacy paths.

Execute:
- Run `tests/contracts/test_structural_dsge_partition.py` first.
- If green, run the two smaller existing model/SVD reference contract tests.

Test:
- Focused tests pass without extended HMC.

Audit:
- Failures should be fixed in the smallest local file only.

Interpretation:
- If D1 passes, D2 is justified.

## Phase D2: BayesFilter gate confirmation

Plan:
- Verify the client metadata through the BayesFilter gate.

Execute:
- Import DSGE models from `/home/chakwong/python/src`.
- Evaluate `dsge_structural_adapter_gate` on SmallNK, Rotemberg, SGU, and EZ.

Test:
- Expected regimes: `all_stochastic`, `mixed_structural`,
  `mixed_structural`, and fail-closed.

Audit:
- No structural path is promoted from state names alone; completion maps are
  required for mixed models.

Interpretation:
- If D2 passes, BayesFilter can move from `blocked_pending_client_metadata` to
  `client_metadata_ready_for_structural_tests`.

## Phase D3: reset memo and blocker update

Plan:
- Record the `/python` commit hash, test results, and remaining DSGE limits.

Execute:
- Update the BayesFilter reset memo and blocker register.

Test:
- Source-map YAML parse and `git diff --check`.

Audit:
- The reset memo must not claim DSGE structural particle correctness or HMC
  convergence.

Interpretation:
- If D3 passes, the next BayesFilter phase may plan DSGE structural particle
  evidence on the certified metadata.
