# LEDH Canonical Batch Fused tf.while_loop Refactor — Master Program

**Program ID**: `ledh-while-loop-refactor-2026-08-30`  
**Target Branch**: `worktree-ledh-canonical-rebuild`  
**Owner Authorization**: Required before Phase 1 execution  
**Created**: 2026-08-30  
**Status**: AWAITING_OWNER_APPROVAL

---

## Executive Summary

Refactor `canonical_batch_fused_value_score` in `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py` from Python-unrolled horizon/substep loops to `tf.while_loop` with explicit multi-direction forward-mode tangent propagation, addressing the measured 6× graph-size explosion (663,766 nodes) and 281.6 s trace time in the surrogate-force HMC gradient path.

**Performance Root Cause** (quantified 2026-08-30):
- Python `for time_index in range(horizon)` unrolling emits ~2,200 nodes per timestep
- At horizon=50, one LEDH call produces 110,628 nodes
- Surrogate-force HMC custom gradient calls LEDH 6 times per gradient (1 value + 5 swept directions for dim=5 theta)
- Total: 663,766 nodes, 61.24 MB GraphDef, 101.8 s trace, 5,245 MB RSS
- Swept form preserves arithmetic efficiency (1.23× base via Grappler CSE) but multiplies graph size 6×
- Naive batched-direction form (theta_directions [B, P]) halves graph size (2.00× base) but defeats CSE (3.51× arithmetic) and is 2.62× slower warm (0.485 s vs 0.185 s/eval)

**Solution Architecture**:
- Replace Python unrolling with `tf.while_loop(maximum_iterations=horizon * substeps)`
- Propagate multi-direction tangents explicitly: forward-mode JVP with K seed vectors, tangent state `[m, K, dim]`
- Full gradient in one `tf.while_loop` body makes work-sharing structural (not optimizer-dependent)
- Hoist invariant tensor construction (UKF sigma-point weights, alpha/beta/kappa constants) outside the loop
- Precompute read-only slices (per-timestep noises, observations) before the loop
- Target: ~2,200-node bounded body × 1 subgraph = O(10³) nodes total vs current O(10⁶); trace time O(10 s) vs 101.8 s

**Hard Constraints** (from user governing directive):
1. NO mid-execution choice questions to the user
2. NO mid-execution plan changes when difficulties arise
3. Mandatory phase repair step after each phase: repair what is repairable, refresh the plan for next phase
4. All allowlists, approvals, and decision rules predeclared before execution

---

## Phase Structure

### Phase 0: Test Audit, Coverage, and Refactor Contract (CURRENT)
- **Objective**: Establish baseline test health, install coverage tooling, write the binding refactor contract
- **Status**: AUDIT COMPLETE, REPAIR PENDING
- **Subplan**: [ledh-while-loop-refactor-phase0-subplan-2026-08-30.md](ledh-while-loop-refactor-phase0-subplan-2026-08-30.md)
- **Repair Gate**: Phase 0 repair MUST complete before Phase 1 execution begins

### Phase 1: Single-Direction tf.while_loop Conversion
- **Objective**: Convert the unrolled loops to `tf.while_loop` with one-direction tangent, preserve all intermediate state
- **Status**: NOT STARTED
- **Subplan**: [ledh-while-loop-refactor-phase1-subplan-2026-08-30.md](ledh-while-loop-refactor-phase1-subplan-2026-08-30.md)
- **Repair Gate**: Mandatory repair-and-refresh after Phase 1

### Phase 2: Multi-Direction Tangent Generalization
- **Objective**: Extend to K-direction tangent state `[m, K, dim]`, full gradient in one loop body
- **Status**: NOT STARTED
- **Subplan**: [ledh-while-loop-refactor-phase2-subplan-2026-08-30.md](ledh-while-loop-refactor-phase2-subplan-2026-08-30.md)
- **Repair Gate**: Mandatory repair-and-refresh after Phase 2

### Phase 3: Closure Hoisting and Constant Precomputation
- **Objective**: Move invariant construction outside the loop, precompute read-only per-timestep slices
- **Status**: NOT STARTED
- **Subplan**: [ledh-while-loop-refactor-phase3-subplan-2026-08-30.md](ledh-while-loop-refactor-phase3-subplan-2026-08-30.md)
- **Repair Gate**: Mandatory repair-and-refresh after Phase 3

### Phase 4: Final Integration and Leaderboard Validation
- **Objective**: Replace the unrolled kernel as the canonical batch-fused entry point, validate on the six-model leaderboard
- **Status**: NOT STARTED
- **Subplan**: [ledh-while-loop-refactor-phase4-subplan-2026-08-30.md](ledh-while-loop-refactor-phase4-subplan-2026-08-30.md)
- **Completion Gate**: Owner acceptance required for promotion to default

---

## Background

### Surrogate-Force HMC Architecture

The target use case is surrogate-force HMC (implemented in `docs/benchmarks/step1_true_surrogate_force.py`):
- MH accept/reject uses EXACT LEDH log-density: λ=0, δ=0 (no ridge on Q or R)
- Leapfrog force uses DAMPED score: λ=1e-3 ridge on process covariance, δ=1e-3 ridge on observation covariance
- Implementation: `tf.custom_gradient` where the forward pass calls the exact model once, and the `grad_fn` calls the damped model P times (once per parameter direction)
- Current implementation sweeps directions: `for p in range(param_dim): direction = one_hot(p); _, scores, _ = canonical_batch_fused_value_score(...)`
- At P=5, batch_size=6: forward 1 call + gradient 5 calls = 6 LEDH calls per HMC gradient evaluation
- Each call traces independently → 6× graph nodes

### Graph-Size Diagnosis (2026-08-30)

Three diagnostic benchmarks (`docs/benchmarks/diagnose_graph_size_20260830.py`, `diagnose_eval_time_20260830.py`, `diagnose_direction_cost_scaling_20260830.py`) established:

**Case A** (one LEDH call, B=1, one dummy direction):
- 110,628 nodes
- 9.98 MB GraphDef
- 16.6 s trace time
- RSS 1,336 MB

**Case B** (value + 5 swept directions, 6 calls total):
- 663,766 nodes (6.00× A)
- 61.24 MB GraphDef
- 101.8 s trace time
- RSS 5,245 MB
- Warm eval: 0.185 s/call

**Case C** (value + 1 batched-direction call, `theta_directions [B, P]`, 2 calls total):
- 221,484 nodes (2.00× A)
- 20.17 MB GraphDef
- 41.5 s trace time
- RSS 6,644 MB
- Warm eval: 0.485 s/call (2.62× slower than swept)

**Scaling by horizon** (linear):
- horizon=5: 11,133 nodes (one call)
- horizon=50: 110,628 nodes (one call)
- ~2,200 nodes per timestep body × 50 timesteps

**Arithmetic efficiency** (D=5, horizon=10, N=252):
- Swept (6 calls): graph 5.00× base, arithmetic 1.23× base → CSE working
- Batched (2 calls): graph 1.00× base, arithmetic 3.51× base → CSE defeated by tiling

**Interpretation**:
- Python unrolling creates replicated subgraphs; Grappler CSE deduplicates primal computation across swept calls
- Batching the direction dimension defeats CSE because the primal is now tiled `[B, K]` instead of scalar, breaking structural identity
- `tf.while_loop` with explicit multi-direction tangent state makes work-sharing structural: one primal path, tangent broadcast/matmul, no optimizer dependence

### TensorFlow Graph and Compilation Policy (CLAUDE.md)

- Repeated TensorFlow scientific kernels MUST execute through `tf.function` with explicit, stable `input_signature`
- Shape polymorphism and retracing must be bounded and justified
- TensorFlow pfor (including `tf.vectorized_map`, `GradientTape.jacobian`, `GradientTape.batch_jacobian`) requires prior written approval
- XLA evaluation required after compatibility, memory, numerical equivalence, compilation cost, steady-state performance, and downstream scientific checks pass under a recorded evidence contract

**Refactor Compliance**:
- `tf.while_loop` is the approved native TensorFlow loop with a single traced body
- No pfor, no `tf.vectorized_map`, no `GradientTape` higher-order methods
- XLA compatibility preserved (all ops in the current kernel are XLA-compatible; `tf.while_loop` is XLA-compatible)
- Final Phase 4 will include XLA smoke tests but XLA promotion remains a separate Phase 5 decision under an experiment plan

### NeuTra Batch-Native Training Rule (CLAUDE.md)

All BayesFilter NeuTra optimizer updates MUST be batched with batch size > 1. The transport, log determinant, target value/score, loss, gradient, and optimizer computation must preserve the leading batch dimension in TensorFlow/XLA.

**Refactor Target Kernel** (`canonical_batch_fused_value_score`):
- Already batch-native: theta [B, P] with theta_directions [B, P] → value [B], score [B]
- Note: score shape is [B], not [B, P] — one directional derivative per row; multi-parameter gradients sweep directions across calls
- Phase 2 extends to theta_directions [B, K, P] → score [B, K]
- Refactor MUST preserve batch-native structure (no Python row loop in the traced graph)
- The non-fused batch lane (`canonical_batch_value_score`) is explicitly PARITY/REFERENCE only and NOT NeuTra-training-eligible (Python row loop, documented in `ledh_canonical_batch_tf.py`)

### Analytical Score Contract (C-9)

The LEDH canonical route uses analytical recursive score, NOT autodiff (contract C-9, recorded in module docstrings). The forward-mode tangent recursion is hand-derived:
- UKF predict tangent (S1): Cholesky Phi differential via `chol_diff`
- Flow tangent (S3): per-particle flow Jacobian `a_matrix`, residual tangent `d_residual_e`, innovation Cholesky tangent
- Weight tangent (S4): Gaussian log-density tangent for transition/observation/proposal, softmax-weighted score increment
- UKF update tangent (S5): gain differential, posterior state/covariance tangent
- Accumulation (S8): `d_total += softmax_weighted_increment`

**Refactor Compliance**:
- The `tf.while_loop` body will propagate the tangent state explicitly (no `GradientTape`, no autodiff)
- Multi-direction extension: tangent loop state gains a LEADING K axis (`[K, m, dim]`), and the K tangent evaluations share one primal inside the loop body. See the [Phase 2 subplan](ledh-while-loop-refactor-phase2-subplan-2026-08-30.md) for why the trailing-K form `[m, K, dim]` originally specified here is not implementable: the model's tangent callbacks (`transition_mean_tangent_fn`, `observation_tangent_fn`) are user-supplied and take rank-2 tangents, so a K axis crossing that boundary breaks the model contract. Folding K into the point axis instead is the already-measured 3.51× CSE failure and is rejected on evidence.
- Parity gates MUST pass before promotion

### LEDH Per-Scope Tuning Rule (CLAUDE.md)

Every claim-bearing LEDH model run requires an offline tuning artifact for the exact model/target, route/reset family, horizon/prepared-data regime, particle count, dimensions, dtype/backend, chunk policy, and route-specific control family. Any changed bound field is a new tuning scope.

**Refactor Implication**:
- The refactored kernel is the SAME route (LEDH-PF-PF canonical with analytical score, UKF lifecycle, Contract E optional)
- Particle count N, dimensions, horizon T, dtype `tf.float32`/`tf.float64`, chunk policy, reset/correction controls ALL unchanged
- **Tuning artifacts remain valid** — the refactor changes only the graph representation (Python unrolling → `tf.while_loop`), not the mathematical program
- If parity gates pass, no retuning is required
- If parity gates fail beyond declared tolerance (rtol 5e-4 for op-order differences), investigate before proceeding

### DPF Transport Chunk Rule (CLAUDE.md)

Active DPF canonical, candidate, benchmark, leaderboard, and production-target routes must use `dpf_transport_exact_divisor_cap3000_v1`. For N≤3000, the only valid chunk extent is K=N; for larger N, use the largest divisor of N no greater than 3000.

**Refactor Compliance**:
- The kernel already delegates to the chunked transport implementation
- The `tf.while_loop` refactor does not change transport chunking
- Chunk policy validation remains in the transport layer

### Backend Rule (CLAUDE.md)

The repository implementation backend is TensorFlow / TensorFlow Probability. NumPy may appear ONLY in explicitly diagnostic code: tests, comparison fixtures, independent reference solutions, closed-form or finite-difference checks, and post-run diagnostic inspection.

**Refactor Compliance**:
- Target kernel: pure TensorFlow/TFP
- Parity tests may use NumPy for reference computation or tolerance checking (already the case)
- No NumPy in the refactored kernel itself

---

## Phase 0: Test Audit, Coverage, and Refactor Contract

### Audit Findings (COMPLETE)

**Canonical test subset run** (2026-08-30, 106 tests, 206.16 s):
- **21 failed**, **83 passed**, **8 errors**, 2396 deselected, 2 warnings

**Failure Root Causes**:

1. **API drift: `flow_substeps` vs `substeps` (6 failures + 1 kernel-side forwarding bug)**
   - `tests/highdim/test_ledh_canonical_batch_fused.py`: 3 failures
     - Lines 111, 119, 136, 143, 161 call `canonical_batch_fused_value_score(..., flow_substeps=10)`
     - Kernel signature (line 43 of `ledh_canonical_batch_fused_tf.py`): `substeps: int` (keyword-only)
     - **Test-side defect**: rename `flow_substeps=` → `substeps=` in all test calls
   
   - `tests/highdim/test_ledh_canonical_batch.py`: 3 failures
     - Line 83 calls `canonical_batch_value_score(..., flow_substeps=10)`
     - Kernel signature (line 43 of `ledh_canonical_batch_tf.py`): `substeps: int` (keyword-only)
     - **Test-side defect**: rename `flow_substeps=` → `substeps=` in test calls
   
   - **Kernel-side forwarding bug** (discovered 2026-08-30):
     - `ledh_canonical_batch_tf.py:75` forwards `substeps=substeps` to `canonical_value_and_analytical_score`
     - Single-cloud authority signature (line 68 of `ledh_canonical_score_tf.py`): `flow_substeps: int = 24` (keyword-only)
     - **Kernel-side defect**: change line 75 to `flow_substeps=substeps`

2. **Missing frozen-fixture JSONs (14 failures)**
   - `tests/highdim/test_ledh_contract_e_canonical_lgssm_phase5.py`: 13 tests fail on missing `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase5-tiny-fixture-freeze-v2-2026-07-14.json`
   - Same file: 1 test fails on missing `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase5-one-step-fixture-freeze-2026-07-14.json`
   - **Status**: These are frozen-fixture contract tests; fixtures were not checked into worktree during Phase 5 work
   - **Disposition**: OUT OF SCOPE for this refactor; record as pre-existing breakage

3. **Missing leaderboard artifact (1 failure + 8 errors)**
   - `tests/highdim/test_complete_highdim_phase1_canonical_targets.py` fails on missing `docs/plans/artifacts/complete-highdim-leaderboard/phase1-canonical-targets-2026-07-11.json`
   - **Status**: Artifact directory does not exist on worktree
   - **Disposition**: OUT OF SCOPE for this refactor; record as pre-existing breakage

4. **Collection errors (2 files, blocks 2396 tests)**
   - `tests/highdim/test_genut_shape_lm_tf.py`: `ModuleNotFoundError: No module named 'bayesfilter.highdim.cubature_genut_batch_tf'`
   - `tests/highdim/test_zhao_cui_austria_sir_lane_b_t2_score_tf.py`: `ImportError: cannot import name '_active_log_weight'`
   - **Status**: Unrelated missing modules
   - **Disposition**: OUT OF SCOPE; run canonical subset with `--ignore=` for both files

**Coverage Tooling**: ABSENT
- Only `pytest.ini` exists; no `pyproject.toml`, `setup.cfg`, or `tox.ini`
- `pytest-cov` not installed/configured
- **Blocks user requirement**: "ensure that we have good test coverage"
- **Phase 0 repair action**: install `pytest-cov`, add coverage config to `pytest.ini`, establish baseline coverage for the refactor target kernel

**Test Inventory** (worktree `tests/highdim/`):
- ~71 canonical LEDH / canonical LGSSM test files (~462 KB total)
- Includes: batch/batch_fused parity tests, UKF lifecycle, score recursion, Fisher identity, flow per-particle, model fidelity, NeuTra target contract, phase-numbered contract tests
- 2396+ tests overall (when collection errors resolved)

### Refactor Contract (BINDING)

The following constraints are **NON-NEGOTIABLE** and MUST be preserved by every phase:

#### Mathematical Contract
1. **Analytical score, no autodiff** (C-9): the score is the hand-derived forward-mode tangent recursion, NOT `GradientTape`
2. **UKF lifecycle**: alpha=1, kappa=0, 2*dim+1 sigma points (predictive, proposal, posterior)
3. **Per-particle flow**: LEDH-PF-PF route with optional Contract E reset and dual-cap correction
4. **Frozen inputs**: initial cloud, noises, observations shared across theta batch (NeuTra target contract)
5. **Parity tolerance**: rtol 5e-4 for batch-size-1 value AND score vs single-cloud authority (op-order differences)

#### API Contract
1. **Signature stability**: `canonical_batch_fused_value_score(model, theta, theta_directions, initial_states, initial_covariances, noises, observations, *, substeps, jitter) -> (value, score, diagnostics)`
2. **Batch-native**: theta [B, P] → value [B], score [B, P], diagnostics["program_valid"] [B]
3. **Keyword-only controls**: all hyperparameters after `observations` are keyword-only
4. **NaN-on-invalid**: if any step produces non-finite values, return NaN for that row (validity gate at the end)

#### Implementation Contract
1. **TensorFlow/TFP only**: no NumPy in the kernel (tests may use NumPy for reference/tolerance)
2. **tf.function compilable**: the refactored kernel MUST compile through `tf.function` with stable `input_signature`
3. **No pfor**: no `tf.vectorized_map`, no `GradientTape.jacobian`, no `GradientTape.batch_jacobian`
4. **XLA-compatible ops only**: all ops in the body must have tf2xla kernels (current kernel already satisfies this)
5. **GPU memory growth**: not a kernel responsibility; handled by the test/runner harness

#### Parity Gates (ALL MUST PASS)
1. **P-1 fused**: batch-size-1 value AND score parity vs single-cloud canonical authority (rtol 5e-4)
2. **Multi-row independence**: rows with different theta produce different outputs; rows with identical theta produce identical outputs
3. **tf.function compilability**: `@tf.function(input_signature=...)` decorator succeeds, no runtime errors

#### Non-Functional Contract
1. **No silent behavior changes**: if a phase introduces a numerical difference beyond rtol 5e-4, STOP and investigate before proceeding
2. **No API breaks**: existing callers (surrogate-force HMC driver, leaderboard runners, NeuTra training) must work without modification
3. **Preserve diagnostics**: `diagnostics["program_valid"]` and any other returned diagnostics must remain available

#### Out-of-Scope (EXPLICITLY EXCLUDED)
1. **Tuning**: existing tuning artifacts remain valid (same route, same mathematical program)
2. **XLA promotion**: XLA smoke tests in Phase 4, but promotion deferred to a separate experiment plan
3. **Fixture repair**: missing phase5/phase1 JSON fixtures are pre-existing breakage, not this refactor's responsibility
4. **Collection-error modules**: `test_genut_shape_lm_tf.py` and `test_zhao_cui_austria_sir_lane_b_t2_score_tf.py` out of scope

---

## Allowlist (Tool/Command Permissions)

The following tools and command patterns are PRE-APPROVED for the duration of this program. Agent MUST NOT ask user permission for these actions during execution (user governing directive: "no mid-execution choice questions").

### File Operations (Read/Write/Edit)
- **READ**: any file under `/home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild/`
- **WRITE/EDIT**: 
  - `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py` (refactor target)
  - `tests/highdim/test_ledh_canonical_batch_fused.py` (parity tests)
  - `tests/highdim/test_ledh_canonical_batch.py` (non-fused parity tests)
  - `bayesfilter/highdim/ledh_canonical_batch_tf.py` (Phase 0 forwarding bug fix)
  - `pytest.ini` (coverage config)
  - Any new test file under `tests/highdim/` for phase-specific regression tests
  - All phase subplan documents under `docs/plans/`
  - All phase result documents under `docs/plans/`

### Bash Commands (CPU-only testing)
- **APPROVED WITHOUT PROMPT**:
  - `cd /home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild && CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/highdim/test_ledh_canonical_batch_fused.py -v`
  - `cd /home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild && CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/highdim/test_ledh_canonical_batch.py -v`
  - `cd /home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild && CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/highdim/ --ignore=tests/highdim/test_genut_shape_lm_tf.py --ignore=tests/highdim/test_zhao_cui_austria_sir_lane_b_t2_score_tf.py -k canonical -v` (canonical subset)
  - `cd /home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild && CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/highdim/test_ledh_canonical_batch_fused.py --cov=bayesfilter.highdim.ledh_canonical_batch_fused_tf --cov-report=term-missing` (coverage measurement)
  - `pip list | grep pytest-cov` (check coverage tooling)
  - `conda list | grep pytest` (check pytest in tftwogpu env)
  - `git status` (worktree state)
  - `git diff <file>` (inspect changes)
  - `git add <file>` (stage changes)
  - `git commit -m "<message>"` (commit phase artifacts)
  - `python -c "import tensorflow as tf; print(tf.__version__)"` (TF version check)
  - Any diagnostic Python script under `docs/benchmarks/` with `CUDA_VISIBLE_DEVICES=-1` and `os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")` in the script

### Bash Commands (GPU testing, REQUIRES ESCALATION per CLAUDE.md GPU policy)
- **APPROVED BUT MUST USE ESCALATION** (user will grant escalation via tool permissions):
  - `cd /home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild && CUDA_VISIBLE_DEVICES=1 python -m pytest tests/highdim/test_ledh_canonical_batch_fused.py -v` (4080 SUPER, tftwogpu env)
  - Any GPU diagnostic script under `docs/benchmarks/` WITHOUT `CUDA_VISIBLE_DEVICES=-1`
- **RATIONALE**: Per CLAUDE.md GPU/CUDA sandbox policy, any command that detects, initializes, benchmarks, or uses GPU/CUDA/NVIDIA devices must run with escalated permissions. Treat non-escalated GPU failures as sandbox evidence only.

### Git Operations
- **APPROVED WITHOUT PROMPT**:
  - Stage and commit files after each phase repair (semantic commits per CLAUDE.md)
  - `git log --oneline -n 10` (recent history)
  - `git show <commit>` (inspect commit)
  - `git cat-file -e HEAD:<path>` (verify file existence)
- **NOT APPROVED** (require explicit user direction):
  - Push to remote
  - Merge worktree → main
  - Rebase
  - Force operations (`git reset --hard`, `git clean -f`)

### Python Package Installation
- **APPROVED**: `pip install pytest-cov` (in tftwogpu conda env, Phase 0 repair)
- **NOT APPROVED**: any other package installation without user direction

---

## Upfront Approval Requests

The following decisions require **OWNER APPROVAL BEFORE PHASE 1 EXECUTION**:

### A1: Phase 0 Repair Execution
**Decision**: Execute Phase 0 repair (fix `flow_substeps`/`substeps` API drift, install coverage tooling)?  
**Required for**: Phase 1 cannot start until parity gates are working and coverage baseline is established  
**User Response Required**: YES / NO / REVISE

### A2: Refactor Contract Acceptance
**Decision**: Accept the binding refactor contract (mathematical, API, implementation, parity gates, out-of-scope)?  
**If NO**: specify required changes  
**User Response Required**: YES / NO / REVISE

### A3: Allowlist Sufficiency
**Decision**: The pre-approved allowlist (file operations, bash commands, git operations) is sufficient for autonomous Phase 1-4 execution?  
**If NO**: specify additional permissions needed  
**User Response Required**: YES / NO / REVISE

### A4: Phase Repair Policy
**Decision**: After each phase, agent MUST repair localized issues (test failures, numerical discrepancies within tolerance, minor bugs) WITHOUT asking user permission, then refresh the plan for the next phase. Stop and ask ONLY for: (1) parity gate failures beyond declared tolerance, (2) fundamental architectural problems, (3) evidence that the tf.while_loop approach is not viable. Acceptable?  
**User Response Required**: YES / NO / REVISE

### A5: Program Execution Model
**Decision**: Once Phase 1 is approved, agent proceeds through Phase 1 → repair → Phase 2 → repair → Phase 3 → repair → Phase 4 autonomously (no mid-phase user questions), pausing only at phase boundaries for mandatory repair-and-refresh OR when a stop condition fires. Acceptable?  
**User Response Required**: YES / NO / REVISE

---

## Stop Conditions (Mandatory User Intervention)

Agent MUST STOP and report to user (do NOT proceed autonomously) if ANY of these occur:

1. **Parity gate failure beyond tolerance**: batch-size-1 value or score differs from single-cloud authority by more than rtol 5e-4 after a phase's implementation
2. **API break**: existing callers (surrogate-force HMC, leaderboard runners) fail due to signature changes
3. **Silent behavior change**: numerical outputs change beyond tolerance without an identified root cause
4. **tf.while_loop compatibility blocker**: a required operation cannot be expressed inside `tf.while_loop` (e.g., dynamic shape dependency, unsupported control flow)
5. **XLA incompatibility introduced**: a refactored op lacks a tf2xla kernel (current kernel is XLA-compatible; refactor must preserve this)
6. **Graph or host memory regression**: host RSS, graph node count, or GraphDef bytes increase by more than 2× at (B=6, horizon=50, N=252). GPU device-memory validation is explicitly deferred as a separate escalated step (see "What This Program Does Not Establish").
7. **Catastrophic performance regression**: warm eval time increases by more than 2× (current swept: 0.185 s/eval; batched: 0.485 s/eval)
8. **Fundamental architectural problem**: evidence emerges that the multi-direction tangent approach or `tf.while_loop` structure cannot satisfy the refactor contract

If a stop condition fires:
- **DO**: Document the failure mode, record which phase it occurred in, preserve all artifacts (broken code, test outputs, diagnostics)
- **DO NOT**: attempt speculative fixes, change the plan, or downgrade requirements
- **REPORT**: failure mode, evidence, phase state, and request user direction

---

## Phase Repair Policy (Binding)

After each phase completes its primary implementation:

### Mandatory Repair Actions (NO USER PERMISSION REQUIRED)
1. **Test failures due to trivial bugs**: off-by-one errors, typos, wrong variable names, missing imports
2. **Numerical discrepancies within tolerance**: if parity tests fail but the error is < rtol 5e-4, investigate and fix (likely op-order or floating-point associativity)
3. **Minor API mismatches**: parameter order, missing optional parameters (as long as no existing caller breaks)
4. **Documentation drift**: docstrings, inline comments, diagnostic messages that no longer match the code
5. **Dead code removal**: if refactor obsoletes helper functions or intermediate variables, remove them
6. **Lint/format issues**: if they block test execution

### Refresh Plan for Next Phase
After repair, agent MUST:
1. **Update the next phase's subplan** if repair revealed new information (e.g., a helper function needs to be written first)
2. **Record lessons learned** in the current phase's result document
3. **Verify pre-conditions** for the next phase (e.g., "parity gates must pass" before Phase 2)
4. **Proceed to next phase** if all pre-conditions satisfied, OR **stop and report** if a stop condition fired

### Example Repair Scenario (Phase 1)
- Phase 1 implements `tf.while_loop` with one-direction tangent
- Parity test fails: batch-size-1 score differs by 3e-4 (within tolerance)
- Investigation finds: tangent accumulation uses `+=` inside loop body, should use `tf.tensor_scatter_nd_add` for immutable update
- **Repair action**: fix the accumulation, rerun parity tests, confirm pass
- **Refresh plan**: Phase 2 subplan already accounts for multi-direction tangent; no changes needed
- **Proceed**: Phase 2 execution begins

---

## Success Criteria (Program-Level)

The program is COMPLETE and SUCCESSFUL when ALL of the following hold:

### Technical Criteria
1. **Parity gates pass**: all three tests in `test_ledh_canonical_batch_fused.py` pass at rtol 5e-4
2. **Graph size reduction**: one gradient evaluation (value + full gradient via multi-direction tangent) produces O(10³) nodes, not O(10⁶)
3. **Trace time reduction**: trace + first eval < 20 s (vs current 101.8 s for swept, 41.5 s for naive batched)
4. **Warm eval time competitive**: warm eval time ≤ 0.3 s (vs current 0.185 s swept, 0.485 s naive batched)
5. **No host memory regression**: host RSS, graph node count, and GraphDef bytes at (B=6, horizon=50, N=252) ≤ 2× current. GPU device-memory validation is deferred as a separate owner-scheduled escalated step (recorded in "What This Program Does Not Establish").
6. **Surrogate-force HMC integration**: `docs/benchmarks/step1_true_surrogate_force.py` runs without modification and produces same acceptance rates ± 5% (stochastic tolerance)

### Process Criteria
7. **All phases complete**: Phase 0 repair + Phase 1 + Phase 2 + Phase 3 + Phase 4 all reached their completion gates
8. **Refactor contract honored**: no silent behavior changes, no API breaks, analytical score preserved, NeuTra batch-native contract preserved
9. **Coverage maintained**: line coverage of the refactored kernel ≥ baseline coverage from Phase 0
10. **Artifacts complete**: all phase subplans, result documents, and the final integration validation recorded under `docs/plans/`

### Governance Criteria
11. **No mid-execution user questions**: agent did not ask user to choose between options during phase execution (pre-declared allowlist and stop conditions only)
12. **No mid-execution plan changes**: plan remained stable; repairs followed the declared repair policy
13. **Owner acceptance**: user reviews Phase 4 result document and approves promotion of the refactored kernel as the canonical batch-fused entry point

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `tf.while_loop` body size limit | Low | High | Python loops already emit ~2,200 nodes/timestep; loop body will be similar size; TF supports large bodies |
| Multi-direction tangent defeats CSE differently | Medium | Medium | Explicit broadcast/matmul over K makes sharing structural; early Phase 2 diagnostic will measure |
| Parity fails due to op-order changes | Medium | Low | Tolerance rtol 5e-4 accounts for this; if beyond tolerance, it's a stop condition |
| Trace time still high due to loop overhead | Low | Medium | `tf.while_loop` with bounded iterations is well-optimized; diagnostic in Phase 4 will measure |
| Warm eval slower due to loop dispatch | Low | High | Unlikely (loop is traced once); if true, it's a stop condition (2× regression) |
| XLA incompatibility introduced | Very Low | High | All current ops are XLA-compatible; agent will verify each new op |
| Memory regression from larger loop state | Low | Medium | Loop state mirrors current intermediate tensors; diagnostic in Phase 4 will measure |
| User rejects refactor contract | Medium | High | A2 approval request addresses this upfront; if rejected, revise before Phase 1 |
| Coverage tooling installation fails | Low | Low | `pytest-cov` is a standard package; fallback: manual coverage via code inspection |

---

## Dependencies and Prerequisites

### Environment
- **Conda env**: `tftwogpu` (TensorFlow GPU build)
- **Python**: 3.9+ (TFP compatibility)
- **TensorFlow**: 2.x (version check in Phase 0 repair)
- **TensorFlow Probability**: compatible with TF version
- **pytest**: installed in tftwogpu env
- **pytest-cov**: to be installed in Phase 0 repair

### Repository State
- **Branch**: `worktree-ledh-canonical-rebuild`
- **Worktree path**: `/home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild/`
- **Git status**: uncommitted changes allowed (will be staged/committed per phase)
- **Disjoint from main**: per memory, do not merge worktree → main

### Code Prerequisites
- **Single-cloud authority**: `canonical_value_and_analytical_score` in `ledh_canonical_score_tf.py` is the correctness reference
- **Parity tests**: `test_ledh_canonical_batch_fused.py` must pass before Phase 1 (Phase 0 repair ensures this)
- **Model factories**: test fixtures use `build_augmented_ksc_target` and LGSSM factories
- **Diagnostic harness**: `docs/benchmarks/` scripts for graph-size, eval-time, direction-cost measurement

---

## Timeline Estimate (Wall-Clock)

**Phase 0 Repair**: 30 min (fix 3 files, install pytest-cov, run canonical subset, record baseline coverage)  
**Phase 1**: 2-4 hours (implement `tf.while_loop` with one-direction tangent, debug parity, document)  
**Phase 2**: 2-4 hours (generalize to K-direction tangent, debug multi-direction parity, document)  
**Phase 3**: 1-2 hours (hoist closures, precompute slices, verify no numerical change, document)  
**Phase 4**: 2-3 hours (integration test, leaderboard smoke, surrogate-force HMC validation, final diagnostics, document)  

**Total Estimate**: 7.5-13.5 hours agent time (assumes no stop-condition blockers)

**User Time**:
- Upfront approvals (A1-A5): 15-30 min review
- Phase boundary reviews (optional): 4 × 5 min = 20 min
- Final acceptance (Phase 4): 15-30 min review

**Note**: Estimates assume the `tf.while_loop` approach is viable (no architectural blockers). If a stop condition fires, timeline extends by investigation + potential pivot time.

---

## What This Program Does Not Establish

This program changes how a computation is represented as a graph. It does not
produce scientific evidence about the computation itself, and none of its
artifacts may be cited as such.

**GPU device memory is not measured.** The entire test and diagnostic lane is
CPU-only `float64` by construction: every test file sets
`CUDA_VISIBLE_DEVICES=-1` before importing TensorFlow, and
`scripts/run_phase_tests.sh` exports it before Python starts. Measuring GPU
device memory requires an escalated run under the GPU/CUDA policy, which would
force a mid-execution escalation prompt and violate the zero-interruption
requirement. Success criterion 5 and stop condition 6 are therefore stated in
host RSS, graph node count, and GraphDef bytes. GPU device-memory validation
remains an open item for a separate owner-scheduled escalated step. The
program's host-side gains are measurable on CPU; the device-side consequence is
not claimed.

**No sampler validity claim.** Phase 4's surrogate-force smoke test checks
finiteness, shapes, and absence of retracing. It establishes nothing about
posterior correctness, convergence, R-hat, ESS, acceptance-rate adequacy, or
the soundness of the exact-value/damped-force construction. Those belong to the
separate surrogate-force validation program with its own evidence contract.

**No tuning claim.** Per the LEDH per-scope tuning rule, a claim-bearing run
needs a tuning artifact for its exact scope. This program produces no
claim-bearing run and no tuning artifact. None of its timing numbers may be
read as tuned performance.

**No statistical ranking.** Timing and memory numbers are single-run
measurements without replication or uncertainty intervals. They are descriptive
engineering measurements. They support statements of the form "graph node count
fell from 663,766 to X" and do not support any claim that one method is better
than another in a statistical sense.

**The estimand warning on the authority still stands.** With
`reset_policy="none"` and `annealed_stages=1`, the single-cloud authority
assumes uniform incoming weights but never resamples, so its returned value is
not a log-likelihood estimator for horizons T > 1 (measured on the frozen LGSSM
anchor: N-independent score bias +4.2 at T=50). The refactor preserves this
behavior exactly — it neither introduces nor repairs the bias. Parity with the
authority means agreement with the authority, not correctness of the estimand.

---

## Codex Audit Memo and Handoff

- [ledh-while-loop-refactor-codex-audit-memo-2026-08-30.md](ledh-while-loop-refactor-codex-audit-memo-2026-08-30.md) — the initial R1–R10 audit request with 12 audit questions
- [ledh-while-loop-refactor-codex-handoff-memo-2026-08-30.md](ledh-while-loop-refactor-codex-handoff-memo-2026-08-30.md) — the full-program review handoff, superseding the audit memo as the active review request

## Phase Subplans

- [Phase 0](ledh-while-loop-refactor-phase0-subplan-2026-08-30.md) — test audit, coverage baseline, API-drift repair
- [Phase 1](ledh-while-loop-refactor-phase1-subplan-2026-08-30.md) — single-direction `tf.while_loop` conversion
- [Phase 2](ledh-while-loop-refactor-phase2-subplan-2026-08-30.md) — multi-direction tangent (contains a design correction to this document)
- [Phase 3](ledh-while-loop-refactor-phase3-subplan-2026-08-30.md) — hoisting and constant precomputation
- [Phase 4](ledh-while-loop-refactor-phase4-subplan-2026-08-30.md) — integration and surrogate-force readiness

---

## References

- **Performance diagnosis**: `docs/benchmarks/diagnose_graph_size_20260830.py`, `diagnose_eval_time_20260830.py`, `diagnose_direction_cost_scaling_20260830.py`
- **Surrogate-force HMC driver**: `docs/benchmarks/step1_true_surrogate_force.py`
- **Correction and validation**: `docs/benchmarks/surrogate_force_correction_and_graph_diagnosis_20260830.md`
- **Refactor target kernel**: `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py`
- **Single-cloud authority**: `bayesfilter/highdim/ledh_canonical_score_tf.py`
- **Parity tests**: `tests/highdim/test_ledh_canonical_batch_fused.py`, `tests/highdim/test_ledh_canonical_batch.py`
- **Policy sources**: `/home/chakwong/.claude/CLAUDE.md`, `/home/chakwong/BayesFilter/CLAUDE.md`
- **Experiment plan template**: `docs/plans/templates/experiment-plan-template.md`

---

**END OF MASTER PROGRAM**
