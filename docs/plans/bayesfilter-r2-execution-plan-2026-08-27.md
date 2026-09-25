# R2 Execution Plan: Literal Resolution + Per-Scope Tuning + Q4 Leaderboard

Date: 2026-08-27  
Branch: `worktree-ledh-canonical-rebuild`  
Governing documents: Parameter justification register (2026-08-27), R2 tuning plan (2026-08-27)

**Problem restated**: Q3 board mechanism-complete but non-claim-bearing. 35 parameters, zero with clean per-scope tuning. Six ASSERTED literals override module defaults silently. Austria tuned on claim data (B2 violation). Sinkhorn ε=2.0 INHERITED, measured as 0.870-nat dlgssm bias driver.

**Objective**: Execute R2-LITERAL → R2-TUNE (6 scopes) → Q4 leaderboard regen. Every cell becomes claim-bearing or honestly labeled. Deploy 3-gate enforcement preventing drift.

**Total budget**: ~3–4 GPU-days. Per-process cap 120 min. Environment: tftwogpu, GPU 1 (4080 SUPER), TF memory growth fail-closed. Output root: `docs/benchmarks/r2_tuning_20260827/`. Stop after 3 consecutive launch failures per scope.

**Execution order**: strictly sequential phases to preserve disjoint data discipline.

---

## Phase 1: R2-LITERAL (code hygiene, CPU, ~4 hours)

**Contract**: Resolve the ASSERTED literals + the flow-discretization contradiction BEFORE tuning. Tuning an untracked parameter is governance-invalid.

**Mechanism**: For each literal, either (a) promote to registry constant with derivation, or (b) expose as tunable parameter in signature + control family.

### CORRECTED RISK PROFILE (2026-08-27 code verification)

The register (P18, P25, P26, P27) described these literals as live value
disagreements inside the production program. **Direct code reading
contradicts that for five of six.** Verified findings:

| Literal | Register claim | Verified reality | Numerics change to fix? |
|---|---|---|---|
| `trust_region_lm_scale_floor` | call 1e-4 vs sig 1e-6 — "two values in one program" | **All production paths run 1e-4.** Registry, all three filter signatures, and both lanes = 1e-4. The leaf's 1e-6 is dead code. | **NO** |
| `coordinate_cap_power` | ASSERTED, no named constant | **Already `OWNER` in registry at 8.** Value lane threads it; score lane hardcodes the same 8. | **NO** |
| `floor` (diagonal) | call 1e-5 vs module 1e-6 | Both lanes hardcode 1e-5; leaf default is dead code. `floor` has **no** module default (required kwarg). | **NO** (Option A) |
| `pairwise_floor` | call 1e-5 vs module 1e-6 | Both lanes hardcode 1e-5; leaf 1e-6 dead. | **NO** (Option A) |
| `flow_substeps` | runner 16 vs sig 24 | **THREE-way split**: value runner 16, **score runner 8**, signature 24. Absent from `LEDH_PRODUCTION_PROGRAM_V1` entirely. | **YES** |

**What this means for the owner's concern.** The stated hesitation was that
"R2-LITERAL changes shipped numerics — promoting a literal changes which
value runs." For four of the five literals that is **not the case**: both
production lanes already pass the same value, and the differing module
default is unreachable dead code. Fixing them is pure threading hygiene
with bit-identical output. Only `flow_substeps` requires a real numerics
decision.

**New finding not in the register — value/score lane program mismatch.**
`run_q3_leaderboard_20260824.py` passes `flow_substeps=16` on the value
path (line 196) and `substeps=8` on the score path (line 262). The Q3
board's value cells and score cells therefore ran **different flow
discretizations of the same model**. `LEDH_PRODUCTION_PROGRAM_V1` contains
no `substeps` field, which is why three values coexist unnoticed: the
wiring gate had nothing to check against. This is the same failure class as
owner findings #1–#3 (a code default silently defining "production"), and
it means the Q3 value and score cells are not two measurements of one
program. Recorded here as **finding #4**; it strengthens rather than
changes the existing non-claim-bearing status.

**Seven decisions** (awaiting owner approval on each):

### Decision 1: `flow_substeps` / `substeps` (THREE-WAY SPLIT)
- **Current state**: value runner passes 16, score runner passes 8, function signature default is 24. The parameter has three different names: `flow_substeps` in value lane, `substeps` in score lane, both feed the same flow-ODE integrator.
- **Proposed**: 
  1. Unify naming: `flow_substeps` everywhere (already in value signature).
  2. Expose in score-lane signature (`canonical_value_and_analytical_score`) as `flow_substeps: int = 24`.
  3. Add to control family, tune per scope via grid {8, 12, 16, 24}.
- **Rationale**: No universal derivation exists; Q3 used 16 for value, 8 for score (inconsistent); models differ in stiffness; tuning resolves this.
- **Alternative**: If owner has universal derivation for a single value, promote to registry constant.
- **Implementation**: Add `flow_substeps` param to score signature; replace `substeps=8` literal in `ledh_canonical_score_tf.py` with `flow_substeps` param; runner grid {8, 12, 16, 24}; register row → `TUNED(scope)` after R2-TUNE.

### Decision 2: `trust_region_lm_scale_floor` — REGISTER MISCHARACTERIZED THIS
- **Verified state** (2026-08-27 code read, corrects register P18):
  - `ledh_canonical_filter_tf.py:103` signature default **1e-4**
  - `genut_guided_proposal_tf.py:714` signature default **1e-4**
  - `ledh_pfpf_genut_initial_rqmc_tf.py:341` signature default **1e-4**
  - `ledh_alg1_contract.py:237` registry **1e-4**
  - `higher_moment_contract_e.py:993` leaf `diagonal_lm_scale_floor` default **1e-6**
  - Value lane at `genut_guided_proposal_tf.py:957` **threads** the registry value (gets 1e-4)
  - Score lane at `ledh_canonical_score_tf.py:449` **hardcodes** 1e-4
- **Finding**: The register claimed "1e-4 at the call site vs 1e-6 in the signature, inside the same production program" — implying a live value divergence. There is none. Every production path runs 1e-4. The leaf's 1e-6 default is **dead code in production**.
- **Actual defect**: the score lane hardcodes instead of threading, so a future registry change would silently not reach the score lane. This is a latent divergence risk, not a live divergence.
- **Proposed**: Thread it (expose in score signature, read from registry). **Zero numerics change.**
- **Implementation**: Add `trust_region_lm_scale_floor: float = 1.0e-4` to score signature; replace literal at line 449; register row → `OWNER` (registry-defined, wiring-gated).

### Decision 3: `coordinate_cap_power` — ALREADY IN REGISTRY, SCORE LANE NOT THREADED
- **Verified state**:
  - `ledh_alg1_contract.py:249` registry defines `"dual_cap_coordinate_cap_power": 8` (OWNER)
  - Value lane `_restore_cloud_primal` signature has `dual_cap_coordinate_cap_power: int = 8` and threads it (line 711, called at 964)
  - Score lane `canonical_value_and_analytical_score` signature **lacks this parameter entirely**
  - Score lane at `ledh_canonical_score_tf.py:456` **hardcodes 8** in the `higher_moment_shape_jvp` call
- **Finding**: Same pattern as Decision 2 — the value is already registry-defined, the value lane threads it, the score lane hardcodes. No live divergence, but a latent wiring defect.
- **Proposed**: Expose in score signature, thread from registry. **Zero numerics change.**
- **Implementation**: Add `coordinate_cap_power: int = 8` to score signature; replace literal at line 456; register row already `OWNER`.

### Decision 4 + 5 + 7: `floor` (diagonal), `pairwise_floor` — BOTH LANES ALREADY 1e-5
- **Verified state**:
  - `higher_moment_shape_jvp` signature defaults: `floor: float` (required), `pairwise_floor: float = 1.0e-6`
  - Value lane at `genut_guided_proposal_tf.py:957,963` **hardcodes** `floor=1.0e-5`, `pairwise_floor=1.0e-5`
  - Score lane at `ledh_canonical_score_tf.py:447,453` **hardcodes** `floor=1.0e-5`, `pairwise_floor=1.0e-5`
  - Both lanes agree on 1e-5; the signature's 1e-6 default is dead code in production.
- **Finding**: No live divergence. Both production lanes tested 1e-5; Q2 relative-ridge work measured 1e-5 >> 1e-12 floor, so 1e-5 vs 1e-6 is not mechanically load-bearing. The defect is that neither lane threads it — signature change wouldn't propagate.
- **Proposed**: Two options, owner decides numerics policy:
  - **Option A (zero change)**: Unify signature defaults to 1e-5, matching tested production values. Register → `DERIVED` ("value+score tested, non-harm verified per S1-S8, Q2 ridge evidence").
  - **Option B (conservative tighten)**: Keep 1e-6 signature, change both call sites to thread it. Registry documents 1e-6 as the derived value. Numerics **change** slightly (tighter floor), requires no-fire regression on oracle gates.
- **Recommendation**: Option A (match tested production). The call sites tested 1e-5 under oracle gates for ~4 GPU-days; changing to 1e-6 is speculative tightening without evidence it's needed.
- **Implementation (Option A)**: Change `floor` signature to `floor: float = 1.0e-5`, `pairwise_floor: float = 1.0e-5`; remove both literals; threads the default. Register rows → `DERIVED`.

**Evidence gate**: `test_parameter_register_completeness` must pass after this phase (every signature param has register row, zero ASSERTED mechanism-affecting params).

**Deliverable**: Updated signatures, threading fixes, passing governance test. Commit references this plan. Wall ~1 session. **Four of five decisions are zero-numerics-change threading fixes.**

---

## Phase 2: R2-TUNE per scope (6 campaigns, ~2–3 GPU-days)

One campaign per model on FRESH tuning data DISJOINT from claim partition.

### 2.1 Tuning data generation (CPU, ~30 min total)

```bash
cd /home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild
conda run -n tftwogpu --no-capture-output bash -c \
  'CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/generate_r2_tuning_data.py \
   --model linear2d --seed 9001 --horizon 5 --replications 20'
```

Repeat for dlgssm (9002, T=50, 20), predator-prey (9003, T=25, 20), KSC (9004, T=1000, 5), gen-SV (9005, T=100, 10), Austria (9006, T=20, 10).

**Output**: `docs/benchmarks/r2_tuning_20260827/{model}_tuning_data.npz` with SHA-256 recorded in manifest.

### 2.2 Control family per scope

**Mandatory (all 6)**:
- `epsilon` ∈ {0.5, 1.0, 2.0, 4.0}
- `sinkhorn_steps` ∈ {8, 16, 24}
- `balance_steps` = 8 (not varied; Q2 evidence shows balance cheap)
- `flow_substeps` ∈ {12, 16, 24} (if exposed by R2-LITERAL)
- `particle_count` ∈ {1008, plus one N-check at 504 and 2016}

**Conditional (nonlinear scopes if plain ESS < 10%)**:
- `temper_stages` ∈ {1, 2, 4, 8}
- `flow_prior_cap` ∈ {8.0, inf}
- `trust_region_lm_damping` = 1e-2 (not varied; Q2 non-harm)
- `trust_region_radius` = 0.5 (not varied; Q2 promotion failed, warm start)

**Fixed (registry status, not tuned)**:
- `dual_cap_enabled=True`, `trust_region_enabled=True` (OWNER, required)
- Dual-cap family P19-P24 (OWNER, 08-07)
- `RELATIVE_PSD_FLOOR=1e-12` (DERIVED, non-harm verified)

### 2.3 Promotion criteria (hard vetoes → selection rule)

**Hard veto** (any cell, any seed):
- Non-finite value or weights
- `program_valid=False`
- Crash
- **Sinkhorn marginal-convergence failure**: max(|row_marginal - uniform|, |col_marginal - uniform|) > 1e-3 after `sinkhorn_steps` iterations. THIS IS NEW vs Q3 (where unconverged cells averaged silently).

**Value-bias budget** (linear scopes only):
- |canonical - exact_kalman| ≤ 0.10 nats (mean over tuning replicates).
- If no (ε, iters) passes, report `tuning_status: WARM_START_ONLY` with reason "bias budget unmet at all tested ε".

**ESS floor** (nonlinear scopes):
- Per-step min ESS ≥ 10% of N (mean over tuning replicates).
- If plain mode (k=1) passes, annealing not required; if k>1 needed, promote smallest k that passes.

**Fisher gate** (score cells, claim scale):
- 40 tuning replications, 3 random directions, |mean score bias| < 3*SE + 0.05 with non-vacuity (SE < 1.0).
- Runs AFTER value config selected, ON tuning data (disjoint from claim).
- If fails: value cell is claim-bearing, score cell marked `Fisher gate failed — deferred`.

**Selection rule**:
- **Epsilon**: SMALLEST ε meeting bias budget (linear) or ESS floor (nonlinear) AND passing Sinkhorn-convergence veto in ALL replicates.
- **Sinkhorn steps**: SMALLEST count converging at selected ε.
- **Annealing k**: SMALLEST k passing ESS floor (prefer k=1 parsimony).
- **Substeps**: prefer 16 (Q3 runner) or smallest passing value.

### 2.4 Execution per scope

```bash
cd /home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild
conda run -n tftwogpu --no-capture-output bash -c \
  'CUDA_VISIBLE_DEVICES=1 python docs/benchmarks/run_r2_tuning_grid.py \
   --model linear2d \
   --tuning-data docs/benchmarks/r2_tuning_20260827/linear2d_tuning_data.npz \
   --output-dir docs/benchmarks/r2_tuning_20260827 \
   --timeout 7200'
```

**Output**: `docs/benchmarks/r2_tuning_20260827/linear2d_tuning_artifact.json` with schema:

```json
{
  "schema": "bayesfilter.r2_tuning.v1",
  "model": "linear2d",
  "tuning_data": {"seed": 9001, "horizon": 5, "replications": 20, "sha256": "..."},
  "claim_partition": "frozen fixture seed 101 (disjoint)",
  "control_family_enumerated": ["epsilon", "sinkhorn_steps", "flow_substeps", "particle_count"],
  "grid_evaluated": {...},
  "veto_cells": [...],
  "promoted_config": {
    "epsilon": 1.0,
    "sinkhorn_steps": 16,
    "balance_steps": 8,
    "flow_substeps": 16,
    "particle_count": 1008,
    "annealed_stages": 1,
    "justification": "smallest epsilon meeting bias budget 0.10 nats; converged all 20 replicates; parsimony"
  },
  "tuning_status": "CLAIM_BEARING",
  "fisher_gate_status": "NOT_RUN",
  "wall_seconds": ...
}
```

**Fisher gate** (if value tuning succeeded):

```bash
conda run -n tftwogpu --no-capture-output bash -c \
  'CUDA_VISIBLE_DEVICES=1 python docs/benchmarks/run_r2_fisher_gate.py \
   --model linear2d \
   --config docs/benchmarks/r2_tuning_20260827/linear2d_tuning_artifact.json \
   --replications 40 --directions 3'
```

Updates `fisher_gate_status` in artifact to `PASSED` or `FAILED`.

**Repeat** for dlgssm, predator-prey, KSC, gen-SV, Austria. Wall ~2–4 hours each, staggered.

**Stop conditions** (per scope):
- 3 consecutive launch failures → record failure, move to next scope.
- Wall cap 120 min per process.
- Tuning-data SHA mismatch → regenerate data, retry.

---

## Phase 3: Q4 Leaderboard Regen (~3 GPU-hours)

**Runner**: `docs/benchmarks/run_q4_leaderboard_20260827.py` (new, derived from Q3 runner with config injection).

**Config injection**: Each row reads `docs/benchmarks/r2_tuning_20260827/{model}_tuning_artifact.json` and passes `promoted_config` to `canonical_value_and_diagnostics`. If `tuning_status: WARM_START_ONLY`, cell manifest records `tuning: UNTUNED — {reason}`.

**Seeds**: 16 paired (D1 fix: np.random.SeedSequence hashing for bootstrap parity).

**Frozen targets**: same SHA-identified observations as Q3 (no data changes).

**Conformance stamp**: `ledh-canonical-conformance-v1-2026-08-27` (updated commit, same contract).

**Execution**:

```bash
cd /home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild
conda run -n tftwogpu --no-capture-output bash -c \
  'CUDA_VISIBLE_DEVICES=1 python docs/benchmarks/run_q4_leaderboard_20260827.py \
   --seeds 16 \
   --output-dir docs/benchmarks/artifacts/ledh_canonical_leaderboard_2026-08/q4_board'
```

**Report builder**: `docs/benchmarks/q4_leaderboard_report_20260827.md` (configuration-status-first enforced):

1. **Configuration table** (before any number):
   - Program: `LEDH_PRODUCTION_PROGRAM_V1`
   - Per-row tuning artifact path or `UNTUNED — {reason}`
   - Nonclaims: "No ranking supported (D2 uncertainty analysis not funded); descriptive only"

2. **Value cells**: 6 models × canonical/bootstrap/UKF, 16 seeds, mean ± seed-std.

3. **Score cells**: linear2d (free exact), dlgssm (if Fisher passed), Austria (if Fisher passed).

4. **Inference-status table**: hard vetoes, statistical ranking, descriptive differences, default-readiness, next evidence.

**Deliverable**: Report cites all 6 tuning artifacts by path, zero unlabeled UNTUNED cells, configuration-status leading.

---

## Phase 4: Enforcement Gates (parallel deployment)

### 4.1 Parameter register completeness test

`tests/governance/test_parameter_register_completeness.py`:
- AST-enumerate every param of `canonical_value_and_diagnostics`, `canonical_value_and_analytical_score`, `higher_moment_shape_jvp`, `_restore_cloud_primal`.
- Assert each has register row in `docs/plans/bayesfilter-parameter-justification-register-2026-08-27.md`.
- Assert no mechanism-affecting param has status ASSERTED.
- Fail if new param appears without register update.

### 4.2 Claim gate in report builder

`docs/benchmarks/ledh_report_builder.py` (new or amended):
- Read cell's tuning artifact.
- Check every param against register.
- If any is INHERITED/ASSERTED/WARM(wrong scope), force `tuning: UNTUNED — not claim-bearing`.
- Render configuration table BEFORE numbers.
- Refuse "production" label unless all cells claim-bearing or honestly limited.

### 4.3 Estimand gate (unchanged)

`tests/governance/test_production_score_lane_value_is_likelihood_estimand`: pins production score to exact reference on linear scopes. Any program change breaking parity fails.

---

## Success Criteria (plan discharge)

1. ✓ R2-LITERAL complete: governance test passes, zero ASSERTED mechanism params.
2. ✓ 6 tuning artifacts: disjoint data, control family enumerated, promotion met or warm-start labeled.
3. ✓ Q4 leaderboard: 16 seeds, tuned configs, configuration table leading, zero unlabeled UNTUNED.
4. ✓ 3-gate enforcement live: completeness test in CI, claim gate in builder, estimand gate green.
5. ✓ Commit trail: every artifact references this plan; Q4 report cites tuning artifacts by path.

**Post-completion**: Every Q4 cell claim-bearing or honestly labeled. First validly claim-bearing LEDH multi-scope leaderboard.

---

## Pre-Execution Checklist

- [ ] Owner approval on R2-LITERAL decisions 1–5 (above)
- [ ] Campaign execution permissions confirmed (added to .claude/settings.json allow list)
- [ ] Worktree HEAD clean or stashed
- [ ] tftwogpu env activated, GPU 1 visible (`nvidia-smi` check)
- [ ] TF memory growth verified in test script
- [ ] Output root `docs/benchmarks/r2_tuning_20260827/` does not exist (fresh)
- [ ] Frozen claim targets SHA-verified unchanged from Q3

**Execution authority**: User message "R2-LITERAL first" + approval on literal decisions → Phase 1 launch. Phase 2 launch after Phase 1 commit. Phase 3 launch after 6 Phase 2 artifacts landed.

**Wall estimate**: ~1 week elapsed (R2-LITERAL 1 session, R2-TUNE 2–3 days GPU, Q4 1 session).
