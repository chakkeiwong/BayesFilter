# R2 Tuning + Leaderboard Rebuild Plan (2026-08-27)

**Governing artifact**: Parameter justification register (2026-08-27).

**Problem**: Q3 board is mechanism-complete but non-claim-bearing. Zero of 35
parameters have clean per-scope tuning artifacts. Sinkhorn epsilon/iters
INHERITED on all 6 rows, measured as dlgssm bias driver (0.870 nats traced
to untuned ε=2.0). Six ASSERTED literals silently override module defaults.
Austria k/c/damping tuned ON claim data (B2 violation). Other 5 models have
zero tuning artifacts (B1).

**Objective**: Per-scope tuned configurations for all 6 models on data
DISJOINT from claim partitions, with the full control family enumerated and
a 3-gate enforcement system preventing future drift. Regenerate the
leaderboard under tuned configs. Every cell becomes claim-bearing or
honestly reports why it is not.

**Budget**: ~3–4 GPU-days total (R2-LITERAL + 6 × R2-TUNE + board regen).
Per-process cap 120 minutes. Environment: tftwogpu conda, GPU 1 (4080
SUPER), TF memory growth fail-closed. Output root:
`docs/benchmarks/r2_tuning_20260827/` (fresh versioned directories per
launch). Stop after 3 consecutive launch failures per scope.

**Dependency order**: R2-LITERAL → R2-TUNE (6 scopes, any order) → board
regen with tuned configs.

---

## Phase 1: R2-LITERAL (resolve asserted literals, ~4 hours CPU)

**Contract**: The 6 ASSERTED literals (P18, P25, P26, P27, flow_substeps
P01, and the two module-vs-callsite disagreements) and the P01
self-contradiction must be resolved BEFORE any tuning campaign, because
tuning an untracked parameter is governance-invalid.

**Mechanism**: For each literal, decide:
1. **Promote to registry constant** with a derivation (preferred if the
   value is derived or has owner rationale), OR
2. **Expose as a tunable parameter** in the signature and add to the
   per-scope control family.

**Evidence gate**: After this phase, the governance test
`test_parameter_register_completeness` must pass (every signature parameter
has a register row, no ASSERTED rows remain for mechanism-affecting params).

**Specific decisions required** (ask owner for pre-approval on each):

| # | Parameter | Current state | Proposed resolution |
|---|---|---|---|
| P01 | `flow_substeps` | runner=16, sig default=24 | Expose in registry with per-scope tuning; or promote 16 to constant if owner has a universal derivation |
| P18 | `trust_region_lm_scale_floor` | call=1e-4, sig=1e-6 | Unify to 1e-4 (score-lane tested) or expose as tunable |
| P25 | `coordinate_cap_power` | 8 asserted | Promote to OWNER constant (part of 08-07 dual-cap family) |
| P26 | `floor` (diagonal) | 1e-5 call, 1e-6 sig | Unify or expose |
| P27 | `pairwise_floor` | 1e-5 call, 1e-6 sig | Unify or expose |
| P28 | `RELATIVE_PSD_FLOOR` | 1e-12 | Already DERIVED (2026-08-27 calibration); register update only |

**Deliverable**: Updated signatures, registry constants with derivations,
and a passing `test_parameter_register_completeness` gate. Commit message
references this plan. Wall: ~1 session.

---

## Phase 2: R2-TUNE per scope (6 tuning campaigns, ~2–3 GPU-days)

One campaign per model: linear2d, dlgssm, predator-prey, KSC, gen-SV,
Austria. Each runs on FRESH tuning data (simulated from the model law or
held-out partition) DISJOINT from the frozen claim observations.

### 2.1 Control family (parameters to tune per scope)

**Mandatory** (every scope):
- P07 `epsilon` (Sinkhorn entropic strength)
- P08 `sinkhorn_steps` (with marginal-convergence veto)
- P09 `balance_steps`
- P01 `flow_substeps` (if exposed)
- P02 `particle_count` (confirm N=1008 or find scope-specific optimum)

**Conditional** (nonlinear scopes only):
- P29 `temper_stages` / `annealed_stages` (if collapse measured)
- P30 `flow_prior_cap` (if annealing used)
- P16 `trust_region_lm_damping` (re-confirm or tune if dual-cap active)
- P17 `trust_region_radius` (re-confirm or tune)

**Fixed** (not tuned, registry status):
- P14/P15 `dual_cap_enabled`, `trust_region_enabled` = True (OWNER, required mechanism)
- P19–P24 dual-cap family constants (OWNER, 08-07 rationale)
- P03–P06 UKF sigma-point weights (INHERITED pending a UKF-tuning contract; not in R2 scope)
- P28 `RELATIVE_PSD_FLOOR` = 1e-12 (DERIVED, non-harm verified)
- P11 reset dtype (fixed float32 pending A4 remedy; noted as a limitation)

### 2.2 Tuning data per scope

| Scope | Tuning data | Claim partition (disjoint) |
|---|---|---|
| linear2d | fresh simulation, model seed 9001, T=5, 20 replicates | frozen fixture seed 101 |
| dlgssm | fresh simulation, model seed 9002, T=50, 20 replicates | frozen benchmark seed 81100 |
| predator-prey | fresh simulation, model seed 9003, T=25, 20 replicates | frozen benchmark (existing) |
| KSC | fresh simulation, model seed 9004, T=1000, 5 replicates (wall cap) | frozen benchmark (existing) |
| gen-SV | fresh simulation, model seed 9005, T=100, 10 replicates | frozen benchmark seed 501 (to be formally frozen with SHA lineage first) |
| Austria | fresh simulation, model seed 9006, T=20, 10 replicates | frozen target (existing, NOT the Q2 k/c target) |

### 2.3 Promotion criteria (pre-declared, per scope)

**Hard veto** (any cell, any seed): non-finite value, non-finite weights,
`program_valid` false, crash, or Sinkhorn marginal-convergence failure
(relative error > 1e-3 on either marginal after `sinkhorn_steps`
iterations). A veto disqualifies that (ε, iters) cell; the surface
interpolates or defaults conservatively.

**Value-bias budget** (linear scopes only): |canonical - exact_kalman| ≤
0.10 nats on tuning data (mean over replicates). If no (ε, iters) cell
passes, the campaign reports "bias budget unmet" and selects the
least-biased cell as a warm start with an UNTUNED flag.

**ESS floor** (nonlinear scopes): per-step min ESS ≥ 10% of N on tuning
data (mean over replicates). If plain mode (k=1) passes, annealing is not
required; if plain mode fails and annealed k>1 passes, that k is promoted.

**Fisher gate** (score cells): claim-scale Fisher identity gate on tuning
data under the tuned value-lane config (40 replications, 3 directions,
|mean| < 3*SE + 0.05 with non-vacuity guard). If the gate fails, the scope
gets a value cell only; score is deferred to a separate contract.

**Selection rule**: for ε, pick the SMALLEST ε that meets the bias budget
(linear) or ESS floor (nonlinear) and passes the Sinkhorn-convergence veto
in all replicates. For `sinkhorn_steps`, pick the smallest count that
converges at the selected ε. For other controls, prefer parsimony (smaller
k, fewer substeps) when multiple configs pass.

### 2.4 Grid design (coarse ladder, per Q2 Austria evidence)

**Epsilon ladder**: {0.5, 1.0, 2.0, 4.0} × sinkhorn_steps {8, 16, 24}.
Post-hoc: if all ε≥2.0 cells fail convergence, add ε=3.0; if ε=0.5 fails
bias budget, try ε=0.75.

**Annealing ladder** (if plain-mode ESS < 10%): k ∈ {2, 4, 8}, c ∈ {8.0,
inf}. Austria Q2 evidence: k=4/c=8 passed, c=inf NaN'd; start there.

**Substeps ladder** (if exposed): {12, 16, 24}. Q3 runner used 16;
signature default is 24; test whether 12 suffices.

**Particle-count check**: run one cell at N=504 and N=2016 to confirm the
bias is ε-driven (N-invariant per A2 measurement), not MC noise.

### 2.5 Artifact per scope

JSON manifest per scope:
```json
{
  "schema": "bayesfilter.r2_tuning.v1",
  "model": "dlgssm",
  "tuning_data": {"seed": 9002, "horizon": 50, "replications": 20, "sha256": "..."},
  "claim_partition": "frozen benchmark seed 81100 (disjoint)",
  "control_family_enumerated": ["epsilon", "sinkhorn_steps", "balance_steps", "flow_substeps"],
  "grid": { ... },
  "veto_cells": [ ... ],
  "promoted_config": {
    "epsilon": 1.0,
    "sinkhorn_steps": 16,
    "balance_steps": 8,
    "flow_substeps": 16,
    "particle_count": 1008,
    "justification": "smallest epsilon meeting bias budget 0.10 nats; converged in all 20 replicates"
  },
  "fisher_gate_status": "PASSED" | "FAILED" | "NOT_RUN",
  "tuning_status": "CLAIM_BEARING" | "WARM_START_ONLY",
  "wall_seconds": ...
}
```

Stored at `docs/benchmarks/r2_tuning_20260827/{model}_tuning_artifact.json`.
The Q4 leaderboard cells cite this path in their `tuning` field.

**Wall budget per scope**: 2–4 hours (linear scopes cheap; KSC/Austria
costlier). Total ~12–20 GPU-hours across 6 scopes.

---

## Phase 3: Leaderboard regeneration (Q4 board, ~3 GPU-hours)

**Runner**: `docs/benchmarks/run_q4_leaderboard_20260827.py` (new file,
derived from Q3 runner with tuned-config injection).

**Configuration injection**: Each row reads its tuning artifact and passes
the promoted config to `canonical_value_and_diagnostics`. If
`tuning_status` is `WARM_START_ONLY`, the cell's manifest records
`tuning: UNTUNED — bias budget unmet` and the report builder renders it
before any number.

**Seeds**: 16 paired seeds (D1 fix: canonical and bootstrap use the same
np.random.SeedSequence-hashed generator streams). Enables paired
uncertainty analysis if D2 is later funded.

**Frozen targets**: same SHA-identified observations as Q3 (no data
changes, only config changes).

**Conformance stamp**: `ledh-canonical-conformance-v1-2026-08-27` (updated
commit, same contract).

**Estimand anchors**:
- linear2d, dlgssm: exact Kalman (value + FD score on dlgssm)
- nonlinear rows: add ONE budgeted reference arm per row (large-N annealed
  bootstrap at N=4032, or SQMC with MCSE; run once per row as a separate
  mini-campaign if not already available). Gap C2 remedy.

**Report structure** (configuration-status-first enforced):
1. **Configuration table** (before any number): program version, per-row
   tuning artifact path or UNTUNED flag, what must NOT be concluded.
2. **Value cells**: 6 models × canonical/bootstrap/UKF, 16 seeds, mean ±
   seed-std reported.
3. **Score cells**: dlgssm (if Fisher passed), Austria (if Fisher passed),
   linear2d (free exact score, gap C1 remedy).
4. **Reference-arm table** (if available): nonlinear value anchors with MCSE.
5. **Inference-status table**: honestly states "no ranking supported"
   unless D2 uncertainty analysis is funded and run.

**Deliverable**: `docs/benchmarks/q4_leaderboard_report_20260827.md` with
all six tuning artifacts cited, configuration-status table leading,
zero UNTUNED cells (or honestly labeled warm-start cells with the
limitation stated).

---

## Phase 4: Enforcement gates (parallel to Phase 2–3)

### 4.1 Parameter register completeness test

`tests/governance/test_parameter_register_completeness.py`:
- Enumerate every parameter of the four core functions via AST inspection.
- Assert each has a register row.
- Assert no mechanism-affecting parameter has status ASSERTED.
- Fail the gate if a new parameter appears without a register update.

### 4.2 Claim gate in the report builder

`docs/benchmarks/ledh_report_builder.py` (new or amended):
- Read each cell's tuning artifact.
- For every parameter the cell uses, check its register status.
- If any parameter is INHERITED, ASSERTED, or WARM(different scope), force
  the cell's `tuning` field to `UNTUNED — not claim-bearing`.
- Render the configuration-status table BEFORE any number table.
- Refuse to build a "production leaderboard" report unless every cell is
  claim-bearing or has an honest limitation label.

### 4.3 Estimand gate (unchanged, Q3 already has it)

`test_production_score_lane_value_is_likelihood_estimand`: pins the
production score lane to the exact-reference anchor on linear scopes. Any
program change that breaks parity fails this gate.

---

## Pre-approvals requested (to avoid mid-campaign blocking)

### A. R2-LITERAL decisions (Phase 1, owner approval on each)

**Question 1**: `flow_substeps` — runner uses 16, signature default is 24.
Which?
- **Option 1a**: Promote 16 to a registry constant (if you have a universal derivation).
- **Option 1b**: Expose as a tunable parameter, add to control family, tune per scope.

**Question 2**: `lm_scale_floor` — score lane calls with 1e-4, signature
default is 1e-6. Unify or expose?
- **Option 2a**: Unify to 1e-4 (the score-lane tested value).
- **Option 2b**: Expose as tunable.

**Question 3**: `floor` and `pairwise_floor` — same pattern (call 1e-5,
sig 1e-6). Unify or expose?
- **Option 3a**: Unify to 1e-5.
- **Option 3b**: Expose as tunable.

**Question 4**: `coordinate_cap_power` = 8 asserted. This is part of the
dual-cap family (08-07 spec). Promote to OWNER constant?
- **Option 4**: Promote to registry as OWNER with 08-07 rationale reference.

### B. Campaign execution permissions (blanket pre-approval)

All commands below are for GPU campaigns (tftwogpu env, CUDA_VISIBLE_DEVICES=1):

1. **Fresh tuning-data generation** (CPU, cheap):
   ```
   CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python \
     docs/benchmarks/generate_r2_tuning_data.py --model {linear2d|dlgssm|...} --seed 900X
   ```

2. **Per-scope tuning grid** (GPU, 2–4 hours each):
   ```
   CUDA_VISIBLE_DEVICES=1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python \
     docs/benchmarks/run_r2_tuning_grid.py --model {model} --output-dir docs/benchmarks/r2_tuning_20260827
   ```

3. **Fisher gate per scope** (GPU, ~20 min each, only if value tuning passed):
   ```
   CUDA_VISIBLE_DEVICES=1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python \
     docs/benchmarks/run_r2_fisher_gate.py --model {model} --config {tuning_artifact.json}
   ```

4. **Q4 leaderboard regeneration** (GPU, ~3 hours):
   ```
   CUDA_VISIBLE_DEVICES=1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python \
     docs/benchmarks/run_q4_leaderboard_20260827.py --seeds 16
   ```

5. **All git operations** (commit tuning artifacts, commit Q4 runner, commit results):
   ```
   git add docs/benchmarks/r2_tuning_20260827/* docs/benchmarks/run_q4_leaderboard_20260827.py
   git commit -m "R2-TUNE {model}: epsilon={...}, converged, bias budget met"
   git add docs/benchmarks/q4_leaderboard_report_20260827.md
   git commit -m "Q4 leaderboard: 6 models, per-scope tuned, claim-bearing"
   ```

6. **Non-destructive inspection** (no approval needed, listed for completeness):
   - Read any file in docs/benchmarks/, docs/plans/, bayesfilter/highdim/
   - grep/find/ls within the repo
   - CUDA_VISIBLE_DEVICES=-1 pytest on any test (CPU smoke checks)

### C. Fallback decisions (if a scope fails promotion)

**If a scope's epsilon ladder fails the bias budget or ESS floor**:
- Record `tuning_status: WARM_START_ONLY` in the artifact.
- The Q4 cell labels itself `UNTUNED — {reason}` and reports the limitation.
- Do NOT silently use an INHERITED default — honest failure is required.

**If Fisher gate fails on a tuned value config**:
- The scope gets a value cell only; score is marked `Fisher gate failed — score cell deferred`.
- This is NOT a value-lane failure; the value cell is still claim-bearing.

---

## Success criteria (plan completion)

1. **R2-LITERAL complete**: governance test passes, no ASSERTED
   mechanism-affecting parameters remain.
2. **6 tuning artifacts**: one per scope, on disjoint data, control family
   enumerated, promotion criteria met or honest warm-start label.
3. **Q4 leaderboard regenerated**: 16 seeds, tuned configs, configuration
   table leading, zero unlabeled UNTUNED cells.
4. **3-gate enforcement live**: register completeness test in CI, claim
   gate in report builder, estimand gate unchanged.
5. **Commit trail**: every artifact references this plan in its commit
   message; the Q4 report cites the 6 tuning artifacts by path.

**Post-completion status**: every Q4 cell is claim-bearing under the
3-gate system, or honestly labeled with why it is not (e.g., warm-start,
Fisher-deferred). The leaderboard becomes the first validly claim-bearing
LEDH evidence across multiple scopes.

**Owner decision points**: R2-LITERAL pre-approvals (4 questions above);
campaign launch timing; whether to proceed if one scope fails (orthogonal
scopes can proceed independently).

**Wall estimate**: ~1 week elapsed (R2-LITERAL 1 session, R2-TUNE 2–3 days
staggered GPU, Q4 regen 1 session). Token budget within this session's
200k.
