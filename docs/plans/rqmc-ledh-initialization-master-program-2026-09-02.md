# RQMC LEDH Initialization Master Program

**Program ID:** `RQMC_LEDH_INIT_V1`  
**Date:** 2026-09-02  
**Updated:** 2026-09-03 (Phase 1 complete)  
**Status:** ACTIVE - PHASE_1_COMPLETE_READY_FOR_PHASE_0_COMPLETION  
**Owner:** chakwong

## Research Goal

**Primary question:** Do randomized quasi-Monte Carlo (RQMC) initialization methods improve LEDH-PFPF-OT particle filter log-likelihood estimates over standard Monte Carlo (MC) initialization?

**Mechanism under test:** Particle cloud initialization at t=0 only. All subsequent LEDH transport, resampling, and covariance updates are identical across arms.

**Decision:** Promote an RQMC method to replace MC as the default initialization if and only if it shows statistically superior terminal log-likelihood across all test models with acceptable dual-cap behavior.

## Background

The LEDH production program (`LEDH_PRODUCTION_PROGRAM_V1`) requires dual-cap trust-region covariance stabilization. RQMC initialization may improve sampling efficiency by reducing Monte Carlo variance at t=0. Four RQMC methods were tested in prior work (rescued from `experiment/genut-guided-initialization`) but predate the dual-cap mechanism. This program tests whether RQMC benefits persist under the production configuration.

## Test Models

Three benchmark models covering different state dimensions and dynamics:

1. **LGSSM T50**: Linear-Gaussian state-space model
   - State dimension: 3
   - Parameter dimension: 5
   - Observation dimension: 3
   - Horizon: T=50
   - Design: Replicated positive GenUT (N=1008)

2. **KSC SV T10**: Kim-Shephard-Chib stochastic volatility
   - State dimension: 1
   - Parameter dimension: 2
   - Observation dimension: 1
   - Horizon: T=10
   - Design: Gaussian GenUT (N=1008)

3. **Predator-Prey T20**: Lotka-Volterra nonlinear dynamics
   - State dimension: 2
   - Parameter dimension: 6
   - Observation dimension: 2
   - Horizon: T=20
   - Design: Gaussian GenUT (N=1008)

**Rationale:** These models are the standard leaderboard models excluding Austria SIR (which has very high dimension D=18 and may dominate the comparison).

## Arms

Five initialization methods:

1. **MC baseline** (control): Standard `tf.random.normal` with `replication_generator(seed)` pattern from `tf-consecutive-from-seed-is-one-stream.md` memory
2. **Sobol-Matousek**: Sobol sequence with Matousek 1998 nested uniform scrambling
3. **Sobol-Owen**: Sobol sequence with Owen 1995 random digital shift + scramble
4. **Halton-Owen**: Halton sequence with Owen 2017 randomized drop-and-permute
5. **GenUT guided**: Ebeigbe et al. 2021 generalized unscented transformation

All arms use identical LEDH-PFPF-OT dual-cap trust-region route after initialization.

## Production Configuration

**Program:** `LEDH_PRODUCTION_PROGRAM_V1`  
**Route:** `ledh_pfpf_ot_contract_e_dual_cap_trust_region`  
**Backend:** TensorFlow/TFP, GPU, TF32 enabled  
**Dtype:** `float32`  
**Particle count:** N=1008  
**Chunk policy:** `dpf_transport_exact_divisor_cap3000_v1` (K=N for N≤3000)  
**Covariance:** Triple {x,P,w} resampling with OT/Sinkhorn carry  
**Dual-cap:** Enabled (pairwise and coordinatewise caps)  
**Trust-region:** Enabled (LM-style damping with scale floor and radius)

## Tuning Requirement

**Per-scope tuning rule:** Every model requires a trust-region tuning artifact with exact scope match:
- Model/target
- Route: `ledh_pfpf_ot_contract_e_dual_cap_trust_region`
- Particle count: N=1008
- Horizon: T (model-specific)
- Dtype/backend: `float32` / TF32-GPU
- Chunk policy: `dpf_transport_exact_divisor_cap3000_v1`
- Dual-cap controls: inherited from dual-cap primal tuning
- Trust-region controls: damping, scale_floor, radius

**Tuning status (as of 2026-09-03):**
- ✅ Austria SIR T20: tuning artifact exists (`docs/benchmarks/artifacts/ledh_trust_region_austria_sir_t20_20260902/result.json`)
- ✅ LGSSM T50: tuning artifact exists (`docs/benchmarks/artifacts/ledh_trust_region_lgssm_t50_20260903/result.json`)
- ✅ KSC SV T10: tuning artifact exists (`docs/benchmarks/artifacts/ledh_trust_region_ksc_sv_t10_20260903/result.json`)
- ✅ Predator-Prey T20: tuning artifact exists (`docs/benchmarks/artifacts/ledh_trust_region_predator_prey_t20_20260903/result.json`)

**Tuning protocol:** Identical Phase 3-style campaign from `ledh-trust-region-tuning-safety-evaluation-phase3-2026-09-02.md`:
- 3 arms: baseline (no dual-cap), dual-cap primal, trust-region grid (27 configs)
- Grid: 3 dampings × 3 scale floors × 3 radii
- Seeds: 2 calibration observations × 2 tuning seeds per config
- Selection criterion: minimal intervention (lowest coordinatewise cap fire rate, then lowest damping, then smallest radius)
- Validity gates: finite, program_valid, residuals ≤ 5.0e-4, displacement ≤ 2.0

## Primary Criterion

**Terminal log-likelihood** at final time T. The LEDH score lane computes `sum_t log(mean_n w_n^t)` which is the standard particle filter likelihood estimand. Higher is better.

## Promotion Criterion

An RQMC arm is **viable for promotion** if:
1. All runs (3 seeds) complete without divergence, NaN, or dual-cap saturation >10%
2. Terminal log-likelihood mean is within 2 standard errors of MC baseline or higher (descriptive comparison)
3. No continuation veto fires

An RQMC arm **replaces MC as default** if and only if:
1. It passes viability on all 3 models
2. Terminal log-likelihood shows statistical superiority over MC on all 3 models (bootstrap 95% CI excludes zero difference, paired by seed)
3. Dual-cap activation rate ≤ MC baseline on all 3 models (not higher instability)

**Statistical note:** The 3-seed design favors the MC baseline. RQMC promotion requires large, unambiguous improvement (effect size >1 SD). Small effects will be called "statistically indistinguishable" - this is conservative by design.

## Promotion Veto

Hard veto (immediate disqualification of that arm on that model):
- Divergence, NaN, or non-finite score at any time step
- Dual-cap coordinatewise saturation >10% of particles at any time step
- Dual-cap pairwise saturation >10% of particles at any time step
- Missing required per-scope tuning artifact
- Program validation failure (`program_valid=False`)

## Continuation Veto

Stop the entire campaign if:
- Trust-region tuning fails to converge for >1 model after Phase 3-style campaign
- All RQMC arms show terminal log-likelihood statistically inferior to MC baseline (bootstrap 95% CI excludes positive difference) on LGSSM (first model tested)
- Infrastructure failure (GPU OOM, serialization corruption, harness crash) affects >20% of runs after 2 repair attempts per configuration
- Estimand gate fails (score lane does not compute likelihood estimand)

## Explanatory Diagnostics

Recorded per arm per model per seed, not used for promotion/veto:
- Per-time-step log-likelihood increment (trace plot)
- Effective sample size (ESS) at T
- Sinkhorn iteration count (mean, max, per time step)
- Dual-cap activation rate (pairwise, coordinatewise, per time step)
- Runtime (wall seconds, GPU utilization)
- Tuning objective value from tuning artifact (for tuning confound check)

## What Will Not Be Concluded

- **Not claiming correctness:** RQMC initialization does not alter the LEDH target distribution; only sampling efficiency and Monte Carlo variance are under test
- **Not claiming horizon-independence:** results are specific to T=10/20/50; longer horizons require fresh evidence
- **Not claiming model-independence:** results are specific to LGSSM, KSC SV, Predator-Prey; other models (especially high-dimensional ones like Austria SIR D=18) require separate validation
- **Not claiming tuning-free:** each model requires per-scope trust-region tuning; RQMC arms use the same tuning artifact as MC baseline
- **Not claiming regime-specific performance:** results are unconditional averages across the full time series; high-volatility vs low-volatility or boundary vs interior performance not separately evaluated
- **Not claiming runtime improvement:** runtime is an explanatory diagnostic, not a promotion criterion; promoted RQMC arms may be slower than MC baseline

## Phase Structure

### Phase 0: Infrastructure Setup (BLOCKING) ⚠️ PARTIAL

**Objective:** Implement the 3 required runners and validate production program wiring.

**Deliverables:**
1. ✅ Production program definition (`bayesfilter/highdim/ledh_production_program_v1.py`)
2. ✅ Trust-region tuning runner template (`docs/benchmarks/run_ledh_trust_region_phase3_austria_sir.py`)
3. ❌ RQMC claim-bearing runner (`docs/benchmarks/run_rqmc_ledh_initialization.py`) — BLOCKING PHASE 2
4. ❌ Result assembler (`docs/benchmarks/assemble_rqmc_ledh_results.py`) — BLOCKING PHASE 3

**Completion criteria:**
- [x] Production program validation function exists and passes on Austria SIR
- [x] Trust-region tuning runner template exists and runs without error
- [ ] MC baseline smoke test passes: LGSSM result matches historical leaderboard within 50%
- [ ] RQMC runner accepts `--model`, `--arm`, `--seed`, `--tuning_artifact`, `--output`
- [ ] Result assembler computes bootstrap 95% CI for paired differences
- [ ] All runners use `replication_generator(seed)` for seed hashing
- [ ] Configuration-status-first reporting enforced in result tables

**Status:** PARTIAL (4/7 complete) — production program and tuning template exist, RQMC runner and assembler missing

**Note:** Phase 1 (tuning) can proceed in parallel with completing Phase 0 items 3-7, since tuning only requires the template (item 2). Phase 0 must be complete before Phase 2 starts.

### Phase 1: Trust-Region Tuning (BLOCKING) — ✅ COMPLETE

**Objective:** Generate per-scope trust-region tuning artifacts for all 3 test models.

**Protocol:** Duplicate Phase 3 Austria SIR campaign for each model:
- 3 arms: baseline, dual-cap primal, trust-region grid (27 configs)
- 2 calibration observations × 2 tuning seeds per config = 116 evaluations per model
- Minimal intervention selection criterion
- Output: model-specific tuning artifact JSON

**Models:**
1. ✅ LGSSM T50 — trust-region tuning complete (2026-09-03, 369s wall time)
2. ✅ KSC SV T10 — trust-region tuning complete (2026-09-03, 228s wall time)
3. ✅ Predator-Prey T20 — trust-region tuning complete (2026-09-03, 359s wall time)

**Completion criteria per model:**
- [x] Campaign completes with ≥1 passing trust-region config (≥1/27 valid)
- [x] Selected config improves on dual-cap primal or is statistically indistinguishable
- [x] Tuning artifact JSON written with exact scope signature
- [x] Completion memo documents selection and Class C verdict

**Results:**
- All 27 configs passed validity gates (100% pass rate) for all 3 models
- All models selected identical configuration: damping=0.001, scale_floor=1e-06, radius=0.1
- Class C verdict: CONDITIONAL PASS (bounded degradation verified, non-harm not checked)

**Artifacts:**
1. `docs/benchmarks/artifacts/ledh_trust_region_lgssm_t50_20260903/result.json`
2. `docs/benchmarks/artifacts/ledh_trust_region_ksc_sv_t10_20260903/result.json`
3. `docs/benchmarks/artifacts/ledh_trust_region_predator_prey_t20_20260903/result.json`

**Completion memo:** `docs/memos/ledh-trust-region-phase4-complete-2026-09-03.md`

**Budget:** ~380 seconds per model (GPU, from Austria SIR precedent) = ~20 minutes total — ACTUAL: 956 seconds total (16 minutes)

**Continuation veto:** None fired. All models produced valid trust-region configs.

**Status:** 3 of 3 models complete (2026-09-03)

### Phase 2: RQMC Claim-Bearing Runs (BLOCKING)

**Objective:** Execute the 3 seeds × 5 arms × 3 models = 45 claim-bearing runs and collect artifacts.

**Protocol:**
```bash
for model in lgssm_T50 ksc_sv_T10 predator_prey_T20; do
  for arm in mc sobol_matousek sobol_owen halton_owen genut_guided; do
    for seed in 98301 98302 98303; do
      python docs/benchmarks/run_rqmc_ledh_initialization.py \
        --model $model \
        --arm $arm \
        --seed $seed \
        --tuning_artifact docs/benchmarks/artifacts/ledh_trust_region_${model}_<date>/result.json \
        --output docs/benchmarks/artifacts/rqmc_ledh_init_v1_<date>/runs/${model}_${arm}_seed${seed}/
    done
  done
done
```

**Per-run artifacts:**
- Manifest: git commit, command, conda env, GPU status, tuning artifact path, seed
- Terminal log-likelihood value
- ESS at T
- Dual-cap activation rates (pairwise, coordinatewise, per time step)
- Sinkhorn iteration counts (per time step)
- Runtime (wall seconds)

**Completion criteria:**
- [ ] All 45 runs complete or explicitly marked as veto/infrastructure-failure
- [ ] ≥80% of runs produce valid artifacts (20% infrastructure-failure budget)
- [ ] No continuation veto fires

**Budget:** ~5-30 minutes per run × 45 runs = 4-23 hours wall time (serial on 1 GPU)

**Infrastructure failure protocol:** After 2 repair attempts on the same configuration, mark cell as "infrastructure failure" and continue. If >20% of runs fail infrastructure (>9/45), stop campaign and diagnose.

**Status:** Not started

### Phase 3: Statistical Analysis and Decision (NON-BLOCKING)

**Objective:** Aggregate results, compute statistical rankings, apply promotion/veto criteria, and make the promotion decision.

**Protocol:**
```bash
python docs/benchmarks/assemble_rqmc_ledh_results.py \
  --input docs/benchmarks/artifacts/rqmc_ledh_init_v1_<date>/runs/ \
  --output docs/benchmarks/artifacts/rqmc_ledh_init_v1_<date>/results.json
```

**Analysis steps:**
1. Aggregate terminal log-likelihood by (model, arm, seed)
2. Compute paired differences (RQMC_arm - MC_baseline) per seed per model
3. Bootstrap 95% CI for mean paired difference (10,000 bootstrap samples)
4. Apply promotion veto: check for NaN, divergence, dual-cap saturation >10%
5. Apply viability criterion: mean within 2 SE of baseline or higher
6. Apply promotion criterion: 95% CI excludes zero on all 3 models
7. Check tuning confound: compare tuning objective values across arms
8. Compute explanatory diagnostics: ESS, dual-cap rates, runtime

**Result table structure (configuration-status-first):**
```
| model | arm | seed | program | tuning_artifact | terminal_ll | ess_T | dual_cap_coord_rate | dual_cap_pair_rate | runtime_s | veto | viable | promoted |
```

**Decision table:**
- For each RQMC arm: PROMOTED / VIABLE / VETOED
- Overall decision: PROMOTE_<arm> / KEEP_MC_DEFAULT / INCONCLUSIVE

**Completion criteria:**
- [ ] Result JSON written with full decision table
- [ ] Bootstrap CIs computed for all arm-model pairs
- [ ] Promotion decision explicitly stated
- [ ] Result memo documents decision rationale and non-conclusions

**Status:** Not started

### Phase 4: Result Documentation (NON-BLOCKING)

**Objective:** Write the result memo with decision rationale, statistical evidence, explanatory diagnostics, and non-conclusions.

**Required sections:**
1. Executive summary (decision in one sentence)
2. Decision table (promoted/viable/vetoed per arm per model)
3. Statistical evidence (bootstrap CIs, viability status)
4. Explanatory diagnostics (ESS, dual-cap rates, runtime, traces)
5. Tuning confound check (tuning objective comparison)
6. Post-hoc red-team: alternative explanations, weakest evidence, regime-specific caveat
7. Non-conclusions (horizon-independence, model-independence, regime-specific, tuning-free)
8. Next steps (if promoted: implementation plan; if not: archive or extended investigation)

**Completion criteria:**
- [ ] Result memo written to `docs/memos/rqmc-ledh-init-v1-result-2026-09-<date>.md`
- [ ] Decision rationale grounded in statistical evidence, not descriptive comparison
- [ ] Non-conclusions explicitly stated
- [ ] Post-hoc red-team section completed

**Status:** Not started

## Seed Allocation

**Tuning seeds:** 98301, 98302 (2 per model, used in Phase 1)  
**Claim-bearing seeds:** 98303, 98304, 98305 (3 per arm per model, used in Phase 2)

**Note:** Seeds are hashed via `np.random.SeedSequence(seed).generate_state(2, dtype=np.uint64)` before passing to `tf.random.Generator.from_seed()` to avoid the consecutive-seed Philox stream sharing bug documented in `tf-consecutive-from-seed-is-one-stream.md`.

## Artifact Locations

**Phase 1 tuning artifacts:**
- `docs/benchmarks/artifacts/ledh_trust_region_lgssm_t50_20260902/result.json`
- `docs/benchmarks/artifacts/ledh_trust_region_ksc_sv_t10_20260902/result.json`
- `docs/benchmarks/artifacts/ledh_trust_region_predator_prey_t20_20260902/result.json`

**Phase 2 run artifacts:**
- Root: `docs/benchmarks/artifacts/rqmc_ledh_init_v1_20260902/`
- Per-run: `<root>/runs/<model>_<arm>_seed<seed>/result.json`

**Phase 3 result artifact:**
- `docs/benchmarks/artifacts/rqmc_ledh_init_v1_20260902/results.json`

**Phase 4 result memo:**
- `docs/memos/rqmc-ledh-init-v1-result-2026-09-<date>.md`

## Pre-Execution Gates

Before proceeding with Phase 2, verify:
- [ ] Estimand gate passing: production program score lane computes `sum_t log(mean_n w_n^t)`
- [ ] GPU memory growth enabled and verified in runner
- [ ] Seed hashing function implemented correctly (test: consecutive seeds produce uncorrelated streams)
- [ ] Configuration-status-first table structure enforced in result assembler
- [ ] Continuation veto logic implemented in runner or supervisor script
- [ ] Fresh output directory created with unique timestamp
- [ ] MC baseline smoke test passing: LGSSM result matches historical leaderboard within 50%

## Dependencies

**Blocking Phase 1 (Tuning):**
- Dual-cap primal tuning artifacts for LGSSM, KSC SV, Predator-Prey (inherit from prior leaderboard work if scope matches)
- Trust-region tuning runner template (`run_ledh_trust_region_phase3_austria_sir.py`)

**Blocking Phase 2 (Claim-bearing runs):**
- Phase 1 complete (all 3 tuning artifacts exist)
- RQMC runner implemented and tested
- Estimand gate passing

**Blocking Phase 3 (Analysis):**
- Phase 2 complete (≥80% of 45 runs valid)
- Result assembler implemented and tested

**Non-blocking:**
- Phase 4 can proceed as soon as Phase 3 decision is finalized

## Authority and Drift Prevention

**This master program is the sole authority for the RQMC LEDH initialization investigation.** The following rules prevent drift:

1. **No user choice:** The program specifies exact models (LGSSM, KSC SV, Predator-Prey), arms (5 methods), seeds (98301-98305), and decision criteria. The user may not substitute different models, change seed counts, or alter promotion criteria mid-execution.

2. **Phase order is strict:** Phase 1 must complete before Phase 2 starts. Phase 2 must complete before Phase 3 starts. No phase may be skipped or reordered.

3. **Continuation vetoes are binding:** If a continuation veto fires, execution stops immediately. The program may not continue by relaxing veto thresholds or "trying one more thing."

4. **Promotion criteria are frozen:** The bootstrap 95% CI threshold and "all 3 models" requirement may not be weakened post-hoc. If results are inconclusive, the decision is KEEP_MC_DEFAULT, not "let's try more seeds" or "let's relax the threshold."

5. **Scope creep is prohibited:** The program tests RQMC initialization only. It may not expand to test different transport methods, different particle counts, different models, or different LEDH routes mid-execution.

6. **Amendments require explicit justification:** Any change to this program (models, criteria, phases, vetoes) requires a written amendment with justification, not an inline user request. The amendment must explain why the original program was insufficient and how the change preserves scientific validity.

## Program Metadata

**Git commit at program creation:** (to be filled at first commit)  
**Conda environment:** `tftwogpu`  
**GPU:** NVIDIA GeForce RTX 4080 SUPER (primary), NVIDIA GeForce RTX 5080 (fallback)  
**TensorFlow version:** (to be recorded at first run)  
**Repository backend rule:** TensorFlow/TFP (NumPy prohibited except diagnostics)

## References

**Governing policies:**
- `CLAUDE.md`: LEDH Per-Scope Tuning Rule, Evidence Contract Before Research Actions, Statistical Evidence Discipline, Research Question Guardian, Safety Guardrail Reversed Burden, Heuristic Dominance Gate
- `LEDH_PRODUCTION_PROGRAM_V1`: Production configuration and wiring gates (`bayesfilter/highdim/ledh_production_program_v1.py`)

**Source plans:**
- `docs/plans/rqmc-genut-ledh-test-plan-2026-09-01.md` (original draft plan)
- `docs/plans/rqmc-genut-ledh-test-plan-review-2026-09-01.md` (skeptical review)
- `docs/plans/ledh-trust-region-tuning-safety-evaluation-phase3-2026-09-02.md` (tuning protocol template)
- `docs/plans/ledh-pfpf-ot-dual-cap-implementation-study-2026-09-01.md` (dual-cap study, Phase 4 now governed by this program)

**Memory:**
- `tf-consecutive-from-seed-is-one-stream.md`: Seed hashing requirement
- `configuration-status-first-reporting.md`: Reporting rule for benchmark tables

**Prior work:**
- Rescued GenUT artifacts: `experiment/genut-guided-initialization` (pre-dual-cap, N=1008, warm-start reference only)

---

**Program status:** ACTIVE  
**Current phase:** Phase 0 (PARTIAL, 4/7 complete) + Phase 1 (Trust-Region Tuning, 0 of 3 models complete)  
**Next action:** Start Phase 1 tuning for LGSSM T50 (can proceed in parallel with Phase 0 runner implementation)  
**Execution strategy:** Phase 1 tuning uses existing template and is independent of Phase 2 runners; complete Phase 1 while finishing Phase 0 items 3-7 in parallel

**Program approved by:** chakwong  
**Execution authorized:** 2026-09-02
