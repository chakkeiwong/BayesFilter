# Surrogate-Force HMC Investigation Plan

## Date
2026-08-31

## Context
User requested surrogate-force HMC investigation after context compaction. We've now discovered the correct batch-native LEDH API and completed an ultra-minimal proof-of-concept.

## Research Question
Can we use surrogate-force HMC (exact value, damped gradient) with batch-native LEDH to:
1. Maintain sampling correctness (samples from exact posterior)
2. Improve numerical stability vs exact gradient
3. Achieve reasonable performance on available hardware

## Background

### Surrogate-Force HMC Theory
- Use p(y|θ) for value (exact posterior)
- Use p̃(y|θ) for gradient (damped/stabilized)
- MH correction ensures exact sampling
- Damping should reduce gradient noise without biasing samples

### Implementation Discovery
- **Wrong API** (used initially): `canonical_value_and_analytical_score` (single-cloud, eager)
- **Correct API**: `canonical_batch_fused_value_score` (batch-native, NeuTra-eligible)
- Located in: `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py`
- Factory: `make_canonical_neutra_target("lgssm")`

### Hardware Constraints Discovered
- Memory limits prevent large HMC runs with N=1008
- Ultra-minimal (1 chain, 20 steps) completes successfully
- Need reduced particle count for meaningful comparisons

## Phases

### Phase 1: Toy Potential Validation ✅ COMPLETE
Validated surrogate-force concept on simple Gaussian potential.

### Phase 2: LGSSM Batch-Native Validation ✅ PROOF-OF-CONCEPT COMPLETE
**Status**: Ultra-minimal run succeeded (1 chain, 10+10 samples)
**Finding**: Batch-native LEDH works with HMC but memory-constrained

**What remains**:
- Implement true surrogate-force (exact value, damped gradient)
- Run with reduced N for longer chains
- Compare exact vs damped force acceptance/ESS

### Phase 3: Three-Arm Comparison (PLANNED)
Compare three configurations on equal footing:
- Arm A: Exact force (λ=1e-5, δ=1e-5)
- Arm B: Damped force (λ=1e-3, δ=1e-3)  
- Arm C: Surrogate-force (exact value, damped gradient)

**Success criteria**:
- All three complete without divergences
- Arm C matches Arm A's posterior (KL divergence, coverage)
- Arm C acceptance rate comparable to Arm B
- Arm C shows improved stability vs Arm A (if Arm A has issues)

## Immediate Next Steps

### Step 1: Implement True Surrogate-Force
- Build adapter that calls exact model for value, damped for gradient
- Fix shape handling for TFP's HMC
- Test on ultra-minimal settings (1 chain, 20 steps)

### Step 2: Reduce Particle Count
- Test N=252 (quarter of 1008)
- Verify reduced N completes longer runs
- Measure memory vs N scaling

### Step 3: Three-Arm Pilot
- Run all three arms with N=252, 2 chains, 100+100 samples
- Compare acceptance rates
- Check posterior agreement (Arm C vs Arm A)
- Measure ESS, runtime

### Step 4: Analysis
- Posterior KL divergence (Arm C vs Arm A should be ~0)
- Gradient stability metrics
- Acceptance rate comparison
- ESS per gradient evaluation

## Evidence Contract

### Promotion Criteria
Surrogate-force is viable if:
1. Arm C samples match Arm A posterior (KL < 0.1, coverage >90%)
2. Arm C completes without divergences
3. Arm C acceptance ≥ 0.5 × Arm A acceptance

### Promotion Vetoes
- Arm C diverges while Arm A doesn't
- Arm C posterior differs materially from Arm A
- Arm C is slower than Arm A with no stability benefit

### Non-Claims
- We do NOT claim surrogate-force is faster (not the goal)
- We do NOT claim it improves ESS (stability is the goal)
- We do NOT claim N=252 is production-ready (it's a test setting)

## Technical Specifications

### Model
- LGSSM d=3, T=50
- Data: benchmark_lgssm_m3_T50_seed81100
- True params: φ=[0.72, 0.55, 0.35], q=0.35, r=0.45

### Damping Configurations
- Exact: λ=1e-5, δ=1e-5 (process/obs ridge)
- Damped: λ=1e-3, δ=1e-3

### Particle Settings
- Phase 2 ultra-minimal: N=1008 (complete, memory-limited)
- Phase 3 pilot: N=252 (reduced for longer runs)

### HMC Settings (Pilot)
- Chains: 2
- Burnin: 100
- Samples: 100  
- Leapfrog steps: 5
- Step size: adaptive (target acceptance 0.65)

## Files
- Plan: `docs/plans/surrogate-force-hmc-investigation-plan-2026-08-31.md`
- Ultra-minimal (complete): `docs/benchmarks/phase2_ultraminimal_fixed.py`
- Implementation summary: `docs/benchmarks/phase2_implementation_summary.md`
- Memory analysis: `docs/benchmarks/phase2_memory_analysis.md`

## Next Action
Execute Step 1: Implement true surrogate-force adapter with exact value + damped gradient.
