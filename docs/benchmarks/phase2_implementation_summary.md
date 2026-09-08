# Phase 2 Surrogate-Force HMC: Implementation Summary

> **RETRACTION NOTICE (2026-08-30).** Claims in this document are partly
> withdrawn. See
> `docs/benchmarks/surrogate_force_correction_and_graph_diagnosis_20260830.md`.
> Withdrawn here: any "validated" / "success" / "samples the exact posterior"
> claim (evidence was 1 chain × 10 retained draws at 100% acceptance, no R-hat,
> no ESS, unmatched seeds); the exact arm's ridge, recorded as λ=δ=1e-5, is
> actually λ=δ=0; and the memory diagnosis attributing the failure to particle
> count is wrong — it is host graph size from Python-loop unrolling. Also note
> the force is 3-parameter, not 5: ∂/∂q and ∂/∂r are structurally zero.

## Date
2026-08-30

## Objective
Validate surrogate-force HMC on LGSSM using batch-native LEDH canonical implementation.

## Key Discovery
The user corrected a major error: I was using the wrong API.
- **Wrong**: `canonical_value_and_analytical_score` (single-cloud, eager mode)
- **Right**: `canonical_batch_fused_value_score` (batch-native, NeuTra-eligible)

## Implementation

### Correct Batch-Native API
- **File**: `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py`
- **Function**: `canonical_batch_fused_value_score(model, theta, directions, ...)`
- **Signature**: 
  - Input: `theta` [B, P], `directions` [B, P]
  - Output: `(values [B], scores [B], diagnostics)`
- **Strategy**: Flatten [B, N] → [B*N] particles, fused operations, analytical score

### Surrogate-Force Adapter
Built two separate `PerPointScoreModel` instances:

1. **Exact model** (for value):
   - Process variance: 0.122500 (q² = 0.35²)
   - Observation variance: 0.202500 (r² = 0.45²)

2. **Damped model** (for gradient):
   - Process variance: 0.123500 (q² + λ, λ=1e-3)
   - Observation variance: 0.203500 (r² + δ, δ=1e-3)

### Adapter Logic
```python
@tf.function(autograph=False)
def compute_value_and_grad(theta_batch):
    # Value from exact model
    exact_vals, _, _ = canonical_batch_fused_value_score(
        exact_model, theta_batch, directions_dummy, ...
    )
    
    # Gradient from damped model (5 directions for 5 parameters)
    gradients = []
    for p in range(5):
        direction = one_hot_direction(p)
        _, scores, _ = canonical_batch_fused_value_score(
            damped_model, theta_batch, direction, ...
        )
        gradients.append(scores)
    
    return exact_vals, tf.stack(gradients, axis=1)
```

## Testing History

### Attempt 1: Full HMC (4 chains, 500+500)
- **Status**: Killed (OOM)
- **Memory**: 45GB RAM before kill
- **Issue**: Too many samples × chains × leapfrog steps

### Attempt 2: Optimized with XLA (2 chains, 200+200)
- **Status**: Failed (XLA compilation OOM)
- **Error**: `LLVM compilation error: Cannot allocate memory`
- **Issue**: XLA tried to compile massive fused graph, ran out of memory

### Attempt 3: Minimal without XLA (2 chains, 50+50, 3 leapfrog)
- **Status**: RUNNING (started 2026-08-31 02:21)
- **Expected runtime**: ~15-30 minutes
- **Settings**:
  - Chains: 2
  - Burnin: 50
  - Samples: 50
  - Leapfrog steps: 3
  - Graph mode with @tf.function (no XLA)

## Technical Validation

### Value/Score Differences (Exact vs Damped)
From single-point test:
- Exact value: -128.996752
- Damped value: -129.058089
- **Value difference**: 0.061338 (small, as expected)

- Exact score[0]: -7.074250
- Damped score[0]: -7.066445
- **Score difference**: -0.007805 (small, damping effect visible)

### Performance
- First call (with tracing): ~25s for 5 parameter directions
- Graph mode prevents retracing on subsequent calls
- Each HMC step requires: 1 value + 5 gradients = 6 LEDH filter calls
- 100 total steps × 6 calls × ~4-5s/call = ~30-45 minutes estimated

## Theory
Surrogate-force HMC samples the **exact posterior** when:
- Value uses exact target density p(y|θ)
- Gradient uses damped/smoothed version p̃(y|θ)
- MH correction uses exact value

The damped gradient provides numerical stability without biasing the samples.

## Next Steps (After Completion)
1. Verify acceptance rate is reasonable (target: 0.65)
2. Check parameter estimates match true values
3. Document as proof-of-concept that surrogate-force works with batch-native LEDH
4. Consider longer runs if minimal test succeeds
5. Update Phase 2 plan with success status

## Files
- Implementation: `docs/benchmarks/phase2_surrogate_minimal.py`
- This summary: `docs/benchmarks/phase2_implementation_summary.md`
- Output: `/tmp/phase2_minimal.txt` (when complete)
