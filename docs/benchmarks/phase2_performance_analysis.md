# Phase 2 Performance Characteristics

> **RETRACTION NOTICE (2026-08-30).** Timings here are descriptive only and their
> stated cause is wrong. See
> `docs/benchmarks/surrogate_force_correction_and_graph_diagnosis_20260830.md`.
> The dominant cost is graph tracing, not per-particle arithmetic: 110,628 graph
> nodes per LEDH call, 6 LEDH calls per gradient evaluation, node count linear in
> horizon and substeps. Reducing N=1008 → N=252 changes tensor extents but not
> node count, which is why it bought only ~2.17× wall time and did not fix the
> memory failure. Any "if successful, proof-of-concept complete" framing is
> withdrawn; no sampling claim was established.

## Batch-Native LEDH Timing

### Single Evaluation (Eager/First Call)
- **Target**: LGSSM, N=1008, substeps=12, T=50
- **Time**: 4-7 seconds per call
- **Memory**: Varies by context

### Graph Mode (@tf.function)
- **First call**: 3-5 minutes (includes tracing + execution)
- **Subsequent calls**: Should be faster (graph cached)
- **Issue**: HMC builds complex nested graph with while_loop

## Memory Consumption by Configuration

| Config | Chains | Burn+Sample | Leapfrog | Memory | Status |
|--------|--------|-------------|----------|--------|--------|
| Full | 4 | 500+500 | 10 | 45GB | OOM killed |
| Optimized+XLA | 2 | 200+200 | 5 | N/A | XLA compilation OOM |
| Minimal | 2 | 50+50 | 3 | Unknown | OOM killed during tracing |
| Ultra-minimal | 1 | 10+10 | 2 | 6.6% (~3-4GB) | Running (3m47s) |

## Key Findings

1. **Memory scales super-linearly** with HMC parameters
   - Not just chains × samples
   - TensorFlow graph includes all loop structure
   - Each leapfrog step multiplies graph size

2. **XLA makes it worse**
   - Tries to compile entire HMC graph upfront
   - Graph is too large for LLVM to compile

3. **Ultra-minimal is viable**
   - 1 chain, 20 total steps, 2 leapfrog
   - 6.6% memory usage (~3-4GB on 48GB machine)
   - Provides proof-of-concept

4. **Production requirements**
   - For meaningful HMC runs (1000+ samples, 4+ chains)
   - Would need: particle count reduction OR more RAM OR GPU

## Particle Count Alternatives

Current: N=1008

| N | Memory Factor | Notes |
|---|---------------|-------|
| 1008 | 1.0x | Current (too large for HMC) |
| 504 | ~0.5x | Half particles |
| 252 | ~0.25x | Quarter particles |
| 126 | ~0.12x | Eighth particles |

Memory reduction not perfectly linear due to:
- Fixed overhead (observations, initial state)
- Weight normalization over particles
- But should help significantly

## Recommendation

For Phase 2 completion:
1. ✅ Ultra-minimal proves concept works
2. For longer runs: reduce N to 252-504
3. Or: enable GPU (more memory bandwidth)
4. Or: use different machine with more RAM

## Current Status

**Awaiting**: Ultra-minimal completion (1 chain, 10+10, running at 6.6% memory)
**ETA**: ~5-10 more minutes for graph tracing + execution
**If successful**: Phase 2 proof-of-concept complete
**If fails**: Need to reduce N or declare hardware-limited

Date: 2026-08-31 04:10
