# Numerical-Design Instrumentation Allocation Audit

Date: 2026-07-14

Status: `STREAMING_REPORT_STATE_O_BND_PLUS_BLOCKS_NO_N_SQUARED_OUTPUT`

The production marginal reporter is
`bayesfilter.highdim.ledh_contract_e_streaming_tf._streaming_column_mass_from_potentials_core`.
It retains:

- final row and column potentials: two `[B,N]` tensors;
- column log normalizer: `[B,N]`;
- result column mass: `[B,N]` through a TensorArray of `[B,col_chunk]` blocks;
- one query/key/cost block: `[B,row_chunk,col_chunk]`; and
- scalar/block loop state.

It returns only `[B,N]` row/column mass, target, and residual tensors plus
`[B]` norms/scales. It does not return or retain a `[B,N,N]` coupling.

Static checks reject `_filterflow_exact_cost`,
`_filterflow_exact_transport_from_potentials`, and any list/tuple shape with
two `particle_count` axes in the reporting helper. Repository source search
found no `num_particles,num_particles` or `particle_count,particle_count`
allocation in the owned streaming module. The existing dense comparator is
used only in the tiny test and is not imported by the owned module.

This is a source/graph-structure engineering audit. It does not measure GPU
memory or establish production-shape runtime feasibility.
