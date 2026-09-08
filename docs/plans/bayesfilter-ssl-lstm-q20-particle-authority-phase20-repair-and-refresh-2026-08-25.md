# Phase 20 Repair and Refresh Note

Status: `PASS_SOURCE_FAITHFUL_GENUT_FIXTURE_REFRESHED_PHASE21`

The phase preserves the source equations, finite-cloud role, and no-density
boundary. A failed feasibility gate is expected to trigger a fixture repair,
not an overclaim or silent clipping.

| Failure | Classification | Repair |
|---|---|---|
| missing/hash-mismatched source or plan | harness | fail closed and create a fresh receipt |
| negative discriminant/offset/central weight | mathematical feasibility | preserve diagnostics; choose a new reviewed fixture |
| moment residual failure | implementation | inspect whitening, offsets, weights, and map orientation |
| finite/status failure | numerical | isolate dtype/shape; no target change |
| good moments but poor global support | explanatory | retain quadrature-only status; no IID/density claim |

After a complete receipt, record source/code hashes, feasibility diagnostics,
moment residuals, and the Phase 21 entry gate.

The 25-point fixture passed with positive central weight and nonzero
asymmetry. Refresh Phase 21 to test the same feasibility equations on the
metadata-bound q20 cloud; if its central weight is negative, retain that as a
mathematical scope result and consider only a reviewed local/per-mode route.
