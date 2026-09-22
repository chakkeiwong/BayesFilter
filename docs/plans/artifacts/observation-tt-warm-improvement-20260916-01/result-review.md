# A09 terminal review

Reviewer: Codex executor self-review. No independent review is claimed.
Status: terminal review complete. PASS for evidence integrity and the narrow
scalar improvement conclusion. WITHHOLD d4 ranking/default promotion because
of the documented guide, reference, evidence and heuristic failures.

## Checked implementation and interpretation

- The optional penalty is lambda times the L1 distance of a core from its
  original coefficients. Its proximal update and reported KKT both use that
  same anchor; the compiler cache includes the policy. Zero-centered behavior
  remains the default. Closed-form and lambda-zero parity checks pass.
- The standalone path retains its own fitted TT between observations. After
  the initial Gaussian step, the actual particle consumer reaches the pair-TT
  sampler. No per-step Gaussian selection is used.
- Polynomial degree and row-count nominees use calibration filtering MSE;
  audit rows do not select them. Both confirmation panels consume the same
  frozen selected-controls.json. Generic-start TT is a historical algorithm
  comparator, not an isolated initialization ablation: its defensive mixture
  and row design also differ from the new warm path.
- The primary comparison is the normalized mean error of the exact-weight
  particle filter. TT regression discrepancies and ESS explain that behavior;
  they do not independently establish downstream superiority.
- Final call-chain review caught an inherited seed-spacing defect after the
  first confirmation. Time-index addition caused random-stream reuse across
  sequences/repetitions. Revision 2 uses disjoint sequence blocks and passes
  both an exhaustive schedule check and a reference-endpoint wiring test.
  First-confirmation intervals are ineligible and raw results are preserved.
- Calibration-02 remains a nomination set with correlated algorithm streams.
  Its fit-loss observations are descriptive. Retuning after inspecting
  confirmation would be invalid; no retuning occurred.
- Fixed four-sweep/128-proximal-step fitting has no solver convergence claim.
  Same-target d4 Gram condition up to 5.24e10 and KKT up to .0137 make solver
  accuracy an unresolved alternative explanation. More basis terms need not
  reduce conditional importance-weight variance.
- Timings include setup/compilation and follow a fixed arm order. They are
  descriptive GPU-kernel/host-orchestration costs, not a statistically supported
  hardware speed ranking or an end-to-end GPU/XLA production benchmark.

## Final result check

audit.py passes against confirmation-02 and calibration-02: 14880 particle-time
records, 6992 TT CDF checks, 1311 fixed fit/seed checks, 19 source snapshots and
54 same-target audits. Scalar contrasts have negative upper limits; d4 is
ineligible because sequence 10 lacks a guide and valid reference. Sequence 4
separately passes the independent reference but fails every guide-dependent
log-evidence screen. Its t=8 guide Cholesky scale is 2-4e-16, confirming a
collapsed proposal chart. Eight d4 conditional heuristic losses restrict
promotion. No default, raw-TT, analytical-total-gradient or source-faithfulness
conclusion is supported. The next repair must address guide scale before
interpreting additional TT capacity as sufficient.

The initial plan review missed time-shift seed reuse. The final call-chain
audit caught it, and the first confirmation was replaced without retuning.
This is a limitation of that review, not evidence that the defect was harmless.
The artifact audit initially assumed every consumer stored a `finite` flag;
analytic heuristic consumers do not. It now checks every stored numeric field
and additionally requires the actual TT sampler flag/CDF diagnostics.
