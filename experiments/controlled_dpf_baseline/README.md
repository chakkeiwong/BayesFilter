# Controlled DPF Baseline

Purpose: develop a clean-room experimental DPF-style controlled baseline owned
by this project for the quarantined student DPF comparison lane.

This project is quarantined under `experiments/`.  It can move faster than the
production implementation and can be used to debug algorithms, compare against
student baselines, and generate stable experimental reports.  It is not a
production BayesFilter implementation.

Allowed uses:

- clean-room fixture generation for `range_bearing_gaussian_moderate` and
  `range_bearing_gaussian_low_noise`;
- a minimal BayesFilter-owned regularized particle-flow/bootstrap-particle
  scaffold with explicit `flow_steps`;
- fixed MP5 smoke execution and MP6 15-record fixed-grid execution;
- proxy metrics, runtime, ESS, resampling, finite-output diagnostics, and
  structured blocker records;
- MP7 proxy-only comparison against frozen student aggregate reports.

Disallowed uses:

- production dependency;
- public BayesFilter API;
- HMC readiness certification;
- monograph evidence without a separate review;
- importing student vendor snapshots, student adapters, or student implementation
  modules;
- copying student classes, functions, control flow, tuning tricks, or numerical
  shortcuts;
- broad EDH/LEDH, soft-resampling, OT, HMC, kernel PFF, DPF, dPFPF, neural OT,
  differentiable-resampling, AR(2), structural-SSM, stiffness, or resampling
  stress experiments unless a later student-lane master revision authorizes
  them explicitly.

Promotion rule:

```text
Experimental success here must be reimplemented or separately audited before
moving into bayesfilter/.
```

Current authority:

- `docs/plans/bayesfilter-student-dpf-baseline-master-program-2026-05-10.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-2026-05-13.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-mp5-clean-room-implementation-plan-2026-05-13.md`.

Planned structure:

```text
prototypes/
fixtures/
runners/
reports/
```
