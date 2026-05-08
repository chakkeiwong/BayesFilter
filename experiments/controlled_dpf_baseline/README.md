# Controlled DPF Baseline

Purpose: develop a clean experimental differentiable particle-filter baseline
owned by this project.

This project is quarantined under `experiments/`.  It can move faster than the
production implementation and can be used to debug algorithms, compare against
student baselines, and generate stable experimental reports.  It is not a
production BayesFilter implementation.

Allowed uses:

- small EDH/LEDH/PF prototypes;
- importance-weight and log-Jacobian experiments;
- soft-resampling and OT experiments;
- common fixture comparisons;
- runtime, ESS, gradient, and failure-mode diagnostics.

Disallowed uses:

- production dependency;
- public BayesFilter API;
- HMC readiness certification;
- importing from `bayesfilter/` production modules in a way that creates
  circular experimental assumptions.

Promotion rule:

```text
Experimental success here must be reimplemented or separately audited before
moving into bayesfilter/.
```

Planned structure:

```text
prototypes/
fixtures/
runners/
reports/
```
