# Student DPF Baselines

Purpose: reproduce and compare student differentiable particle-filter
implementations as internal experimental baselines.

This project is `comparison_only`.  Student code may be used for testing and
comparison because permission has been granted by the students working on this
project.  It must not be imported by production BayesFilter modules or treated
as production-quality implementation.

Allowed uses:

- reproduce reported student results;
- compare two independent implementations on common fixtures;
- record filtering metrics, runtime, ESS, and failure modes;
- identify ideas that may later be reimplemented behind BayesFilter contracts.

Disallowed uses:

- production implementation;
- public BayesFilter API;
- HMC target construction;
- correctness certification;
- direct dependency of `bayesfilter/`.

Planned structure:

```text
adapters/
fixtures/
runners/
reports/
vendor/
PERMISSIONS.md
sources.yml
```
