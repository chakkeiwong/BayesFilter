# Superseded: standalone score-master execution plan

Date: 2026-09-14

All implementation and execution work belongs to the
[score master program](younis-kdm-score-master-program-2026-09-14.md).
Treating it as a separate program was a planning error. The master now includes
its own prerequisite implementation phases, their tests, study execution,
and repair-and-refresh steps. Start with Phase 0A in the master; no additional
preparatory program is required.

The former work packages have these destinations:

| Former package | Work now included in the master |
|---|---|
| E0: reconcile specifications | Phase 0A: source, mathematical specification, manuscript, and environment reconciliation. |
| E1: coordinator | Phase 0B: registry, runner, validation, failure accounting, repair, resume, and reporting services. |
| E2: real baseline study | Phase 0C: baseline implementations, exact oracles, scoped tuning, integrated tests, and real baseline execution. |
| E3: KDM/IWSG | Phase 0D: KDM/IWSG, control variates, estimator combinations, and their numerical tests. |
| E4: proposal alternatives | Phase 0E: covariance lifecycle, SGQF, twisting/iAPF, and corrected-proposal tests. |
| E5: finite differences and error attribution | Phase 0F: deterministic and stochastic FD, reconstruction, normalization, and consistency diagnostics. |
| E6: systematic comparisons | Phase 0G builds model adapters, configurations, capacity pilots, and reports; Phases 2--4C and 7 execute the applicable studies. |
| E7: replication and closeout | Phase 0G builds final-run and audit support before Phases 8--9 execute replication and terminal review. |

The initial compute allocation, implementation-default audit, proposed CLI,
phase closeout fields, and repair/resume tests are also in the master.
Dependencies apply to the methods and models each study actually uses.
Completing later implementations is not a prerequisite for starting an already
supported study. Every active method still needs its implementation and tests
before the active matrix is complete.

The [unaltered former plan](artifacts/younis-score-master-prerequisite-phases-20260914-01/execution-plan-before.md)
is retained as historical evidence. It creates no separate execution or
approval requirement. The [integration checkpoint](artifacts/younis-score-master-prerequisite-phases-20260914-01/checkpoint.md)
records the current state and validation.
