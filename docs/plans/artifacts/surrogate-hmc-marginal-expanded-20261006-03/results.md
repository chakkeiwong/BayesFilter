# Expanded marginal-weight verification — running results

Purpose: test the likelihood/analytical-score component needed for surrogate HMC. This campaign does not yet test posterior sampling.

Updated: 2026-10-06T13:23:26.120391+00:00. Campaign: ARCHIVED_FOR_REPAIRED_SOURCE.
Measured worker time so far: 8.72 hours of the 24.0-hour ceiling.
Completed scope jobs: 16/90. A rejected control pair counts as a completed attempt, never as a successful accuracy comparison.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Engineering | Passed | Batch parity, finite-program score, T=1 reference checks | Larger scopes | Continue the same master | Model accuracy or HMC convergence |
| Precision | Passed | Nonfinite evaluations | Different datasets/scopes | Keep FP32 and inspect larger scopes | Agreement with true model likelihood |
| Fresh-data replication | 16/90 scope jobs complete | See per-scope records | Dataset sampling and reference error | Complete planned 4→8→16 looks within budget | Universal improvement |
| Nonlinear dimension extension | Pending | Model equations undefined | KSC is scalar; original range-bearing is 4D | Resolve requested new model definitions | Requested nonlinear dimension grid covered |

| Inference status | Current interpretation |
|---|---|
| Hard veto screen | Engineering and precision verdicts above; accuracy scopes below |
| Statistically supported ranking | Only the predeclared per-cell dataset intervals can support a ranking |
| Descriptive-only differences | Timing, precision deltas and heuristic point estimates; FP64 timing from attempt 01 is superseded |
| Default-readiness | Not established; ancestor remains the default |
| Next evidence needed | Fresh datasets, reference refinement, particle-count and matched-FP32 cost comparisons, then actual surrogate-HMC tests |

Independent accuracy authorities use FP64. In the active continuation simple comparators and LEDH use FP32/TF32/GPU/XLA. The equal-runtime bootstrap comparison measures likelihood accuracy; the bootstrap produces no score, while LEDH timing includes its analytical score. It is not an equivalent HMC-force workload. Recorded times also include unequal report-serialization overhead. The synchronized-call diagnostic confirms unequal serialization overhead; no computational-value ranking is established.

Automatic covariance/timing diagnostics: DIAGNOSTICS_FINISHED. The supervisor drains the active short worker and resumes master dispatch afterward. See covariance-probe-supervisor.json for the exact state.

Bounded covariance-product repair evaluation: DIAGNOSTIC_FINISHED. The unchanged master resumes automatically after this diagnostic.

Invalid comparison attribution (original evidence is preserved):

- m13-t20 N=1024 point 1: INVALID_BASELINE; counts {'ancestor': 1, 'marginal_mixture': 0, 'reference': 0, 'incomplete': 0}.

The GPU engineering fixtures used distinct parameter rows for all three models and both policies. Maximum batch/separate-call discrepancy: 0; maximum relative score/finite-difference discrepancy: 1.77863e-05. The T=1 range-bearing quadrature/PF comparison passed its recorded uncertainty check.

| N=4096 precision arm | Parameter point | Absolute log-value delta | Scaled score delta | Repeat spread |
|---|---:|---:|---:|---:|
| ancestor | 0 | 0.0049715 | 0.00165195 | 0 |
| ancestor | 1 | 0.00491015 | 0.00217172 | 0 |
| marginal_mixture | 0 | 0.00529032 | 0.000240125 | 0 |
| marginal_mixture | 1 | 0.00143087 | 0.00339039 | 0 |

Pilot result: {'status': 'COMPLETE', 'outcome': 'NO_VIABLE_CONTROLS', 'candidate_rejected': True, 'scope': 'lg20-t240', 'particles': 1024}

| Scope | Complete fresh datasets | Completed paired parameter/design rows |
|---|---:|---:|
| ksc-t10-n1024 | 16 | 1536 |
| ksc-t20-n1024 | 16 | 1536 |
| ksc-t50-n1024 | 16 | 1536 |
| lg1-t10-n1024 | 16 | 1536 |
| lg1-t20-n1024 | 16 | 1536 |
| lg1-t50-n1024 | 16 | 1536 |
| lg3-t10-n1024 | 16 | 1536 |
| lg3-t20-n1024 | 8 | 768 |
| lg3-t50-n1024 | 8 | 768 |
| m13-t10-n1024 | 16 | 1024 |
| m13-t20-n1024 | 16 | 1024 |
| m13-t50-n1024 | 1 | 121 |

| Scope | N | Datasets | Parameter point | Value verdict | Score verdict | Heuristic screen |
|---|---:|---:|---:|---|---|---|
| lg1-t10 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t10 | 1024 | 16 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t10 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg3-t10 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg3-t10 | 1024 | 16 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg3-t10 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| ksc-t10 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| ksc-t10 | 1024 | 16 | 1 | INCONCLUSIVE | INCONCLUSIVE | PROMOTION_VETO |
| ksc-t10 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| m13-t10 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| m13-t10 | 1024 | 16 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg10-t10 | 1024 | 0 | — | NO_VIABLE_CONTROLS | Not assessed | Not assessed |
| lg20-t10 | 1024 | 0 | — | NO_VIABLE_CONTROLS | Not assessed | Not assessed |
| lg1-t20 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t20 | 1024 | 16 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t20 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg3-t20 | 1024 | 8 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg3-t20 | 1024 | 8 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg3-t20 | 1024 | 8 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| ksc-t20 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| ksc-t20 | 1024 | 16 | 1 | INCONCLUSIVE | NONINFERIOR | PROMOTION_VETO |
| ksc-t20 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| m13-t20 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| m13-t20 | 1024 | 16 | 1 | INVALID_BASELINE | INVALID_BASELINE | Not assessed |
| lg10-t20 | 1024 | 0 | — | NO_VIABLE_CONTROLS | Not assessed | Not assessed |
| lg20-t20 | 1024 | 0 | — | NO_VIABLE_CONTROLS | Not assessed | Not assessed |
| lg1-t50 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t50 | 1024 | 16 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t50 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg3-t50 | 1024 | 8 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg3-t50 | 1024 | 8 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg3-t50 | 1024 | 8 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| ksc-t50 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| ksc-t50 | 1024 | 16 | 1 | INCONCLUSIVE | INCONCLUSIVE | PROMOTION_VETO |
| ksc-t50 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg10-t50 | 1024 | 0 | — | NO_VIABLE_CONTROLS | Not assessed | Not assessed |

Post-run red-team note: exact finite-program derivatives do not establish accuracy relative to the statistical model. Four datasets can leave large uncertainty; particle-reference refinement and MCSE do not bound all remaining bias. The diagonal/pairwise correction controls remain scope-specific hypotheses. Low process-noise range-bearing performance must be reported separately from the nominal parameter point. The current parameter regimes are likelihood-evaluation points on data generated at the recorded truth; they are not a factorial grid of generating parameters.

The next phase is part of the same authorized master. Consult next-phase.json and campaign.json for current execution and unfinished work. An active worker is not a completed result.
