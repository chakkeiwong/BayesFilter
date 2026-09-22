# Phase 0E relative-shape repair: result and next step

No underflow failure occurred in the fresh shape-objective run, but the repair
did not produce an admissible score-estimation candidate.
One weak-regime validation fit did not converge within its iteration budget.
The curved-regime candidate reached untouched claims, where its observed error
exceeded EKF, UKF and the local-linear proposal and its fit hit a bound. No
method is promoted. This rejects the tested settings; it does not reject iAPF
or invalidate the numerical harness.

Execution used clean commit `e6298503bf923a2aa8ee9fff53732b79eac205cb`, isolated
worktree `.localresources/worktrees/younis-score-iapf-density-repair-20260917`.
The initial repair plan and mathematical derivation are in
`docs/plans/younis-score-iapf-relative-shape-repair-2026-09-17.md`.
The original Eq. (15) comparator is unchanged. The optional objective is
profiled residual norm divided by Gaussian-density norm. Its analytical
gradient includes the denominator derivative. This is an Algorithm-3 fitting
adaptation, not an implementation of the published Eq. (15) minimizer.

The final score remains the analytical derivative of the finite filter with
fitted coefficients, realized particle count, ancestor labels and mixture
labels frozen. That is the quantity actually computed. It is compared with
the refined-grid model score but is not claimed to be an unbiased model-score
estimator. The likelihood correction is unchanged. Source Proposition 1
preserves the ideal corrected value target for positive bounded twists;
it does not establish score unbiasedness or broad scientific validity.

## Executed evidence

The 32 planned rows yielded 30 attempts: 29 accepted numerical results, one
rejected validation fit, and two weak-regime claim rows blocked because their
source study was incomplete. There were no underflow failures. Seven of eight
repair source rows completed; both curved repaired claims completed. All eight
baseline source rows, four baseline claim rows, and eight heuristic rows
completed. Failed rows were retained and did not block independent work.

The failure was validation dataset 1212, first backward sweep, first time step:
the fit remained valid and the coefficient cast passed, but after 2,000 steps
its projected gradient was `1.9728630005377834e-5`, above the unchanged `1e-7`
threshold. Its shape residual was `3.595909215438848e-8`; its bound flag was
false. A small residual cannot replace the required convergence check.
The other time step converged in 33 steps. Full data are retained in
`launch02/invalid-fits.json`, including seeds and fitted coefficients.

Maximum observed shape residual among the seven accepted repair source rows
was 0.141936; the baseline source maximum was 0.855644. One repair source row
and five baseline source rows showed boundary activity. These are descriptive
diagnostics across accepted rows; excluding the rejected row cannot establish
superiority. Both curved repaired claims shared a bound-active fit with maximum
shape residual 0.106113. All four baseline claim rows had bound activity.

Each table entry is squared error relative to the refined-grid six-parameter
model score. iAPF entries average two independent final replications on the
same dataset; stochastic heuristics have one replication. This is an observed
conditional screen, not a confidence interval or statistical ranking.

| Regime / fresh claim data | EKF | UKF | Bootstrap | Local-linear | Original density iAPF | Relative-shape iAPF |
|---|---:|---:|---:|---:|---:|---:|
| Weak / 1220 | 0.00321049 | 0.00686002 | 2.23642 | 0.0563318 | 0.609396 | blocked |
| Curved / 1221 | 0.214633 | 0.0437179 | 7.74137 | 0.291364 | 14.4813 | 1.63934 |

The repaired curved result is descriptively lower than its density comparator,
but it fails three cheap heuristic comparisons. That is the headline, not a
promotion based on the comparison between two complex methods. The weak
baseline loses descriptively to EKF, UKF and local-linear; the curved baseline
loses to all four. Realized counts were 32 for weak baseline, 16 for curved
baseline and repaired candidate. Work is recorded; this is not a matched-cost
superiority claim.

## Verification and reproducibility

The isolated source passed 32 focused CPU/XLA tests, including the prior 19-test
suite, finite differences of the new analytical gradient, exact Gaussian
recovery, target-amplitude invariance, the saved underflow witness, real adapter
wiring and rejection of a mislabeled objective. Three oracle contract tests
passed at commit time. CPU runs intentionally hid GPU devices. The main
checkout additionally passed all 13 repair/driver checks after integration.
All six changed source/test files were verified byte-for-byte against the clean
executed source before closeout; unrelated working-tree edits were preserved.

`saved-run-audit.json` passes all nine study/source fingerprints, all 29 result
digests, matching observations and reference scores, declared objective
identities, frozen fits across claim replications, and independent final random
streams. Maximum reference mesh/domain/tail discrepancies were respectively
`5.13325e-16`, `1.61933e-15`, and `5.55112e-16`, within the declared gates.
No numerical comparison was rerun by this audit.

Trusted GPU execution used NVIDIA RTX 5080, UUID
`GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a`, PCI `00000000:01:00.0`.
Memory growth, FP32/TF32 filtering, FP64 offline fitting and XLA were verified.
Another workload was active, so runtime cannot support a ranking. The full
command is in `execution-command.txt`; manifests, physical-device details,
seeds, fit/control identities and allocator measurements are in `launch02/`.
The new objective and appended optimization-loss diagnostic are scoped and
repository-emitted; historical artifacts are not silently upgraded.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep the optional shape implementation for further diagnosis | Objective/derivative/call-chain checks pass | Initial convergence failure resolved in diagnostic replay; curved boundary and heuristic vetoes remain | Solver sufficiency across fresh scopes and Gaussian-family adequacy | Design fresh scope-specific fitting calibration using the recorded solver-budget finding | Score superiority or default readiness |
| Reject tested candidate settings for promotion | Untouched score screen fails in curved regime; weak claims unavailable | Explicit heuristic and bound vetoes | Very few claim replications and only T=2 scalar regimes | Address fit geometry/bounds and downstream adequacy before powered comparison | Research-direction rejection |
| Preserve corrected harness and comparator | Fingerprints, observations, references and streams pass | No harness continuation veto observed; campaign launch limit reached | Author-code parity remains unchecked | Prepare the next bounded campaign before further launches | Canonical LEDH or HMC completion |

| Inference status | Finding |
|---|---|
| Hard veto screen | One fit fails the declared convergence threshold; curved repair and both baselines fail heuristic/bound screens |
| Statistically supported ranking | None |
| Descriptive-only differences | Table entries, shape residuals, counts and timing |
| Default readiness | No; original and repaired settings remain unsuitable for promotion |
| Next evidence needed | Valid fresh control calibration, complete untouched conditional heuristic checks, and powered replicated score comparisons |

The strongest alternative explanation for the weak fit failure is inefficient
optimization rather than an inadequate objective. For curved score error, the
small particle budget, bound restriction, Gaussian family, floor and frozen
finite-program score can each contribute. The present run does not identify
which dominates. A successful optimizer replay only addresses the first
explanation; it cannot repair the curved claim failure. The weakest evidence
is the downstream comparison: one fresh claim dataset per regime and two final
iAPF replications. Wider fresh replicated evidence would be needed to support
a ranking or overturn the tested scope's promotion rejection.

## Final solver-budget diagnostic

The separate pre-run plan is
`docs/plans/younis-score-iapf-solver-budget-diagnostic-2026-09-17.md`.
The final authorized launch replayed validation dataset 1212 with 10,000 steps
and the same `1e-7` threshold, all other controls and first offline streams
unchanged. The first likelihood matched exactly, as did all four first-iteration
random streams. The formerly failing fit converged after **4,201 steps**, with
projected gradient `9.884881624344644e-8` and shape residual
`5.873210095867432e-9`; it was finite, interior to the bounds and cast-valid.
The second offline fit also converged (82 and 538 steps), and the full finite
iAPF replay completed. Thus the 2,000-step cap was insufficient for this saved
case. This is evidence about solver sufficiency, not a newly validated setting.
Ten thousand steps is an upper cap used for diagnosis, not a promoted default.
No claim data were used and the curved promotion veto remains unchanged.

`launch03-solver-diagnostic/comparison.json` preserves both first-iteration
records and the source fingerprint. Its manifest records driver hash,
environment, seeds, device, exact Python invocation and resources. The full
shell command is also preserved in `execution-command.txt`.

| Diagnostic decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| The saved convergence failure was a solver-budget failure | 4,201 steps meet the unchanged 1e-7 threshold and full replay completes | No numerical/stream/reference veto | Whether longer budgets suffice across fresh scopes | Calibrate solver budget or optimizer geometry on fresh source data | Score improvement, validated new controls, or rescue of the curved claim |

## Campaign close and master refresh

Total campaign consumption is **249/280 charges, 3/3 GPU launches,
252.98/3,000 conservatively charged GPU wall seconds, and
1,074.95/7,200 charged CPU seconds**, including prior consumption and documented
allowances. There are 31 unused charges, but the three-launch limit is exhausted.
`campaign-accounting.json` preserves the calculation; no extra experiment was
launched beyond that limit.

The next master task is a new bounded Phase 0E fitting-control calibration,
with fresh calibration/validation/claim partitions. Use the demonstrated
iteration requirement to justify a solver-budget ladder or a separately
checked optimization repair. Also diagnose the curved Gaussian-family,
bound/floor and small-particle limitations before claiming progress on model
score error. Keep the original density fit, the plain repaired fit, and the
conditional heuristic set visible. Preserve disjoint claim data and require
uncertainty evidence before ranking. This phase does not complete fitted-moment
integration into LEDH, wider model coverage or powered replication.
