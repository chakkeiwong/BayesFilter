# Lorenz precision repair: checked result, 30 September 2026

The bounded repair stage is complete; Lorenz replication is not. We confirmed
a genuinely invalid unconstrained fit, tested three explicit safeguards,
rejected all three for further empirical use, and repaired a rank check that
had accepted a numerically unresolved fit. No campaign job remains running.

The [master](controlled-smc-kdm-replication-master-program-20260928.md) and
[repair plan](controlled-smc-lorenz-precision-repair-20260930.md) remain the
authoritative scope and evidence contract. All failed attempts are preserved.

## What failed and why

Independent full-rank SVD77 reproduced the first invalid precision in the third
refinement at forcing2.5, repetition0, time index12: minimum eigenvalue
−1730.16, normalized margin−0.132509. The previous policy was valid. Both scalar
and batched consumers failed on this design. This is an integrability failure
of the unconstrained quadratic update; retracing, XLA and QR alone do not
explain it. Twelve matched fits and36 diagnosis checks support that conclusion.

The paper, §6.2 p2924, equations44 and footnote4, conditions tractability on
positive-definite twisted covariance and suggests an SPD projection.
The pinned author source cSMC_Resample.m:213–215 leaves its multivariate fit
unconstrained; covariance EnsureSPD calls at56/98 are commented out.
Its active full-quadratic Figure6 lane therefore does not specify an executed
general safeguard. This is an implementation/specification gap; it does not
disprove the paper's conditional Gaussian formulas or controlled-SMC theory.
Local paper/source anchors and the inspected Higham author note are in the plan.

| Optional safeguard | Healthy replay | Failed-design result | Verdict |
|---|---|---|---|
| Project precision | Bitwise unchanged, no intervention | Initial proposal mean about2.45e8; RK4 overflows in refinement3 | Rejected for non-harm |
| Backtrack the whole correction until precision is positive | Bitwise unchanged, no intervention | Proposal mean about412; RK4 overflows in refinement4 | Rejected for non-harm |
| Project covariance and transform the whole policy consistently | Bitwise unchanged, no intervention; finite replay80 | Independent validation82 fails11/112 checks, including fit coefficients, actual weights/logZ and numerical rank | Rejected for correctness |

The first two failures are localized by saved-trajectory diagnosis79.
Covariance projection80 was a safety-screen pass only. It is superseded for
candidate eligibility by82's independent rejection; it must never be reported
as a validated repair.83 now makes its unresolved fourth fit fail closed.

The final localization84 finds the first coefficient mismatch exactly when
projection first fires: time12 in refinement3 and time13 in refinement4.
At the first event the projected quadratic matrix has entries as large as
4.26e10. Differences propagate backward;13 and14 time points respectively fail
the original coefficient tolerance. The fourth fitting cloud has independent
singular ratio8.11e-9. Near-machine-precision flooring has therefore not
established a stable complete recursion.84 identifies the first discrepancy;
it does not settle how much comes from eigenvector sensitivity, coordinate
conversion or cancellation in Gaussian integrals.

## Implemented engineering repair

The old QR route checked numerical rank through eigenvalues of the squared
design matrix. Squaring its condition number can hide a weak direction in
roundoff. The new check uses a conservative reciprocal-condition lower bound
from the triangular QR factor and its inverse. It retains the intended
sqrt(64 eps64) margin, rejects unresolved fits and never changes accepted
coefficients. Both condition diagnostics remain available in saved outputs.

GPU/XLA validation83 passed37 checks across all101 times of eight healthy fits
and the rejected case. Healthy coefficients remain unchanged; the actual
refinement endpoint rejects the unstable fourth fit. All tested graphs trace
once. GPU memory growth was verified; FP64, TF32 off and XLA remained explicit
reference-lane settings. NumPy/SciPy were used only in independent CPU diagnostics.
No new precision safeguard was made the default.

The guard fixes a false validity verdict, not the unstable candidate.
An unresolved numerical rank is not a claim of exact algebraic rank deficiency.

## Evidence and decisions

Campaign artifacts are under
[the preserved campaign root](artifacts/controlled-smc-kdm-replication-20260928).
Key evidence:77 diagnosis,78 safety comparison,79 trajectory diagnosis,
80 covariance safety,82 independent correctness,83 guard validation,
84 first-discrepancy localization.81 failed JSON serialization of a diagnostic
boolean;82 repeats it with that host-only repair. Manifests preserve commands,
Git/source snapshots, seeds, data hashes, hardware, memory policy and wall time.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Keep QR rank guard |37 focused checks pass | No healthy regression; known unstable consumer rejected | Broader scopes remain untested | Retain guard and diagnostics | No policy-quality improvement |
| Reject the three safeguard candidates | Healthy non-harm alone insufficient |78 nonfinite;82 correctness/rank failures | Stable full-covariance constraint still unresolved | Isolate the first projection with fixed inputs before proposing another policy | No rejection of controlled SMC theory |
| Hold full Figure6 execution | No validated fresh-design repaired consumer | Numerical continuation veto | Correctness and cost both open | Renew repair contract, then batch parity and timing | No completed Lorenz replication |

| Inference status | Assessment |
|---|---|
| Hard veto screen | Indefinite precision, nonfinite trajectories and failed independent correctness are observed |
| Statistically supported ranking | None; these are deterministic fixed-design diagnostic comparisons |
| Descriptive differences | Magnitudes, conditioning and runtime explain failure; they do not rank methods |
| Default readiness | No safeguard qualifies; the new fail-closed guard is an engineering validity repair |
| Next evidence needed | Independently validated constrained fit and complete forward consumer, followed by fresh-seed conditional comparisons |

## Remaining work and budget

The next repair must define its matrix representation, projection or constrained
fit objective, scale and floor before execution. Use the first failed projection
at time12 as a fixed arithmetic fixture; preserve both independently fitted and
identical-coefficient comparisons to separate fitting sensitivity from projection
and Gaussian-integral arithmetic. A conditioning/calibration curve must justify
any numerical protection parameter; likelihood variance is not a safety tuning
target. After local agreement, require a complete healthy no-fire replay and
the failed-seed consumer, followed by untouched seeds. No tolerance relaxation
or silently changed approximation class may substitute for those checks.
Batch parity and the full Figure6 cost forecast follow only after correctness.

The current attempt allowance84 is consumed. Compute is not exhausted:
116553.44 GPU seconds (32.38h) remain in the inherited grant; the current
GPU18000-second allocation has3602.13 seconds left. CPU balance is422.03 seconds,
with168.86 seconds left in the CPU500 allocation. Further attempts require a
refreshed bounded experiment specification within those balances, not another
48-hour grant or per-command approval ceremony.

Other replication gaps are unchanged: full Lorenz Figure6/Table1, source-M16
multimodal Table2, unresolved conditional filtering underperformance, and KDM
AppendixB parameters/training schedule/full empirical learning. iAPF remains
incomplete and deferred. The old85.89h neural-bearings estimate does not apply
to the parametric AppendixB problem.

Post-run self-review: the strongest alternative explanation for coefficient
disagreement is amplified floating-point differences after projection rather
than different underlying formulas. However, the actual weight discrepancies
and independently unresolved fitting rank already veto this implementation.
A stable independent arithmetic replay and complete valid fresh-design
consumer would overturn the candidate rejection. No independent reviewer was
launched; the report distinguishes source claims, checked local results and
remaining numerical uncertainty.
