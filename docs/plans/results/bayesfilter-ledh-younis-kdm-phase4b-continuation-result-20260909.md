# Phase 4B continuation: GPU checks and mathematical audit

Date: 2026-09-09
Branch: `kdm-total-score-continuation-20260909`
Baseline commit: `284d65fbc3d75b379ea96b3f090e902ca7a048ed`

## Result so far

The repaired raw-IWSG program passes all four prescribed GPU/XLA smoke cells.
Additional compiled negative controls show that the KDM consumer rejects
invalid bandwidth, bandwidth tangent, component label, proposal density,
stratified offset, and higher-moment trajectory. XLA ignores the graph Assert
operation, but the returned validity flags survive compilation and the host
checks them. The warning alone was insufficient evidence of a KDM guard bug.

The campaign runner is now implemented using the existing full analytical
endpoints and the specified two-state LGSSM. Six new CPU reference checks
pass, including public canonical-endpoint wiring, host validity consumption,
matrix-model total derivatives, fixed-anchor replay for both mark policies,
conditional heuristic vetoes, and power-cap enforcement. The earlier 41-test
KDM suite also passed on this checkout before these additions. CUDA was
intentionally hidden for every CPU reference check.

## GPU execution evidence

All cells use `N=8,D=2,T=2`, physical GPU 1 (RTX 4080 SUPER), TensorFlow
`2.20.0-dev0+selfbuilt` in `tftwogpu`, XLA on, TF32 off, and verified memory
growth before tensor initialization. The error below is
`abs(analytical-FD)/max(1,abs(FD))`.

| Receipt directory under the dated root | Mark policy | Dtype | FD error | Wall seconds |
|---|---|---|---:|---:|
| `attempt01_float64_responsibility` | Responsibility mean | float64 | 9.22e-11 | 25.74 |
| `attempt02_float64_selected` | Fixed label | float64 | 1.45e-10 | 24.86 |
| `attempt03_float32_responsibility` | Responsibility mean | float32 | 7.63e-5 | 27.34 |
| `attempt04_float32_selected` | Fixed label | float32 | 5.54e-5 | 28.17 |
| `attempt05_float64_responsibility_guards` | Responsibility mean, invalid-input controls | float64 | 9.22e-11 | 24.49 |
| `attempt06_float32_selected_guards` | Fixed label, invalid-input controls | float32 | 5.54e-5 | 28.34 |

Dated root:
`docs/benchmarks/artifacts/ledh_younis_kdm_phase4b_20260909/`.
Each `result.json` retains its exact command, seeds, original source hashes,
memory policy, allocator peak, route/target identity, and diagnostics. The
first four cells used smoke schema v4; the two negative-control cells used
v5. Later formatting of the smoke harness does not change those historical
hashes or overwrite their receipts.

Every cell executed 128 full-mixture pairs, a nonzero off-diagonal GenUT
correction, and both cap diagnostics. The incoming raw log weight and tangent
equal the preceding outgoing values exactly. A nonzero sum of raw weight
tangents is preserved. Peak TensorFlow allocator usage was below 0.3 MiB for
these tiny fixtures; this is not a large-N capacity measurement.

## Mathematical document and code audit

The relevant current note is
`docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex`, not the old
Section 3.6 producer worktree. Its TeX and Bib sources existed only as ignored
local files after consolidation. Narrow ignore exceptions now retain them
and the compact receipts in Git while keeping downloaded literature and raw
execution rows local.

| Finding | Verdict and action |
|---|---|
| Earlier LaTeX says the zero-bandwidth cloud tends to repeated categorical component locations | Wrong for the actual uniform, stratified route. N equal strata with masses 1/N select component j once. Corrected the note and added an executable label check. |
| Forward location limit was used to explain absence of score parity | Incomplete reasoning. With `B=h^2 Q`, fixed-anchor IWSG can have a leading `epsilon' L_Q^-1 dc/h` tangent even as locations approach their centres. Added the derivation with distinct-centre and fixed-Q assumptions. |
| LaTeX still says repaired GPU smokes are pending | Stale. Updated it from the four new raw-IWSG receipts. |
| XLA ignores the higher-moment assertion | Observed, but the Phase 4B consumer also retains `higher_moment_valid` and includes it in `valid`. Actual invalid-input GPU controls reject through that flag. Broader canonical-consumer rejection is not certified by this test. |
| Bootstrap comparator validates offsets only with graph assertions | A real status-path omission under XLA. Added offset-domain validity to its returned flag without altering accepted values. The campaign preflight exercises invalid offsets. |
| Power formula did not state its relation to the ten-percent improvement boundary | Repaired in the amendment and runner: use the paired margin `e_B^2-0.9 e_A^2` and explicitly design for a 20% true reduction. Counts above 500 remain underpowered and cannot promote. |
| Old Section 3.6 code uses a different producer absent on current main | No blind port. Current campaign calls the public canonical analytical endpoint and the current integrated/resampling factories. An executable wiring test verifies the consumer path. |

The raw mixture density, uncentered weight differential, responsibility and
covariance-mark differentials, and next-normalizer equations were compared
with the current code. Their one-at-a-time and joint numerical tests remain
the supporting evidence. These checks establish the total directional
derivative of `RESKDM-IWSG-FINITE` in the declared scope. They do not establish
that it equals the unchanged `ATOM-FINITE` derivative or the exact model score.

The complete DSGE chart/retraction derivative and a multi-parameter Phase 4B
assembly endpoint remain unimplemented. The current full-rank Gaussian
fixture cannot close either gap. The broader heuristic ladder also includes
an innovation-jitter arm and an unproved zero-mean control variate that this
bounded comparison does not supply.

The corrected LaTeX note builds to 16 pages using `pdflatex`, `bibtex`, and
two subsequent `pdflatex` passes. The final log has no unresolved citations,
references, or overfull boxes. Rendered pages 11 and 15 were inspected for the
changed small-bandwidth argument and implementation status; no overlap or
clipping was observed.

## Decision and next evidence

| Ledger | Status |
|---|---|
| Engineering | Four valid-input GPU smokes, two GPU negative-control suites, CPU campaign checks, and `matrix_preflight01` GPU/XLA checks pass. |
| Numerical | Finite-difference and replay tolerances pass in the stated tiny scopes. TF32 and large-N behavior remain unevaluated here. |
| Scientific | Phase 4B score quality is still unmeasured. The paired matrix campaign is required. |

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Continue | Implementation and matrix-runner preflight passed | None in tested KDM cells | Campaign cost and score-error dispersion | Run amended calibration/pilot/validation within 45 GPU minutes | Score improvement, DSGE/HMC validity, default readiness |

The strongest alternative explanation for a future good result is favorable
fixed numerical controls or a weak finite baseline. Controls are explicit
unpromoted hypotheses, and the exact Kalman plus bootstrap comparisons help
expose that explanation. A failure of the current candidate is not a rejection
of the KDM research direction. Campaign outcomes will be appended separately
from these implementation checks.
