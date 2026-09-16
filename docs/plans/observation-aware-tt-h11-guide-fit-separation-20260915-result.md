# A07 result: separate SGQF guide validity from TT fitting

Date: 2026-09-15. Status: **COMPLETE; representation screen vetoed**.

This result closes the reviewed A07 amendment
([protocol](observation-aware-tt-master-amendment-07-guide-fit-separation-20260915.md),
reviewed `VERDICT: AGREE` at
`artifacts/observation-tt-h11-separation-20260915-01/review-proposal-01.txt`).
The run was authorized by the H11 continuation ledger and used a fresh,
versioned output root:
`../benchmarks/artifacts/observation_tt_h11_separation_20260915/attempt-01/`.

## Question and evidence contract

The question was whether the signed SGQF guide remains valid over the whole
horizon and, conditional on a valid guide, where the TT representation loses
the exact correlated SGQF joint. The exact analytic SGQF joint was the target
reference. The comparator ladder was the generic degree-3/rank-3 TT start, the
SGQF-converted warm start, and the validation-selected warm or generic fit.
Validation selected the candidate; the untouched audit panel supplied the
predeclared candidate veto. Guide failures, fit failures, non-finite records,
and audit losses were not averaged away.

The run used six fresh sequences for each of d=1 and d=4, horizon T=20,
degree 3, rank 3, four sweeps, 128 proximal steps, L1 grid
`{0, 1e-5, 1e-3}`, and 1,024/4,096/8,192 train/validation/audit rows. The
manifest records the seed formulas, A06 observations were not reused, and
selection did not inspect audit rows.

## Execution record

The GPU preflight completed in 12.321 seconds. The admitted attempt completed
in 369.122 seconds (12 sequences, 228 time targets) on the RTX 5080 with
TensorFlow `2.20.0-dev0+selfbuilt`, float64, TF32 enabled, XLA enabled, and
verified memory growth. The exact command, commit, device policy, allocator
bytes, source hashes and output paths are in
`../benchmarks/artifacts/observation_tt_h11_separation_20260915/attempt-01/run_manifest.json`;
the complete log is in `../benchmarks/artifacts/observation_tt_h11_separation_20260915/attempt-01.log`.

| Dimension | Fresh sequences | Guide failures | Fit failures | Audit losses of selected TT vs exact SGQF joint |
| --- | ---: | ---: | ---: | ---: |
| d=1 | 6 | 0 | 0 | 1 |
| d=4 | 6 | 0 | 0 | 5 |
| **Total** | **12** | **0** | **0** | **6** |

All six guides completed all signed levels 2–5, and every attempted TT target
had finite rows, weights, amplitudes, fit diagnostics and Hellinger scores.
The six losses occurred at d=1 sequence 4, t=18, and d=4 sequences 1–5 at
t=1, 3, 6, 17 and 13 respectively. In every loss the validation-selected
candidate was the SGQF-converted warm TT; the exact analytic SGQF joint was
available as the reference. The selected family counts were descriptive only:
the d=4 sequences selected the warm arm for 17–19 of 19 times, while d=1
selected a mixture of exact reference, warm and generic arms.

## Decision and inference status

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| SGQF guide robustness | Passed in this six-sequence screen | No guide failure | Small fresh sample; this is not universal robustness | Retain as a valid guide for a separately reviewed diagnostic | Universal SGQF validity or exact filtering |
| Current TT representation | **Failed candidate screen** | 6/228 selected-TT audit losses vs exact joint | Finite audit panels and frozen rank/degree | Repair fitting/selection/capacity only under a new reviewed amendment | That the TT research direction is invalid |
| Downstream filtering/default | Not admitted | Representation veto active | No valid basis for ranking filters | Do not launch filtering; write the next reviewed protocol | Statistical superiority, production, HMC or default readiness |

| Inference status | Result |
| --- | --- |
| Hard veto screen | **FAILED** by six selected-TT audit losses; guide and fit validity screens passed |
| Statistically supported ranking | **NOT TESTED**; no uncertainty analysis supports a method ranking |
| Descriptive-only differences | Recorded in `result.json`, including selected family and per-time Hellinger values |
| Default readiness | **NOT ESTABLISHED** |
| Next evidence needed | A reviewed fitting/selection or capacity repair, followed by an untouched audit and only then a conditional filtering phase |

## Interpretation

The guide result and the fitting result are separable. The current SGQF guide
did not fail in this fresh A07 sample, so the earlier non-SPD event remains a
real observed failure mode rather than a universal outcome. Conditional on a
complete guide, the TT route still failed to preserve the exact SGQF joint in
six audit cases. The losses are evidence against promoting the current fitted
TT candidate under the frozen degree/rank/optimizer contract. They do not show
that the SGQF joint is worse: the exact SGQF joint is the analytic reference,
and a finite warm TT is the object that lost information relative to it.

The SGQF initialization was therefore useful as a diagnostic and was actually
tested, but it is not a certified lower bound. The result supports the narrow
statement that the warm start can remain worse than the exact SGQF target after
conversion, fitting, or selection. The empirical Hellinger values are
descriptive; this run does not establish a population ranking.

The heuristic-dominance verdict is `PROMOTION_VETO` in the machine-readable
result. That veto applies to the candidate representation. It does not reject
the research direction, and it does not authorize an ad hoc repair or a
downstream filtering launch.

## Post-run red-team note

The strongest alternative explanation is that the six audit losses reflect
finite-panel noise or the frozen rank/degree and L1 selection scope rather than
a general conversion failure. A larger, independently reviewed capacity and
selection study with untouched audit data would distinguish those explanations.
The weakest evidence is any ordering among the finite Hellinger values; no
ranking is claimed. The result would be overturned for the current veto only
by a reviewed repeat in which the selected candidate has no audit loss under
the same target and a predeclared, adequately powered uncertainty analysis.

The machine-readable aggregate is
`../benchmarks/artifacts/observation_tt_h11_separation_20260915/attempt-01/result.json`.
