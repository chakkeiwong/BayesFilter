# Actual-SV Overcomplete Analytical Chart Repair Result

Date: 2026-07-17

Status: `COMPLETE_NARROW_ACTUAL_SV_TP_CHART_AND_OWN_SCALAR_REPAIR`

Plan:
`docs/plans/bayesfilter-actual-sv-overcomplete-analytic-chart-repair-plan-2026-07-17.md`

Failure-ledger action: close `CE-07` only as the experimental Actual-SV TP
fixed-square chart/own-scalar engineering defect.  `CE-06`, `CE-11`, and
`CE-12` remain active.

## Outcome

The program achieved its scoped engineering objective.  The historical
four-anchor fixed-square chart failed two frozen `1e-5` endpoints at `T=1000`.
The repaired candidate keeps the same Actual-SV law, observations, four
features, lookahead, and FD neighborhood, but uses a fixed overcomplete
equality-constrained Pearson projection with an analytical `q x q` solve and
explicit total JVP.  The smallest globally constant capacity passing the
design ladder was `K=23`.

The selected candidate passed the untouched held-out chart audit, own-scalar
derivatives through `T=1000`, and trusted GPU/XLA on all five frozen FD points
under an `8192 MiB` logical-device cap.  Therefore the local chart and
own-scalar defect is fixed.  This result does not establish scientific score
equivalence, canonical Contract E--Chol correctness, a trajectory-relevant HMC
region, a default, or leaderboard admission.

## LaTeX Documentation Closure

The previously missing execution and certification material is now part of
`docs/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex`, in the subsection
labelled `sec:bf-ledh-actual-sv-overcomplete-certification`.  It records, in
the same notation as the analytical amendment:

- the frozen design and untouched held-out parameter sets;
- the complete `K=5,...,25` capacity disposition and the rule selecting the
  smallest global pass, `K=23`;
- the weakest design and held-out weights and their 50/100/200-decimal sign
  audits;
- a proposition and proof showing why finite-point chart audits establish
  only pointwise finite-program claims, not box, convex-hull, neighborhood, or
  HMC-trajectory validity;
- the FD definition, norm, owner-selected `0.05*sqrt(p)` FD-only tolerance,
  and the explicit correction that this deterministic threshold is not a 95%
  confidence interval or standard-deviation rule;
- center/endpoint reverse-mode failures, the failed custom reverse pullback
  and its removal, eager forward-AD checks, and the reason the explicit manual
  JVP remains the production-shaped XLA route;
- dense-reference value and score comparisons, adjacent quadrature
  refinement, and the absence of a predeclared scientific-equivalence margin;
- exact GPU/XLA memory policy, loop-graph evidence, replay and CPU/GPU
  diagnostics, allocator limitations, runtime, and exception-recording scope;
  and
- the preparation hash, artifact provenance, narrow `CE-07` closure, and all
  canonical, HMC, default, scientific-equivalence, and leaderboard nonclaims.

The empirical certificate is kept separate from the propositions defining the
finite chart.  In particular, observed passes are not used as assumptions in
the analytical projection or KKT/JVP proofs, and own-scalar derivative
agreement is not represented as dense-target score equality.

## Engineering Ledger

| Question | Evidence | Verdict |
| --- | --- | --- |
| Is the analytical primitive implemented as specified? | Specialized diagonal-`P` `q x q` solve, JVP/VJP checks, fixed-shape recursive factories, and `40` focused tests | correct for the checked finite program |
| Is the production-shaped route loop-native and XLA-default? | Source guard has no reachable Python loop; `T=1000` graph has three functional `StatelessWhile` nodes | pass |
| Does the fixed candidate cover the required local chart points? | Nine design and eight untouched held-out points at `T=1000`, with stable high-precision weakest-case signs | pass for those declared points only |
| Does the total manual score differentiate its own finite scalar? | FD relative error through `T=1000` at most `5.7988816510053705e-8`; eager forward-AD relative difference at `T=1000` `2.6364004959309425e-12` | pass |
| Does trusted GPU/XLA execute the same prepared route? | Bound preparation SHA-256, GPU-resident outputs, exact `8192 MiB` cap, no memory growth, replay identity, descriptive CPU parity | pass |

## Numerical Ledger

| Item | Result | Boundary |
| --- | --- | --- |
| Selected capacity | Smallest design-passing `K=23` | `K=24,25` have zero center Voronoi/reference mass; not alternative candidates |
| Weakest design weight | `5.395467979032687e-165` | positive at 50/100/200 decimal audit; not an interval proof |
| Weakest held-out weight | `5.396877738726166e-165` | positive at 50/100/200 decimal audit; held-out was not used for retuning |
| GPU weakest FD-point weight | `5.396496430930526e-165` | positive finite-program result at the frozen points only |
| GPU FD score check | relative error `6.037244233113808e-8` | frozen FD-only tolerance `0.05*sqrt(2)`; not a general scientific tolerance |
| CPU/GPU score diagnostic | maximum absolute difference `1.4408030324375432e-11` | descriptive; no CPU/GPU equivalence margin |
| GPU allocator telemetry | peak `1,995,776` bytes | excludes CUDA context, libraries, and driver allocations |

## Scientific Ledger

Claimed target: repair the known Actual-SV experimental TP local chart while
preserving the declared finite four-feature program and its total derivative.

Quantity actually computed: a deterministic `K=23` overcomplete finite TP
scalar and explicit total JVP on the frozen center/design/held-out/FD point
sets, plus descriptive dense-filter comparisons at `T=2,10,100`.

Relation: the engineering target is satisfied.  Scientific equality to the
dense filtering target is **unsupported**, because no target-specific
equivalence margin was predeclared.  The largest reported Phase 6
componentwise symmetric-relative score difference was
`1.2128187577154448e-5` at `T=100`; there were no sign reversals and adjacent
order-129/order-257 refinement was near floating-point roundoff.  Those facts
are descriptive evidence, not a proof or promotion gate.

What remains unproved or unevaluated:

- behavior between or outside the finite frozen local points;
- a trajectory-relevant HMC region and value/score continuity there;
- canonical Contract E--Chol/Sinkhorn correctness;
- Actual-SV leaderboard or default eligibility;
- superiority, posterior correctness, or generalization to other models.

## Evidence Artifacts

- Mathematical amendment:
  `docs/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex`, subsection
  `sec:bf-ledh-actual-sv-overcomplete-certification`
- Built document:
  `docs/main.pdf` (422 pages after the documentation amendment)
- Frozen design:
  `docs/benchmarks/artifacts/actual_sv_overcomplete_analytic_chart_repair_20260717/phase-01-specification/design_specification.json`
- Selected full preparation:
  `docs/benchmarks/artifacts/actual_sv_overcomplete_analytic_chart_repair_20260717/phase-03-capacity/attempt-05-t1000-k23-preparation.json`
  (`e7958ba79da7f584b5e761faa22a9ed4fdc53cd90027b0c13389388e130c6a8f`)
- Design and held-out audits:
  `docs/benchmarks/artifacts/actual_sv_overcomplete_analytic_chart_repair_20260717/phase-03-capacity/attempt-06-t1000-k23-design-chart-xla-result.json` and
  `docs/benchmarks/artifacts/actual_sv_overcomplete_analytic_chart_repair_20260717/phase-04-held-out/attempt-01-t1000-k23-held-out-chart-xla-result.json`
- Full-horizon derivative:
  `docs/benchmarks/artifacts/actual_sv_overcomplete_analytic_chart_repair_20260717/phase-05-derivative/attempt-05-t1000-k23-score-fd-xla-result.json`
- Dense-reference results:
  `docs/benchmarks/artifacts/actual_sv_overcomplete_analytic_chart_repair_20260717/phase-06-dense-reference/`
- Trusted GPU/XLA:
  `docs/benchmarks/artifacts/actual_sv_overcomplete_analytic_chart_repair_20260717/phase-07-gpu/attempt-01-t1000-k23-trusted-gpu-xla-result.json`
- Phase close records:
  `docs/plans/bayesfilter-actual-sv-overcomplete-analytic-chart-phase0-result-2026-07-17.md`
  through
  `docs/plans/bayesfilter-actual-sv-overcomplete-analytic-chart-phase7-result-2026-07-17.md`.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` |
| Worktree | dirty; preserved, no unrelated changes reverted |
| Environment | conda `tf-gpu`, TensorFlow `2.19.1`, Python `3.11.14` |
| CPU reference mode | GPU deliberately hidden, float64, two TensorFlow intra-op threads, one inter-op thread |
| GPU mode | RTX 4080 SUPER, one logical GPU, `8192 MiB`, no memory growth, XLA on, float64, TF32 setting recorded enabled but float64 arithmetic used |
| Data | deterministic seed `81101`; target/preparation hashes in structured artifacts |
| Random seeds | no runtime randomness; deterministic quadrature and frozen data |
| GPU wall time | `69.50322806398617 s` for charged attempt 1 |
| Attempt budget | one of two GPU attempts used; CPU and GPU budgets not exhausted |
| Exact commands | stored in each JSON artifact's execution record |
| Output root | `docs/benchmarks/artifacts/actual_sv_overcomplete_analytic_chart_repair_20260717/` |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | No selected-candidate chart, derivative, graph, GPU, OOM, identity, or budget veto fired |
| Viable candidates | `K=23` remains viable for this experimental finite route and declared local points |
| Statistically supported ranking | None; no stochastic candidate ranking was performed |
| Descriptive-only differences | Phase 6 dense-reference and Phase 7 CPU/GPU magnitudes |
| Default readiness | Not established |
| Next evidence needed | Resolve the separate canonical wiring ledger, then predeclare a trajectory-relevant accuracy/continuity region for any HMC or admission study |

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close narrow `CE-07` defect | Selected smallest design-pass candidate also passes held-out, own-scalar, endpoint, full-horizon, and GPU/XLA gates | clear | Scientific equivalence and nonzero-region validity are not established | Preserve this experimental route and continue only under separate canonical/HMC plans | No canonical, HMC, default, leaderboard, posterior, or superiority claim |

## Post-Run Red Team

The strongest alternative explanation is that the selected sparse point sets
miss an interior chart failure or score discontinuity relevant to HMC.  Such a
failure would overturn any future region-readiness claim but would not undo the
finite-point engineering result recorded here.  The dense reference may also
share unmeasured truncation bias across both quadrature orders.  The weakest
part of the scientific evidence is therefore the absence of a predeclared
equivalence margin and rigorous reference error bound.  The weakest numerical
margin is the extremely small positive tail weight; its high-precision audits
support the saved cases but are not interval certificates.

Terminal verdict: `CORRECT_FOR_THE_CHECKED_EXPERIMENTAL_FINITE_PROGRAM` and
`UNSUPPORTED_FOR_SCIENTIFIC_EQUIVALENCE_OR_HMC_READINESS`.

## Terminal Review

The local terminal audit checked the result against the plan decision cases,
phase artifacts, active failure ledger, and explicit scientific nonclaims.  It
found no material inconsistency, unsupported threshold, or scope drift.

Claude health probing returned `CLAUDE_PROBE_OK`, but the subsequent bounded
one-file read-only review was rejected at the external workspace-data-export
boundary before Claude read the file.  No workaround or broader prompt was
attempted.  Under the repository's advisory review policy, this procedural
unavailability does not override the completed local scientific and artifact
checks.

Review verdict: `AGREE_LOCAL_INDEPENDENT_AUDIT`; external Claude substantive
review: `NOT_PERFORMED_EXPORT_BOUNDARY`.

## Final Verification

- `40` affected tests passed across the diagonal KKT, Actual-SV
  overcomplete, primitive, derivative, and scalar-SV loop modules.
- The three benchmark scripts pass `py_compile`.
- All Phase 6 and Phase 7 JSON artifacts parse successfully.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeds from
  `docs/`, and every label introduced by the Actual-SV certification subsection
  resolves in the resulting 422-page PDF.
- The repository-wide LaTeX log still contains pre-existing duplicate-label
  warnings and 11 undefined citations outside this amendment; they are not
  represented as resolved by this program.
- The selected preparation SHA-256 remains
  `e7958ba79da7f584b5e761faa22a9ed4fdc53cd90027b0c13389388e130c6a8f`.
- `git diff --check` passes for the changed implementation, mathematical
  chapter, benchmark, plan, result, and ledger paths.
