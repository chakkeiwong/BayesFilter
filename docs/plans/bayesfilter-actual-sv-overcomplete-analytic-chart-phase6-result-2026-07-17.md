# Actual-SV Overcomplete Analytical Chart Phase 6 Result

Date: 2026-07-17

Status: `COMPLETE_PHASE_6_DESCRIPTIVE_SAME_TARGET_DIAGNOSTIC`

Plan:
`docs/plans/bayesfilter-actual-sv-overcomplete-analytic-chart-repair-plan-2026-07-17.md`

## Result

The selected `K=23` overcomplete candidate was compared at the frozen center
with the existing deterministic dense Actual-SV filtering reference at
`T=2,10,100`.  The reference used adjacent Legendre orders `129` and `257` on
radius `10`, matching the predecessor comparison.  The candidate values and
manual total scores came directly from the Phase 5 artifacts, so this phase did
not rerun or replace the own-scalar derivative evidence.

| `T` | Candidate minus order-257 value | Absolute score difference `(gamma, log_beta)` | Componentwise symmetric-relative score difference |
| ---: | ---: | ---: | ---: |
| 2 | `+1.3014182620452175e-8` | `(2.6158827104438842e-8, 2.8765310133849198e-9)` | `(1.187693931512109e-7, 5.435712744272272e-9)` |
| 10 | `+2.4500970141616563e-8` | `(1.604391193232857e-7, 7.053514550214857e-9)` | `(2.2686520413738664e-7, 1.1943543223104596e-8)` |
| 100 | `-6.553175069257122e-7` | `(1.235704860413911e-5, 3.52904263500875e-6)` | `(1.2128187577154448e-5, 1.192559771676947e-6)` |

There were no score sign reversals.  The adjacent order-129/order-257 value
differences were `4.71e-14`, `2.34e-13`, and `2.39e-12`; the largest adjacent
score difference was `1.07e-13`.  These observations show that the reported
candidate/reference differences are not explained by the observed adjacent
quadrature refinement.  Adjacent refinement is still not a rigorous error
bound.

Artifacts:

- `docs/benchmarks/artifacts/actual_sv_overcomplete_analytic_chart_repair_20260717/phase-06-dense-reference/attempt-01-t2-k23-dense-reference-result.json`
- `docs/benchmarks/artifacts/actual_sv_overcomplete_analytic_chart_repair_20260717/phase-06-dense-reference/attempt-02-t10-k23-dense-reference-result.json`
- `docs/benchmarks/artifacts/actual_sv_overcomplete_analytic_chart_repair_20260717/phase-06-dense-reference/attempt-03-t100-k23-dense-reference-result.json`

## Interpretation

Claimed target: descriptive agreement of the repaired finite scalar and its
total score with the independent same-model dense filtering reference.

Quantity computed: deterministic center differences between the Phase 5
candidate and two adjacent dense quadrature orders.

Verdict: the differences are directly measured and small in the reported
coordinates, but scientific equivalence is **unsupported** because no
target-specific equivalence margin was justified before the run.  This phase
therefore neither passes nor fails scientific score equivalence and does not
rank methods.  It completes the required explanatory diagnostic and permits
the independent GPU/XLA engineering phase.

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | No nonfinite value, identity mismatch, or dense-reference execution failure |
| Statistically supported ranking | None; deterministic differences do not define a ranking |
| Descriptive-only differences | Reported in the table and structured artifacts |
| Default readiness | Not established |
| Next evidence needed | Trusted full-horizon GPU/XLA certification; a separately justified accuracy margin would be needed for scientific equivalence |

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Complete Phase 6 | All three planned same-target comparisons and adjacent refinements are preserved | No Phase 6 execution/identity veto fired | No predeclared scientific equivalence margin | Run selected `T=1000,K=23` trusted GPU/XLA arm under the 8192 MiB cap | No scientific equivalence, superiority, HMC, canonical, default, or leaderboard claim |

## Post-Run Red Team

The strongest alternative explanation is that both dense quadrature orders
share truncation or discretization bias that adjacent refinement does not
reveal.  A wider-radius or independently derived reference could overturn the
descriptive magnitude assessment.  The weakest evidence is therefore the
reference's lack of a rigorous error bound, not the already checked
own-scalar derivative.
