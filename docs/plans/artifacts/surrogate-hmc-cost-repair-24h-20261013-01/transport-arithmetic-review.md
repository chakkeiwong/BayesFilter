# SIR transport arithmetic repair review

Question: can algebraically equivalent transport tangent arithmetic close the
FP32 versus same-program FP64 gap without changing the mathematical target?
Diagnostic source variants are not claim-bearing runtime implementations.

The saved, identical-input observations 2–6 yield these maximum scaled tangent
errors, using abs(error)/max(1,abs(FP64)):

| Observation | Baseline | Gauge | Direct | Both |
|---|---:|---:|---:|---:|
| 2 | .0032925 | .0048604 | .0018277 | .0041189 |
| 3 | .0062138 | .0112860 | .0059108 | .0106833 |
| 4 | .0144301 | .0178633 | .0149489 | .0184203 |
| 5 | .0403497 | .0504296 | .0322625 | .0515282 |
| 6 | .0331630 | .0282112 | .0333538 | .0286842 |

No candidate achieves the predeclared fivefold reduction on observations 4–6.
For observation 5, three warmed repeats take .124–.126s for baseline,
.125–.127s for gauge, 2.067–2.106s for direct, and 2.234–2.253s for both.
These timings are descriptive. FP64 tiny-XLA equivalence and master wiring
checks pass; the focused CPU suite has 150 passing tests. No runtime transport
rewrite is adopted. Continue fresh scope tuning, with existing mixed recursion
only for SIR and uniform FP32 for the other four models.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Reject these candidates as the SIR residual repair | Fivefold reduction fails | No observed local numerical-validity veto | Local-to-full propagation | Fresh SIR mixed-precision tuning and consumer checks | Rejection of the research direction |

| Inference status | Finding |
|---|---|
| Hard veto screen | Local finite/equivalence checks pass; full SIR uniform-FP32 score fails 1% |
| Statistically supported ranking | None |
| Descriptive-only differences | Three warmed timing repetitions and local error differences |
| Default-readiness | No transport rewrite promoted; FP32 remains default with explicit SIR exception |
| Next evidence needed | Current-source selected precision, consumer parity/cost, and actual posterior inference |

Red-team note: componentwise scaling emphasizes small reference components;
smaller relative L2 error does not erase the declared 1% criterion. A
current-source full-horizon uniform-FP32 pass would remove the need for this
mixed-precision workaround. Local diagnostic equivalence does not establish
fused full-horizon propagation or posterior accuracy.

Evidence: `sir-transport-arithmetic-01/sir_d18/transport-arithmetic.json`,
`sir-reset-arithmetic-01/sir_d18/reset-arithmetic.json`,
`fixed-design-full-precision-01/precision.json`, and
`transport-arithmetic-cpu-tests.log`. The corresponding manifests and budget
ledger preserve exact commands, source identities, device settings and wall time.
