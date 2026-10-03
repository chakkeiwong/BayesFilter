# Frozen R reference: small-dimension completion

All 96 new repetitions pass the prespecified dataset-level accuracy, fit,
tail-validity and conditional heuristic screens. Together with the separately
preserved d40/d80 results, all five dimensions now have bounded evidence for
the same optional log-quadratic/floor8 reference. This is not equation-15
conformance or the published 1000-repeat study.

| Dimension | Mean likelihood ratio to Kalman | Bootstrap 95% interval | Minimum relative tail margin | Combined screen |
|---|---:|---|---:|---|
| 5 | 0.997983 | [0.987329, 1.008255] | 0.368077 | Pass |
| 10 | 1.000777 | [0.988857, 1.013202] | 0.348272 | Pass |
| 20 | 0.996912 | [0.976710, 1.017693] | 0.329383 | Pass |

All three intervals include one and lie wholly inside [0.9,1.1]. This is a
finite-sample accuracy screen, not proof of unbiasedness. All 57,600 QR fits
have the expected rank and finite diagnostics; all 9,600 saved-guide
Gaussian-limit tail checks pass. No conditional heuristic veto fires against
BPF, fully adapted PF or SIS on ordinary or large-innovation prefixes.
The complete comparison table, including those heuristics, is in summary.json.
No success-only observations were selected, and no numerical setting changed.

The final particle count is 2000 in d5/d10 and 1000 or 2000 in d20. This
reconstructed controller differs from the paper's reported average N=1000
at the smaller dimensions; passing the likelihood screen does not erase that
difference. The methods also use unequal computational budgets. No efficiency
ranking is supported. Poor training-relative density residuals remain visible
in the report; they were explanatory, not the predeclared downstream criterion.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Limit |
|---|---|---|---|---|---|
| Retain the optional R reference for diagnostics | Every dataset's likelihood interval passes | Fits, tails, complete records and conditional heuristics pass | Rare weights and data dependence with 32 repeats | Compare mathematically common R/TF operations | Not a universal validation or equation-15 replication |
| Keep exact-paper replication unresolved | Original numerical specification incomplete | Source identity gap persists | Original optimization/floor/controller choices | Reconcile those choices before a same-method full study | Do not substitute this successful alternative silently |

| Inference status | Finding |
|---|---|
| Hard veto screen | No observed invalidity, failed fit, tail or conditional heuristic veto |
| Statistically supported ranking | No overall method ranking established |
| Descriptive-only differences | Runtime, particle counts, variance ratios, condition numbers and finite-sample tails |
| Default readiness | No default or TensorFlow runtime promotion |
| Next evidence needed | More independent datasets/repeats for stronger accuracy claims; a matched source-defined method for paper replication |

Red-team note: a finite positive Gaussian-limit tail margin does not establish
a finite empirical variance estimate is precise, and 32 repeats can miss rare
large weights. A repeatable accuracy failure on fresh data would overturn the
present qualification. The weakest remaining evidence is limited sampling
and dataset coverage, rather than a failed mathematical identity in this run.

The three captured-source commands are preserved in attempt manifests;
post-run commands and input/output hashes are in diagnostic/report manifests.
CPU-only R4.1.2 was intentional, with GPU hidden and BLAS/OMP threads fixed to
one. No package, algorithmic default or numerical R core changed. Eleven
focused tests pass in 3.79 seconds (routine mechanics, separately recorded).
All three launches completed; no retry occurred and no worker remains active.

Filtering used 447.480744111 seconds and reporting/diagnostics
used 14.151159658, total 461.631903769
of 1333.492512594 summed worker seconds. Remaining:
871.860608825. Transfer 180 to the automatically continuing
component-parity plan; 691.860608825 stays in the unspent repair reserve.
Prior source allowances must not be spent again.
