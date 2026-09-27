# Factor anchor selection after clipping

Run04578 reproduces an early cause of the actual CDF factor-fit differences.
Six two-factor loading rows exceed the same clipping radius. Their clipped
norms are equal in real arithmetic, but recomputing norms from rounded row
components differs by one or two binary64 units. Original, graph and XLA
therefore choose different anchor rows before optimization. This is separate
from isotropic eigenspace ambiguity and from the principal-angle SVD defect.

The inspected authority is `_initial_factor_state` in
`bayesfilter/inference/factor_correlation_geometry.py` at3582b4ac (lines429--486)
and the current shared implementation. Both clip each row `u` by the scalar
`min(1,c/||u||)` and then choose a row of greatest clipped norm using argmax.
For nonzero finite rows the exact norm is `min(||u||,c)`; zero rows remain zero.
Thus selecting from that scalar quantity preserves the real-arithmetic rule
and its first-index tie convention while avoiding a round-trip norm error.
No epsilon, new rank threshold, covariance family, loading radius, random
stream or optimizer setting is introduced.

First evaluate the one-factor and two-factor anchor selections as a diagnostic
source transformation, leaving the runtime unchanged. Replace only the
argmax inputs with the existing pre-clipping row norms capped at the existing
radius. Compare graph/XLA against the exact original source, preserving full
initialization/encoding/covariance records. The fixtures are the original and
current saved CDF weighted covariance matrices, a separated no-clipping SPD
case, exact identity and nearby repeated spectra. For the clipped actual case,
the mathematically tied first anchor must be the smallest clipped-row index;
the candidate must not invent an eigenspace orientation for the isotropic case.

Primary gates: first clipped maximum is chosen consistently; row clipping and
decoded initial covariance are unchanged within binary64 forward error on the
identical eigensystem; healthy cases without ambiguous maxima preserve their
anchors and complete initialized state at the existing1e-10 bounds. Compare
the projected raw charts only after aligning their anchor definition, since
different anchor coordinates are not the same parameter vector. Archive every
original/current/proposed record. A changed covariance beyond rounding, changed
unique maximum, nonfinite accepted fixture, or unexpected source change vetoes
the candidate and triggers localization. No full optimizer or consumer
equivalence follows from an initialization test.

If these gates pass, the next repair phase may implement this exact algebraic
identity in the shared authority and qualify real fitting and CPU/GPU calls.
Do not adopt a near-tie tolerance, change the isotropic loading floor, select
an arbitrary eigenbasis, or relax the original full-record comparator under
this plan. Any such proposal requires a separate stated numerical target and
bounded comparison. Failed unchanged fits must not be repeated for luck.

Reserve at most eight300-second CPU/GPU workers and1800 combined seconds from
the existing global56CPU/52GPU-hour caps, after the active locator workers.
The extra24CPU hours are already included. Use one numerical worker at a time,
the existing supervised campaign runner, frozen input hashes and fresh numbered
artifacts. GPU work waits for an eligible non-display device. No source-admission,
live MacroFinance, training or HMC change is part of this initialization unit.

Skeptical review: choosing the largest pre-clipping norm without capping it
would change the method, because all clipped rows are exact ties; reject that
shortcut. Merely forcing the observed anchor13 would overfit the CDF fixture;
test permutations and both factor counts. Stable tie handling does not repair
a repeated eigenspace or prove that the later optimizer converges. The proposed
identity is a numerical implementation repair with an explicit mathematical
target, not an alternative chart-selection heuristic. This is local review;
no independent review or downstream admission is claimed.

## Diagnostic result and runtime repair

Run 04587 passed in 18.459 seconds: 22 fixture/factor combinations, each in
graph and XLA mode, including reversed coordinate orders. The largest decoded
covariance difference was 6.94e-17. Every clipped case chose the first clipped
row and the separated case preserved its complete state at the original
1e-10 bounds. This supports implementing the exact capped-norm identity in
the shared `_initial_factor_state`; it does not establish optimizer parity.

The runtime repair changes only the first-anchor argmax operand. Existing
clipping, rotation, second-anchor rule, covariance model, loading floor,
optimization and admission settings remain authoritative. The regression now
loads the pre-repair function from 5d398a45b and compares the installed shared
function to the independently transformed diagnostic. Continue with
`factor_clipped_anchor_trial_cpu`, `factor_clipped_anchor_regression_cpu`, and
`factor_clipped_anchor_fit_cpu` through the existing runner with 300-second
limits; GPU counterparts wait for an eligible device. All belong to the same
eight-worker/1800-second allocation, including 04587. Preserve the saved-fit
full-record differences and compare selected precision/covariance separately
before any claim of downstream equivalence. No tolerance relaxation follows.

Review: the identity fixes a real-arithmetic tie, but a changed chart can still
alter a finite optimizer trajectory. Require the existing factor-recovery,
domain, rank-deficiency and structured-policy regressions and an actual CDF
saved-input fit. Isotropic eigenspace ambiguity remains a separate unresolved
issue. Runtime tests now constitute mandatory checks; the fit localization
remains explanatory because it records rather than rejects every old mismatch.

Also run `factor_clipped_anchor_graph_fit_cpu` (300 seconds, same allocation)
on the identical saved callback inputs. Unlike failed graph run 04575, this
uses the repaired first-anchor rule; it tests whether clipping-induced chart
choice explains that loading-domain assertion. Keep the assertion intact. A
new failure must be preserved and localized, without an unchanged retry.

Runtime CPU qualification 04588 passed (25.069 seconds), followed by all
factor-geometry regressions in 04589 (52.044 seconds). Saved-input XLA fit
04591 passed (51.783 seconds) and exactly reproduces every field of the
pre-anchor-repair 04582 result, including selected precision/covariance. The
11 earlier principal-angle report differences are unaffected.

Graph fit 04592 failed in 17.448 seconds at the retained loading-domain
assertion. Inspection shows that graph-reference `_decode_covariance` raises
immediately, while the XLA path counts forbidden evaluations and rejects the
completed factor fit. In this actual case, XLA records three invalid covariance
evaluations for factor 1 / replicate 0 and returns `factor_optimizer_failed`;
the pinned original fitter 04574 also rejects that same factor fit. The anchor
repair therefore does not remove the separate graph interruption. Preserve
04592; do not retry it unchanged or turn off its validity check.

| Decision | Primary criterion | Veto status | Next action | Not concluded |
|---|---|---|---|---|
| Keep the algebraic runtime repair on the repair branch | Initialization invariants and existing CPU regressions pass | No changed selected CDF geometry in XLA replay | Qualify eligible GPU checks | Full optimizer, isotropic eigenspace or consumer equivalence |
| Preserve graph-reference failure | Same invalid factor fit remains rejected | Immediate graph domain assertion is active | Classify this diagnostic interruption separately from anchor correctness | A need to relax the loading boundary |
| Keep main unmerged | Broader master criteria remain incomplete | GPU, actual-consumer and terminal gaps remain | Renew policy and preserve evidence | Whole-program completion |

No performance ranking is supported by these correctness runs. The strongest
remaining limitation is that an equal initial covariance need not produce an
equal finite optimizer path after a chart change; GPU and complete-consumer
checks remain necessary before any broader equivalence statement.

Evidence through 04595 is archived with the locator investigation in
`artifacts/filter-gradient-repair-20260917/locator-anchor-cpu-04595-evidence.tar.gz`:
84 verified members, SHA-256
`a756686f6164ddeca63ac4e17f654a84d9d3540ad16bc7f8791c096baec293ff`.
The frozen clipping fixture is also committed as a standalone regression input.
