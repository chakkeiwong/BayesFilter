# iAPF public-reference paper conformance

2026-09-19. Owner requests executable tests that establish whether the public
implementation used as a comparator computes the paper's stated object.

## Question and evidence contract

Compare pinned `Sempreteamo/iAPF/iapf.R` at
`a88114395f6c11075fedc653db9480791b43391a` with Guarniero, Johansen and Lee's
paper, not merely with BayesFilter. Its original-author provenance is unverified.
Primary paper: `.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.pdf`.
Inspect definitions (5)--(6), Proposition 1, Algorithms 3--5 and Section 5.1
equations (15)--(16). Preserve the distinction between the general twisted-PF
construction, the outer iAPF procedure and the paper's example fitting scheme.

Pass criteria: independent closed-form Gaussian products and the path-density
telescoping identity agree with executed source; actual APF resampling and
non-resampling branches produce the prescribed weights and likelihood for a
deterministic supplied noise stream; recursive fitting targets use the next
twist; iteration decisions match the paper after explicit zero-/one-based
index conversion; the fitting objective and positive floor match the stated
example scheme if that stronger comparison is claimed. A full-paper reference
label requires every required check and complete declared coverage. Missing
checks, nonfinite arithmetic, unexpected source changes, or a substantive
deviation block that label. Matching adapted operations remain usable only
with an explicit restricted comparison scope.

A green regression suite means the auditor detects the known deviations and
rejects inappropriate reference admission; it must not relabel deviations as
paper-conforming. Numerical fitting accuracy is a separate quality diagnostic.
No stochastic ranking, nonlinear 1900 causal claim, default, LEDH or HMC
admission follows. Exact Gaussian calculations are the independent authority;
no method-performance or heuristic-dominance comparison is being conducted.

## Implementation and checks

Extend the existing restricted R diagnostic loader/probes and add a stdlib
Python conformance report with named requirements, source anchors, discrepancies
and a fail-closed reference check. Exercise actual public function bodies and
controller expressions; do not replace the implementation with its expected
formula. Keep dependency substitutions visible. Wire the current comparison
consumer to the conformance assessment and label its restricted object.

Tests cover positive identities, known differences, rejection of missing or
drifted evidence, consumer wiring, and selected mutations of the executed R
source that would corrupt a mathematical invariant. Do not require the private
reference checkout for unrelated repository tests; an absent R/source may skip
the integration fixture, but cannot issue a passing conformance report.

## Defaults, limits and skeptical audit

Use CPU-only independent-reference diagnostics, `CUDA_VISIBLE_DEVICES=-1`;
stdlib Python and installed base R need no framework, GPU or package changes.
Small Gaussian fixtures are chosen for exact answers, not representative
performance. Nonzero transition and unequal variances avoid trivial cancellation;
prescribed samples and ancestors isolate bookkeeping without claiming RNG
distribution parity. Include both resampling branches and an exact-twist
likelihood check. Numerical tolerances reflect FP64 arithmetic; source/procedure
classifications are exact. A positive test floor represents equation (16), not
a tuned default. Keep the pinned source unchanged.

Skeptical audit: PASS. The previous comparison's wrong-reference risk is real:
agreement with a different objective is not equation-(15) compliance, and an
always-resampled independent-state example does not exercise Algorithm 5's
retained-weight branch. These gaps are explicit targets here. Tests must not
confuse an allowed Algorithm-3 approximation with fidelity to Section 5.1.
The code's one-based loop must be normalized before classifying stopping.
No prior particle-study allocation is reopened.

Budget: up to 600 CPU wall seconds and four focused test/diagnostic attempts,
with no scientific fits beyond deterministic unit fixtures and no GPU launches.
Stop for invalid source loading or artifacts, unresolved paper ambiguity, or
budget exhaustion; repair localized harness failures within this allocation.
Save commands, logs, source/paper hashes, environment and report under fresh
directories within `docs/plans/artifacts/iapf-paper-conformance-20260919-01/`.
Run the new CPU tests plus directly affected existing checks. Record terminal
findings and update the active checkpoint and master reference classification.

Execution audit, 2026-09-20: the ideal-twist fixture has unit terminal weights.
Removing its terminal likelihood contribution therefore cannot change the
answer. The terminal-normalizer mutation must be detected by the nonideal,
fixed-noise Algorithm-5 fixture; the ideal fixture retains its separate
Proposition-2 role. The first suite run exposed this test-oracle mistake, not
a new source discrepancy. Correct it before the fourth and final validation
attempt; retain the failed test log. The T=1 probe tests the general paper's
domain, beyond the source's fixed T=100 example, and is not evidence that the
original example fails.
