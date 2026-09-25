# M16 engineering preparation: sequential validation wrapper

This bounded software work can proceed independently while M15 samples on
immutable source. No M16 research campaign starts before M15's result/refresh.
Reserve at most1200 CPU-reference seconds within M16's20,000-second allocation
for focused tests; GPUs are hidden. No new inference or tuning default changes.

Question: does the validation infrastructure implement Gandy--Scott Algorithm3
faithfully, including fresh experiments at each look? Sources inspected are the
local paper's Section3, Algorithm3, Theorem3.1, AppendixA proof and AppendixB,
and official CRAN `mcunit` commit9e3fd08e41c1d4cd19664e95d2baf646286a2b88,
`R/expect_mc.R`. M13 preserves source hashes and inspection in
`sequential-wrapper-source-preflight.json`. The R diagnostic-message typo is not
part of the mathematical algorithm and is not copied.

Use beta=alpha/k, gamma=beta**(1/k), q=d*min(raw p-values). Reject q<=beta;
stop without rejection if q>gamma+beta; otherwise beta/=gamma and continue.
Increase sample count once after the first look, by the predeclared multiplier.
Every look is a fresh independent frozen-kernel experiment with separate anchors,
momenta, insertion positions, ties and null simulations. Reusing cumulative
samples is forbidden. Keep all look records and underlying diagnostics. A failed
or nonfinite experiment is invalid execution, never statistical rejection.

Implementation scope is the diagnostic invariance engine. Require explicit
observables and a fixed p-vector dimension. Apply multiplicity once to raw
p-values; preserve a larger caller-declared family conservatively. Validate
finite p-values, dimensions and configuration, including enough Monte Carlo
resolution to reach the first rejection threshold. The alpha/k allocation is
derived; k and sample multiplier must be supplied, with no copied mcunit default.
Actual size/power settings and budgets will be resolved in the M16 phase refresh.

Acceptance evidence and posterior stopping do not acquire this wrapper's
guarantee. The theorem requires independent looks and superuniform component
p-values under the null. Unit tests can check decisions and wiring; they cannot
prove those properties of an arbitrary callback or certify sampler mixing.
Expose the theorem as conditional, not an unconditional accuracy claim.

Skeptical audit: the wrong baseline would wrap correlated running p-values;
the implementation must construct new child experiments. Double multiplicity
would alter the specified test, so consume raw values and record the single
Bonferroni operation. Updating n after every look would differ from both source
and paper, so test the complete sample-count sequence. Optional stopping can
inflate error without these assumptions; test independent discrete null examples
and the threshold boundaries, and estimate actual kernel size/power later.
The reviewed scope passes for implementation and bounded mechanics tests.
