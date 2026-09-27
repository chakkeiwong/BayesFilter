# Filter and gradient repair resume checkpoint

Checkpoint through 04410, September 27. The complete prepared posterior
initializer passes nine distinct checks on both CPU and GPU 2: full pinned public
payloads, physical callback order/counts, changed starts/scales, replay, one trace,
stable HLO, frozen derivatives and real compiler failure without eager fallback.
The construction-suppression failure 04390 and diagnostic NaN-formatting failure
04395 are repaired and preserved. Policy 04410 passes 141 checks. The evidence
archive has 106 verified SHA-256 members; see the enclosing result and
posterior-enclosing-04410-verification.json. Public dispatch is still unchanged.

Next install the reviewed public-integration drafts only after this checkpoint
is committed. Drafts in /tmp cover the public wrapper, completed reporting,
bounded owner, row-count and factor-configuration error precedence, explicit
frozen-module reference loading and exported-API tests. The plan also requires
an identifiable larger two-factor fixture: D1/D3 with factor_max=2 does not test
an identifiable two-factor family. Public comparisons and original API checks
precede matched original/graph/XLA costs. No numerical worker is active.

Remote main 06590cb5a is merged into the repair branch as 035e19fdd without
conflicts. Commit and push this completed internal-controller checkpoint;
main promotion remains blocked. The enclosing reservation remains 10,800 combined
seconds inside global 56 CPU / 52 GPU-hour caps. The extra 24 CPU hours are already
included. Charges through 04410: 89855.983997 CPU / 82380.530717 GPU seconds,
leaving 31.040004 CPU / 29.116519 GPU hours.

All F01--F20 terminal findings remain open. Remaining work includes full public
initializer integration/costs, actual DZ5 consumers, GenUT reporting, native
capacity and terminal repeated-cost/source review. Canonical NeuTra remains
bayesfilter_neutra_iaf_author_v1; canonical LEDH rebuilding is excluded. No
tolerance or ill-conditioning waiver is installed.

