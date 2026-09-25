# M18 independent performance and source preparation

This engineering work may proceed while M15 runs unchanged frozen source.
M18's terminal numerical comparison and book build still follow the phase
refreshes. Reserve 1800 CPU seconds within its 6000-second ceiling for source
inspection, a bounded host profile/parity study and focused tests. No new GPU
experiment is part of this preparation.

M7's actual profiles recorded 992 checkpoint calls, 237 seconds in checkpoint
writing and 201 cumulative seconds in generic hashing in a beta fit. These
overlapping profiled times do not predict end-to-end speedup. M7 already
optimized built-in scalar dispatch. The current writer still repeatedly
normalizes JSON-decoded execution specifications, evidence and partial chunks
through the generic arbitrary-Python-payload normalizer before encoding them.
Inspect measured saved records before changing anything.

Question: can checkpoint hashing skip redundant normalization of already
JSON-native records while preserving exact bytes, checksums, corruption checks,
durable resume and numerical outputs? Compare the current generic hash with
ordinary sorted compact JSON plus SHA-256 on every saved evidence/specification
from the three M15 CPU pilots. All three payload families are stored after
`_json_copy`; the fast helper must be restricted to that explicit boundary.
Do not change general candidate identity hashing, source closure, seeds, work
ordering, checkpoint cadence or scientific decisions. Continue rehashing each
live record at every existing check; this is not a mutation-blind cache.

Pass criterion: exact hash agreement on the saved inventory and new native
fixtures, preserved corruption rejection, and a measured reduction in the
same host workload. Alternate implementations for five repeats; report timing
descriptively without a statistical sampler-speed ranking. The later M18 GPU
comparison uses the same source, seeds and inputs, changing only which hash
function the diagnostic harness invokes. Compare all numerical tensors and
candidate decisions with timing excluded. A hash mismatch, lost check or
numerical difference vetoes the optimization. If the CPU study has no useful
benefit, retain the baseline rather than refactor for appearance.

Skeptical review: an arbitrary Python mapping is the wrong baseline for this
specialization; integer keys and custom objects have different generic
normalization semantics. Restrict the new function to previously JSON-normalized
records and test that precondition. Updating source identity changes numerical
streams, so the later GPU comparison must hold source closure fixed and switch
the diagnostic hash hook only. M7 profile totals overlap and are historical;
new unprofiled same-payload timing is required. This scoped study passes the
review. It establishes neither broad runtime speedup nor default readiness.

Separately locate full technical sources for the three unresolved bibliography
keys (Afshar2015, Gorinova2020, Pakman2014), using local ResearchAssistant when
available. The old shared-drive paths are absent. Preserve retrieved sources
under `.localresources/`, inspect relevant method/theory sections, and correct
the nearby claims if they overstate those sources. Bibliography metadata alone
cannot establish that reflection/refraction repairs an ordinary continuous kink.
No sampler implementation is inferred from those papers in this phase.

## Broader regression review, September 21

The first broad selection was interrupted after 147 passes, four failures and
19 setup errors. Inspection found three obsolete callback-based fixed-transport
tests calling the public API, which correctly rejects that retired route. Their
unchanged historical telemetry/cap assertions now call the explicitly historical
scheduler, as the neighboring historical scheduler tests already do. No public
API is weakened. One test requires an absent external c603 checkout; 19 errors
require an absent July09 private migration certificate. These are unavailable
historical fixtures, not current numerical failures. Preserve their results
and omit those two files from the new bounded batch with explicit reasons.
The automatic-search extended test was interrupted while its subprocess ran;
it supplied no passing evidence. Serious current public searches have their
phase-specific matrix/SBC budgets; exclude extended tests from the routine
batch. A fresh batch still includes all self-contained HMC/fixed-transport
regressions and the full inference-validation test directory. Review remaining
failures before making a final regression claim.

The second batch passed 591 cases (including the three repaired historical
step-cap tests), then found eight failures/errors in July private replay and
identity-adoption tests. The missing files and drifted historical bytes are
preserved; do not fabricate or regenerate those retired ceremony records to
satisfy old tests. The current AGENTS policy explicitly supersedes that
governance. The next batch resumes after `test_hmc_identity.py`, retains
self-contained identity checks, and records individual missing replay tests
and superseded authority/migration files as excluded coverage. This is an
active HMC regression audit, not a claim that the repository's historical
campaign test suite passes. All failed/partial batches stay in accounting.

The metered continuation reproduced four fixed-mass grid/cache failures and
one boundary-checkpoint test failure. A bounded diagnostic found no hard veto:
the first four use exactly .65, whose trace conversion `exp(log(.65))` is
.6499999999999999 on this environment. These tests are about grid-edge repair
and runner reuse, not acceptance endpoints. Change their scripted viable arm
to .67, inside the unchanged pass band; retain their grid/cache assertions.
The checkpoint test patches the facade although the implementation now lives
in `hmc_mass_adaptation`. Patch the actual owner of both collaborators and
retain the failure/boundary assertions. Skeptical review: neither correction
may relax runtime validity or public candidate membership, and these old
single-pair fixtures cannot establish modern candidate-set behavior. Focused
reruns must pass before these failures are classified as repaired tests.

The continuation also exposes obsolete single-winner assumptions in the
Gaussian posterior-oracle tests. Migrate the ordinary and frozen-transport
public tests to candidate sets, identity-based selection, durable member
reload and fresh discarded/retained archives. Use the supported affine codec
for that public transport test; preserve the independent dense affine
value/Jacobian oracle separately. The two deliberately historical efficiency
selection tests retain their original assertions under explicitly historical
names and scheduler calls. Their old single-winner protocol must not be
reintroduced into the public API. Fixed moment tolerances in these tiny CPU
tests are inherited mechanics checks, not MCSE calibration or convergence
claims. Inspecting the unlaunched GPU parity harness also found its missing
required acceptance-policy argument; supply the unchanged public policy before
execution. These corrections leave numerical package source unchanged.

September 21 regression allocation refresh: the broad self-contained selection
requires more than the original 2400-second test allowance, including the
preserved interrupted batch. Reallocate 2000 CPU worker seconds from unused
campaign reserves to M18, increasing its phase ceiling to 8000 CPU seconds;
the total authorization and 2000-second GPU ceiling are unchanged. This funds
a bounded 1800-second continuation, final documentation checks and cleanup.
The prior total ledger leaves over 86,000 CPU seconds, so this does not expand
authorized compute. The skeptical review retains the same test scope and
failure criteria; missing private historical fixtures remain explicit exclusions.

The next batch reproduced two further historical-fixture assumptions: the
uncertainty admission must preserve the source candidate's survival flag, not
require it to be false; a tamper test must invert that flag rather than assign
its existing value. Preserve exact payload round-trip and veto tests. The lazy
import subprocess inherited pytest's opt-in custom-op preload, which expressly
imports TensorFlow. Test normal lazy import with that opt-in disabled, matching
the package's documented environment. The public-oracle fixture also needs
matching preparation/execution telemetry policies and centered acceptance bands.
These are test configuration repairs. No runtime criterion is changed.

The isolated lazy-import rerun remained a real implementation failure even
after disabling custom-op preload. Direct import of the uncertainty module
does not load TensorFlow, but the public export scanner first imports unrelated
numerical modules. Add a direct export for the affected admission function,
using the existing lazy-export mechanism. This changes import routing only,
not its callable or numerical behavior. The focused subprocess test must pass.
Preserve the prepared M18 source-r1 and create source-r2 from the new current
package before parity; include both identities and the exact one-line routing
difference in the terminal audit. All M17 numerical source remains immutable.

Final reconciliation found two reporting issues, not numerical mismatches.
The first test inventory preferred any saved pass over a later failed report.
The long r5 batch had loaded the old oracle fixtures before their focused
repairs, but wrote its XML afterwards. Preserve that inventory as superseded;
the replacement must use the latest complete named outcome, with a failure
overriding an earlier pass. Rerun the affected oracle module after all prior
batches have ended, recording source hashes and the result. This bounded check
resolves the ambiguous chronology without rerunning unrelated passing tests.
The final inventory must retain both documented skips and all explicit historical
exclusions. No anonymous progress dots count as evidence.

Rendered inspection also found the long constraints chapter title overflowing
its running header and colliding with the page number. Add a short running
header while preserving the full title and table-of-contents entry. Build a
fresh guide-r2, preserve guide-r1 and the original PDF, and inspect the changed
tuning evidence pages, coverage table, and repaired header before installation.
These localized repairs leave the numerical source, criteria, and phase budget
unchanged. The skeptical review accepts the precise latest-outcome rule and
layout repair; a remaining failure must be resolved before terminal completion.

Both public oracle cases passed the terminal focused rerun. The preceding
whole-module retry exceeded its 120-second allowance; its anonymous dots are
excluded and 125 seconds are charged conservatively (timeout plus cleanup).
Final current-source evidence now reconciles to 2090 named passes and two
documented skips under the corrected rule.

The rendered table exposed a reader ambiguity: its original external
eight-schools suite still says unavailable, while the separate M17 pinned
posteriordb comparison ran successfully. Clarify that these are distinct
designs and that the generated table includes suite reports, not all separately
executed diagnostics. This agrees with the existing API reference and
validation README. Preserve guide-r2 and use a final guide-r3 for this prose
repair; no numerical evidence or coverage classification changes.
