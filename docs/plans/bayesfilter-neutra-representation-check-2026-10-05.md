# Conditional representation check after controlled IAF fitting

This is P4 of the authorized source-and-fit remedy. Launch only after the full
matched optimizer study if the same canonical procedure still fails the declared
distribution screen. The existing `naf_dsf` configuration is a comparator, not
a replacement default. No new normalizing-flow implementation is introduced.

Question: does replacing affine scalar conditionals by monotone sigmoid-mixture
conditionals change the persistent exact-teacher density failure at affordable
cost? Huang et al. (2018) §3.1 Eq. (8) defines
`y=logit(sum_k w_k sigmoid(a_k*x+b_k))` with positive slopes and weights. Its
Theorem 1 concerns a sequence of expressive networks and convergence in
distribution; it does not guarantee that a finite four-component network trains
well or gives good HMC geometry. Technical source inspected locally:
`.localresources/q20-flow-training-literature-20260923/papers/huang-2018-naf.txt`
§3.1 and §4; `code/naf-flows.py:221`, `:251` and `code/naf-iaf_modules.py:135`.
The checked local mapping and inverse derivative are in
`docs/reference/neutra-implementation.md:80` and `:114`.

The existing log-domain implementation preserves the paper's full real support
instead of the clipped author's sigmoid output. This recorded adaptation stays
explicit. The source cMADE initializer and four mixture components follow the
current author-derived constructor. Width 64, two hidden layers, three stages,
batch 64 and 8,192/16,384 updates are inherited comparison hypotheses; they are
not sufficient-capacity claims. Inverse tolerances 1e-11 and 100 bisections are
existing FP64 numerical choices whose exhaustion vetoes the affected result.

First price two blocks of 512 updates on the three-mode development target at seed 11, after
the configured-map inverse, density/gradient and frozen-consumer mechanics tests.
The second block distinguishes warmed update cost from compilation. This run
is a timing and validity check only. Its 240-second ceiling is an
engineering containment cap for unfamiliar inverse/compilation cost. Require
finite density, parameter gradients and roundtrips; a failed inverse triggers
implementation/conditioning diagnosis, not a conclusion against NAF.

Forecast each full worker as twice the measured whole pricing-worker cost plus
16,384 warmed update costs, rounding upward to a whole second. This deliberately
includes two sets of setup/endpoint overhead for the two optimizer stages; it
remains an extrapolation until full data shapes execute. Reserve six such worker
ceilings and twice that many CPU-core seconds before full launch.

The complete comparison, if affordable, uses both development targets and seeds
11/37/73. Each NAF trains fresh forward KL at LR .001 for 8,192 updates, then
continues for 8,192 with reset Adam at .0003. This is the same predeclared recipe
as the IAF fresh-parent/lower-LR arm, reusing the exact saved fresh and heldout
tensors after checksum verification, with equal batch sizes and update allocations.
The observed LR preference differs by target; this is a matched recipe, not a
claim that the lower-LR arm is the strongest IAF competitor. Retain the original-
LR arm alongside it in the result table and do not declare architecture
superiority from comparison to the weaker IAF endpoint. Separate initial architectures cannot share
identical parameters. Report parameter counts and measured time; equal updates
do not equal cost. Price all six jobs and endpoint checks before launching any
scientific representation comparison. The two-block timing run is not a
successful substitute for the full recipe.

The primary stage record is independent heldout density/coverage, with all
existing numerical vetoes and the 1,000-point post-training score diagnostic.
Three fit seeds give descriptive evidence only. A candidate passing a coarse
screen may proceed to target-specific tuning; it is not promoted to posterior
correctness. Fixed-map HMC, native-teacher transfer and untouched final targets
remain separate requirements. If the full comparison exceeds remaining compute,
preserve the timing result and request one measured extension covering the next
complete comparison; do not quietly shorten it.

Pre-execution skeptical audit: IAF optimization has received a matched data and
LR check; testing a different scalar deformation can discriminate representation
without asserting a disconnected-support impossibility. NAF inverse/training
cost, finite-family limitations and initialization are unresolved hypotheses.
The comparison will remain unexecuted if its trigger or complete budget fails.

October 5 continuation: the full optimizer comparison failed its shape screen,
activating this discriminator. The timing run and 110 focused tests passed.
The owner has now added 24 GPU-process and 24 CPU-core hours, so the measured
six-fit reservation of 6,816 / 13,632 seconds is funded. Previous charges and
all paired datasets remain preserved; this is the same scientific comparison.

## Numerical-failure localization before continuation

The first full two-mode/seed-11 worker completed 8,192 forward updates and its
endpoint checks, then rejected continuation update 2,828 as nonfinite. The
optimizer retained iteration 2,827. The failed attempt is preserved and charged.
Replay only that lower-LR branch from its saved initial checkpoint with the exact
saved data and stateless seeds. Save the last accepted state, failing batch,
objective/gradient validity, and per-coordinate inverse residuals and iterations.
This is debugging evidence, not another fit comparison or map nomination.
The 240-second cap inherits the pricing cap and exceeds the observed 77.639-second
failed branch plus diagnostic setup. Reserve at most 240 GPU-process / 480
CPU-core seconds. Do not change LR, architecture, data, inverse tolerance or
scientific criteria to conceal the failure. A demonstrated numerical defect
triggers a focused regression and repaired replay before the remaining fits.

Skeptical audit: the successful 1,024-update pricing run could not establish
late-training numerical validity. The saved intermediate forward result cannot
substitute for the intended 16,384-update endpoint. Replaying the failing batch
distinguishes objective/gradient/inverse failure from resource exhaustion; the
worker exited after 284.732 seconds, well before its 1,190-second limit. An
invalid common inverse stops dependent use until localized and repaired.

The replay reproduced the exact rejected update with finite parameters. The
last stage's second coordinate, row 2, returned invalid after 25 iterations,
although its loop condition had already accepted the root. Its final residual
was -1.2116974090758958e-11 versus tolerance 1.2116901911200309e-11, a difference
of approximately 7.22e-17. Separate XLA evaluations at the stopping boundary
disagreed in final bits. Iteration exhaustion and optimizer overflow do not
explain this case. The preserved scalar data and full checkpoint/batch provide
regressions, in addition to existing inverse and parameter-derivative checks.

Repair: for f(x)=y, retain final requirements |f(x)-y|<=tau_y and bracket
width<=2*tau_x. Stop internal iterations only at half those bounds, then apply
the unchanged final checks. The factor one-half is an engineering slack choice
to avoid last-bit boundary disagreement, not a new scientific tolerance or a
global roundoff theorem. It typically adds one bisection; the 100-iteration cap
and invalid-result behavior remain. The same saved branch must pass through the
formerly rejected update on GPU/XLA before restarting the full comparison.
No target, loss, architecture, optimizer setting or data changes in this repair.
