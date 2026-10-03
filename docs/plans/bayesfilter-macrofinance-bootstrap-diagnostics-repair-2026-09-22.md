# MacroFinance bootstrap and public diagnostics repair

## Question and checked baseline

At main `2c2419c096ce744e23a37e0ab19e65840334119d`, does ordinary
preparation select epsilon using the recorded proposal probabilities, and do
both public posterior diagnostic APIs reach the repaired Stan/ArviZ ESS?
The answer from the call-chain inspection is no in both cases.

The supplied September 22 MacroFinance report gives a sixteen-proposal screen
with mean probability 0.6846376853336343 and binary acceptance 13/16. Bootstrap
reads the binary statistic and increases epsilon despite the probability being
inside [0.65, 0.75]. Its next screen fails a target finiteness assertion. The
reported run and full handoff paths under `/home/ubuntu/workspace/MacroFinance`
are unavailable in this checkout. Treat the numbers as supplied evidence;
reproduce the decision with controlled fixtures, not as a replay of that run.

The current public ordinary route reaches `hmc_preparation` and
`hmc_bootstrap.run_hmc_bootstrap_screen`; historical imports alias this same
implementation. The convergence API still calls `hmc_diagnostic_math`'s
deliberately preserved TFP precision estimator, uses `x >= q95`, and computes
percentiles independently. This is an integration omission, not grounds to
remove the separately named TFP mean/quantile precision option.

## Repair and evidence contract

1. Compute bootstrap mean probability from every recorded log acceptance ratio,
   including rejected proposals and excluding discarded burnin. Use this field
   for classification and directional repair. Preserve binary acceptance as
   an explicitly explanatory field, with no fallback to it when probability
   evidence is absent, empty, malformed or nonfinite. Check the requested
   recorded count and sample/trace shapes. Keep the existing bootstrap roles:
   fixed-kernel band versus startup floor; neither issues a final tuning artifact.
2. Preserve exceptions and first-failure attribution. A declared target-domain
   exception may nominate a smaller *fresh* bootstrap trial only when the
   first-failure record locates it in a proposal target callback, after a finite
   pre-transition state. Initial-state, retained/trace, unclassified TensorFlow,
   device, programming and classifier errors remain terminal. Reuse the existing
   exception classification vocabulary, repair multiplier and total round cap.
   Keep numerical-failure bounds separate from measured acceptance endpoints;
   never assign a fictitious zero acceptance. Evict a terminal failed reusable
   runner before a retry. Only a subsequently completed screen may be handed off.
3. Delegate public convergence bulk/tail ESS and split/folded R-hat to the
   repaired posterior implementations, with the public axis order transposed
   explicitly. Preserve schema, threshold values and split/unsplit conventions;
   record ESS method identity. The TFP precision estimator and covariance-window
   ESS heuristic remain separate, explicitly named computations.
4. Fix unrealistic bootstrap fixtures; add opposing-probability/binary cases,
   the reported numbers, missing/nonfinite/shape/count evidence, bounded unsafe
   trials, L clamping, terminal runner replacement and error preservation.
   Test the public convergence endpoint against local ArviZ 0.21 on tied,
   antithetic, constant and odd-length fixtures, plus public call-chain tests.
5. Update the official LaTeX book and agent/API reference together. Record
   commands, results, source identity and limitations in this note; refresh the
   master without modifying any frozen source or consumed consumer evidence.

Pass criteria are exact decision/invariant checks and independent numerical
parity at existing reference-test tolerances (rtol/atol 2e-11). A failed target
assertion, bad call chain, lost failure record, accepted unscreened pair, or
unexplained reference discrepancy vetoes this repair's completion. Statistical
uncertainty, sixteen-transition acceptance, missing native divergence telemetry
and diagnostic runtime are explanatory, not posterior or performance evidence.
No previous round is retroactively certified. The MacroFinance 22/180/22
adaptation schedule and final candidate/posterior execution still need a fresh,
consumer-qualified run. No R-hat/ESS/MCSE rule is added to tuning membership.

## Defaults, safety search and skeptical review

The acceptance band, multiplier 2, five repairs and L limits are inherited
bootstrap controls, retained here to isolate the wrong statistic. They are
heuristics, not stability or coverage guarantees. The smallest failure check
is a trace whose binary and probability statistics disagree. An unsafe epsilon
is a proposal bound under this fixed geometry, not a measured low-acceptance
endpoint or universal stability ceiling; every new epsilon/L pair is screened.

The existing four-momentum startup helper is optional and requires qualified
batched value/score, valid initial/retained states and the declared telemetry
policy. Runtime exceptions there currently stop execution. The fixed-L warmup
search has separate lifecycle and seed requirements. Do not transplant either
as an alternative bootstrap authority or enable it globally to hide this bug.
The bounded bootstrap retry above is limited to explicitly attributed proposal
failures; an adapter must narrowly declare its target-domain exceptions.

Skeptical audit before execution: the supplied run predates the resolved merge,
so verify current call sites rather than accepting a byte-comparison claim.
Posterior and adaptation ESS contexts differ. Changing one public report must
not silently change the intentionally selected TFP precision method. Binary
rate is not a probability estimator without additional accept/reject randomness.
A healthy fixture must represent probabilities in its actual trace. A successful
repair screen cannot establish adaptation or posterior success. Existing L
clamps must remain in records. Error text alone cannot establish a recoverable
proposal. These checks resolve the identified plan flaws; proceed with bounded
engineering tests before any consumer rerun.

## Execution limits and live campaign

This is a routine correctness repair within M24 and the existing 48 CPU/24 GPU
hour allowance. Reserve at most 1800 CPU worker-seconds for focused tests and
local reproductions (an engineering timeout, not a statistical sample size).
Use one diagnostic process at a time, GPUs deliberately hidden, TensorFlow
intra/inter-op, OMP and OpenBLAS threads set to one. The two live confirmation
workers continue from their immutable `m21-r2/source-r1`. For this bounded
repair only, permit one additional single-process diagnostic worker; this
revises the earlier two-worker overall concurrency ceiling, not the campaign
inventory or its worker count. The machine has ample CPU capacity; no concurrent
runtime result will support a performance comparison. Charge the measured test
wall time separately, then restore the two-worker ceiling. No GPU run or new
MacroFinance attempt is planned. Store results in fresh `m24-r2/` paths.

Continuation vetoes: the focused repair allocation is exhausted, test evidence
is corrupted, or the scientific target would need to change. A regression
failure triggers localized repair inside the same allocation. Keep unrelated
q20 and training changes intact.

Execution audit found an additional constant-draw edge case: an XLA reduction
can round the mean of an exactly constant array and leave a tiny positive
centered variance. Checking only positive variance then reports a finite ESS.
The repaired ESS now also checks exact constancy of the input draws, which is
the stated existing nonpromotion rule. Add non-power-of-two lengths and several
constant scales to the reference regression; this does not change the estimator
on nonconstant inputs. Some old bootstrap fixtures also supplied four or twelve
rows for a sixteen-row screen. Repair these fixtures rather than relaxing the
new evidence-count check.

## Execution result and terminal audit

The final combined run passed **413 tests**, with two disclosed skips, in
175.31597336998675 worker-seconds (pytest body: 170.08 seconds). The skips were
the optional external DZ5 reproducer, which is absent, and an existing tiny
Gaussian mass-stage smoke whose bootstrap exhausted its acceptance repairs.
Neither skip is counted as a successful full fit. Independent reference tests,
the actual traced target-domain recovery, public preparation wiring, checkpoint
tests and real operational rotated-Gaussian mass-stage tests passed.

Exact command and environment are in
`artifacts/hmc-repair-master-2026-09-16/m24-r2/tests-final-r1/result.json`.
That command runs the bootstrap, convergence, posterior-reference, preparation,
checkpoint, mass-stage, public API and interface-registry modules plus the
before/after reproduction. All tests use deliberately hidden GPUs and one
numerical thread per worker. Total charged time, including five intermediate
attempts, is **444.9127681890968 CPU seconds**; GPU time is zero. The temporary
diagnostic-worker exception has ended. The live queue still has at most two
numerical workers and its source is unchanged.

Intermediate attempts found stale mock payloads, inconsistent requested/returned
bootstrap counts and the constant-array XLA edge case. One collection attempt
also required the existing reference environment's explicit
`matplotlib.style.core` import before ArviZ. These were fixture/harness or local
implementation failures, not evidence against HMC or the MacroFinance target.
All previously failing selected tests passed in the final run. Original attempt
logs and manifests remain in their distinct directories.

The official book built successfully and changed PDF pages 411 and 487 were
visually inspected. A stale source-directory bibliography initially left seven
citations unresolved; regenerating BibTeX in the new build directory and forcing
the final build resolved them. Existing unrelated layout warnings remain. The
updated PDF is installed as `docs/main.pdf`; its source and build receipt are
preserved, while the generated PDF remains ignored by Git.

Terminal red-team: the fix demonstrably changes the supplied decision and makes
public bulk/tail diagnostics agree with the independent reference. It does not
show the old consumer run was successful, certify its third screen retroactively,
identify its first overflowing primitive, or prove the 180-transition window is
adequate. A fresh consumer-qualified attempt is still required. The
[reply memo](bayesfilter-macrofinance-bootstrap-diagnostics-repair-reply-2026-09-22.md)
records migration details, decision and inference-status tables, and the remaining
limits. No tuning/posterior thresholds, candidate inventory, metric heuristic,
TFP precision option or active campaign source was changed.
