# Selected-consumer arithmetic check: completion of phase 3

Question: does each freshly selected finite filter retain the owner's 1%
arithmetic accuracy on the saved consumer data at its four initial positions?
The completed baseline precision check used earlier controls, while fresh tuning
tests two positions on independent tuning data. Neither answers this question.

Freeze each current selection, source, data, fixed particle design, chart,
controls and beta. Evaluate q=-.15,-.05,.05,.15 in every coordinate, matching the
existing four-chain mechanics and consumer checks. Compare the actual force
binding with a reconstruction from the selected B2 filter, and compare that
filter's likelihood and every score with the same finite program in FP64.
The FP64 reference receives the exact candidate physical parameters, observations,
design and mixture weights, cast to FP64. This isolates arithmetic from changes
in chart rounding or input data. Range bearing uses its checked FP64 model factory.
Retain the actual chart Jacobian when constructing the reference force.

Primary arithmetic screen: finite complete trajectories, all rows valid, and
abs(candidate-reference)/max(1,abs(reference)) <= .01 for likelihood, every
physical score and every q-coordinate force. Report each row and component.
Check actual binding versus reconstructed candidate force with a separate
1e-6 scaled wiring tolerance and report the finer error. Match the consumer's
multiply-before-cast order. A wiring unit test must reject a deliberately
different binding. No omitted chart/prior terms are permitted.

Numerical failure is a repair trigger and promotion veto, not a reason to cancel
independent model checks. Source/selection mismatch, invalid inputs, corrupted
artifacts and exhausted budget stop the affected execution. These are diagnostic
checks, not tuning data: do not select controls using them or overwrite a failure.
Any subsequent tuning uses fresh partitions and preserves this holdout result.

FP64 is a same-program arithmetic reference, not a model-likelihood or posterior
oracle. This check cannot establish canonical LEDH admission, posterior accuracy,
convergence, a precision default change or a statistical speed ranking. Initial
positions do not cover the posterior. Timing is operational information only.
No new heuristic ranking is attempted; the finite-program identity is the exact
engineering comparator, with model-reference comparisons remaining separate.

One serial worker per model, maximum 3600 seconds each, under the existing 86400s
ledger and 120-attempt limit. Run after paired-consumer checks; no overlapping
campaign workers. Preserve unique output directories, source snapshots, inputs,
selection identity, per-row diagnostics, GPU memory policy and wall time.

Skeptical audit: this closes a real changed-controls/changed-data gap. It does not
substitute a narrower proxy for the primary score comparison. Physical inputs
are held exactly equal, and B2 grouping matches the actual force consumer. FP64
compilation and SIR arithmetic cost justify the cap; a timeout remains unresolved,
not a numerical failure. Runtime scientific sources remain frozen during tuning.
The implementation and bounded tests precede execution. This is phase 3 of the
authorized computation repair, not a change to the inference target or direction.
