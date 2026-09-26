# Filter and gradient repair resume checkpoint

Checkpoint through 04207, September 26. Pushed commit `5c9aa438e` contains the
bounded GenUT reverse-compilation repair and preceding remote integration.
Remote main `5eb6dcff5` is contained through `07996dd81`. No worker is active.
The Gram/triangular precision candidate remains uninstalled. Main promotion
and F01--F20 terminal dispositions remain open.

The two runtime GenUT loops now declare fixed reverse-storage bounds, retaining
zero actual iterations for zero configured steps. Complete current-runtime
primal and reset/callback renewals 04194--04199 pass on CPU/GPU. Policy 04200
passes 129 checks; coverage remains 270 sources/1,424 exact allowances.
No new NumPy/numerical-loop allowance, default TF32 change, tolerance waiver,
canonical LEDH rebuild or MacroFinance edit is installed.

TF32 causes two localized defects: diagonal Gram primal rounding in graph mode
and the triangular-solve factor pullback in XLA. Individual attribution 04188
selects only these two changes. Other trial substitutions are unnecessary for
the tested defect. The uninstalled candidate passes all six GPU XLA extent
cells (dimensions 1/3/18, zero/four iterations), including changed inputs,
against independent FP64 smooth values/derivatives. FP64 CPU/GPU pass all
12 graph/XLA cells (04191/04193). Primitive GPU renewal 04207 passes eight cases.

FP32 CPU 04190 still fails dimension-18/four-step weight gradients in graph
and XLA, also failing in the original implementation. FP32 GPU 04192 fails
graph dimension-18 changed zero-step and four-step weight gradients. At the
four-step GPU graph fixture, several original forward fields are also wrong
against independent FP64; the candidate values pass. All original comparisons
are retained. Small gradient magnitude alone does not establish ill-conditioning.
The candidate cannot yet support general precision or cost qualification.

Cross-mode complete-record gates 04201/04202 still fail on the thresholded cap
report. 04203--04206 reproduce its prior operand/lowering cause. No reporting
semantics changed. Highest-dot pullbacks pass earlier primitive/full-program
checks, but remain diagnostic and inherit unresolved full-program precision.

See `filter_gradient_genut_reverse_precision_result_20260926.md` and immutable
`genut-precision-extent-verification-04207.json`. Archives contain every raw run file,
including rejected trials, and available source variants. Next execute
`filter_gradient_genut_weight_precision_followup_20260926.md`: split weight-use
branches with exact-forward/FP64-total checks to localize cancellation and
rounding. No new runner group or test for that unit has been written yet.
The prepared `filter_gradient_genut_bounded_gradient_cost_20260926.md` cohort
is deferred until precision repair/disposition; no new candidate memory or
speed claim is established.

Charges through 04207: 86432.436514 CPU / 78463.779462 GPU seconds,
leaving 31.990990 CPU / 30.204506 GPU hours under unchanged
56/52-hour caps. The user's added 24 CPU hours are already included. Broader
staged/public API, initializer, full reset, DZ5, native-retention and terminal
source-freeze work remains. Canonical NeuTra stays
`bayesfilter_neutra_iaf_author_v1`; canonical LEDH rebuilding remains excluded.
