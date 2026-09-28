# Filter and gradient repair resume checkpoint

Branch repair/filter-gradient-xla-validation-20260918 in
/tmp/bayesfilter-filter-gradient-xla-validation-20260918. Use git HEAD for the
latest committed checkpoint. Main remains unmerged.

Through 04839; active worker runs: none.
Charged/reserved CPU 111374.689551s / GPU 97813.826679s.
Remaining CPU 25.062586h / GPU 24.829493h.
Global caps56 CPU/52 GPU process-hours include the extra24 CPU hours.
Active allocation: docs/plans/filter_gradient_fitted_apf_execution_20260929.md (closed); at most
6 workers/600 CPU/600 GPU seconds.
Used/reserved 6 workers/29.727734 CPU/12.207573 GPU seconds.
One numerical worker at a time.

Fitted-APF RNG diagnosis04834–04839 is complete: raw Philox state/words agree, but built-in compiled floating draws differ. Shared explicit normal conversion is exact on CPU/GPU;04839 passes162 readback/policy checks. No numerical runtime change or stream migration. Reuse existing stateless_random_tf FP64 compatibility primitives.

Next: Implement and qualify the fixed fitted-APF enclosing owner with preserved stream labels, feature ordering, fit histories, failure behavior and frozen-fit derivatives. FP32 remains a distinct compatibility obligation; adaptive iAPF follows separately.

Nonlinear base04783–04805 qualifies healthy EKF/UKF and six refused LEDH-family
variants on CPU/GPU. The original four/four reset fails its second balance
check; severe ill-conditioning is not established. Fresh independently
calibrated scopes04806–04821 qualify all three providers, both ordinary and
diagnostic endpoints, CPU/GPU, at eight/eight reset counts. Complete parity
max2.22e-16; five-point error max7.10e-12. Original failure remains preserved.
See filter_gradient_nonlinear_directions_result_20260929.md and
filter_gradient_nonlinear_scope_result_20260929.md. This is mechanics evidence,
not a canonical tuning artifact or scientific admission.

Committed Gaussian qualification/cost evidence through04782 is in d9a534584.
The full-filter streaming CPU regression remains unaccepted: paired warm
ratios1.08239/1.09678 and upper bounds1.11687/1.15608 atT32/T128. RNG component
profile04750–04755 does not explain that slowdown. F14's identified pfor sites
are closed through04727; the broader F01–F20 dispositions remain open.

Remaining work: nonlinear/other matched owner cost acceptance; memory counter
interpretation, native/compiler residency and uncontended GPU capacity; one
bounded full-filter compiler intervention; actual public analytical-score
coverage and current-source measurement applicability; DZ5 locator121 strict
trajectory differences and unconverged optimizers;4539 fitted-geometry CPU/GPU
differences and isotropic angles; remaining terminal dispositions and eventual
integration/merge. Work order: filter_gradient_terminal_gap_queue_20260928.md.
Reuse unchanged actual-DZ504618–04628/import04629–04630/index04631 evidence.

GPU0 is remote desktop, GPU1 display. Recheck GPU2/3 availability before use;
shared numerical checks do not establish uncontended costs/capacity. GPU growth
and trusted placement are required; CPU is explicit reference. Preserve shared
analytical authorities, invalidity errors, LEDH streams and canonical NeuTra
IAF. No subagents, training/HMC, live MacroFinance edits, package/system/cache
changes, relaxed gates, canonical LEDH rebuild or main merge at this stage.
