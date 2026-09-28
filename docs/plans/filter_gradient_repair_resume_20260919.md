# Filter and gradient repair resume checkpoint

Branch repair/filter-gradient-xla-validation-20260918 in
/tmp/bayesfilter-filter-gradient-xla-validation-20260918. Use git HEAD for the
latest committed checkpoint. Main remains unmerged.

Through 04862; active worker runs: none.
Charged/reserved CPU 111537.539357s / GPU 98006.280826s.
Remaining CPU 25.017350h / GPU 24.776033h.
Global caps56 CPU/52 GPU process-hours include the extra24 CPU hours.
Active allocation: docs/plans/filter_gradient_fitted_apf_cost_20260929.md (closed); at most
6 workers/1800 CPU/0 GPU seconds.
Used/reserved 5 workers/45.472254 CPU/0.000000 GPU seconds.
One numerical worker at a time.

Fixed fitted-APF CPU cost screen04858–04862 passes all four measured arms and162 readback/policy checks. Gaussian/nonlinear warm medians8.078/8.103ms become1.298/1.424ms; cold ratios0.977/1.034. Extra sampled RSS99.105/100.504MiB triggers attribution. These are descriptive single-process results, not cost acceptance. Numerical repair is committed at4629f3bd8; adaptive iAPF and broad endpoint preparation remain open.

Next: Preserve the fixed-fit cost cohort; add its approximately100MiB enclosing-owner residency to the shared compiler/lifetime attribution work. Inspect and prepare the separate adaptive iAPF controller repair, including exact stopping, changing particle shapes, fit/cast vetoes and work accounting.

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

Remaining work: nonlinear/other matched owner cost acceptance; native/compiler residency and uncontended GPU capacity; a new bounded
mechanism for the streaming regression (the scalar-state candidate was rejected); actual public analytical-score
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
