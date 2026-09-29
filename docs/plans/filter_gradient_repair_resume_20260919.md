# Filter and gradient repair resume checkpoint

Branch repair/filter-gradient-xla-validation-20260918 in
/tmp/bayesfilter-filter-gradient-xla-validation-20260918. Use git HEAD for the
latest committed checkpoint. Main remains unmerged.

Through 05059; active worker runs: none.
Charged/reserved CPU 114810.056762s / GPU 100216.229576s.
Remaining CPU 24.108318h / GPU 24.162158h.
Global caps56 CPU/52 GPU process-hours include the extra24 CPU hours.
Active allocation: docs/plans/filter_gradient_ssl_lstm_replay_execution_20260929.md (complete_qualified_with_resource_followup); at most
120 workers/12000 CPU/12000 GPU seconds.
Used/reserved 119 workers/384.029219 CPU/666.028694 GPU seconds.
One numerical worker at a time.

Fixed replay qualification passes17 CPU/17 GPU tests04981--04982; all72 renewed costs04983--05054 and independent readbacks05055--05056 pass. Lifetime05057--05058 passes2000-call reuse and20-specialization bounds, but native RSS is retained after cache clear. Final164 readback/policy checks05059 pass. GPU T8 already-XLA warm ratio1.235 remains open and motivates one conditional-split intervention; no algorithm/seed change or gate relaxation. Source guard317 sources/1465 exact exceptions.

Next: Preserve and push the completed replay unit, then register a bounded conditional-split intervention to investigate the matched-XLA cost trigger. Keep terminal current-caller/resource/integration gates open.

The September29 scope correction in
[the terminal queue](filter_gradient_terminal_gap_queue_20260928.md) controls
current work and supersedes broader historical next-action lists. This campaign
repairs numerical Python loops, runtime NumPy and incomplete/default-off XLA
execution in active filtering/analytical-gradient call paths, verifies affected
values, scores, decisions and error semantics, and compares before/after memory
and performance. Fixed fitted-APF remains in scope; adaptive iAPF and KDM remain
owner-deferred. No numerical gate, policy allowance or raw evidence was changed
by the scope correction.

Further convergence/trajectory research on the unselected factor fits and the
historical DZ5 locator is separate from rewrite closure. Both backends select
the same accepted geometry in the saved factor cohort; tested objective
gradients pass independent references through04930. Rank-cut refusal qualifies
through04935. The4534 factor-record and121 historical locator differences remain
failed diagnostic comparisons, not accepted equality. Reopen a rewrite blocker
when evidence connects one to a changed in-scope caller's usable result,
selection, score or failure status. Broader isotropic initialization/parameter
redesign is not required merely to explain historical diagnostics.

Reduced GenUT FP32 reverse-gradient precision and cap-report failures remain
unresolved and unsupported; GPU XLA has source/reset gradient failures in the
saved cohort too. Live wiring04936–04939 separates that authority from the
registered analytical score. Terminal review must establish affected active
consumers and preserve explicit unsupported-use/admission blocks. A reachable
XLA regression still requires repair or explicit refusal; optional/noncanonical
status alone is not an exemption. General precision research and the excluded
canonical LEDH rebuild are not blanket rewrite requirements.

Remaining work: current-call-path and F01–F20 dispositions, affected numerical
regression/error checks, applicable matched memory/runtime and bounded capacity
acceptance, then remote integration and affected tests before main merge.
Streaming CPU ratios1.08239/1.09678 have95% upper bounds1.11687/1.15608 above1.10;
that finding remains unaccepted. Include fixed fitted-APF, input preparation,
remaining-SVD and angle/subspace-guard cost triggers. Compiler residency needs
measurement and a practical lifetime/capacity disposition, not a general proof
that TensorFlow releases all native allocations. No zero-overhead claim follows.

Reuse unchanged qualified evidence: Gaussian through04782, nonlinear through
04821, input preparation through04913, factor/angle/subspace through04935,
GenUT wiring through04939 and actual-DZ504618–04628/import04629–04630/index04631.
Keep each source/data/dtype/device scope explicit. F14's identified pfor sites
are closed through04727;19 broad findings await terminal dispositions, which is
not a count of19 known unfixed bugs. The317-source/1465-exception guard is scoped
coverage, not a repository-wide compliance claim.

GPU0 is remote desktop, GPU1 display. Recheck GPU2/3 availability before use;
shared numerical checks do not establish uncontended costs/capacity. GPU growth
and trusted placement are required; CPU is explicit reference. Preserve shared
analytical authorities, invalidity errors, LEDH streams and canonical NeuTra
IAF. No subagents, training/HMC, live MacroFinance edits, package/system/cache
changes, relaxed gates, canonical LEDH rebuild or main merge at this stage.
