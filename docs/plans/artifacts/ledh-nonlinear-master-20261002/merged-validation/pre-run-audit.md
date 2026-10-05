# Merged-code validation

Development commit: 1ccf9375e. Fetched main: 3673aebf1.
Question: does integrating upstream compiled model/random-input changes preserve
our shared analytical score, selected residual design and guarded nonlinear execution?
This is an engineering regression, not a filter-accuracy comparison.

Resolution: retain upstream native RK loops, fixed SIR adjacency helper and
compiled random inputs. Retain predator-prey dtype support, observation dimensions,
moment-safety forwarding and selectable fixed residual designs. The richer design
is theta-independent configuration constructed once outside XLA and captured by
value/score kernels, identically to trace diagnostics; repeated axes remain native.
The private kernel keeps its old default argument for upstream callers/tests.

Skeptical audit: checked both parents and the consumer call chain; an automatic
merge would have silently dropped the richer design. Added an executable design
assertion. Baselines are the premerge tests and preserved exact input streams,
not moment-loss proxies. No parameters or scientific thresholds are being tuned.
Stop on broken invariants, nonfinite previously valid fixtures, trace mismatch or
budget exhaustion; local harness repairs remain within the original budget.
CPU devices deliberately hidden for reference regressions; GPU smokes use XLA,
TF32 and verified memory growth. Remaining budgets before integration: about
1358 GPU seconds and 465 CPU seconds. No accuracy, HMC or default promotion.
