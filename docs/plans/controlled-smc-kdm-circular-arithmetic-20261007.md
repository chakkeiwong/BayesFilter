# KDM circular arithmetic localization — 7 October 2026

Continue the accepted replication campaign while the complete development pair
trains. This is an explicitly CPU-only, non-XLA reference diagnostic, limited to
600 CPU seconds and three attempts. It uses one thread and no GPU. It neither
changes the running workers nor consumes their reserved training allocation.
Charge actual time separately against the existing campaign CPU balance.

Question: do source Normal arithmetic and dimension-addition order account for
the preserved circular float32 discrepancies, and do cancellation-sensitive
gradient scales explain why a final-value-relative bound is fragile?

Baseline: immutable `kdm-circular-20261006-02/author.json` from pinned author
commit b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18, plus the current TF consumer.
Use the same eight ordinary/seam/tiny-weight/far-tail float32/float64 fixtures.
The eight diagnostic arms independently toggle (a) variance-first Normal log
density, (b) broadcasting Normal bandwidth to particle shape before computing
variance/log scale, and (c) left-associated dimension addition. These are source operations:
`torch/distributions/normal.py:87-103` in the recorded tf-gpu2 environment;
author `kernel_density_estimator.py:232-269`. The author IWSG consumer is
`src/models/kde_particle_filter/kde_particle_filter.py:738-758`. Preserve
normalized resampling weights, stopped samples, the two bandwidth conventions,
and posterior density flooring. Save hashes of all inspected source files.

The arithmetic forms are mathematically equal for positive bandwidth. Their
finite-precision programs differ. The alternative arms are diagnostic subclasses;
the live numerical implementation remains frozen during complete training.

## Evidence contract and research intent

The existing per-array comparison bounds remain exactly
`128*2**-23*(1+abs(author))` for float32 and `1e-9*(1+abs(author))` for float64.
They report the preserved screen; none is replaced or tuned here. Primary
diagnostic output is the complete per-arm, per-fixture comparison, including
worsened comparisons. A failed float32 comparison is evidence, not a run error.
Float64 disagreement, invalid density/gradient, changed fixture/source hashes,
or an incorrect analytic derivative is a diagnostic validity veto. Missing
artifacts, CPU time limit, or any attempted GPU use stop the attempt.

For stopped samples x_m, let rho_j=(w_j+epsilon)/sum_k(w_k+epsilon),
pi_mj=rho_j f_j(x_m)/q(x_m), and a_m be the loss derivative with respect to
log q(x_m). Then g_wj=[sum_m a_m pi_mj-rho_j sum_m a_m]/(w_j+epsilon).
For the raw objective a_m=h(x_m). For the connected posterior loss, with
normalized model weights nu, posterior responsibilities r and density d,
a_m=(nu_m-r_m)*d/(d+epsilon)/(B*(1+epsilon)).
The bandwidth derivative sums a_m*pi_mj times
`residual**2/b**3-1/b` (Normal) or
`-kappa**2*(cos(residual)-d log I0(kappa)/d kappa)` (von Mises).
Check these independently assembled derivatives against float64 autodiff on the
exact cast float32 inputs. Save the sum of absolute positive/negative terms and
the cancellation condition ratio at discrepant coordinates. Ratios of observed
error to epsilon times that term scale are explanatory only, not proven bounds.

Defaults/assumptions: old fixtures are localization cases, never fresh validation;
float64 is a checked derivative reference, not exact arithmetic; source operation
order is a hypothesis, not automatically a better GPU result; non-XLA CPU cannot
establish GPU/XLA equivalence. The eight arms form a factorial check of three
concrete source differences. No seed/model/ranking inference is made. The
simple analytic derivative is the relevant sanity comparator; heuristic tracking
baselines concern later filter qualification and are outside this diagnostic.

## Execution and review

Run `docs/benchmarks/diagnose_kdm_circular_arithmetic_cpu.py --output PATH` with
tftwogpu Python; CUDA_VISIBLE_DEVICES=-1 is set before TensorFlow import. Unique
output prefix: `docs/plans/artifacts/controlled-smc-kdm-replication-20260928/
kdm-circular-arithmetic-20261007-XX`. Save command, environment, Git provenance,
source hashes, data hash, complete comparisons, analytic checks, conditioning,
wall time and terminal status. No random generation is used.

Skeptical pre-execution audit: the baseline is the actual author fixture, not a
port-generated target; formulas preserve the source scalar and stopped-sample
derivative. Proxy conditioning reports cannot promote the filter or replace
tolerances. Source-order changes can worsen another endpoint and every endpoint
must be retained. The plan passes for localization only. The earlier training
checkpoint's no-overlapping-numerical-launch rule is narrowed here to exclude
this bounded single-thread CPU reference; no additional GPU job is permitted.

Next action follows evidence: if an arm helps, require fresh fixed-input and
optimizer-consumer checks plus GPU/XLA confirmation after current training.
If none resolves the discrepancy, retain the failure and investigate reduction
conditioning rather than repeating identical runs. No outcome here completes
replication, justifies a tolerance waiver, or rejects the KDM method.
