# MCSE scale and interval-shape diagnostic

18 October 2026. Status: all 768 records completed; the independent terminal
audit passed at 07:30 UTC. The single scientific attempt launched at
06:32:23 UTC (14:32 China time); do not relaunch this completed study.
This is a bounded CPU-reference phase of the modern-HMC program. It leaves
the live SIR campaign, tuning membership, posterior thresholds and all runtime
defaults unchanged. Terminal interpretation is recorded in
`bayesfilter-mcse-dependence-interval-result-2026-10-18.md`.

Terminal interpretation audit: the comparator, conditions and 231-screen
family remain frozen. Separate fixed endpoints from conditional stopped
endpoints, and keep nonstoppers and unavailable estimates in the denominator
tables. Known-variance and Gaussian Student controls are fixed-endpoint
references only. Passing their screens cannot prove calibration, and this
study has no estimator-selection or ranking criterion. The descriptive
reader checks the independently audited result identity before summarizing;
the full distributions remain in the original result. Settlement must charge
the submission overhead once and reconcile the separately charged saved-data
replay; the supervisor's historical balance predates that concurrent work.

Continuation audit, 18 October 05:47 UTC: the checkout, numerical overlay,
allocation and live SIR processes were rechecked. The comparator remains the
actual public retained-posterior assessment; exact-variance controls are never
stopping inputs, and neither coverage screens nor a smoke can promote a policy.
The reader now also checks saved sample counts, with a corruption fixture.
Focused tests must run against the frozen source without skipped integration
checks. Preflight, the single scientific attempt and closeout are all charged
within the existing 7,200-second reservation. No SIR source or cap is changed.

Execution checkpoint, 06:35 UTC: 89 frozen-source tests and five bounded
supervisor fault tests passed without skips. The fresh seed-75116 smoke
completed all six conditions; its independent audit passed for six records,
84 source dependencies and all 231 screen definitions. The auditor regenerates
streams, means, batch components, controls and first-pass histories; it does
not claim independent full ESS/R-hat recomputation. Preflight charged
`53.66638870770112` seconds. The tested driver, auditor and tests are frozen in
`source-01/`; `frozen-sources-01.json` also binds the supervisor.

Service `bayesfilter-mcse-interval-study-20261018-01.service` is active and has
committed records across all six conditions under `attempt-01/`. Exact launch
arguments and the `0.041790587361902`-second submission receipt are preserved
in `launch-submission-01.json`; reconcile that exclusive submission overhead
once at terminal settlement. The supervisor's scientific cooperative and
watchdog caps are 6,600/6,660 seconds, with at most 420 seconds for the
independent terminal audit subject to the remaining inclusive allocation.
No second scientific attempt is authorized by this study. Initial progress is
engineering evidence only; coverage results await terminal audit.

## Research question and evidence contract

The stationary-stress study found fixed-endpoint undercoverage and unavailable
precision estimates under strong dependence, while the antithetic diagnosis
showed that the local positive-pair ESS arithmetic matches installed TFP 0.25
for the inspected cases. This study asks which part of the remaining error is
visible in two separate controls: (1) does a batch variance estimate have the
right scale for the known finite-length variance of the sample mean, and (2)
does a normal interval fail even when supplied that exact variance? It also
tests a twice-square-root batch-size sensitivity on fresh paths. This is a
diagnostic, not selection of an estimator or posterior stopping rule.

The comparator is the actual public retained-posterior assessment in the
immutable `e68cdf1c4` numerical source with the already-reviewed precision
overlay at
`N5/generic-adaptation/priced-comparison-01/precision-repair-01/source`.
All methods see the same four independent stationary chains, latent paths,
law transforms and looks. Compare the four existing policies—square-root
lugsail, its local maximum safeguard, square-root batch means and TFP positive-
pair autocorrelation—with two explicitly experimental policies: raw lugsail
and the local maximum safeguard using
`b(n) = min(floor(2 sqrt(n)), floor(n/20))`. The latter cap guarantees at
least twenty complete large batches. The factor two and cap are diagnostic
hypotheses, not literature defaults or a proposed production policy. The
usual public R-hat, ESS and `MCSE/SD <= .02` rules remain frozen and are shared
by every arm.

The six arm IDs are frozen as follows. An arm ID is not an extra public
estimator name; the last two arms explicitly pass a different batch-size input
to the existing public estimator.

| Arm ID | Public estimator | Batch-size input at look `n` |
| --- | --- | --- |
| `lugsail` | `lugsail` | `None` |
| `lugsail_max_batch_means` | `lugsail_max_batch_means` | `None` |
| `batch_means` | `batch_means` | `None` |
| `autocorrelation` | `autocorrelation` | `None` |
| `lugsail_twice_sqrt_cap20` | `lugsail` | `min(isqrt(4*n), n//20)` |
| `lugsail_max_batch_means_twice_sqrt_cap20` | `lugsail_max_batch_means` | `min(isqrt(4*n), n//20)` |

The experimental pair shares exactly the same schedule, so changing batch
size is distinguishable from adding the maximum safeguard. `isqrt(4*n)` is
`floor(2 sqrt(n))`; `2*isqrt(n)` is a different schedule. Preserve the actual
large/small batch sizes, counts, discarded remainders and per-chain variance
components emitted by the public assessment for every evaluated look.

The study has no promotion criterion: no MCSE or interval policy will be
selected or become a default from this result. Engineering validity requires
complete source-bound records, exact input/seed identity, reproduced analytic
moments, preserved nonstoppers and unavailable intervals, and a terminal
audit. For each fixed endpoint, report all 6 conditions × 6 methods × 3
quantities; for stopped endpoints report the same cells conditional on a
valid interval, retaining nonstoppers and unavailable results separately.
Use exact binomial intervals with Bonferroni alpha `.05/231` for one declared
family: 216 method screens (6 conditions × 6 arms × 3 quantities × 2
endpoints), 12 fixed-endpoint exact-variance-normal controls (6 conditions ×
coordinate/square) and 3 Gaussian-coordinate Student controls (one per rho).
Controls are stored once per common path, not copied for each method. This is
a falsification screen, not proof of
nominal or anytime coverage.

At the fixed endpoint, compare the estimated-variance interval against (a) a
normal interval using the exact finite-length variance for the coordinate and
square quantities, and (b) a four-chain Student interval for the Gaussian
coordinate control. Under independent stationary Gaussian AR(1) chains, the
four chain means are independent normal observations, so the latter is an
exact finite-sample t interval with three degrees of freedom. The exact-
variance normal control isolates the normal approximation from variance-
estimation error; it does not provide an operational interval because its
variance is known only in this synthetic reference. The t interval is an exact
control only for the Gaussian coordinate, not a general solution for skewed
or nonlinear posterior quantities.
Preserve its four chain means, standard error and endpoints. Constant chain
means yield an unavailable Student control, not an invented zero-width
interval. The plain batch-means arm, exact-variance normal control and
chain-mean Student control form the cheap constructed comparator set, assessed
separately under independence, persistence, antithetic dependence and skewness.

For `X=Z`, use `gamma_X(h)=rho^h`; for `X=Z^2`, use
`gamma_X(h)=2 rho^(2h)`. For `X=exp(Z)^k`, `k=1,2`, use
`gamma_X(h)=exp(k^2) expm1(k^2 rho^h)`. The fixed-length variance for `C`
independent chains and `n` draws per chain is
`[gamma(0)+2 sum_{h=1}^{n-1}(1-h/n) gamma(h)]/(C n)`.
Check these formulas against direct finite covariance matrices at small `n`,
constructed independently using Gaussian moments or the multivariate-normal MGF,
before using them as study references. The event quantity retains the actual
public interval assessment; no exact event-variance result is claimed here.

Hard continuation vetoes are source/input mismatch, corrupt checkpoint or
record hashes, invalid streams, changed method definitions, missing required
diagnostics, unexpected GPU visibility, or a failed independent moment
oracle. These trigger a localized repair under the same total allocation or
stop with the failure preserved. Noncoverage, nonstopping, negative or
unavailable variance estimates, and a poor candidate policy are research
findings, not continuation vetoes. The strongest alternative explanation is
high replication-to-replication variation in skewed-square estimates; the
report must show the full distribution and uncertainty, not just its mean or
extreme quantiles. No ranking or HMC/burn-in claim follows.

## Intent ledger and assumptions

| Item | Frozen choice and role |
| --- | --- |
| Main question | Separate finite-batch variance-scale error from normal-interval-shape error in known stationary references. |
| Mechanism tested | Fresh fixed and actual sequential assessment under the four existing policies plus the twice-square-root diagnostic pair. |
| Expected failure | Strong persistence can bias square-root batch estimates; antithetic paths can make raw lugsail unavailable; skewed square quantities can defeat normal intervals even with correct variance. |
| Promotion criterion | None. Results nominate the next discriminating study only. |
| Promotion veto | Any policy result alone is insufficient for default promotion; direct failure of the engineering/source contract blocks interpretation. |
| Continuation veto | Source, data, stream or artifact integrity failure; missing required outputs; invalid framework/device provenance; or the hard allocation. |
| Repair trigger | Localized reader, serialization, environment or harness defect that leaves the scientific contract and total cap unchanged. |
| Explanatory diagnostics | Exact-variance ratios, estimator availability, fixed/stopped misses, stopping frequency, batch metadata and empirical error shape. |
| Must not conclude | Burn-in sufficiency, HMC invariance, posterior correctness, general MCSE calibration, superiority, or an acceptable runtime/default policy. |

Material assumptions and early checks:

| Choice | Provenance/status | Failure mode | Earliest check |
| --- | --- | --- | --- |
| Four independent stationary Gaussian AR(1) chains; `rho=0,.99,-.99` | Reused stress design and exact covariance model | Confusing stationary initialization with burn-in evidence or independent replication units | Seed-layout and small covariance-trace tests |
| Gaussian and lognormal margins on common latent paths | Reused diagnostic design; exact moments exist | Shared latent paths could be miscounted as independent replications; lognormal-square tails may dominate 128 records | One record per latent replication and report paired law labels |
| 128 replications per each of six conditions | Matched to the completed stress study; bounded and underpowered for small coverage defects | Wide uncertainty can hide modest undercoverage | Exact intervals and explicit no-certification conclusion |
| Existing four methods, plus `2 sqrt(n)` raw/max variants | Existing methods are baselines; factor two is an unproven sensitivity hypothesis | New batches could worsen variance or become unavailable at early looks | Assert `b >= 1`, `n//b >= 20`, exact metadata; compare to analytic variance |
| Existing sequential thresholds and ten looks | Inherited public policy, held fixed to expose interval changes | Repeated looks are not anytime-valid confidence intervals | Report fixed and stopped coverage separately and state no anytime claim |
| CPU, two framework/OMP threads, GPUs hidden, XLA on | Prior stress environment and repository CPU-reference exception | CPU load can slow the concurrent GPU campaign; package or source mismatch | Check process load before launch, verify actual imports and preserve SIR caps |
| 7,200-second total allocation from the existing released SGQF balance | Convenience ceiling; scaled from the measured 2,422.045-second four-method study: 1.5× for six methods plus compilation/closeout headroom | A cap can leave an incomplete denominator | Preflight one-record smoke, monitor progress, stop and report all incomplete rows at cap |
| Fresh seed namespace `75117`; smoke namespace `75116` | Convenience identifiers, chosen disjoint from prior 74116/74117 study | Accidental reuse or seed reassignment | Test exact `[base, correlation_group, replication]` identities and store all seeds |

## Execution and resource limits

Use fresh root
`docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/mcse-dependence-interval-20261018-01/`.
Do not overwrite any stress or antithetic artifact. The single attempt covers
at most 128 replications in each of six conditions, 6 methods, four chains,
and looks 1,000 through 10,000. The total supervisor-inclusive CPU cap is
7,200 seconds from the existing released SGQF balance of `134851.5684849512`
seconds. The planned linear estimate is `2422.0453545683995 * 6/4 =
3633.068...` seconds; the remainder covers new XLA shape compilations,
preflight, focused tests and terminal audit. This is a spending cap, not an
ETA. It does not change or charge the active SIR reservation. Do not launch a
second scientific attempt if the cap is reached; preserve incomplete
denominators and reassess from measured costs.

Run with `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`,
`CUDA_VISIBLE_DEVICES=-1`, `TF_FORCE_GPU_ALLOW_GROWTH=true`,
`TF_NUM_INTEROP_THREADS=2`, `TF_NUM_INTRAOP_THREADS=2`, `OMP_NUM_THREADS=2`,
and XLA enabled. This phase explicitly allocates 7,200 CPU-worker seconds from
the previously released balance under the owner's continuation authorization;
the unallocated balance becomes `127651.5684849512` seconds while this
reservation is active. Verify the preceding receipt hashes and arithmetic and
write `allocation-01.json` before metered preflight. Do not include the separate
held 3,600-second reserve or historical 1,579.592954-second balance. The
immutable numerical source and imported numerical dependencies must be captured
in the manifest. The manifest must include checkout commit,
source overlay identity/hashes, exact command, environment and package
versions, CPU/GPU status, seeds, plan/result paths, wall time, method IDs,
replication/condition denominators and whether every output passed integrity
checks. NumPy/SciPy are permitted only in the explicitly diagnostic synthetic
reference generator and independent analytical audit.

Planned implementation and commands:

1. Add `scripts/diagnose_hmc_precision_intervals.py` as an explicitly
   diagnostic-only driver, with one ordinary checksummed JSON file per complete
   condition/replication, a small atomic index, and an explicit interrupted
   record. Do not repeatedly rewrite a large cumulative record file or archive
   raw paths. Use a 6,600-second cooperative numeric cap and a 6,660-second
   watchdog inside the 7,200-second total; reduce both if measured preflight
   leaves less headroom. It must verify that all
   imported numerical modules resolve beneath the immutable source root,
   freeze source hashes before sampling, and recheck them at closeout.
2. Add focused tests for the exact finite-covariance formula versus explicit
   covariance matrices; Gaussian-coordinate t intervals; `2 sqrt(n)` batch
   eligibility and public assessment metadata; tied/constant paths staying
   unavailable; same-path matching across methods; strict seed/source
   identity; tampered records; first-pass histories; and incomplete/nonstopping
   denominators. Tests may use tiny CPU/XLA fixtures, but they confer no scientific
   status.
3. Run the focused suite with GPUs hidden. Run one-replication smoke under a
   fresh `smoke-01/` directory; inspect every condition, look, policy, output
   and source identity before the serious run.
4. Run at most 128 complete replications per condition, one attempt, in the
   fresh result directory with `--seed 75117 --max-wall-seconds 6600`. Write a
   manifest before the first synthetic draw and atomically checkpoint every
   complete replication. Preserve incomplete replications as explicit
   in-progress/unfinished records rather than treating them as missing-free.
5. Independently regenerate the synthetic streams and audit record checksums,
   prefix fingerprints, means, batch variance components, exact controls,
   first-passage histories, fixed/stopped denominators, all 231 exact-binomial
   screens, MCSE/interval roles, source hashes
   and inclusive cost. Write a result note with decision and inference-status
   tables and a post-run red-team discussion. Charge cost once and release the
   unused reservation to the same SGQF balance.

## Skeptical pre-execution audit

**Verdict: proceed after the focused tests and one-record smoke pass.**

The initial draft omitted controls from multiplicity, left the experimental
raw-lugsail arm ambiguous, and had not made an explicit released-funds
allocation. This revision fixes all three before execution. Its independent
reader regenerates paths and batch components; it does not claim a second full
TensorFlow ESS/R-hat replay. Prior antithetic arithmetic parity remains separate
evidence. No fixed-length variance is interpreted as a stopped-conditional
variance. Shared Gaussian/lognormal paths are paired and never pooled as
independent replications. Negative/unavailable precision and nonstopping are
findings, not reasons to change the frozen arms or thresholds.

The baseline is the actual public posterior assessment from the immutable
source, not a hand-coded approximation; the only changed numerical input is
an explicitly named diagnostic batch schedule. The same paths, chain count,
targets, looks, thresholds and public assessment are used in each arm, so the
comparison is paired. The exact covariance is used only as a fixed-length
diagnostic oracle, never as the stopping estimator. Coverage under sequential
stopping is reported separately and cannot be upgraded to anytime validity.
No mean variance ratio, MCSE threshold, pass rate, or short smoke is treated
as promotion evidence. The allocation and stop condition are explicit. The
reference starts stationary by construction; this design cannot answer
burn-in, HMC, multimodality or posterior validity questions. The numerical
snapshot is frozen and device policy explicitly hides GPUs, so this cannot
compete for the live SIR GPU or be mistaken for GPU-performance evidence.
The artifact paths and independent checks answer both the variance-scale and
interval-shape questions. A partial cap outcome is preserved as incomplete,
not interpreted as a pass.
