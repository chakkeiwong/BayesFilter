# C2 fixed and defensive UKF preparation result

The fixed K2/K4 and defensive local K1/K2/K4 compilers now enclose the
numerical time recurrence, covariance lifecycle, compatible Philox random
inputs and exact-prefix analytical APF feedback in one retained TensorFlow
owner, with a fixed signature and XLA enabled by default. Their public
signatures, algorithms, diagnostics and error contracts are preserved. The
model cache is capped at four configurations; observations, theta and seed
are live inputs. Static family/topology dispatch is resolved when tracing.
Completed host diagnostic/manifest formatting remains outside the owner.

The plan is filter_gradient_c2_mixture_preparation_20261001.md. The original
source is Git385a348b9 and all original/current arms use the frozen D4 C2
fixture recovered in05401. Unique run manifests05420--05445 preserve exact
commands, hashes, TensorFlow2.19.1 environment, inputs/seeds, CPU reference or
trusted GPU3 placement, verified growth and elapsed time. No new RNG, score
backend, numerical bound, tuning control or scientific classification was used.

05420/05421 freeze ten complete original cases and ordered invalid-input
records per CPU/GPU before runtime edits.05422 CPU and05424 GPU pass20 checks
each: full branches, diagnostic fields, manifests, values/analytical scores,
exact ancestors/component choices, original errors, live input reuse, one
trace and enclosing XLA/no callbacks. Maximum errors are5.684341886080802e-14
CPU and7.105427357601002e-14 GPU, at unchanged1e-10 FP64 bounds. The varied
cases include T3/N16 and T4/N20, changed observations, negative and large seeds.

05423 passes163 graph-growth/policy/runner checks. Computational graph size
is stable across T3/T7/T11; only constant/Fill construction may differ. Guard
coverage is331 sources/1514 exact exceptions. The two added allowances format
completed diagnostics, without numerical feedback.05425 CPU and05426 GPU each
pass22 affected K=1/shared-prefix, strict moment/density and finite-difference
score checks. This renews numerical applicability after generalizing the owner;
K=1 resource numbers in its earlier result remain measurements of7151ab5fb.

05427--05444 are18 uncontended fresh-process complete-public cost workers:
three counterbalanced original/graph/XLA blocks for fixed K4 and defensive K4,
T3/N16/D4/seed9104. Each worker includes20 synchronized warm calls. Full
numerical records agree within the same bounds in every block.05445 terminal
readback passes provenance, exact sources, actual GPU, worker exit, numerical
comparisons and policy checks. No failed worker occurred in this unit.

Medians across independent processes (warm calls are not independent process
replicates; graph is an explicit reference exception):

| Family | Execution | Warm ms | Cold s | Host RSS MiB | TF device peak KiB |
|---|---|---:|---:|---:|---:|
| mixture K4 | original | 6276.537 | 7.726 | 3061.750 | 9364.500 |
| mixture K4 | graph | 58.743 | 5.631 | 1267.910 | 9366.500 |
| mixture K4 | xla | 19.490 | 8.598 | 1348.926 | 564.250 |
| defensive K4 | original | 6541.754 | 8.558 | 3158.242 | 9364.500 |
| defensive K4 | graph | 93.545 | 6.953 | 1319.426 | 9375.250 |
| defensive K4 | xla | 21.305 | 11.156 | 1417.156 | 572.000 |

Paired geometric warm ratios with three-block log-t intervals:

- mixture XLA/original: 0.003068 [95% interval 0.002824, 0.003333].
- mixture XLA/graph: 0.327498 [95% interval 0.294280, 0.364466].
- defensive XLA/original: 0.003267 [95% interval 0.003237, 0.003297].
- defensive XLA/graph: 0.223308 [95% interval 0.196297, 0.254035].

Both measured XLA families substantially improve warm cost and host RSS over
the original Python-controlled public preparation. The XLA versus graph cold
and host-residency increase is an explicit bounded tradeoff for lower warm
latency and live device peak. Graph is not a promoted default. The scope does
not support a universal speed, capacity or overhead ceiling.

All six XLA workers complete128 additional public reuses with one owner/trace.
Late64-call RSS growth is20480--28672 bytes; live GPU allocation is
unchanged in each worker. Parent exit observations verify cleanup after process
termination. These results do not prove native executable eviction from cache
removal or fit at arbitrary horizon, count, dimension or heterogeneity.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action / nonclaim |
|---|---|---|---|---|
| Close fixed/defensive UKF preparation unit | Full CPU/GPU numerical/API/XLA and scoped resource gates pass | None in this unit | Tiny complete-public cost scope | Preserve checkpoint; no whole-C2/repository closure |
| Retain compiled owners | Major warm/RSS improvement over original, stable reuse | XLA cold/RSS tradeoff versus graph recorded | Broader shapes and native residency unmeasured | Bounded owner/worker lifetime; no universal release claim |
| Continue other C2 construction | Confirmed Python prefix feedback remains | F06/F19 still open | Bootstrap, independent Gaussian/Hermite, transformed Student and DMIS | Repair those boundaries, then final caller/integration gates; main unmerged |

Terminal primary-agent review checks the complete public boundary, frozen
originals before edits, exact random/discrete decisions, independent score
checks and real identities. The strongest remaining alternative is that tiny
fixtures hide broader compiler or numerical problems; the result is expressly
scoped. No independent reviewer was launched under the no-subagent rule.
Adaptive iAPF/KDM remain deferred and canonical LEDH rebuilding excluded.
