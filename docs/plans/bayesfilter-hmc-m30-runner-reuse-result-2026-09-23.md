# M30 compiled runner reuse: result and audit

The optional compiled-runner reuse path passed exact GPU/XLA replay and two
complete public-pipeline comparisons. It preserves all verified candidates and
keeps R-hat, ESS and MCSE in posterior assessment. The option is
`execution_config.reuse_leapfrog_graphs=True`; its default remains false.

| Target | Candidates / verified | Numerical records | Saved tensor records | Static seconds | Reuse seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian | 94 / 24 | 157 | 84 | 627.135 | 346.588 |
| Beta-binomial | 100 / 16 | 158 | 94 | 575.165 | 285.510 |

Every pair has identical candidates, streams, observations, verification
receipts, complete states and traces, and posterior decisions. Both selected
posteriors passed: Gaussian at 30000 warmup / 4000 retained transitions per
chain; beta-binomial at 2000 / 5000. Fixed-count comparators also completed.
All other verified members remain unassessed by design. One pair per target
does not establish a stochastic speed ranking or posterior calibration.

## Mechanism and comparison validity

Existing execution cached four scalar-chain runners by (L, chunk count), with
state, seed and epsilon already dynamic. The new option also makes L a scalar
tensor argument and caches by chunk count inside one frozen binding. Its target,
geometry, shape, dtype, topology, trace policy and backend cannot cross the
cache boundary. Independent chains keep the original stateless seed folding;
no chain batching, thinning, reduced grid or weaker acceptance screen was used.
Invalid runtime L values fail before tracing. The conditional position-field
route rejects this exact-score-specific option.

The execution config and full binding record the option, and checkpoint/resume
preserves it. It is excluded from numerical seed policy for same-source paired
comparisons. Source checksums and paths still bind the scope. The audit caught
the original plan's error: a changed source snapshot changes fresh-fit random
streams, even with identical design seeds. Accordingly, M29's saved chunks
were used for exact cross-source transition replay, while complete static and
dynamic fits used the same new snapshot. Source validation was never bypassed.

The twelve archived M29 cells cover both targets at L=3/25 and counts=8/136/256.
Static and dynamic workers each executed three identical calls per cell.
All samples and every trace tensor, including acceptance, proposal health and
endpoint momenta, matched the archived checksums. An independent read-only
audit reconstructed those fingerprints directly from the saved bytes. Dynamic
execution used three runner groups instead of six, each scalar graph tracing
once. CPU diagnostics checked same-backend repeatability only.

Dynamic warmed calls were slower on these fixtures. Across the six cells,
the two warmed repeats totaled 3.426 versus 2.323 Gaussian seconds and 3.007
versus 1.881 beta-binomial seconds. The complete fits nevertheless consumed
less time in these observed pairs because they avoided many first calls. This
tradeoff is why the option remains explicit and workload-specific.

Gaussian terminal host RSS was about 6.18 GiB static / 2.60 GiB dynamic, with
1251 / 411 registered TensorFlow functions. Beta-binomial was about 5.80 /
2.60 GiB and 1084 / 440 functions. GPU allocator peaks were roughly 42 / 39 MB
and 8.5 / 8.6 MB. These are descriptive process/allocator measurements, not
peak-memory guarantees. Every GPU run used trusted GPU1 access, float64,
XLA, TensorFlow 2.20.0 / TFP 0.25.0 and verified incremental memory growth.
CPU reference workers deliberately hid GPU devices.

## Source, verification and documentation

Baseline replay source identity is
`71ef22422903ca463337c99801ca632b91c7cba8675cb7d437bcfe43f70cce53`.
The exact full pairs used source-r7 identity
`628a0a3a2d1a6305c1df1b38b196bb45b12d2930f7712df0d9f7520f772e39c7`,
from commit `2921c2ffd` plus listed owned overlays. Final source-r10 identity is
`77786c75e2ae414e3b0883ff1b300b095d9e967a41d31c0e52e07ab0ea99746f`.
The only later numerical-package delta is a three-line completed-fit guard
rejecting reuse-policy relabeling; it cannot affect a fresh fit and has its own
test. Unrelated Q20, NeuTra and governance edits were excluded from all numerical
snapshots.

The primary regression set has 120 passing cases, including dynamic serial and
batched member export/reload, corruption, health, target/geometry/count isolation,
fresh evidence and exact resume. Additional checked suites have 73 pipeline/
process/profiling cases, 5 frozen-map/unsupported-route cases, 24 existing runner
compatibility cases and 15 documentation cases. These sets overlap. The two
initial regression failures were invalid zero-reserve fixtures and were repaired;
they did not execute incorrect sampler logic. Three early diagnostic setup
failures (helper import, CPU/GPU identity comparison, JSON tuple/list comparison)
remain preserved and charged.

The agent reference and official tuning chapter describe the same option and
its limits. `docs/main.tex` rebuilt to 574 pages with BibTeX and no unresolved
citations or references. PDF page 424 / printed 406 was visually checked; an
adjacent overfull identifier was rewritten without losing its meaning.

Exact commands, environments, root and folded seeds, target/data/prior lineage,
source checksums, wall time and artifacts are under
`artifacts/hmc-repair-master-2026-09-16/m30-r1/`. The compact terminal audit,
attempt-manifest summary, source manifests, raw-evidence hashes and guide
validation are committed; large raw tensor archives and source copies remain
local. Both full pairs use unchanged M29 design files; their replays contribute
no confirmation denominator.

## Decisions, budget and next phase

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit optional reuse for tested exact-score paths | Exact chunk and complete-fit parity passed | No failed numerical invariant or abnormal full-fit exit | Other targets/backends and workload costs | Keep default false; assess actual workload | Universal exact arithmetic or speed guarantee |
| Close M30 cost mechanism | Complete fits and resources measured | Source/stream comparisons valid | One paired fit per target | Use prices conservatively in next design | Statistically established speed superiority |
| Keep stopped coverage and subtle full-fit power open | Adequate fresh replication absent | Existing criterion unchanged | Operating characteristics and affordability | Execute reviewed M31 design investigation | Small passing pairs establish calibration |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Engineering parity, source/geometry, trace integrity, GPU policy and normal-exit checks passed |
| Statistically supported ranking | None |
| Descriptive differences | Observed runtime reduction about 45% / 50%; RSS and graph counts |
| Default readiness | No global tuning, posterior, validation or execution default changed |
| Next evidence needed | Affordable, source-grounded full-procedure calibration design; broader optional-path evidence |

All 30 outer attempts are terminal. Total charge is **552.365 CPU / 2084.990
GPU worker-seconds**, below the 1800/4800 phase ceilings. Failed attempts and
book-build retries are included once; nested child time is not double-counted.
Remaining campaign allowance is **74427.314 CPU / 75209.499 GPU seconds**
(20.67 / 20.89 hours).

At the observed reuse costs, 384 fits imply 36.97 Gaussian or 30.45 beta-binomial
GPU hours, each still above the remaining GPU allowance. These extrapolations
are not runtime ceilings. The reviewed
[M31 plan](bayesfilter-hmc-post-m30-next-phase-2026-09-23.md) examines the precise
statistical question and affordable alternative designs before further HMC
calibration. It keeps the original stopped-coverage requirement open.

Post-run review: the strongest alternative explanation for the timing difference
is order or host-load variation; the exact replay and graph inventory identify
a plausible mechanism but do not remove that uncertainty. Another target may
lose more in warmed execution than it saves in compilation. A complete-trace,
seed, health or posterior-decision mismatch would invalidate optional reuse for
that scope. The tested engineering mechanism passed; global exploration,
adequate stopped-interval/MCSE evidence, subtle full-fit power, exact MacroFinance
inputs and upstream learned-map quality remain distinct open requirements.
