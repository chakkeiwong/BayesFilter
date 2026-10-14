# iAPF phase 18: repaired TF32 consumer revalidation

## Question and scope

Does the repaired shared fixed-guide GPU/XLA consumer still have a systematic
likelihood error attributable to TF32, after separating arithmetic error from
changes in discrete resampling decisions? This follows master item 2. The
phase-17 QR rescue and iteration-cap remedies are already recorded in the
master; they remain diagnostic R options, not a change of fitting default.

Work exclusively in `.localresources/worktrees/ledh-iapf-execution-20260922`,
branch `ledh-refactor-with-policy-fix`, numerical base `89065bc6`. The root
surrogate-HMC checkout belongs to another agent. Keep all older results intact.

The target is the local six-parameter linear Gaussian model implemented by
`gaussian_tf.parameterized_model`, with an x0 draw and transition before y1.
It differs from the paper's first study. The claim-bearing consumer for this
diagnostic is the existing `make_fitted_twist_kernel`, reached by both local
fitted/adaptive adapters. No alternative runtime filter is to be constructed.
All gradients inspected here are fixed-label finite-program derivatives;
they are not the marginal model score.

## Evidence contract and intent ledger

- Baselines: independent R transition/filter calculations; GPU FP64; GPU FP32
  with TF32 disabled. Candidate: the SAME consumer in FP32 with TF32 enabled.
- Mechanism: `m + K(c-m)` is now `m - Km + Kc`, including the analytical
  tangent, to avoid a demonstrated time-center reuse in a Triton GEMM fusion.
- Engineering pass: independent old-fixture FP64 and ordinary FP32 value,
  cloud and derivative comparisons at their unchanged prior tolerances;
  center-probe controls at the existing 0.003 TF32 bound; finite results,
  verified GPU placement, XLA compilation and memory growth. Preserve strict
  TF32 full-path failures without relabeling them as passes.
- Primary precision criterion: within each fixed dataset/guide/scope, paired
  TF32-minus-FP64 AND TF32-minus-FP32 mean normalized likelihood differences
  must have simultaneous 99% intervals wholly inside [-0.02, 0.02]. This is
  a declared 2% engineering error budget, not a paper tolerance or an assertion
  of unbiasedness. Use two-sided Student intervals with Bonferroni allocation
  across all planned contrasts; bootstrap/tail diagnostics can veto inference.
  Failure to establish equivalence is an unresolved precision veto, not proof
  of a compiler defect or a continuation veto.
- Explanatory diagnostics: normalized likelihood mean, RMSE, MC standard
  error, log error, maximum contribution to squared paired error, actual
  ancestor/mixture changes, and cloud error before the first changed decision.
  If one replicate contributes >20% of squared paired deviation, intervals are
  explicitly tail-unresolved and cannot establish the precision criterion.
- Validity/continuation veto: nonfinite outputs, missing/corrupted evidence,
  non-GPU execution, missing XLA/growth provenance, failed independent FP64
  controls, exhausted phase/campaign budget, or a broken model/timing contract.
  Local harness/serialization problems trigger repair and retry with a fresh
  attempt directory within the same budget. A candidate precision failure
  triggers localization; it does not invalidate unaffected R/FP64 evidence.
- No conclusions about a new default, all historical results, adaptive
  stopping equivalence, high-dimensional paper replication, marginal score,
  KDM, canonical LEDH or HMC. Only scopes actually rerun gain new evidence.

## Fixed design and cheap adversaries

First replay the old d=1,2,5, N=64, T=5 R bridge fixture, including constant
and positive-floor guides and the time-varying-center reproducer. Do not run
the old bridge supervisor: it now has a stale assertion that the already
removed multidimensional adapter restriction must still be present.

For distributional checks use 512 fresh particle streams per cell. Cells:
the three old datasets; two new datasets at each old d=1,2,5 scope; and two
new d=5, N=128, T=25 datasets. Each new dataset is simulated from the actual
local model, with theta=(.55, log(.6), log(.7), 1, .2, log(.8)). Seeds are
2026092300+d and 2026092400+d for T=5; 2026092505 and 2026092605 for T=25.
Particle seeds bind the cell and replicate, independent of dataset generation.

Freeze four guide arms before particle evaluation: the old positive-floor
fixture guide (new datasets use the same declared random-center construction),
constant/BPF (no future information), current-observation Gaussian (cheap
analytic adaptation), and full-future Gaussian oracle (conditional exact
lookahead). These three controls are constructed from this model, not named
as generic categories. They test poor guide, short/long horizon and dimension
conditions separately. The arbitrary fixture guide is a compiler stress case,
not a trained candidate. Record the conditional heuristic-dominance table for
each precision; any observed loss blocks filter-quality promotion, which is
not sought here. Oracle agreement is also checked with an independent R
Kalman likelihood using the correct first propagated mean/covariance.

Use shared FP32-representable normal draws and 23-bit uniform lattice draws,
cast to FP64 for that arm, so draw quantization cannot masquerade as TF32
error. Fixed parameters, observations and guides likewise use identical
FP32-representable inputs for distributional comparisons. Retain unrounded
old-fixture inputs in its separate strict R replay. Separate worker processes
for each precision avoid cached graphs inheriting another global TF32 mode.
Trace the first 8 streams per cell/arm and verify tracing is numerically neutral.

## Assumptions, pre-mortem and skeptical review (before implementation)

| Choice | Provenance/status | Failure risk | Early check |
|---|---|---|---|
| Repaired shared kernel | Previously checked algebraic repair | Endpoint bypass or stale source | Source closure/hash and actual factory call |
| Small d and N | Old failing compiler fixtures; diagnostic scope | Cannot transfer to paper d80 | Scope-specific report; no transfer claim |
| T25 extension | Hypothesis: accumulation/branching amplifies precision error | Low particle count creates heavy tails | Report tails and oracle baseline before precision interpretation |
| Random-center positive floor | Reproduction of old stress construction | Poor likelihood efficiency | Three conditional controls; no guide-quality claim |
| No clipping/damping changes | Freeze the existing numerical object | Nonfinite precision/covariance | Fail closed on nonfinite results; no silent numerical repair |
| 512 streams; 2% margin | Bounded diagnostic precision budget | Low power or tail-dominated mean | Simultaneous intervals and explicit tail veto; unresolved is allowed |
| Independent R Kalman | Exact linear Gaussian model | x0/y1 timing mismatch | Compare full-joint Gaussian likelihood independently on small scopes |
| Student intervals | Approximate Monte Carlo uncertainty | Heavy tails invalidate approximation | Tail contribution guard, descriptive-only flags and bootstrap sensitivity |
| RTX4080, XLA, growth | Available prior failing hardware and repository policy | Compiler/device differences | Elevated probe, version/HLO/device/memory records |

Skeptical audit: PASS for this bounded diagnostic. The initial proposed reuse
of the old bridge supervisor is rejected because its scalar-veto assertion is
stale; reuse its independent fixtures and primitive probes instead. A changed
resampling path cannot be used as evidence of likelihood bias. Conversely,
mean agreement cannot erase strict path/derivative mismatches. Poor random
guides and initial Monte Carlo variation mean even the full-guide consumer is
not a zero-variance estimator here; do not incorrectly demand deterministic
Kalman equality. Frozen FP64/FP32 baselines and three controls distinguish
precision defects from finite-particle error. No independent-review claim.

## Execution, budget and artifacts

Versioned root: `docs/plans/artifacts/iapf-tf32-revalidation-20260923-01/`.
Keep source snapshots, exact commands, inputs, JSON/CSV results, full logs,
source/output SHA256, device/compiler provenance, decision and inference tables.
Python: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`; R: `/usr/bin/Rscript`.
Use GPU UUID `GPU-68251639-fe82-8f81-3ccc-2953c32e805b`; the RTX5080 is occupied.
Set `TF_FORCE_GPU_ALLOW_GROWTH=true`, configure and verify the repository growth
helper before initialization, `BAYESFILTER_PRELOAD_CUSTOM_OP=0`, CPU BLAS/OMP
threads=1, TF intra/inter-op threads=2. GPU probes/runs require escalation.
R/reference/report jobs explicitly hide GPU with `CUDA_VISIBLE_DEVICES=-1`.

Phase caps: 7,200 GPU process seconds and 3,600 CPU single-thread worker seconds,
maximum 12 attempts including repairs. GPU process wall includes host overhead;
CPU reference/report wall is charged separately. Reserve/settle each attempt;
all failed attempts consume budget. Record a conservative 30 GPU seconds for
the initial NVIDIA availability probe. Starting campaign balances from phase17:
64923.84917570048 CPU seconds, 171882.52032958285 GPU seconds, no wall expiry.
Existing owner execution authorization applies; no renewed local approval.

Implement a dedicated diagnostic exporter/runner/reporter, syntax-check it,
run R fixtures and trusted GPU probes, replay old controls, then run the three
precision workers. Review integrity and uncertainty before interpreting.
If a localized implementation failure occurs, preserve it, fix only its cause,
rerun focused checks, and continue. Finish by refreshing the master and concise
checkpoint with the result and next discriminating action.
