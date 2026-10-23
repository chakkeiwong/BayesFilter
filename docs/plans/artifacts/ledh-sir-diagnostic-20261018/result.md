# SIR diagnostic result — 18 October 2026

Scope correction after owner feedback: this diagnostic investigated an
agent-selected iteration comparison instead of the requested SIR test of
LEDH improvements against Zhao–Cui with OT fixed at (1,5). Its numerical
results remain recorded below, but it does not answer whether controlled-SMC
or KDM augmentation improves SIR score estimation.

The requested epsilon 1 / five Sinkhorn iterations has lower observed mean
per-run score error than 40 iterations on both SIR data sets. Improvement is
not statistically established: only three and two primary pairs completed.
On the second data set, the requested setting has higher observed error than
the simpler prior-map, defensive-local and equal-mixture comparators. The
equal-mixture comparison is 70.86 versus 45.78. These are descriptive results;
no candidate is promoted.

## Target and evidence scope

The target is `canonical_sir_d18_x0_then_transition_observe_v1` at theta zero,
T=50, N=1008, state dimension 18, observation dimension 9 and three parameters.
The consumer is `covariance_proposal_tf.make_filter`, with analytical recursive
scores and route `covariance_guided_defensive_mixture_finite_v1`. This is an
untuned, noncanonical diagnostic. No SIR-fitted controlled-SMC or KDM guide
exists in this test, so augmentation improvement was not evaluated.

The comparison changes the finite OT iteration count from 40 to five at
epsilon 1, holding the other consumer settings fixed. Both arms use beta
`[0.1, 0.04913405758312983, 0.8508659424168701]` for prior/global/local proposal
densities, flow count 16 and correction zero. These inherited controls are
hypotheses, not tuned SIR defaults. Exact transport chunks are K=N=1008.

The saved October Zhao–Cui rank-20 numerical comparator uses two fits and
100,000 paths per fit, with fit seeds 41 and 53. It is approximate, not an
exact oracle; rank, support and approximation bias are not bounded by its
between-fit standard errors. Reference:
`../ledh-independent-validation-20261008-01/report-20261009-011148/comparison.json`,
SHA-256 `3f0f6e4f591a724db52abe8a22435b85d5bcafd888efe566f27f9519ab806e01`.
Prepared data are the sibling campaign's
`inputs/sir_d18-d26100831/dataset.json` and `inputs/sir_d18-d26100832/dataset.json`.
No pre-21-August-2026 LEDH results are used.

## Scores and conditional errors

Scores below are means across the completed replications. MCSE is the
estimated standard error across those replications and is itself imprecise
with two or three observations. It excludes reference approximation error.

| Data seed | Score component | Numerical reference | Five iterations, mean ± MCSE | 40 iterations, mean ± MCSE |
| --- | --- | ---: | ---: | ---: |
| 26100831 | log kappa scale | -157.1104 | -149.4996 ± 33.9627 | -207.6012 ± 34.3750 |
| 26100831 | log nu scale | 104.5783 | 116.2436 ± 21.4590 | 143.7611 ± 16.8744 |
| 26100831 | log observation-noise scale | 56.8607 | 57.4181 ± 5.2029 | 60.9464 ± 4.9011 |
| 26100832 | log kappa scale | -135.1908 | -134.4702 ± 70.2752 | -181.3115 ± 71.3356 |
| 26100832 | log nu scale | 56.6963 | 64.9024 ± 0.0360 | 80.5482 ± 0.4810 |
| 26100832 | log observation-noise scale | 5.8875 | 2.2867 ± 1.4095 | 8.9347 ± 5.9883 |

The primary metric is the mean of each replication's Euclidean score error,
not the error of the mean score. Contrasts are paired within each data set.

| Data seed | Completed pairs / planned | Five iterations | 40 iterations | Paired five-minus-40 difference |
| --- | ---: | ---: | ---: | ---: |
| 26100831 | 3 / 8 | 58.1107 | 72.8357 | -14.7250 |
| 26100832 | 2 / 8 | 70.8590 | 77.6318 | -6.7727 |

The planned eight-pair design was not completed. No confidence intervals or
statistically supported rankings are issued. In particular, proximity of the
second data set's mean kappa score to the reference hides substantial
replication variability.

For fair comparison with the four simple arms, the following table uses only
the two common paired seeds per data set, including for the requested arm.
All entries are mean per-run Euclidean score errors; smaller is closer to
the fixed approximate reference.

| Arm; beta for prior/global/local | Data 26100831 | Data 26100832 |
| --- | ---: | ---: |
| Requested five; (0.1, 0.0491341, 0.8508659) | 62.23 | 70.86 |
| Prior-map; (1, 0, 0) | 63.52 | 65.21 |
| Defensive-global; (0.1, 0.9, 0) | 424250.19 | 11222.18 |
| Defensive-local; (0.1, 0, 0.9) | 73.73 | 69.97 |
| Equal-mixture; (1/3, 1/3, 1/3) | 1297.53 | 45.78 |

Prior-map still uses the OT reset; it is not a bootstrap particle filter.
Observed heuristic losses on data 26100832 prevent interpreting the smaller
five-versus-40 errors as adequate evidence of progress. The machine-readable
complete-design heuristic verdict remains `not_checked`, not a pass.

## Validity, repairs and budget

All 26 completed rows passed the recorded finite-value, horizon and validity
checks. The N32/T2 analytical-score smoke agreed with coordinate finite
differences: errors were about 2.46e-6 and 6.14e-7 at steps 1e-4 and 5e-5.
The focused CPU-only SIR analytical-recursion regression passed 2/2 tests.
These checks support implementation mechanics, not score accuracy at T50.

Across the completed traces, minimum ESS was 8.6364, maximum particle weight
0.26105, maximum Sinkhorn row-mass error 0.25022 and maximum column error
8.88e-16. Maximum reset mean error was 1.99e-13 and maximum relative covariance
error 7.01e-15, with no moment-roundoff warnings. The roundoff-sized moment
errors are not a blocker. The very large global-arm scores remain finite:
numerical validity does not imply an accurate score. OT mass errors and these
trace diagnostics are explanatory, not substitutes for score-error evidence.

| Attempt | Outcome | Charged GPU-worker seconds |
| --- | --- | ---: |
| 01 | Native exit 139 before a full row; WSL libcuda fault | 180.0000, conservative charge |
| 02 | GPU pinned by UUID; SIGSEGV (-11), zero full rows | 47.4224 |
| 03 | Command buffers disabled; normal worker exit, 26 full rows | 913.0771, supervisor wall time |
| Total / cap | Three of three launches used | 1140.4996 / 1200 |

Attempt 03's worker wall time was 911.2626 seconds and its status is
`budget_partial`: 26 of 96 planned arm evaluations completed. Its GDB wrapper
returned 1 because the final `bt` command found no stack after the worker
exited normally; this is not a fourth native failure. The remaining 59.5004
seconds do not authorize another launch; no fourth launch or scope extension
was made. Analysis took 0.0070 seconds within the separate 180-second CPU
analysis allowance. Prior CPU checks are recorded separately in the plan.
Disabling command buffers is a successful retry hypothesis, not proof of the
original crash's cause or default runtime performance.

## Reproducibility

Plan: `../../ledh-sir-diagnostic-20261018.md`. Branch
`ledh-refactor-with-policy-fix`; commit
`ea1018de1da55ed02a29a8d60f08a6827fce8a42`, with working-tree source hashes in
each manifest. Environment `tftwogpu`, TensorFlow `2.20.0-dev0+selfbuilt`,
float64 GPU/XLA with TF32 off: an explicit reference/debug exception to the
FP32/TF32 production direction. GPU was RTX 4080 SUPER, UUID
`GPU-68251639-fe82-8f81-3ccc-2953c32e805b`, with memory growth verified before
logical-device initialization and `TF_FORCE_GPU_ALLOW_GROWTH=true`.
Attempt 03 set `XLA_FLAGS=--xla_gpu_enable_command_buffer=` while retaining JIT.
Planned seeds were 261018301–308 and 261018401–408; actual seeds and every
completed row are preserved in [rows.jsonl](attempt-03/rows.jsonl).

The exact launch, worker command and elapsed time are preserved in
[attempt-03-supervisor.json](attempt-03-supervisor.json). The launch was:

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/plans/artifacts/ledh-sir-diagnostic-20261018/supervise_attempt.py --attempt attempt-03 --seconds 972.5 --native-backtrace
```

Analysis was deliberately CPU-only:

```text
env CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/run_ledh_sir_diagnostic.py analyze --output docs/plans/artifacts/ledh-sir-diagnostic-20261018/attempt-03
```

The runner was amended after worker completion only to make incomplete-design
inference explicit and include heuristic losses in the promotion condition.
The run manifest retains the original runner SHA-256
`bb0578e497e68f1fc0c0fbcbcc6f643f0b5450b8b63d781e737f3a319e011e8a`;
the amended analysis runner is
`8e5543c8a55221f844f85759cf16cb5f358eacd7addf59d1eec400dfde2e58a5`.
No GPU calculation was rerun after that analysis-only change.

## Decision and inference status

| Decision | Primary criterion | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Diagnostic only; no promotion | Not established; incomplete paired design | No completed-row hard validity veto; observed simple-arm losses; complete-design heuristic check unavailable | Few seeds and approximate reference | Plan a SIR-specific fitted-guide comparison with independent validation and sufficient replication | Canonical LEDH, exact-oracle agreement, default readiness, HMC or cSMC/KDM augmentation improvement |

| Evidence category | Status |
| --- | --- |
| Hard veto screen | Completed rows valid; two preceding infrastructure failures preserved |
| Statistically supported ranking | None; no candidate's superiority established |
| Descriptive-only differences | Lower five-versus-40 errors on both data sets; simple-arm losses on the second; extreme global-arm scores |
| Candidate viability | No completed-row validity rejection; requested arm is not promoted and heuristic adequacy is unresolved |
| Default readiness | Not evaluated: untuned, noncanonical route |
| Next evidence needed | SIR-specific fitted guidance, untouched evaluation, sufficient paired replications and stronger reference-error assessment |

Post-run red team: seed variability and reference approximation are the
strongest alternatives to an algorithmic interpretation. More independent
paired replications or a stronger reference could reverse these observed
contrasts. The weakest evidence is the two-pair conditional comparison. This
test rejects neither future fitted guidance nor the SIR research direction;
it does not establish improvement from the present settings.

Detailed machine-readable results: [analysis.json](attempt-03/analysis.json),
[run manifest](attempt-03/manifest.json), [smoke](attempt-03/smoke.json) and
[per-attempt result](attempt-03/result.md). Preflight and regression logs are
preserved alongside them. No commit, push or cross-branch sync was performed.
