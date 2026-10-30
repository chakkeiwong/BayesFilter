# Tuned N=1008 range-bearing result

The new covariance-guided defensive-mixture filter has conditional evidence of
smaller score error than the tuned previous marginal-mixture LEDH route at the
difficult parameter point on two of three datasets. It is not uniformly better:
truth-point differences are unresolved, the third difficult cell has high
variance, and the simple prior proposal has lower observed mean error in four
cells. This is a filtering/value/score diagnostic preparing surrogate-HMC
testing. No HMC chain was run here.

## Target and tuning

M13 is the existing wrapped range-bearing constant-velocity model: state
dimension 4, observation dimension 2, parameter dimension 2, T=20, N=1008, K=N.
Parameters are log process/observation noise scales. Truth is (0,0); the difficult
point is (-0.73272085,-0.01465508). Shared model callback value/tangent parity is
exact in the executable check. The intended score is the derivative of the model
log marginal likelihood. The filters compute analytical total derivatives of
their own finite fixed-design value programs. Equality to the model score is
not established; the bootstrap/Fisher reference retains finite-particle bias.

Both arms were tuned for this exact FP64/TF32-off GPU/XLA scope on calibration
datasets 18010/18011 and validation datasets 18012/18013, then frozen before
held-out datasets 1704/18020/18021. New controls: 16 flow steps, 80 reset steps,
epsilon 1, correction_steps 0; beta=(0.1,0.0251769507,0.8748230493). This does not
test optional GenUT moment correction. Previous-arm selected controls 1 are
preserved in tuning.json. Validation differences among 40/80/120 reset steps
were about 1e-10; selecting 80 is not evidence that 80 is intrinsically better.

## Conditional results

Errors are mean score-L2 distances to the same independent FP64 reference,
over eight paired fixed designs. Negative paired new-minus-old intervals
favor the new method.

| Data | Point | New error | Previous error | Prior error | Paired 95% interval | Interpretation |
|---|---|---:|---:|---:|---|---|
| 1704 | truth | 1.2735 | 1.2476 | 1.2136 | [-0.836, 0.888] | no supported ranking |
| 1704 | difficult | 5.5841 | 26.4270 | 3.5094 | [-36.144, -5.542] | conditional improvement |
| 18020 | truth | 0.9823 | 1.6885 | 1.4643 | [-1.504, 0.092] | no supported ranking |
| 18020 | difficult | 1.3996 | 22.5433 | 2.2369 | [-33.706, -8.582] | conditional improvement |
| 18021 | truth | 1.6403 | 3.1934 | 1.5851 | [-3.673, 0.567] | no supported ranking |
| 18021 | difficult | 4.4376 | 133.5798 | 3.9305 | [-270.652, 12.368] | no supported ranking |

The two supported difficult cells have about 79% and 94% smaller observed mean
score errors. The third has about 97% smaller mean error, but high variance
leaves its interval crossing zero. Mean absolute log-likelihood errors are
smaller for the new arm in all six cells; value differences are descriptive.

These are nominal small-sample Student intervals, conditional on each dataset.
The predeclared extension from four to eight designs did not retune. These are
exploratory intervals, without sequential/multiplicity correction; they do not
establish an across-dataset or all-regime ranking. Each sensitivity envelope
expands by twice the larger of the reference refinement displacement and three
score-MCSE vector norms. That is not a rigorous bound on reference particle bias.

Independent references have 16 bootstrap/Fisher replicates at N=16384 and
65536, plus same-N1008 references for the cheap comparator. Raw values, scores,
replicate references and uncertainty remain in the final artifact directory.

| Data | Point | Score MCSE components | Reference refinement displacement |
|---|---|---|---:|
| 1704 | 0 | (0.0650, 0.0227) | 0.1494 |
| 1704 | 1 | (0.2372, 0.2197) | 0.2093 |
| 18020 | 0 | (0.0796, 0.0298) | 0.2151 |
| 18020 | 1 | (0.1478, 0.0618) | 0.4760 |
| 18021 | 0 | (0.0583, 0.0514) | 0.1711 |
| 18021 | 1 | (0.3985, 0.1520) | 0.1116 |

## Numerical validity and interpretation

All 48 new and 48 previous evaluations satisfy the validity guards. The first
four designs reproduce exactly in the extension (maximum difference
0.0). The global-guide adversary fails 5/48 evaluations, all at
the difficult point; its partial means cannot pass its screen. That failure
does not invalidate complete primary pairs. The prior adversary is valid 48/48
and has lower descriptive mean error in four cells: the declared broad-promotion
veto, not a statistical ranking. Neither method uniformly dominates heuristics.

Before repair, FP32/TF32 failed covariance restoration; FP32 without TF32 and
FP64 failed mean restoration. More reset iterations alone did not repair it.
The shared reset now explicitly recentres transformed deviations and their
analytical tangents, an identity in exact arithmetic; guards stay unchanged.
Existing healthy reset/recursive finite-derivative tests, including LGSSM/KSC,
pass 20/20. M13 N1008 FP32-no-TF32 T20 smoke passes; FP64 T2 finite differences
have maximum scaled errors 1.29e-8 and 2.83e-9. Repaired FP32/TF32 has not had a
fresh tuned held-out run, so the supported result is FP64 diagnostic evidence.

Reporter completeness was repaired to separate primary and heuristic arms.
The initial four-design report withheld two primary intervals solely because
global-guide failed. Prior artifacts remain preserved; corrected eight-design
report has separate primary/heuristic flags. Five synthetic reporter tests pass.
The first synthetic fixture accidentally made the bootstrap adversary exact,
correctly firing a veto; that fixture was repaired and both attempts retained.

## Decisions

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Scoped comparison completed | Six complete eight-design primary cells | New/previous pass | Conditional finite-design uncertainty | Preserve result | HMC convergence or posterior accuracy |
| Help in some difficult cases | Two negative reference-sensitive intervals | No primary validity failure | Sequential look, multiplicity, reference bias | Candidate for later target-specific testing | Uniform or all-model improvement |
| Broad/default promotion withheld | Truth unresolved; third difficult cell variable | Prior losses in four cells; guide screen incomplete | Generalization/FP32 | Fresh tuned FP32 scope is a remaining item | Default readiness |

| Inference status | Verdict |
|---|---|
| Hard veto screen | New/previous pass 48/48; global-guide fails 5/48 |
| Statistically supported ranking | Exploratory conditional evidence in two difficult cells, subject to Student/reference assumptions |
| Descriptive-only differences | Other cells, value errors, runtime, heuristic means |
| Default-readiness | Not established |
| Next evidence needed | Repaired FP32/TF32 scope; independent replication/heuristics before broad promotion; actual HMC for posterior claims |

## Provenance and terminal audit

Interpreter: /home/chakwong/anaconda3/envs/tftwogpu/bin/python.
TensorFlow 2.20.0-dev0+selfbuilt, escalated RTX4080SUPER GPU0, XLA,
FP64/TF32off; verified memory growth. HEAD 10884528ca8b1a7463895256d30b9655d69c0955;
dirty-source checksums preserved in each worker manifest. Extension checked all
numerical dependency hashes against original tuning before reuse. Peak final
TF allocator 271305984 bytes. Original comparison
536.404s; extension 385.568s. Aggregate numerical
attempts plus healthy CPU regression about 1991.832s (33.20min)
of 4h: 23 numerical workers plus one healthy CPU regression. Routine reporting
tests are separately logged. The planned four-to-eight design stopping point
is reached; unused time does not require launching HMC.

Original tuning/checks: artifacts/ledh-range-bearing-covariance-20261009-06/.
Final report/raw rows/references/manifests:
artifacts/ledh-range-bearing-covariance-20261009-07/.
Healthy reset regression:
artifacts/ledh-range-bearing-centering-regression-20261009-01/.
Reporter regression:
artifacts/ledh-range-bearing-report-regression-20261010-02/.
Exact commands are in execution.json and worker manifests.

Skeptical audit: comparator is the tuned previous repair, not the earlier broken
score; heldout cannot retune; derivative parity is not model-score accuracy;
FP64 is not relabelled FP32; failed heuristics do not erase primary pairs; simple
prior losses remain prominent. Strongest alternative explanation is heavy-tailed
previous-filter errors in selected difficult regimes, controlled by defensive
sampling that can also work with a simple prior proposal. New local flow
necessity is unproved. Fresh default-precision or independent data reversing
contrasts would overturn promotion arguments. Genealogical reference bias and
small paired sample size are the weakest accuracy evidence.

Variance and particle-count assessment: [2026-10-10 follow-up](ledh-range-bearing-variance-assessment-2026-10-10.md).
