# Controlled SMC / KDM: first execution tranche

29 September 2026, Hong Kong. Owner authorization: review and execute the
20260928 master. This activates that successor; iAPF remains incomplete and
deferred. No new approval tokens or per-attempt scientific approvals apply.

## Question and evidence contract

Can the pinned controlled-SMC author implementation execute on its neuroscience
data, and can an independently derived TensorFlow consumer reproduce its
quadratic policies, proposals and weights on fixed inputs? This tranche is
engineering and numerical validation, not a replicated performance ranking.

Reference: Heng et al., arXiv:1708.08396v3, Algorithms 1–3, section 4.4 and
Appendix I.3; author commit 2696f6d611aae172d10102cb7bfbfd314bb6b88f.
The repository NeuroScience Demo uses N=1024, I=2, all 3000 observations.
Section 6.1 instead uses I=3 and 100 independent repetitions. Passing the demo
does not complete that experiment. Lorenz and ToyMultimodal remain separate,
unexecuted scope cells. Published-versus-manuscript experiment mapping remains
open; no journal-wide replication claim is eligible.

The author reference is GNU Octave 6.4.0, an explicit MATLAB compatibility
exception. Isolated statistics 1.5.0 (GNU Octave upstream commit
bfb1b9f2e5b38a9200fba77e20d5e5517b839edf) requires Octave >=6.1.0. Only its
local M-file paths are added; no global package/environment mutation. Preserve
upstream code unchanged. Headless model configuration is derived from paper
Appendix I.3 and checked against Model_NeuroScience.m. No source-code license
was found for controlledSMC: execute it as an external reference; do not copy
or translate its implementation into BayesFilter. Derive the TF implementation
from the published mathematical expressions and document that provenance.

The reference uses alpha=.99, process variance=.11, initial N(0,1), binomial
counts with 50 trials and logistic response, and the pinned Data_NeuroScience.mat.
Seed 20260929 is a newly declared reproducibility seed, not a recovered author
seed. An initial N=256, 128-observation, I=2 prefix is a compatibility/cost
pilot only. Preserve its policy and SMC intermediates for fixed-input checks.
Run the full demo only if the pilot predicts it fits the remaining CPU tranche;
otherwise record the resource blocker without reducing the replication claim.

Primary numerical criterion: finite outputs; positive controlled precision;
Gaussian integral and normalized proposal identities; fixed-cloud residual-fit
and forward-weight agreement within 1e-7 absolute + 1e-7 relative in FP64.
Boundary tests must reject invalid precision and identify rank deficiency.
Check the actual TF consumer call chain, cumulative policy addition (equivalent
to multiplication of twists), initial normalizer, terminal potential, ancestry
and every-time resampling. A failed identity vetoes correctness of that lane
and triggers localization; it does not invalidate the paper.

Source normal equations and its scalar .001 precision projection are recorded
choices, not universal runtime defaults. Normal equations are a parity baseline;
condition/rank and projection diagnostics explain discrepancies. A candidate
solver or numerical safeguard must be explicitly classified and checked before
adoption. Never silently replace the source target or claim parity after tuning
a tolerance on the observed discrepancy.

Explanatory diagnostics: ESS, individual logZ, residuals, coefficient sizes,
condition estimates, runtime, tracing and allocator memory. None ranks methods.
The bootstrap iteration is the immediate comparator. Kalman/analytic Gaussian
fixtures certify identities only. The master requires EKF/UKF proposal and source
APF comparators before performance promotion; their absence here blocks that
claim, not this fixed-input validation. No trained-KDM, superiority, default,
HMC or complete-paper claim follows from this tranche.

K work stays independent: inspect/reuse existing normalized-density and Appendix
B code only under their actual model definitions. An unresolved coefficient or
training schedule remains an explicitly labeled hypothesis. Do not run a
surrogate experiment under Appendix B's name. A separate exact executable
contract is required before any K research-decision run.

## Budget, commands, artifacts and stop conditions

Live old-ledger reconciliation: 37 attempts, no running process;
CPU remaining 753.1669356211562 seconds, GPU remaining 130951.30035072038
seconds (36.375361 hours). The initial tranche caps are 300 numerical CPU
seconds, 7200 GPU seconds and four numerical attempts, including failures.
Development, source inventory and dependency fetching are not numerical trials.
Charge full subprocess wall time conservatively, including imports/compilation.

Root: docs/plans/artifacts/controlled-smc-kdm-replication-20260928/.
Use unique attempt directories and the existing runner's lock, process identity,
timeout, hashing and manifest helpers; retain the old ledger. Each launch records
exact command, source/data hashes, seed, Git commit and dirty source snapshots,
environment/device policy, wall charge, plan and result paths. No overwritten
evidence. Scripts and commands are registered in run_controlled_smc_kdm.py.

Stop an affected run on a wrong target/data, invalid reference output, corrupt
artifact, absent required diagnostics, timeout or resource exhaustion. Classify
the failure and repair locally within the same budget where possible. At four
attempts reassess cost and the diagnosed blocker before any further tranche.
GPU work must use escalated access, verified memory growth, stable tf.function
signatures and XLA. No pfor or NumPy runtime implementation.

## Skeptical pre-execution review

PASS for this bounded validation tranche. Material defects repaired: the draft
had no executable phases; MATLAB compatibility was assumed; demo I=2 was not
the section 6.1 I=3/100-repetition experiment; no package version was pinned;
source copying terms and independent mathematical implementation were conflated;
and completion could otherwise confuse source execution with replication.
The explicit comparator, fixed tolerance, cost pilot, artifact requirements and
stop conditions answer these defects. Exact author RNG equivalence, the journal
version mapping and full experiment costs remain open and cannot be inferred
from a successful command. KDM's measured ten-trial cost exceeds the budget.

Pre-mortem: a run could succeed while using the wrong time indexing or omitting
the initial twist integral; intermediate fixed-input comparisons catch this.
It could fail due to Octave compatibility or poor conditioning rather than the
method: retain logs, rank/precision diagnostics and the unchanged reference.
Repeated stochastic runs without an independent contract will not repair those
issues and are not authorized as blind retries.

Attempt 01: Octave lacks rng; no filter execution. Charged .214463 seconds.
Use separately seeded native rand/randn generators; author RNG equality is not
claimed. This localized compatibility repair leaves model/method/budget unchanged.

Attempt 02 reached reference completion but Octave lacks jsonencode, so output
serialization failed. The supervisor also failed to parse that empty file.
Repair: native MAT export and an explicitly diagnostic SciPy JSON adapter;
catch malformed JSON in supervision and classify failed artifacts. No runtime
NumPy dependency or numerical method change. Charge the unobserved-exit attempt
conservatively through the existing recovery rule.

## Four-attempt reassessment and C2 contract refinement
C1 pilot and full source demo PASS. Source algorithm cost was 26.0133 seconds
for H=3000,N=1024,I=2; no performance ranking follows. Two preceding runtime/
serialization failures were localized and repaired. Reassess PASS: permit four
further attempts (eight total) within the SAME 300 CPU / 7200 GPU-second tranche;
no compute expansion and no unresolved reference blocker.
C2 initially implements scalar affine Gaussian dynamics with an injected
observation likelihood. This is partial C2, not its promised multidimensional
model-adapter completion. Source code is not translated: square completion,
twisted potentials and backward expectation ratios are derived from the paper.
The primitive integral is checked by numerical quadrature, the actual forward
endpoint by recovered source draws/ancestry, and backward policy by saved
reference coefficients. Rank deficiency and zero precision must fail closed.
Normal equations and the explicit source_scalar_projection mode are parity
baseline choices, not general numerical defaults. The .001 margin is chosen
solely to reconstruct that declared source setting; broader adoption needs
target-specific non-harm calibration. Stable binomial log arithmetic is
algebraically equivalent and must satisfy the same frozen 1e-7 tolerances on
all reference clouds. No pfor or NumPy runtime path is present.

Attempt 05 stopped in reference-index ingestion: Octave exports integer-valued
ancestry as doubles, and TensorFlow refuses direct int32 constant conversion.
Use an explicit tensor cast; this is a harness repair, not a kernel change.

## Full-horizon GPU execution diagnostic
After unchanged fixed-input checks pass, run the scalar TF endpoint with all
3000 source observations, N=1024, I=2 and fresh stateless streams at each
refinement. Seed 20260929; streams 31–36. The primary criterion is finite valid
forward and fit outputs at the source horizon with one trace per signature.
This extends shape/resource validity; it is not another source-RNG parity test
or a statistical likelihood comparison. Record end logZ, ESS, design condition,
projection count, compile/steady execution times, host RSS and GPU allocator
peak. Cap 600 GPU seconds within the same tranche. Data MAT-to-JSON conversion
is an isolated diagnostic process; the TF candidate uses only TF arrays.
Review PASS: small-shape parity cannot certify full-horizon execution, so this
larger diagnostic answers a new question without repeating passed tests.

## Comparison ingestion correction
Post-run self-review found that tf.cast on expected Python lists could first
create float32 values. Kernel inputs were FP64, but several expected-output
lists were rounded before checking. Replace with tf.convert_to_tensor(...,
dtype=float64), preserve attempt 06, and run c2_parity_fp64. The frozen
tolerance is unchanged. This is required before closing fixed-input parity.
Attempt 07 full-horizon execution passed; it does not depend on list-comparator
accuracy and remains valid for its limited execution/finite-output claim.

## Eight-attempt reassessment
Attempt 08 corrected FP64 parity PASS; attempt 07 full-horizon GPU execution
PASS. Numerical CPU charged about 72.83 seconds (including conservative 45
seconds for the unobserved serialization exit), GPU about 52.28 seconds.
No continuation veto fired. Permit independent K1 plus up to three localized
repairs (12 total attempts) within the UNCHANGED 300 CPU/7200 GPU-second tranche.
K1 has its own exact source/gradient contract. No repeated empirical campaign,
compute increase or scientific promotion is implied.

## Full-horizon fixed-input closure
After attempt11, run c2_full_parity capped at 300 GPU seconds (including CPU
reference export), within the unchanged tranche and twelfth-attempt ceiling.
Extract only the required clouds/policies/weights/ancestries from attempt04's
preserved author-result.octave. Do not rerun or change the author algorithm.
Record that input's hash; keep the exported MAT/JSON and source snapshots.
Replay all three rounds and both backward fits at H=3000, N=1024. The frozen
1e-7 absolute plus relative tolerances and exact ancestry equality apply.
Primary criterion is every full-horizon replay/fit/weight check passing with
one trace per signature. A failure triggers localization, not tolerance
relaxation. Do not repeat the already-passed analytic, fresh-stream or rank
controls in this phase. This extends the parity horizon, not the empirical
claim; Figure 4's I=3/N=128 grid and independent repetitions remain unexecuted.
Skeptical review PASS: the prior short fixture could hide the same long-horizon
fit sensitivity that remained unresolved for iAPF. Replay at the actual demo
horizon answers that concern without statistical ranking or an altered target.

## Tranche closure
All declared validation questions now have passing artifacts, including the
separate Kalman and KDM contracts and full-horizon source replay. Twelve attempts
used 72.828086 CPU seconds and 105.044156 GPU-charged seconds; compute caps never
expanded. Four-to-eight and eight-to-twelve attempt reassessments are preserved
in opening-budget.json. The first-execution result records repairs, exact
evidence, limitations and remaining phases. This closes the first tranche only;
no empirical ranking or whole-replication completion follows.
